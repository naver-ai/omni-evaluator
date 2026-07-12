# OmniEvaluator
# Copyright (c) 2026-present NAVER Cloud Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
import httpx
import itertools
import json
import logging
import numpy as np
import os
from pathlib import Path
import PIL
from PIL import Image
import random
import re
import requests
import time
import threading
import torch
import traceback
from transformers import AutoTokenizer
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Iterable
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

from omni_evaluator.inference.vllm.completions import (
    batch_completion_sync, batch_completion_async,
)
from omni_evaluator.inference.vllm.chat_completions import (
    batch_chat_completion_sync, batch_chat_completion_async,
)
from omni_evaluator.inference import (
    NUM_DEBUG_SAMPLES,
    DEFAULT_MULTIPLE_CHOICE_OPTIONS,
)
from omni_evaluator.modules.image_generation import T2IGenerator
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.generation_options import VllmGenerationOptions
from omni_evaluator.schemas.task import TaskConfig
from omni_evaluator.inference import healthcheck

STOP_WORDS = ["<|im_end|>", "<|endofturn|>", ]
    
    
def main(
    rank: int,
    world_size: int,
    run_index: int,
    num_runs: int,
    inference_engine: str,
    evaluation_engine: str,
    benchmark: str,
    evaluation_method: str,
    benchmark_idx: int,
    num_benchmarks: int,
    benchmark_dataset: Iterable,
    task_config: TaskConfig,
    default_generation_options: Dict[str, Any],
    url: Optional[str],
    benchmark_dataset_size: Optional[int] = None,
    api_version: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    model_name_or_path: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    hf_hub_cache: Optional[str] = None,
    skip_chat_template: Optional[bool] = False,
    reasoning: Optional[Union[str, bool]] = False,
    add_generation_prompt: Optional[bool] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    media_io_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    max_video_frames: Optional[int] = None,
    fps: Optional[float] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    allowed_local_media_path: Optional[str] = None,
    seed: Optional[str] = None,
    t2i_generator: Optional[T2IGenerator] = None,
    batch_size: Optional[int] = 1,
    timeout: Optional[Union[int, float]] = 30,
    socket_timeout: Optional[Union[int, float]] = 900,
    max_retry: Optional[int] = 5,
    wait_between_retry: Optional[int] = 2,
    do_async: Optional[bool] = False,
    debug: Optional[bool] = False,
    verbose: Optional[bool] = True,
    thread_lock: Optional[threading.Lock] = None,
    thread_barrier: Optional[threading.Barrier] = None,
    *args, **kwargs,
) -> Optional[List[Dict[str, Any]]]:
    # Orchestrate batch inference for a benchmark dataset using vLLM (local or remote).
    # Args: benchmark_dataset - iterable of evaluation records, task_config - benchmark task configuration,
    #   default_generation_options - sampling params, url - vLLM server endpoint (None for local LLM),
    #   skip_chat_template - if True use raw prompt instead of chat format, do_async - run requests concurrently
    # Returns: list of record dicts with predictions, or None on complete failure
    if world_size > 1 and thread_lock is not None:
        thread_lock.acquire()

    model, tokenizer = None, None
    if isinstance(url, str):
        if not healthcheck(
            url=url,
            max_retries=20,
            interval=30,
            token=os.getenv("VLLM_API_KEY"),
        ):
            raise RuntimeError(f'Failed to healthcheck: {url}')
    else:
        from omni_evaluator.utils.optional_import import require_package
        require_package("vllm", feature="vLLM local inference")
        from vllm import LLM, SamplingParams
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=hf_hub_cache,
            trust_remote_code=trust_remote_code,
        )
        model = LLM(
            model=model_name_or_path,
            enable_sleep_mode=True,
            tensor_parallel_size=1,
            distributed_executor_backend="external_launcher",
            dtype=torch_dtype,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            max_model_len=4096,
            seed=seed,
            download_dir=hf_hub_cache,
        )

    if world_size > 1:
        if thread_lock is not None:
            thread_lock.release()
        if thread_barrier is not None:
            thread_barrier.wait()

    # Single-pass prep loop accumulating only `records`. messages / prompts /
    # generation_options / options / tools are derived inside chat_completion /
    # batch_*_completion from each record, so we do not maintain parallel
    # per-request lists here.
    _tqdm_total = benchmark_dataset_size if benchmark_dataset_size is not None else task_config["num_records"]
    records = list()
    for record_idx, record in tqdm(
        enumerate(benchmark_dataset),
        initial=0,
        total=_tqdm_total,
        desc=f'({benchmark_idx+1:02}/{num_benchmarks:02}) (rank: {rank+1:02}/{world_size:02}) (run: {run_index+1:02}/{num_runs:02}) Prepare inference {inference_engine} - {benchmark}/{evaluation_method}',
    ):
        if debug and record_idx >= NUM_DEBUG_SAMPLES:
            break
        if record is None:
            logger.warning(f"({rank+1:03}/{world_size:03}) invalid record: {benchmark}/{record_idx}")
            continue

        _record = copy.deepcopy(record) # to prevent pointer intervention
        _generation_options = copy.deepcopy(default_generation_options)
        if _record.generation_options:
            for _k, _v in _record.generation_options.items():
                if (
                    _k not in _generation_options
                    or _generation_options[_k] is None
                ):
                    _generation_options[_k] = _v
        _record.generation_options = VllmGenerationOptions.from_dict(obj=_generation_options)
        records.append(_record)

    # Resolve media_io_kwargs and mm_processor_kwargs.
    # Precedence (highest to lowest):
    #   1. CLI per-field overrides (max_video_frames, fps, min_pixels, max_pixels)
    #   2. CLI raw dicts passed in (mm_processor_kwargs, media_io_kwargs)
    #   3. task_config.inference.{media_io_kwargs, mm_processor_kwargs}
    # None at any layer means "don't touch" — downstream falls through to the next.
    _task_inference = getattr(task_config, "inference", None)
    # media_io_kwargs: nested dict keyed by modality; merge sources in precedence order (lowest first).
    _media_io_kwargs: Dict[str, Dict[str, Any]] = dict()
    for _modality_kwargs in (
        getattr(_task_inference, "media_io_kwargs", None),
        media_io_kwargs,
    ):
        if not isinstance(_modality_kwargs, dict):
            continue
        for _modality, _kwargs in _modality_kwargs.items():
            if isinstance(_kwargs, dict):
                _media_io_kwargs.setdefault(_modality, dict()).update(_kwargs)
    if max_video_frames is not None:
        _media_io_kwargs.setdefault("video", dict())["num_frames"] = max_video_frames
    if fps is not None:
        _media_io_kwargs.setdefault("video", dict())["fps"] = fps
    media_io_kwargs = None
    if _media_io_kwargs:
        media_io_kwargs = _media_io_kwargs

    # mm_processor_kwargs: flat dict; merge sources in precedence order (lowest first).
    _mm_processor_kwargs: Dict[str, Any] = dict()
    for _modality_kwargs in (
        getattr(_task_inference, "mm_processor_kwargs", None),
        mm_processor_kwargs,
    ):
        if isinstance(_modality_kwargs, dict):
            _mm_processor_kwargs.update(_modality_kwargs)
    if min_pixels is not None:
        _mm_processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        _mm_processor_kwargs["max_pixels"] = max_pixels
    mm_processor_kwargs = None
    if _mm_processor_kwargs:
        mm_processor_kwargs = _mm_processor_kwargs

    try:
        if not skip_chat_template:
            if do_async:
                records = asyncio.run(batch_chat_completion_async(
                    url=url,
                    records=records,
                    evaluation_method=evaluation_method,
                    reasoning=reasoning,
                    api_version=api_version,
                    chat_template_kwargs=chat_template_kwargs,
                    mm_processor_kwargs=mm_processor_kwargs,
                    media_io_kwargs=media_io_kwargs,
                    allowed_local_media_path=allowed_local_media_path,
                    verbose=verbose,
                    semaphore_size=batch_size,
                    timeout=timeout,
                    socket_timeout=socket_timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                ))

            else:
                records = batch_chat_completion_sync(
                    url=url,
                    records=records,
                    evaluation_method=evaluation_method,
                    reasoning=reasoning,
                    api_version=api_version,
                    chat_template_kwargs=chat_template_kwargs,
                    mm_processor_kwargs=mm_processor_kwargs,
                    media_io_kwargs=media_io_kwargs,
                    allowed_local_media_path=allowed_local_media_path,
                    verbose=verbose,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                )

        else:
            if do_async:
                records = asyncio.run(batch_completion_async(
                    url=url,
                    records=records,
                    tokenizer=tokenizer,
                    evaluation_method=evaluation_method,
                    reasoning=reasoning,
                    api_version=api_version,
                    verbose=verbose,
                    semaphore_size=batch_size,
                    timeout=timeout,
                    socket_timeout=socket_timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                ))

            else:
                records = batch_completion_sync(
                    url=url,
                    records=records,
                    tokenizer=tokenizer,
                    evaluation_method=evaluation_method,
                    reasoning=reasoning,
                    api_version=api_version,
                    verbose=verbose,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                )

    except Exception as ex:
        logger.warning(f'{benchmark}/{evaluation_method} failed API request - {type(ex)}: {str(ex)}')
        traceback.print_exc()
        records = None

    if records is None or len(records) < 1:
        logger.error(f"{benchmark}/{evaluation_method} failed to batch_inference")
        return None
    return records