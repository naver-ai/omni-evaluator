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
import itertools
import json
import logging
import numpy as np
import os
from pathlib import Path
import time
import threading
import torch
import traceback
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Iterable
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

from omni_evaluator.inference import (
    NUM_DEBUG_SAMPLES,
    DEFAULT_MULTIPLE_CHOICE_OPTIONS,
    apply_chat_template_kwargs,
)
from omni_evaluator.inference.sglang.completions import (
    batch_completion_sync, batch_completion_async,
)
from omni_evaluator.inference.sglang.chat_completions import (
    batch_chat_completion_sync, batch_chat_completion_async,
)
from omni_evaluator.modules.image_generation import T2IGenerator
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.generation_options import SglangGenerationOptions
from omni_evaluator.schemas.task import TaskConfig
from omni_evaluator.inference import healthcheck


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
    model_name_or_path: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    skip_chat_template: Optional[bool] = False,
    reasoning: Optional[Union[str, bool]] = False,
    add_generation_prompt: Optional[bool] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
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
    # Orchestrate batch inference for a benchmark dataset using SGLang server.
    # Args: benchmark_dataset - iterable of evaluation records, task_config - benchmark task configuration,
    #   default_generation_options - sampling params, url - SGLang server endpoint,
    #   skip_chat_template - if True use raw prompt instead of chat format, do_async - run requests concurrently
    # Returns: list of record dicts with predictions, or None on complete failure
    if not healthcheck(
        url=url,
        max_retries=20,
        interval=30,
        token=os.getenv("SGLANG_API_KEY"),
    ):
        raise RuntimeError(f'Failed to healthcheck: {url}')

    # Derive chat_template_kwargs from reasoning/add_generation_prompt if not explicitly provided
    if chat_template_kwargs is None:
        _temp = apply_chat_template_kwargs(
            {}, reasoning=reasoning, add_generation_prompt=add_generation_prompt,
        )
        chat_template_kwargs = _temp.get("chat_template_kwargs") or None

    # Single-pass prep loop accumulating only `records`. messages / prompts /
    # generation_options / options / tools are derived inside chat_completion /
    # batch_*_completion from each record, so we do not maintain parallel
    # per-request lists here.
    _tqdm_total = benchmark_dataset_size if benchmark_dataset_size is not None else task_config.num_records
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
        _record.generation_options = SglangGenerationOptions.from_dict(obj=_generation_options)
        records.append(_record)

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
