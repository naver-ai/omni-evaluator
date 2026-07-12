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
import logging
import time
import threading
from tqdm import tqdm
import traceback
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Iterable

logger = logging.getLogger(__name__)

from omni_evaluator.api.chat_completions import (
    batch_chat_completion_sync, 
    batch_chat_completion_async,
)
from omni_evaluator.inference import NUM_DEBUG_SAMPLES
from omni_evaluator.modules.image_generation import T2IGenerator
from omni_evaluator.schemas.task import TaskConfig

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
    api_name: str,
    benchmark_dataset_size: Optional[int] = None,
    reasoning_options=None,
    t2i_generator: Optional[T2IGenerator] = None,
    batch_size: Optional[int] = 1,
    timeout: Optional[Union[int, float]] = 30,
    max_retry: Optional[int] = 5,
    wait_between_retry: Optional[int] = 2,
    do_async: Optional[bool] = False,
    debug: Optional[bool] = False,
    verbose: Optional[bool] = True,
    thread_lock: Optional[threading.Lock] = None,
    thread_barrier: Optional[threading.Barrier] = None,
    *args, **kwargs,
) -> Optional[List[Dict[str, Any]]]:
    # Orchestrate batch inference for a benchmark dataset using external API providers (OpenAI, Anthropic, Google).
    # Args: benchmark_dataset - iterable of evaluation records, task_config - benchmark task configuration,
    #   default_generation_options - sampling params, api_name - API provider identifier,
    #   do_async - run requests concurrently, batch_size - concurrency limit for async mode
    # Returns: list of record dicts with predictions, or None on complete failure

    # Single-pass prep loop accumulating only `records`. messages /
    # generation_options / options / tools are derived inside chat_completion /
    # batch_*_completion from each record, so we do not maintain parallel
    # per-request lists here.
    _tqdm_total = benchmark_dataset_size if benchmark_dataset_size is not None else task_config.num_records
    records = list()
    _num_dropped_prep = 0
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

        # Per-record try/except: a single multimodal fetch failure (e.g. a
        # transient ``ChunkedEncodingError`` while downloading an image)
        # used to abort the entire batch via the outer ``except`` below.
        # With this guard the failed record is dropped and the rest of the
        # batch proceeds — only the inference engine layer's catastrophic
        # failures fall through to the outer try/except.
        try:
            _record = copy.copy(record) # to prevent pointer intervention (shallow — see above)
            _generation_options = copy.deepcopy(default_generation_options)
            if _record.generation_options:
                for _k, _v in _record.generation_options.items():
                    if (
                        _k not in _generation_options
                        or _generation_options[_k] is None
                    ):
                        _generation_options[_k] = _v
            _record.generation_options = _generation_options
            records.append(_record)
        except Exception as ex:
            _num_dropped_prep += 1
            logger.warning(
                f"{benchmark}/{evaluation_method} rec_idx={record_idx} dropped during prep: "
                f"{type(ex).__name__}: {ex}"
            )
            traceback.print_exc()
            continue

    if _num_dropped_prep:
        logger.warning(
            f"{benchmark}/{evaluation_method} dropped {_num_dropped_prep} records during prep; "
            f"{len(records)} remaining"
        )

    if not records:
        logger.error(f"{benchmark}/{evaluation_method} failed to batch_inference (no records after prep)")
        return None

    # Narrow the outer try to the actual batch call. ``batch_chat_completion_*``
    # already swallows per-record failures internally (each fail returns an
    # ``*InferenceOutput(error_message=...)`` merged onto the record); the only
    # exceptions that bubble up here are catastrophic (asyncio loop failure,
    # SDK construction error, etc.), which warrant the whole-task abort.
    try:
        if do_async:
            records = asyncio.run(batch_chat_completion_async(
                api_name=api_name,
                records=records,
                reasoning_options=reasoning_options,
                evaluation_method=evaluation_method,
                verbose=verbose,
                semaphore_size=batch_size,
                timeout=timeout,
                max_retry=max_retry,
                wait_between_retry=wait_between_retry,
            ))
        else:
            records = batch_chat_completion_sync(
                api_name=api_name,
                records=records,
                reasoning_options=reasoning_options,
                evaluation_method=evaluation_method,
                verbose=verbose,
                timeout=timeout,
                max_retry=max_retry,
                wait_between_retry=wait_between_retry,
            )
    except Exception as ex:
        logger.warning(f"{benchmark}/{evaluation_method} failed API request - {type(ex)}: {str(ex)}")
        traceback.print_exc()
        records = None

    if records is None or len(records) < 1:
        logger.error(f"{benchmark}/{evaluation_method} failed to batch_inference")
        return None
    return records