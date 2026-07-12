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

import argparse
import asyncio
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime
import logging
import os
from pathlib import Path
import PIL
from PIL import Image
import re
import shutil
import sys
import threading
import traceback
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Iterable

from omni_evaluator import ApiGroup, EvaluationEngine, EvaluationMethod, InferenceEngine, T2IGeneratorType
from omni_evaluator.api.chat_completions import (
    chat_completion_sync as chat_completion_sync_api,
    chat_completion_async as chat_completion_async_api,
)
from omni_evaluator.args import get_parser
from omni_evaluator.clients.s3_client import S3Client
from omni_evaluator.inference import DEFAULT_NUM_RUNS
from omni_evaluator.inference.vllm.chat_completions import (
    chat_completion_sync as chat_completion_sync_vllm,
    chat_completion_async as chat_completion_async_vllm,
)
from omni_evaluator.inference.vllm.completions import (
    completion_sync as completion_sync_vllm,
    completion_async as completion_async_vllm,
)
from omni_evaluator.inference.sglang.chat_completions import (
    chat_completion_sync as chat_completion_sync_sglang,
    chat_completion_async as chat_completion_async_sglang,
)
from omni_evaluator.schemas.generation_options import (
    ApiGenerationOptions, VllmGenerationOptions, SglangGenerationOptions, HuggingfaceGenerationOptions,
)
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.utils.data import split_iterator
from omni_evaluator.utils.resource import get_num_cuda_devices
from omni_evaluator.schemas.inference import Record, VllmInferenceOutput, SglangInferenceOutput
from omni_evaluator.schemas.task import TaskConfig
from omni_evaluator.utils.common import get_custom_module, set_seed
from omni_evaluator.utils.io import ensure_per_run_format, get_output_filename, read_file, write_file
from omni_evaluator.modules.image_generation import T2IGenerator

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> Tuple[OrderedDict, OrderedDict, OrderedDict]:
    # Run inference across benchmarks using the configured inference and evaluation engines.
    # Args: args - parsed CLI arguments with model config, benchmarks, engine settings, and resume options
    # Returns: tuple of (inference_outputs, inference_metas, inference_runtimes) OrderedDicts keyed by benchmark name;
    #   inference_runtimes[benchmark] is a list of per-run inference wall-clock seconds
    output_filenames = list()
    for benchmark, evaluation_method in zip(args.benchmarks, args.evaluation_methods):
        output_filename = get_output_filename(
            benchmark=benchmark,
            evaluation_method=evaluation_method,
        )
        output_filenames.append(output_filename)

    s3_client = None
    try:
        s3_client = S3Client(
            bucket_name=args.s3_bucket_name,
            access_key=args.s3_access_key,
            secret_key=args.s3_secret_key,
            endpoint_url=args.s3_endpoint_url,
            region=args.s3_region,
            verbose=True,
        )
    except Exception as ex:
        logger.warning('Failed to connect s3 storage')
        traceback.print_exc()

    inference_outputs, inference_metas = OrderedDict(), OrderedDict()
    inference_runtimes = OrderedDict()
    for benchmark in args.benchmarks: # to keep order
        inference_outputs[benchmark] = None
        inference_metas[benchmark] = None
        inference_runtimes[benchmark] = None
    
    left_benchmarks = copy.deepcopy(args.benchmarks)
    left_evaluation_methods = copy.deepcopy(args.evaluation_methods)
    left_output_filenames = copy.deepcopy(output_filenames)
    if args.resume: # skip if result exist and resume
        left_benchmarks, left_evaluation_methods, left_output_filenames = list(), list(), list()
        for task_idx, (benchmark, evaluation_method, output_filename) in tqdm(
            enumerate(zip(args.benchmarks, args.evaluation_methods, output_filenames)), 
            initial=0, 
            total=len(args.benchmarks), 
            desc=f'Resume inference: {args.inference_engine}'
        ):
            _output_filepath = os.path.join(args.output_dirpath, "output", output_filename)
            if os.path.exists(_output_filepath):
                _inference_output = read_file(filepath=_output_filepath)
                inference_outputs[benchmark] = ensure_per_run_format(_inference_output["inference"])
                # Hydrate cached config to instance + reattach the raw yaml
                # snapshot persisted under ``_output["yaml"]``. Keeps the
                # downstream evaluate.py merge yaml-aware (explicit ``null``
                # honored) instead of falling back to non-None overlay.
                _cfg = _inference_output["config"]
                _raw_yaml = _inference_output.get("yaml")
                try:
                    _instance = TaskConfig.ensure(
                        _cfg,
                        mode="direct" if not args.reasoning else "reasoning",
                    )
                    if isinstance(_raw_yaml, dict):
                        _instance._raw_yaml = _raw_yaml
                    inference_metas[benchmark] = _instance
                except Exception:
                    # Defensive: fall back to the raw dict — evaluate.py's
                    # ensure call will re-hydrate (without _raw_yaml).
                    inference_metas[benchmark] = _cfg
                inference_runtimes[benchmark] = _inference_output.get("runtime_inference", None)
                continue
            else:
                left_benchmarks.append(benchmark)
                left_evaluation_methods.append(evaluation_method)
                left_output_filenames.append(output_filename)
                logger.warning('Cannot resume inference %s since output not exist. output_filepath: %s', benchmark, _output_filepath)
        logger.info('Loaded previous job: %d/%d', len(args.benchmarks)-len(left_benchmarks), len(args.benchmarks))
    
    if not (len(left_benchmarks) == len(left_evaluation_methods) == len(left_output_filenames)):
        raise ValueError(
            f'Length not match between left_benchmarks, left_evaluation_methods, and left_output_filenames: {len(left_benchmarks)} vs. {len(left_evaluation_methods)} vs. {len(left_output_filenames)}'
        )

    for task_idx, (benchmark, evaluation_method, output_filename) in tqdm(
        enumerate(zip(left_benchmarks, left_evaluation_methods, left_output_filenames)),
        initial=0, 
        total=len(left_benchmarks), 
        desc=f'Engines for inference: {args.inference_engine}/{args.evaluation_engine}',
    ) :
        inference_output = list()
        task_config = None
        custom_module = get_custom_module(
            evaluation_engine=args.evaluation_engine,
            task_name=benchmark,
        )
        
        num_runs = args.num_runs
        if not num_runs:
            num_runs = DEFAULT_NUM_RUNS
        
        default_generation_options = {
            "inference_engine": args.inference_engine,
            "num_beams": args.num_beams,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        }
        
        _runtime_inference_per_run = []
        if (
            args.evaluation_engine == EvaluationEngine.builtin
            and custom_module
            and hasattr(custom_module, "infer")
        ): # custom inference
            inference_output = []
            for _run_index in range(0, num_runs):
                logger.info('(%02d/%02d) Run inference', _run_index+1, num_runs)
                set_seed(seed=args.seed if _run_index ==0 else None)

                _started_at = datetime.now()
                _inference_run_output, task_config = custom_module.infer(
                    args=args,
                    task_name=benchmark,
                    run_index=_run_index,
                )
                _completed_at = datetime.now()
                _runtime_inference = (_completed_at - _started_at)
                _runtime_inference = _runtime_inference.seconds + _runtime_inference.microseconds * 1e-6
                _runtime_inference_per_run.append(_runtime_inference)

                task_config.meta.num_runs = num_runs
                # Reconcile num_records with the actually yielded record count.
                # Lazy dataset paths (see prepare_dataset.iter_file) initialize
                # task_config.num_records from the file row count, which can
                # over-count when dataset.subset filters rows in-stream.
                task_config.num_records = len(_inference_run_output)
                # Serialize Records to plain dicts (drops PIL/np.ndarray/bytes
                # so the downstream evaluate.py / jsonl-write path can consume
                # them safely; evaluate.py uses dict `.get()` access).
                _inference_run_output = [
                    _record.to_dict(template="json") if isinstance(_record, Record) else _record
                    for _record in _inference_run_output
                ]
                inference_output.append(_inference_run_output)

        else:
            inference_output = []
            for _run_index in range(0, num_runs):
                logger.info('(%02d/%02d) Run inference', _run_index+1, num_runs)
                set_seed(seed=args.seed if _run_index ==0 else None)
            
                # Acquire dataset (list for lmms_eval/lm_eval_harness/vlm_eval_kit,
                # factory callable for builtin). split_iterator handles both.
                _benchmark_dataset, task_config = None, None
                if args.evaluation_engine == EvaluationEngine.builtin:
                    from omni_evaluator.evaluation.builtin import get_data_iterator as get_builtin_data_iterator
                    _benchmark_dataset, task_config = get_builtin_data_iterator(
                        evaluation_engine=args.evaluation_engine,
                        task_name=benchmark,
                        evaluation_method=evaluation_method,
                        subtask_type=args.subtask_type,
                        system_prompt=args.system_prompt,
                        task_prompt=args.task_prompt,
                        num_ocr_tokens=args.num_ocr_tokens,
                        num_subtitle_cues=args.num_subtitle_cues,
                        num_entity_tokens=args.num_entity_tokens,
                        reasoning=args.reasoning,
                        num_fewshot=args.num_fewshot,
                        fewshot_image_max_size=args.fewshot_image_max_size,
                        do_cot=args.do_cot,
                        cache_dirpath=args.cache_dirpath,
                        local_dirpath=args.local_dirpath,
                        batch_size=args.inference_concurrency,
                        run_index=_run_index,
                        debug=args.debug,
                        dataset_subset_override=args.dataset_subset,
                    )

                elif args.evaluation_engine == EvaluationEngine.lmms_eval:
                    from omni_evaluator.evaluation.lmms_eval import get_data_iterator as get_lmms_data_iterator
                    _benchmark_dataset, task_config = get_lmms_data_iterator(
                        evaluation_engine=args.evaluation_engine,
                        task_name=benchmark,
                        system_prompt=args.system_prompt,
                        task_prompt=args.task_prompt,
                        num_ocr_tokens=args.num_ocr_tokens,
                        num_subtitle_cues=args.num_subtitle_cues,
                        task_manager=None,
                        output_path=args.cache_dirpath,
                        process_with_media=False,
                        cache_requests=True,
                        rewrite_requests_cache=False,
                        apply_chat_template=False,
                        fewshot_as_multiturn=False,
                        num_fewshot=args.num_fewshot if args.num_fewshot >= 0 else None,
                        run_index=_run_index,
                        override_generation_kwargs=getattr(args, "generation_kwargs", None),
                        debug=args.debug,
                        # gen_kwargs=gen_kwargs,
                        # predict_only=predict_only,
                        # fewshot_random_seed=fewshot_random_seed,
                    )
                    if evaluation_method != task_config["evaluation"]["method"]:
                        _new_method = task_config["evaluation"]["method"]
                        logger.warning(
                            'Set `evaluation_method` from `%s` to `%s` per task config',
                            evaluation_method, _new_method,
                        )
                        # Sync four locations so on-disk filename, the current
                        # iteration, the remaining-task list, and the args view
                        # (which evaluate.py rebuilds filenames from) all
                        # agree on the new method.
                        left_evaluation_methods[task_idx] = evaluation_method = _new_method
                        left_output_filenames[task_idx] = output_filename = get_output_filename(
                            benchmark=benchmark, evaluation_method=_new_method,
                        )
                        try:
                            _args_idx = args.benchmarks.index(benchmark)
                            args.evaluation_methods[_args_idx] = _new_method
                        except ValueError:
                            pass

                elif args.evaluation_engine == EvaluationEngine.lm_eval_harness:
                    from omni_evaluator.evaluation.lm_eval_harness import get_data_iterator as get_lm_eval_harness_data_iterator
                    _benchmark_dataset, task_config = get_lm_eval_harness_data_iterator(
                        evaluation_engine=args.evaluation_engine,
                        task_name=benchmark,
                        system_prompt=args.system_prompt,
                        task_prompt=args.task_prompt,
                        task_manager=None,
                        output_path=args.cache_dirpath,
                        process_with_media=False,
                        cache_requests=True,
                        rewrite_requests_cache=False,
                        apply_chat_template=False,
                        fewshot_as_multiturn=False,
                        trust_remote_code=True,
                        num_fewshot=args.num_fewshot
                        if isinstance(args.num_fewshot, int) and args.num_fewshot >= 0 else None,
                        run_index=_run_index,
                        override_generation_kwargs=getattr(args, "generation_kwargs", None),
                        # gen_kwargs=gen_kwargs,
                        # predict_only=predict_only,
                        # fewshot_random_seed=fewshot_random_seed,
                    )
                    if evaluation_method != task_config["evaluation"]["method"]:
                        _new_method = task_config["evaluation"]["method"]
                        logger.warning(
                            'Set `evaluation_method` from `%s` to `%s` per task config',
                            evaluation_method, _new_method,
                        )
                        left_evaluation_methods[task_idx] = evaluation_method = _new_method
                        left_output_filenames[task_idx] = output_filename = get_output_filename(
                            benchmark=benchmark, evaluation_method=_new_method,
                        )
                        try:
                            _args_idx = args.benchmarks.index(benchmark)
                            args.evaluation_methods[_args_idx] = _new_method
                        except ValueError:
                            pass

                elif args.evaluation_engine == EvaluationEngine.vlm_eval_kit:
                    from omni_evaluator.evaluation.vlm_eval_kit import get_data_iterator as get_vlm_eval_kit_data_iterator
                    _benchmark_dataset, task_config = get_vlm_eval_kit_data_iterator(
                        evaluation_engine=args.evaluation_engine,
                        task_name=benchmark,
                        task_prompt=args.task_prompt,
                        system_prompt=args.system_prompt,
                        num_ocr_tokens=args.num_ocr_tokens,
                        num_subtitle_cues=args.num_subtitle_cues,
                        model_name=None,
                        config=None,
                        fps=args.fps,
                        nframe=args.max_video_frames,
                        run_index=_run_index,
                        debug=args.debug,
                    )

                task_config.meta.num_runs = num_runs
                if _benchmark_dataset is None:
                    continue

                # Engine-level capability gate — get_data_iterator above may have
                # rewritten `evaluation_method` to `perplexity` (lmms_eval /
                # lm_eval_harness task that only supports loglikelihood). If the
                # active inference_engine cannot compute ppl (e.g. OpenAI /
                # Anthropic / Google chat APIs expose no prompt-logprobs), skip
                # this benchmark instead of silently producing empty perplexities.
                from omni_evaluator.inference.capabilities import INFERENCE_ENGINE_FEATURES
                _engine_features = INFERENCE_ENGINE_FEATURES.get(args.inference_engine)
                if (
                    _engine_features is not None
                    and evaluation_method == EvaluationMethod.perplexity
                    and not _engine_features.support_compute_perplexity
                ):
                    logger.warning(
                        "Skipping benchmark %s: evaluation_method=perplexity not "
                        "supported by inference_engine=%s (no prompt_logprobs).",
                        benchmark, args.inference_engine.value if hasattr(args.inference_engine, 'value') else args.inference_engine,
                    )
                    continue

                # CPU guard: the HF engine on CPU (explicit device_map=cpu, or no
                # CUDA visible) can't use the world_size thread-split — that path
                # assigns GPUs per rank and forces device_map=auto. Thread-parallel
                # doesn't help on CPU anyway (GIL + core contention); batch_size is
                # the lever. Collapse to a single worker before the split.
                if args.inference_engine == InferenceEngine.huggingface and args.world_size > 1:
                    _device_map = getattr(args, "device_map", None)
                    _cpu_only = (
                        _device_map == "cpu"
                        or (isinstance(_device_map, dict) and _device_map
                            and all(str(_v) == "cpu" for _v in _device_map.values()))
                        or get_num_cuda_devices() == 0
                    )
                    if _cpu_only:
                        logger.warning(
                            "device_map=%s / cuda_devices=%d -> forcing world_size=1 "
                            "(CPU thread-parallel disabled; use batch_size / concurrency instead)",
                            _device_map, get_num_cuda_devices(),
                        )
                        args.world_size = 1

                # Engine-agnostic contiguous-block split. split_iterator branches
                # internally on type: factories produce a fresh generator per rank
                # (builtin), Sequences are sliced via islice (lmms_eval
                # /lm_eval_harness/vlm_eval_kit list outputs). Both are race-safe.
                _slices, _rank_sizes = split_iterator(
                    _benchmark_dataset,
                    total_size=task_config.num_records,
                    world_size=args.world_size,
                )
                benchmark_datasets_rank = {r: _slices[r] for r in range(args.world_size)}
                benchmark_datasets_rank_size = {r: _rank_sizes[r] for r in range(args.world_size)}
            
                main__inference, main__inference_kwargs = None, None
                if args.inference_engine == InferenceEngine.huggingface:
                    from omni_evaluator.inference.huggingface import main as main__inference_huggingface
                    main__inference = main__inference_huggingface
                    main__inference_kwargs = dict(
                        inference_engine=args.inference_engine,
                        evaluation_engine=args.evaluation_engine,
                        benchmark=benchmark,
                        evaluation_method=evaluation_method,
                        benchmark_idx=task_idx,
                        num_benchmarks=len(left_benchmarks),
                        benchmark_dataset=benchmark_datasets_rank[0],
                        benchmark_dataset_size=benchmark_datasets_rank_size[0],
                        task_config=task_config,
                        default_generation_options=default_generation_options,
                        model_name_or_path=args.model_name_or_path,
                        model_group=args.model_group if hasattr(args, "model_group") else None,
                        hf_hub_cache=args.hf_hub_cache,
                        skip_chat_template=args.skip_chat_template,
                        reasoning=args.reasoning,
                        device_map=args.device_map,
                        torch_dtype=args.torch_dtype,
                        low_cpu_mem_usage=args.low_cpu_mem_usage,
                        trust_remote_code=args.trust_remote_code,
                        min_pixels=args.min_pixels,
                        max_pixels=args.max_pixels,
                        max_video_frames=args.max_video_frames,
                        fps=args.fps,
                        use_audio_in_video=args.use_audio_in_video,
                        debug=args.debug,
                        verbose=args.verbose,
                        **args.model_kwargs,
                    )
                    
                elif args.inference_engine == InferenceEngine.vllm:
                    default_generation_options["logprobs"] = args.logprobs
                    if evaluation_method == EvaluationMethod.perplexity:
                        default_generation_options["logprobs"] = True

                    from omni_evaluator.inference.vllm import main as main__inference_vllm
                    main__inference = main__inference_vllm
                    main__inference_kwargs = dict(
                        inference_engine=args.inference_engine,
                        evaluation_engine=args.evaluation_engine,
                        benchmark=benchmark,
                        evaluation_method=evaluation_method,
                        benchmark_idx=task_idx,
                        num_benchmarks=len(left_benchmarks),
                        benchmark_dataset=benchmark_datasets_rank[0],
                        benchmark_dataset_size=benchmark_datasets_rank_size[0],
                        task_config=task_config,
                        default_generation_options=default_generation_options,
                        url=args.url,
                        api_version=args.vllm_api_version,
                        model_name_or_path=args.model_name_or_path,
                        hf_hub_cache=args.hf_hub_cache,
                        trust_remote_code=args.trust_remote_code,
                        torch_dtype=args.torch_dtype,
                        skip_chat_template=args.skip_chat_template,
                        reasoning=args.reasoning,
                        add_generation_prompt=args.add_generation_prompt,
                        chat_template_kwargs=args.chat_template_kwargs if hasattr(args, "chat_template_kwargs") else None,
                        mm_processor_kwargs=args.mm_processor_kwargs if hasattr(args, "mm_processor_kwargs") else None,
                        max_video_frames=getattr(args, 'max_video_frames', None),
                        fps=getattr(args, 'fps', None),
                        min_pixels=getattr(args, 'min_pixels', None),
                        max_pixels=getattr(args, 'max_pixels', None),
                        allowed_local_media_path=args.allowed_local_media_path if hasattr(args, "allowed_local_media_path") else None,
                        seed=args.seed,
                        batch_size=args.inference_concurrency,
                        timeout=args.request_timeout,
                        max_retry=args.max_retry,
                        wait_between_retry=args.wait_between_retry,
                        do_async=args.do_async,
                        verbose=args.verbose,
                        debug=args.debug,
                    )

                elif args.inference_engine == InferenceEngine.sglang:
                    default_generation_options["logprobs"] = args.logprobs
                    if evaluation_method == EvaluationMethod.perplexity:
                        default_generation_options["logprobs"] = True

                    from omni_evaluator.inference.sglang import main as main__inference_sglang
                    main__inference = main__inference_sglang
                    main__inference_kwargs = dict(
                        inference_engine=args.inference_engine,
                        evaluation_engine=args.evaluation_engine,
                        benchmark=benchmark,
                        evaluation_method=evaluation_method,
                        benchmark_idx=task_idx,
                        num_benchmarks=len(left_benchmarks),
                        benchmark_dataset=benchmark_datasets_rank[0],
                        benchmark_dataset_size=benchmark_datasets_rank_size[0],
                        task_config=task_config,
                        default_generation_options=default_generation_options,
                        url=args.url,
                        api_version=args.vllm_api_version,
                        model_name_or_path=args.model_name_or_path,
                        skip_chat_template=args.skip_chat_template,
                        reasoning=args.reasoning,
                        add_generation_prompt=args.add_generation_prompt,
                        seed=args.seed,
                        batch_size=args.inference_concurrency,
                        timeout=args.request_timeout,
                        max_retry=args.max_retry,
                        wait_between_retry=args.wait_between_retry,
                        do_async=args.do_async,
                        verbose=args.verbose,
                        debug=args.debug,
                    )
                    
                elif args.inference_engine in [
                    InferenceEngine.api__openai,
                    InferenceEngine.api__anthropic,
                    InferenceEngine.api__google,
                ]:
                    if evaluation_method == EvaluationMethod.perplexity:
                        default_generation_options["logprobs"] = True
                        default_generation_options["top_logprobs"] = 1
                        default_generation_options["max_new_tokens"] = 1

                    from omni_evaluator.inference.api import main as main__inference_api
                    main__inference = main__inference_api
                    main__inference_kwargs = dict(
                        inference_engine=args.inference_engine,
                        evaluation_engine=args.evaluation_engine,
                        benchmark=benchmark,
                        evaluation_method=evaluation_method,
                        benchmark_idx=task_idx,
                        num_benchmarks=len(left_benchmarks),
                        benchmark_dataset=benchmark_datasets_rank[0],
                        benchmark_dataset_size=benchmark_datasets_rank_size[0],
                        task_config=task_config,
                        default_generation_options=default_generation_options,
                        api_name=args.api_name,
                        reasoning_options={
                            "reasoning_effort": args.reasoning_effort,
                            "thinking_budget": args.thinking_budget,
                        },
                        logprobs=args.logprobs,
                        batch_size=args.inference_concurrency,
                        timeout=args.request_timeout,
                        max_retry=args.max_retry,
                        wait_between_retry=args.wait_between_retry,
                        do_async=args.do_async,
                        verbose=args.verbose,
                        debug=args.debug,
                    )
            
                _started_at = datetime.now()
                if args.world_size == 1:
                    t2i_generator = None
                    if args.t2i_generator_type is not None:
                        t2i_generator = T2IGenerator(
                            generator_type=args.t2i_generator_type,
                            hf_cache_dir=args.hf_hub_cache,
                            cache_dirpath=args.cache_dirpath,
                            rank=0,
                            world_size=1,
                            **args.t2i_generator.__dict__,
                        )

                    _inference_run_output = main__inference(
                        rank=0,
                        world_size=args.world_size,
                        run_index=_run_index,
                        num_runs=num_runs,
                        t2i_generator=t2i_generator,
                        **main__inference_kwargs,
                    )
                else:
                    _inference_run_output = list()
                    thread_lock = threading.Lock()
                    thread_barrier = threading.Barrier(args.world_size)
                    with ThreadPoolExecutor(max_workers=args.world_size) as executor:
                        futures = list()
                        for _rank in range(args.world_size):
                            _rank_inference_kwargs = dict(main__inference_kwargs)
                            _rank_inference_kwargs["benchmark_dataset"] = benchmark_datasets_rank[_rank]
                            _rank_inference_kwargs["benchmark_dataset_size"] = benchmark_datasets_rank_size[_rank]
                            t2i_generator = None
                            if args.t2i_generator_type is not None:
                                t2i_generator = T2IGenerator(
                                    generator_type=args.t2i_generator_type,
                                    hf_cache_dir=args.hf_hub_cache,
                                    cache_dirpath=args.cache_dirpath,
                                    rank=_rank,
                                    world_size=args.world_size,
                                    **args.t2i_generator.__dict__,
                                )

                            _kwargs = _rank_inference_kwargs
                            futures.append(executor.submit(
                                lambda rank=_rank, kwargs=_kwargs: main__inference(
                                    rank=rank,
                                    world_size=args.world_size,
                                    run_index=_run_index,
                                    num_runs=num_runs,
                                    t2i_generator=t2i_generator,
                                    thread_lock=thread_lock,
                                    thread_barrier=thread_barrier,
                                    **kwargs,
                                )
                            ))
                        for _rank, _future in enumerate(futures):
                            try:
                                _partial = _future.result()
                            except Exception as ex:
                                logger.error('Thread was terminated unexpectedly: %s', _rank)
                                traceback.print_exc()
                                executor.shutdown(cancel_futures=True)
                                raise
                            # A rank that returns None means its main__inference
                            # gave up (e.g. all prep failed, batch call raised);
                            # skip just that rank's contribution rather than
                            # crashing the run with ``list += None``.
                            if _partial is None:
                                logger.warning(
                                    '%s/%s rank=%s returned None from main__inference; '
                                    'skipping this rank in the run aggregation',
                                    benchmark, evaluation_method, _rank,
                                )
                                continue
                            _inference_run_output += _partial
                _completed_at = datetime.now()
                _runtime_inference = (_completed_at - _started_at)
                _runtime_inference = _runtime_inference.seconds + _runtime_inference.microseconds * 1e-6
                _runtime_inference_per_run.append(_runtime_inference)

                # A None here means the single-rank main__inference gave up
                # (catastrophic batch failure). Skip this run iteration cleanly
                # — the outer ``len(inference_output) < 1`` check (below) still
                # routes the task to the standard ``failed to get inference_result``
                # path when every run yields nothing.
                if _inference_run_output is None:
                    logger.error(
                        '%s/%s run=%s skipped: main__inference returned None '
                        '(see prior errors)',
                        benchmark, evaluation_method, _run_index,
                    )
                    continue

                # Reconcile num_records with the actually yielded record count.
                # Lazy dataset paths (see prepare_dataset.iter_file) initialize
                # task_config.num_records from the file row count, which can
                # over-count when dataset.subset filters rows in-stream.
                task_config.num_records = len(_inference_run_output)
                # Serialize Records to plain dicts (drops PIL/np.ndarray/bytes
                # so the downstream evaluate.py / jsonl-write path can consume
                # them safely; evaluate.py uses dict `.get()` access).
                _inference_run_output = [
                    _record.to_dict(template="json") if isinstance(_record, Record) else _record
                    for _record in _inference_run_output
                ]
                inference_output.append(_inference_run_output)

        if (
            len(inference_output) < 1
            or all(len(run) < 1 for run in inference_output)
        ):
            logger.error('%s/%s failed to get inference_result', benchmark, evaluation_method)
        else:
            if args:
                task_config["arguments"] = dict(vars(args))
                if isinstance(task_config["arguments"].get("t2i_generator", None), argparse.Namespace):
                    task_config["arguments"]["t2i_generator"] = dict(vars(task_config["arguments"]["t2i_generator"]))
            # to_dict goes into the local dump dict only — leave the
            # ``task_config`` variable itself as the original instance so the
            # transient ``_raw_yaml`` attr survives into ``inference_metas``
            # (and from there into the evaluate.py merge path).
            _raw_yaml = getattr(task_config, "_raw_yaml", None) if isinstance(task_config, TaskConfig) else None
            _config_for_dump = (
                task_config.to_dict() if isinstance(task_config, TaskConfig) else task_config
            )

            output = {
                "config": _config_for_dump,
                "yaml": _raw_yaml,
                "inference": inference_output,
                "runtime_inference": _runtime_inference_per_run,
            }
            if isinstance(args.output_dirpath, str):
                _output_filepath = os.path.join(args.output_dirpath, "output", output_filename)
                write_file(
                    filepath=_output_filepath,
                    obj=output,
                )
                logger.info('Saved inference output: %s', _output_filepath)
            if (
                s3_client is not None
                and isinstance(args.remote_output_dirpath, str)
            ):
                _remote_output_dirpath = os.path.join(args.remote_output_dirpath, "output")
                try:
                    s3_client.upload_file(
                        filepath=_output_filepath,
                        remote_dirpath=_remote_output_dirpath,
                    )
                    logger.info('Uploaded inference output to S3: %s', _remote_output_dirpath)
                except Exception as ex:
                    logger.warning('Failed to upload inference output to S3: %s', _remote_output_dirpath)

        # collect output
        inference_outputs[benchmark] = inference_output
        # Keep instance form so the transient ``_raw_yaml`` (attached by the
        # yaml-driven ``from_engine`` path) survives into evaluate.py's merge
        # input. ``TaskConfig.ensure`` accepts either instance or dict.
        inference_metas[benchmark] = task_config
        inference_runtimes[benchmark] = _runtime_inference_per_run

    if os.path.exists(args.cache_dirpath):
        _cache = Path(args.cache_dirpath).resolve()
        # Safety: only delete if the cache path is at least 3 levels deep (e.g. /mnt/tmp/exp)
        if len(_cache.parts) >= 4:
            shutil.rmtree(args.cache_dirpath)
        else:
            logger.warning(
                f"Skipping cache cleanup: path is too shallow and may be unsafe to delete: {args.cache_dirpath}"
            )
    return inference_outputs, inference_metas, inference_runtimes

def infer_record(
    args: argparse.Namespace,
    record: Record,
    generation_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    do_async: Optional[bool] = False,
    verbose: Optional[bool] = True,
    hf_inferencer: Optional[Any] = None,
    evaluation_method: Optional[str] = None,
    output_modality: Optional[List[str]] = None,
) -> Optional[Union[VllmInferenceOutput, SglangInferenceOutput, Dict[str, Any]]]:
    # Run synchronous inference on a single record, dispatching to the appropriate engine.
    # Args: record - input Record with messages and generation_options; tools - optional tool definitions for function calling;
    #   hf_inferencer - HuggingfaceInferencer instance (required when inference_engine == huggingface);
    #   evaluation_method - "generation" or "perplexity" (required for huggingface);
    #   output_modality - list of output modalities e.g. ["text"] (used for huggingface)
    # Returns: engine-specific inference output, or None if unsupported
    _record = copy.deepcopy(record) # to prevent pointer intervention

    if not generation_options:
        generation_options = dict()
    if _record.generation_options:
        for _k, _v in _record.generation_options.items():
            if (
                _k not in generation_options
                or generation_options[_k] is None
            ):
                generation_options[_k] = _v

    output = None
    if args.inference_engine == InferenceEngine.huggingface:
        if hf_inferencer is None:
            raise ValueError('hf_inferencer must be provided for huggingface inference engine')
        _generation_options = HuggingfaceGenerationOptions.from_dict(obj=generation_options).to_dict()
        output = hf_inferencer(
            messages=[_message.to_dict(template="hf") for _message in _record.messages],
            generation_options=_generation_options,
            evaluation_method=evaluation_method,
            output_modality=output_modality,
        )
    elif args.inference_engine == InferenceEngine.vllm:
        generation_options = VllmGenerationOptions.from_dict(obj=generation_options).to_dict()
        _chat_template_kwargs = getattr(args, "chat_template_kwargs", None)
        _mm_processor_kwargs = getattr(args, "mm_processor_kwargs", None)

        prediction, generated_text = None, None
        tool_calls, function_call, annotations = None, None, None
        reasoning_content = None
        if not args.skip_chat_template:
            messages = list()
            for _message in _record.messages:
                _message = _message.to_dict(template="openai")
                messages.append(_message)
            output = chat_completion_sync_vllm(
                url=args.url,
                messages=messages,
                model_name=args.vllm_model_name,
                api_version=args.vllm_api_version,
                generation_options=generation_options,
                tools=tools,
                reasoning=args.reasoning,
                chat_template_kwargs=_chat_template_kwargs,
                mm_processor_kwargs=_mm_processor_kwargs,
                max_video_frames=getattr(args, 'max_video_frames', None),
                fps=getattr(args, 'fps', None),
                min_pixels=getattr(args, 'min_pixels', None),
                max_pixels=getattr(args, 'max_pixels', None),
                allowed_local_media_path=args.allowed_local_media_path,
                timeout=args.request_timeout,
                max_retry=args.max_retry,
                wait_between_retry=args.wait_between_retry,
            )
            if isinstance(output, Record):
                prediction = output.prediction
                reasoning_content = output.reasoning_content
                generated_text = output.generation
                tool_calls = output.tool_calls or list()
                function_call = output.function_call
                annotations = output.annotations or list()
                output = {"latency": output.latency}
        else:
            prompt = ""
            for _message in _record.messages:
                if _message["role"] == "system":
                    _query = ChatMessage.get_query(message=_message)
                    _query = f'{_query}\n'
                else:
                    _query = ChatMessage.get_prompt(message=_message)
                prompt += f'{_query}\n\n'
            prompt = prompt.rstrip()
            output = completion_sync_vllm(
                url=args.url,
                prompt=prompt,
                model_name=args.vllm_model_name,
                api_version=args.vllm_api_version,
                generation_options=generation_options,
                tools=tools,
                reasoning=args.reasoning,
                timeout=args.request_timeout,
                max_retry=args.max_retry,
                wait_between_retry=args.wait_between_retry,
            )
            if isinstance(output, Record):
                prediction = output.prediction
                reasoning_content = output.reasoning_content
                generated_text = output.generation
                output = {"latency": output.latency}

        output = VllmInferenceOutput(
            prediction=prediction,
            reasoning_content=reasoning_content,
            generated_text=generated_text,
            tool_calls=tool_calls,
            function_call=function_call,
            annotations=annotations,
            latency=output.get("latency", None) if isinstance(output, dict) else None,
        )

    elif args.inference_engine == InferenceEngine.sglang:
        generation_options = SglangGenerationOptions.from_dict(obj=generation_options).to_dict()
        messages = [_message.to_dict(template="openai") for _message in _record.messages]
        output = chat_completion_sync_sglang(
            url=args.url,
            messages=messages,
            model_name=getattr(args, "sglang_model_name", None),
            api_version=getattr(args, "vllm_api_version", None),
            generation_options=generation_options,
            tools=tools,
            reasoning=args.reasoning,
            timeout=args.request_timeout,
            max_retry=args.max_retry,
            wait_between_retry=args.wait_between_retry,
        )
        prediction, generated_text, reasoning_content = None, None, None
        tool_calls, function_call, annotations = None, None, None
        if isinstance(output, Record):
            prediction = output.prediction
            reasoning_content = output.reasoning_content
            generated_text = output.generation
            tool_calls = output.tool_calls or list()
            function_call = output.function_call
            annotations = output.annotations or list()
            output = {"latency": output.latency}
        output = SglangInferenceOutput(
            prediction=prediction,
            reasoning_content=reasoning_content,
            generated_text=generated_text,
            tool_calls=tool_calls,
            function_call=function_call,
            annotations=annotations,
            latency=output.get("latency", None) if isinstance(output, dict) else None,
        )

    elif args.inference_engine in [
        InferenceEngine.api__openai,
        InferenceEngine.api__anthropic,
        InferenceEngine.api__google,
    ]:
        output = chat_completion_sync_api(
            api_name=args.api_name,
            messages=_record.messages,
            generation_options=generation_options,
            tools=tools,
            timeout=args.request_timeout,
            max_retry=args.max_retry,
            return_dict=True,
        )

    return output


async def infer_record_async(
    args: argparse.Namespace,
    record: Record,
    generation_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    verbose: Optional[bool] = True,
    hf_inferencer: Optional[Any] = None,
    evaluation_method: Optional[str] = None,
    output_modality: Optional[List[str]] = None,
) -> Optional[Union[VllmInferenceOutput, SglangInferenceOutput, Dict[str, Any]]]:
    # Run asynchronous inference on a single record, dispatching to the appropriate engine.
    # Args: record - input Record with messages and generation_options; semaphore - asyncio.Semaphore for concurrency control;
    #   hf_inferencer - HuggingfaceInferencer instance (required when inference_engine == huggingface, called synchronously);
    #   evaluation_method - "generation" or "perplexity" (required for huggingface);
    #   output_modality - list of output modalities e.g. ["text"] (used for huggingface)
    # Returns: engine-specific inference output, or None if unsupported
    _record = copy.deepcopy(record) # to prevent pointer intervention

    if not generation_options:
        generation_options = dict()
    if _record.generation_options:
        for _k, _v in _record.generation_options.items():
            if (
                _k not in generation_options
                or generation_options[_k] is None
            ):
                generation_options[_k] = _v

    output = None
    if args.inference_engine == InferenceEngine.huggingface:
        if hf_inferencer is None:
            raise ValueError('hf_inferencer must be provided for huggingface inference engine')
        _generation_options = HuggingfaceGenerationOptions.from_dict(obj=generation_options).to_dict()
        output = hf_inferencer(
            messages=[_message.to_dict(template="hf") for _message in _record.messages],
            generation_options=_generation_options,
            evaluation_method=evaluation_method,
            output_modality=output_modality,
        )
    elif args.inference_engine == InferenceEngine.vllm:
        generation_options = VllmGenerationOptions.from_dict(obj=generation_options).to_dict()
        _chat_template_kwargs = getattr(args, "chat_template_kwargs", None)
        _mm_processor_kwargs = getattr(args, "mm_processor_kwargs", None)

        prediction, generated_text = None, None
        tool_calls, function_call, annotations = None, None, None
        reasoning_content = None
        if not args.skip_chat_template:
            messages = list()
            for _message in _record.messages:
                _message = _message.to_dict(template="openai")
                messages.append(_message)
            output = await chat_completion_async_vllm(
                url=args.url,
                messages=messages,
                generation_options=generation_options,
                tools=tools,
                reasoning=args.reasoning,
                chat_template_kwargs=_chat_template_kwargs,
                mm_processor_kwargs=_mm_processor_kwargs,
                max_video_frames=getattr(args, 'max_video_frames', None),
                fps=getattr(args, 'fps', None),
                min_pixels=getattr(args, 'min_pixels', None),
                max_pixels=getattr(args, 'max_pixels', None),
                allowed_local_media_path=args.allowed_local_media_path,
                semaphore=semaphore,
                timeout=args.request_timeout,
                socket_timeout=args.socket_timeout,
                max_retry=args.max_retry,
                wait_between_retry=args.wait_between_retry,
            )
            if isinstance(output, Record):
                prediction = output.prediction
                reasoning_content = output.reasoning_content
                generated_text = output.generation
                tool_calls = output.tool_calls or list()
                function_call = output.function_call
                annotations = output.annotations or list()
                output = {"latency": output.latency}
        else:
            prompt = ""
            for _message in _record.messages:
                if _message["role"] == "system":
                    _query = ChatMessage.get_query(message=_message)
                    _query = f'{_query}\n'
                else:
                    _query = ChatMessage.get_prompt(message=_message)
                prompt += f'{_query}\n\n'
            prompt = prompt.rstrip()
            output = await completion_async_vllm(
                url=args.url,
                prompt=prompt,
                generation_options=generation_options,
                tools=tools,
                reasoning=args.reasoning,
                semaphore=semaphore,
                timeout=args.request_timeout,
                socket_timeout=args.socket_timeout,
                max_retry=args.max_retry,
                wait_between_retry=args.wait_between_retry,
            )
            if isinstance(output, Record):
                prediction = output.prediction
                reasoning_content = output.reasoning_content
                generated_text = output.generation
                output = {"latency": output.latency}

        output = VllmInferenceOutput(
            prediction=prediction,
            reasoning_content=reasoning_content,
            generated_text=generated_text,
            tool_calls=tool_calls,
            function_call=function_call,
            annotations=annotations,
            latency=output.get("latency", None) if isinstance(output, dict) else None,
        )

    elif args.inference_engine == InferenceEngine.sglang:
        generation_options = SglangGenerationOptions.from_dict(obj=generation_options).to_dict()
        messages = [_message.to_dict(template="openai") for _message in _record.messages]
        output = await chat_completion_async_sglang(
            url=args.url,
            messages=messages,
            model_name=getattr(args, "sglang_model_name", None),
            api_version=getattr(args, "vllm_api_version", None),
            generation_options=generation_options,
            tools=tools,
            reasoning=args.reasoning,
            semaphore=semaphore,
            timeout=args.request_timeout,
            socket_timeout=args.socket_timeout,
            max_retry=args.max_retry,
            wait_between_retry=args.wait_between_retry,
        )
        prediction, generated_text, reasoning_content = None, None, None
        tool_calls, function_call, annotations = None, None, None
        if isinstance(output, Record):
            prediction = output.prediction
            reasoning_content = output.reasoning_content
            generated_text = output.generation
            tool_calls = output.tool_calls or list()
            function_call = output.function_call
            annotations = output.annotations or list()
            output = {"latency": output.latency}
        output = SglangInferenceOutput(
            prediction=prediction,
            reasoning_content=reasoning_content,
            generated_text=generated_text,
            tool_calls=tool_calls,
            function_call=function_call,
            annotations=annotations,
            latency=output.get("latency", None) if isinstance(output, dict) else None,
        )

    elif args.inference_engine in [
        InferenceEngine.api__openai,
        InferenceEngine.api__anthropic,
        InferenceEngine.api__google,
    ]:
        output = await chat_completion_async_api(
            api_name=args.api_name,
            messages=_record.messages,
            system_message=None,
            generation_options=generation_options,
            tools=tools,
            semaphore=semaphore,
            timeout=args.request_timeout,
            max_retry=args.max_retry,
            wait_between_retry=args.wait_between_retry,
            return_dict=True,
        )

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser, validations = get_parser(parser=parser)
    args = parser.parse_args()
    for _validation_func in validations:
        args = _validation_func(args=args)
    main(args)