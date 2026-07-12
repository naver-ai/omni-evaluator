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
from collections import OrderedDict
import copy
from datetime import datetime
from functools import partial
import json
import logging
import os
from pathlib import Path
import time
from tqdm import tqdm
import traceback
from typing import Any, Dict, List, Tuple, Union

from omni_evaluator import EvaluationEngine, EvaluationMethod, InferenceEngine
from omni_evaluator.args import get_parser
from omni_evaluator.clients.s3_client import S3Client
from omni_evaluator.evaluation.metrics.verifier import Verifier
from omni_evaluator.infer import main as infer_main
from omni_evaluator.postprocess import get_postprocess_functions
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.evaluation import EvaluationOutput
from omni_evaluator.schemas.task import (
    TaskConfig, TaskPostprocessLogic,
)
from omni_evaluator.submission.leaderboard import LeaderboardFormatter
from omni_evaluator.utils.common import get_custom_module
from omni_evaluator.utils.io import ensure_per_run_format, get_output_filename, read_file, write_file
from omni_evaluator.utils.optional_import import require_package

logger = logging.getLogger(__name__)


_MM_CONTENT_TYPES = ("audio", "image", "video")


def _drop_message_multimodal_values_inplace(runs):
    """Drop base64/PIL multimodal payloads from per-run inference records.

    Mirrors ``AudioContent/ImageContent/VideoContent.to_serializable(remove_unserializable=True)``
    on already-dict-form records — necessary because builtin
    ``evaluate_task`` re-loads multimodal items into ``records[i]["messages"]``
    via ``load_dataset`` (so the in-memory state at dump time has base64
    audio/image inflated again). Every evaluation engine re-loads multimodal
    items from the dataset on its own evaluate path (builtin via the message
    overwrite; lmms_eval/lm_eval_harness/vlm_eval_kit via their own
    task/dataset builders), so the payload is redundant on disk.

    runs: per-run list — ``[[record, ...], [record, ...], ...]``.
    Mutates in place; safe at dump time after evaluate_task has finished.
    """
    if not isinstance(runs, list):
        return
    for _run in runs:
        if not isinstance(_run, list):
            continue
        for _rec in _run:
            if not isinstance(_rec, dict):
                continue
            _msgs = _rec.get("messages")
            if not isinstance(_msgs, list):
                continue
            for _m in _msgs:
                _content = _m.get("content") if isinstance(_m, dict) else None
                if not isinstance(_content, list):
                    continue
                for _item in _content:
                    if isinstance(_item, dict) and _item.get("type") in _MM_CONTENT_TYPES:
                        if "value" in _item:
                            _item["value"] = None


def main(args: argparse.Namespace) -> None:
    # Orchestrate the full evaluation pipeline: inference, postprocessing, metric computation, and submission formatting.
    # Args: args - parsed CLI arguments with benchmarks, engine configs, output paths, and evaluation options
    # Returns: None (writes results to disk and optionally uploads to S3)
    output_filenames = list()
    for task_name, evaluation_method in zip(args.benchmarks, args.evaluation_methods):
        output_filename = get_output_filename(
            benchmark=task_name,
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

    # inference
    inference_outputs = None
    if (
        not args.skip_inference # iterate all benchmarks
        or args.resume # iterate all benchmarks except completed
    ):
        logger.info('Execute inference')
        inference_outputs, task_configs, inference_runtimes = infer_main(args=args)
    else:
        inference_outputs, task_configs = OrderedDict(), OrderedDict()
        inference_runtimes = OrderedDict()
        for task_name, output_filename in zip(args.benchmarks, output_filenames):
            _output_filepath = os.path.join(args.output_dirpath, "output", output_filename)
            logger.info('Load inference_output: %s', _output_filepath)
            _output = read_file(filepath=_output_filepath)
            inference_outputs[task_name] = ensure_per_run_format(_output["inference"])
            # Always rebuild task_config from the current source (config.yaml / engine
            # factory) so schema changes (new judges entries, postprocess pipeline
            # tweaks, prompt edits, etc.) take effect even when re-evaluating cached
            # inference outputs. The output's config snapshot is the BASE — runtime-
            # populated fields the yaml can't know about (e.g. ``num_records`` set
            # by infer.py from the actual yielded record count) survive — and the
            # rebuild OVERLAYS only the fields it authored (yaml edits). See
            # ``SchemaInterface.merge`` for the policy + known limits.
            try:
                _rebuilt = TaskConfig.from_engine(
                    evaluation_engine=args.evaluation_engine,
                    task_name=task_name,
                    reasoning=args.reasoning,
                )
                task_configs[task_name] = TaskConfig.merge(
                    base=_output.get("config"),
                    overlay=_rebuilt,
                )
            except Exception as ex:
                logger.warning(
                    'Failed to rebuild task_config for %s (%s); falling back to cached config snapshot',
                    task_name, ex,
                )
                # Reattach the cached snapshot's raw yaml so the next save
                # still persists a ``_output["yaml"]`` (history continuity)
                # and so a later resume's merge stays yaml-aware.
                _cfg = _output["config"]
                _cached_raw_yaml = _output.get("yaml")
                try:
                    _instance = TaskConfig.ensure(
                        _cfg,
                        mode="direct" if not args.reasoning else "reasoning",
                    )
                    if isinstance(_cached_raw_yaml, dict):
                        _instance._raw_yaml = _cached_raw_yaml
                    task_configs[task_name] = _instance
                except Exception:
                    task_configs[task_name] = _cfg
            inference_runtimes[task_name] = _output.get("runtime_inference", None)
    if not (len(args.benchmarks) == len(inference_outputs) == len(task_configs)):
        raise ValueError(
            f'Length not match between benchmarks and inference_outputs: {len(args.benchmarks)} vs. {len(inference_outputs)} vs. {len(task_configs)}'
        )

    # evaluation
    evaluation_outputs = None
    if not args.skip_evaluation:
        evaluation_outputs = OrderedDict()
        for task_name in args.benchmarks: # to keep order
            evaluation_outputs[task_name] = None
        
        left_task_names = copy.deepcopy(args.benchmarks)
        left_evaluation_methods = copy.deepcopy(args.evaluation_methods)
        left_output_filenames = copy.deepcopy(output_filenames)
        if args.resume: # skip if result exist and resume
            left_task_names, left_evaluation_methods, left_output_filenames = list(), list(), list()
            for task_idx, (task_name, evaluation_method, output_filename) in tqdm(
                enumerate(zip(args.benchmarks, args.evaluation_methods, output_filenames)), 
                initial=0, 
                total=len(args.benchmarks), 
                desc=f'Resume evaluation: {args.evaluation_engine}'
            ):
                _output_filepath = os.path.join(args.output_dirpath, "output", output_filename)
                if os.path.exists(_output_filepath):
                    _output = read_file(filepath=_output_filepath)
                    if isinstance(_output.get("evaluation", None), dict):
                        logger.info('Skip evaluation %s with loading output: %s', task_name, _output_filepath)
                        evaluation_outputs[task_name] = _output["evaluation"]
                        continue
                left_task_names.append(task_name)
                left_evaluation_methods.append(evaluation_method)
                left_output_filenames.append(output_filename)
                logger.warning('Cannot resume evaluation %s since evaluation key not found in output: %s', task_name, _output_filepath)
            logger.info('Loaded previous job: %d/%d', len(args.benchmarks)-len(left_task_names), len(args.benchmarks))
        
        if not (len(left_task_names) == len(left_evaluation_methods) == len(left_output_filenames)):
            raise ValueError(
                f'Length not match between left_benchmarks, left_evaluation_methods, and left_output_filenames: {len(left_task_names)} vs. {len(left_evaluation_methods)} vs. {len(left_output_filenames)}'
            )

        for task_idx, (task_name, evaluation_method, output_filename) in tqdm(
            enumerate(zip(left_task_names, left_evaluation_methods, left_output_filenames)),
            initial=0,
            total=len(left_task_names),
            desc=f'Engines for evaluation: {args.inference_engine}/{args.evaluation_engine}'
        ):
            _inference_outputs = inference_outputs[task_name]
            # Fail-graceful skip: a benchmark whose inference produced nothing
            # (e.g. infer.py:628 "failed to get inference_result", upstream API
            # quota error, network timeout) leaves an empty list here. Without
            # this guard, the per-run `_inference_outputs[_run_index]` access
            # below raises IndexError and kills the entire evaluation phase,
            # losing the metrics of every benchmark that finished successfully.
            # Skip cleanly and let the other benchmarks proceed.
            if (
                _inference_outputs is None
                or len(_inference_outputs) < 1
                or all(not _run for _run in _inference_outputs)
            ):
                logger.warning(
                    'Skip evaluation %s: empty inference_outputs (inference likely '
                    'failed upstream; check infer logs and re-run with '
                    '--resume=True --benchmarks=%s to retry just this task)',
                    task_name, task_name,
                )
                evaluation_outputs[task_name] = None
                continue
            # Single entry point: coerce dict → TaskConfig + mode unwrap
            # (direct/reasoning) + apply_reasoning_defaults. JSON-hydrated
            # configs (resume path) get the same treatment as fresh yaml builds.
            _task_config = TaskConfig.ensure(
                task_configs[task_name],
                mode="direct" if not args.reasoning else "reasoning",
            )
            _inference_runtimes = inference_runtimes.get(task_name, None)

            # postprocess
            _runtime_postprocess_per_run = [0.0] * _task_config.meta.num_runs
            if evaluation_method in [
                EvaluationMethod.perplexity,
            ]:
                pass
            else:
                postprocess_variants, postprocess_cond_key = get_postprocess_functions(
                    evaluation_engine=args.evaluation_engine,
                    task_name=task_name,
                    task_config=_task_config,
                    postprocess_pipeline=args.postprocess_pipeline,
                    postprocess_version=args.postprocess_version,
                    postprocess_api_name=args.postprocess_api_name,
                    postprocess_allow_api=args.postprocess_allow_api,
                    parse_boxed=args.parse_boxed,
                    verbose=args.postprocess_verbose,
                )

                if any(len(_v) > 0 for _v in postprocess_variants.values()):
                    _desc_variant_label = (
                        f"conditional[{postprocess_cond_key}]={list(postprocess_variants.keys())}"
                        if postprocess_cond_key
                        else list((postprocess_variants.get(None) or {}).keys())
                    )
                    for _run_index, _run_records in enumerate(_inference_outputs):
                        _started_at_postprocess = datetime.now()
                        for _record_idx, _record in tqdm(
                            enumerate(_run_records),
                            initial=0,
                            total=len(_run_records),
                            desc=f'Postprocessing run {_run_index}: {_desc_variant_label}'
                        ):
                            _prediction = _record.get("prediction", None)
                            if isinstance(_prediction, str):
                                _prediction = _prediction.strip()
                                if _prediction.endswith("<|im_end|>"):
                                    _prediction = _prediction[:-len("<|im_end|>")].strip()

                            _query = ChatMessage.get_query(message=_record["messages"][-1])
                            if (
                                not isinstance(_prediction, str)
                                or len(_prediction) < 1
                            ):
                                continue

                            # Per-record postprocess kwargs collected from each
                            # TaskPostprocessLogic.extra. `None` entries are looked up
                            # in record.meta (e.g. `data_info: null` → record.meta.data_info).
                            # Built once per record and applied across every chain step;
                            # processors that don't use a particular kwarg absorb it via **kwargs.
                            _pp_new = _task_config.postprocess
                            _postprocess_kwargs = dict()
                            if _pp_new is not None and _pp_new.chain:
                                for _proc_name, _entry in _pp_new.chain.items():
                                    if isinstance(_entry, TaskPostprocessLogic):
                                        _logic_entries = [_entry]
                                    elif isinstance(_entry, dict):
                                        _logic_entries = [v for v in _entry.values() if isinstance(v, TaskPostprocessLogic)]
                                    else:
                                        _logic_entries = []
                                    for _logic in _logic_entries:
                                        if not _logic.extra:
                                            continue
                                        for _k, _v in _logic.extra.items():
                                            if _v is None:
                                                _postprocess_kwargs[_k] = _record.get("meta", dict()).get(_k, None)
                                            else:
                                                _postprocess_kwargs.setdefault(_k, _v)

                            # Pick the chain for this sample. When conditional_on is set,
                            # look up sample.meta[key]; on miss, the chain is empty (no-op).
                            _variant_key = None
                            if postprocess_cond_key:
                                _variant_key = (_record.get("meta", {}) or {}).get(
                                    postprocess_cond_key, None
                                )
                                _chain = postprocess_variants.get(_variant_key, OrderedDict())
                            else:
                                _chain = postprocess_variants.get(None, OrderedDict())

                            _has_postprocessed = False
                            for _function_name, _function in _chain.items():
                                _prediction_postprocessed = _function(
                                    prediction=_prediction,
                                    query=_query,
                                    options=_record.get("options", None),
                                    option_contents=_record.get("option_contents", None),
                                    **_postprocess_kwargs,
                                )
                                # parse_think returns {"prediction": ..., "reasoning_content": ...};
                                # unwrap so downstream postprocessors keep receiving strings.
                                if isinstance(_prediction_postprocessed, dict):
                                    _reasoning_content = _prediction_postprocessed.get("reasoning_content", None)
                                    if _reasoning_content is not None:
                                        _inference_outputs[_run_index][_record_idx]["reasoning_content"] = _reasoning_content
                                    _prediction_postprocessed = _prediction_postprocessed.get("prediction", None)
                                if _prediction_postprocessed is not None:
                                    _prediction = _prediction_postprocessed
                                    _has_postprocessed = True
                            if _has_postprocessed:
                                _inference_outputs[_run_index][_record_idx]["prediction_postprocessed"] = _prediction
                        _completed_at_postprocess = datetime.now()
                        _runtime_postprocess = (_completed_at_postprocess - _started_at_postprocess)
                        _runtime_postprocess_per_run[_run_index] = _runtime_postprocess.seconds + _runtime_postprocess.microseconds * 1e-6

            # evaluate
            custom_module = get_custom_module(
                evaluation_engine=args.evaluation_engine,
                task_name=task_name,
            )
            
            evaluation_output = EvaluationOutput(
                inference_engine=args.inference_engine,
                evaluation_engine=args.evaluation_engine,
                task_name=task_name,
                evaluation_method=evaluation_method,
                num_runs=_task_config.meta.num_runs,
            )
            _evaluation_run_output, _sample_metrics_run = None, None
            for _run_index in range(0, _task_config.meta.num_runs):
                _run_records = _inference_outputs[_run_index]

                _evaluation_run_output = None
                _started_at = datetime.now()
                if (
                    args.evaluation_engine == EvaluationEngine.builtin
                    and custom_module
                    and hasattr(custom_module, "evaluate")
                ): # custom inference
                    # Multimodal restoration was hoisted out of
                    # ``builtin.evaluate_task`` into a shared helper so custom
                    # evaluate paths (which skip evaluate_task entirely) also
                    # get real image/audio in messages before hitting
                    # judge/verifier API renderers — otherwise the sample-level
                    # multimodal cleanup (value=None on dump) causes crashes.
                    from omni_evaluator.evaluation.builtin.engine import restore_multimodal_items
                    restore_multimodal_items(
                        records=_run_records,
                        task_name=task_name,
                        task_config=_task_config,
                        system_prompt=args.system_prompt,
                        task_prompt=args.task_prompt,
                        num_ocr_tokens=args.num_ocr_tokens,
                        num_entity_tokens=args.num_entity_tokens,
                        num_subtitle_cues=args.num_subtitle_cues,
                        local_dirpath=args.local_dirpath,
                        cache_dirpath=args.cache_dirpath,
                    )
                    _evaluation_run_output, _sample_metrics_run = custom_module.evaluate(
                        args=args,
                        evaluation_method=evaluation_method,
                        task_name=task_name,
                        task_config=_task_config,
                        records=_run_records,
                    )

                else:
                    if args.evaluation_engine == EvaluationEngine.builtin:
                        from omni_evaluator.evaluation.builtin import evaluate_task
                        _evaluation_run_output, _sample_metrics_run = evaluate_task(
                            evaluation_engine=args.evaluation_engine,
                            task_name=task_name,
                            task_config=_task_config,
                            evaluation_method=evaluation_method,
                            records=_run_records,
                            system_prompt=args.system_prompt,
                            task_prompt=args.task_prompt,
                            num_ocr_tokens=args.num_ocr_tokens,
                            num_subtitle_cues=args.num_subtitle_cues,
                            num_entity_tokens=args.num_entity_tokens,
                            reasoning=args.reasoning,
                            num_fewshot=args.num_fewshot,
                            fewshot_image_max_size=args.fewshot_image_max_size,
                            cache_dirpath=args.cache_dirpath,
                            local_dirpath=args.local_dirpath,
                            do_async=args.do_async,
                            debug=args.debug,
                        )

                    elif args.evaluation_engine == EvaluationEngine.lmms_eval:
                        require_package("lmms_eval", extras='.[lmms_eval]', feature="lmms_eval evaluation engine")
                        from omni_evaluator.evaluation.lmms_eval import evaluate_task
                        _evaluation_run_output, _sample_metrics_run = evaluate_task(
                            evaluation_engine=args.evaluation_engine,
                            task_name=task_name,
                            evaluation_method=evaluation_method,
                            task_config=_task_config,
                            records=_run_records,
                            task_manager=None,
                            output_path=args.cache_dirpath,
                            process_with_media=False,
                            cache_requests=True,
                            rewrite_requests_cache=False,
                            system_instruction=args.system_prompt,
                            apply_chat_template=False,
                            fewshot_as_multiturn=False,
                            num_fewshot=args.num_fewshot if args.num_fewshot > 0 else None,
                            # gen_kwargs=gen_kwargs,
                            # predict_only=predict_only,
                            # fewshot_random_seed=fewshot_random_seed,
                            log_samples=True,
                            bootstrap_iters=100000,
                            debug=args.debug,
                            # num_fewshot=num_fewshot,
                        )

                    elif args.evaluation_engine == EvaluationEngine.lm_eval_harness:
                        require_package("lm_eval", extras='.[lm_eval]', feature="lm_eval_harness evaluation engine")
                        from omni_evaluator.evaluation.lm_eval_harness import evaluate_task
                        _evaluation_run_output, _sample_metrics_run = evaluate_task(
                            evaluation_engine=args.evaluation_engine,
                            task_name=task_name,
                            task_config=_task_config,
                            evaluation_method=evaluation_method,
                            records=_run_records,
                            task_manager=None,
                            output_path=args.cache_dirpath,
                            process_with_media=False,
                            cache_requests=True,
                            rewrite_requests_cache=False,
                            system_instruction=args.system_prompt,
                            apply_chat_template=False,
                            fewshot_as_multiturn=False,
                            trust_remote_code=True,
                            num_fewshot=args.num_fewshot
                            if isinstance(args.num_fewshot, int) and args.num_fewshot >= 0 else None,
                            # gen_kwargs=gen_kwargs,
                            # predict_only=predict_only,
                            # fewshot_random_seed=fewshot_random_seed,
                            log_samples=True,
                            bootstrap_iters=100000,
                            debug=args.debug,
                        )

                    elif args.evaluation_engine == EvaluationEngine.vlm_eval_kit:
                        require_package("vlmeval", extras='.[vlmeval]', feature="vlm_eval_kit evaluation engine")
                        from omni_evaluator.evaluation.vlm_eval_kit import evaluate_task
                        _evaluation_run_output, _sample_metrics_run = evaluate_task(
                            evaluation_engine=args.evaluation_engine,
                            dataset_name=task_name,
                            evaluation_method=evaluation_method,
                            benchmark_config=_task_config,
                            records=_run_records,
                            model_name=None,
                            config=None,
                            fps=args.fps,
                            nframe=args.max_video_frames,
                            api_nproc=args.inference_concurrency,
                            retry=None,
                            judge=None,
                            judge_args=None,
                            use_verifier=False,
                            use_vllm=False,
                            verbose=True,
                        )

                # update _sample_metrics_run
                if (
                    isinstance(_sample_metrics_run, list)
                    and len(_sample_metrics_run) == len(_run_records)
                ):
                    for _record_idx, _record in enumerate(_run_records):
                        if not isinstance(_inference_outputs[_run_index][_record_idx]["metrics"], dict):
                            _inference_outputs[_run_index][_record_idx]["metrics"] = dict()
                        if isinstance(_sample_metrics_run[_record_idx], dict):
                            _inference_outputs[_run_index][_record_idx]["metrics"].update(_sample_metrics_run[_record_idx])

                if (
                    not args.enable_verifier
                    or evaluation_method == EvaluationMethod.perplexity
                ):
                    # Perplexity-mode tasks compare option NLL — even when the
                    # inference module fills `prediction` (e.g. qwen2_omni's
                    # compute_perplexity → argmin letter), the judge prompt is
                    # essentially redundant with the multiple-choice acc already
                    # produced. Skip the judge call and emit explicit `null` so
                    # downstream readers can distinguish "skipped for ppl" from
                    # "judge not requested" (`enable_verifier=False` leaves
                    # the key absent).
                    if isinstance(_evaluation_run_output.metrics, dict):
                        _evaluation_run_output.metrics.setdefault("verifier_score", None)
                    if isinstance(_evaluation_run_output.sample_metrics, list):
                        for _sm in _evaluation_run_output.sample_metrics:
                            if isinstance(_sm, dict):
                                _sm.setdefault("verifier_score", None)
                
                else: # enable verifier
                    # Verifier prompt is ALWAYS prompts/verifier.py; only
                    # --verifier-reasoning toggles the CoT variant. Both backends
                    # (huggingface / api/*) run through ONE Verifier instance
                    # configured entirely from VerifierArgs — no _DEFAULT_JUDGE_LOGICS
                    # and no task_config["judges"]["judge_score"] (lang fixed "en";
                    # reason_format is an internal postprocess gate in Verifier).
                    # Prompt formatting (query/label extraction + template) is owned
                    # by Verifier._format_verifier_prompt — here we only filter to
                    # eligible records and keep their original indexes for merge-back.
                    _judge_records, _record_indexes = list(), list()
                    for _record_idx, _record in enumerate(_run_records):
                        # skip if prediction or label is empty
                        if (
                            not _record["prediction"]
                            or not _record["label"]
                        ):
                            continue
                        _judge_records.append(_record)
                        _record_indexes.append(_record_idx)

                    if not _judge_records:
                        # all records were skipped (missing prediction/label) -> nothing to score
                        logger.warning(
                            "enable_verifier: no eligible records (all skipped due to empty prediction/label); skipping verifier"
                        )
                        verifier_score_results = {"metrics": {}, "sample_metrics": [], "group_metrics": {}}
                    else:
                        # ONE Verifier for both backends. Config precedence is
                        # PER FIELD: task config (``verifier:`` block) wins when set,
                        # else the VerifierArgs (CLI/default) value.
                        # `del` -> __del__ releases any GPU model after the pass
                        # (repo convention, cf. engine.main's `del hf_inferencer`).
                        # NOTE: api/* concurrency is governed by --verifier-batch-size
                        # (default 1 = sequential), not the legacy do_async batch_size=16.
                        # Start from VerifierArgs (CLI/default), then overlay the
                        # task's ``verifier:`` block — only its explicitly-set
                        # (non-None) fields win, per field.
                        _verifier_kwargs = {
                            "engine": args.verifier_engine,
                            "model_name_or_path": args.verifier_model_name_or_path,
                            "model_group": args.verifier_model_group,
                            "device_map": args.verifier_device_map,
                            "gguf_filename": args.verifier_gguf_filename,
                            "alias": args.verifier_alias,
                            "api_name": args.verifier_api_name,
                            "reasoning": args.verifier_reasoning,
                            "num_concurrency": args.verifier_num_concurrency,
                            "num_cpu_threads": args.verifier_num_cpu_threads,
                            "max_seq_len": args.verifier_max_seq_len,
                        }
                        # Generation params flow through evaluate(generation_options=...),
                        # not the Verifier constructor (they are normalized per engine there).
                        _verifier_generation_options = {
                            "temperature": args.verifier_temperature,
                            "max_new_tokens": args.verifier_max_new_tokens,
                        }
                        _task_config_verifier = _task_config.get("verifier")  # TaskVerifier | None
                        if hasattr(_task_config_verifier, "to_dict"):
                            _task_config_verifier = _task_config_verifier.to_dict()
                        _task_config_verifier = _task_config_verifier or dict()
                        # per-field task override (task value wins when set): split across
                        # constructor kwargs and generation options.
                        _verifier_kwargs.update({
                            _k: _v for _k, _v in _task_config_verifier.items()
                            if _v is not None and _k in _verifier_kwargs
                        })
                        _verifier_generation_options.update({
                            _k: _v for _k, _v in _task_config_verifier.items()
                            if _v is not None and _k in _verifier_generation_options
                        })
                        _verifier = Verifier(
                            torch_dtype=getattr(args, "torch_dtype", None),
                            cache_dir=getattr(args, "hf_hub_cache", None),
                            verbose=args.verifier_verbose,
                            **_verifier_kwargs,   # device_map included (VerifierArgs default cpu)
                        )

                        try:
                            verifier_score_results = _verifier.evaluate(
                                records=_judge_records,
                                target_metrics=["verifier_score", ],
                                generation_options=_verifier_generation_options,
                            )
                        finally:
                            del _verifier

                    _evaluation_run_output.metrics.update(verifier_score_results["metrics"])
                    if isinstance(verifier_score_results.get("group_metrics", None), dict):
                        if not _evaluation_run_output.group_metrics:
                            _evaluation_run_output.group_metrics = dict()
                        for _group_name, _group_metrics in verifier_score_results["group_metrics"].items():
                            if _group_name not in _evaluation_run_output.group_metrics:
                                _evaluation_run_output.group_metrics[_group_name] = dict()
                            _evaluation_run_output.group_metrics[_group_name].update(_group_metrics)
                    if isinstance(verifier_score_results.get("sample_metrics", None), (list, tuple)):
                        _cursor = 0
                        # some custom_module.evaluate impls return sample_metrics shorter
                        # than records (zip-truncated); skip out-of-bounds records to avoid
                        # IndexError.
                        _run_sample_metrics = _evaluation_run_output.sample_metrics or []
                        # ``responses`` carries the judge model's raw generated_text
                        # per record (parallel to sample_metrics). Mirroring it onto
                        # the record-level metrics gives downstream audit a way to
                        # inspect the actual judge response, not just the parsed
                        # rating — useful when the parser silently fails or the
                        # judge model emits malformed output.
                        _run_responses = verifier_score_results.get("responses") or []
                        for _record_idx in _record_indexes:
                            _sample_metrics = verifier_score_results["sample_metrics"][_cursor]
                            _resp_map = _run_responses[_cursor] if _cursor < len(_run_responses) else None
                            _cursor += 1
                            if not _sample_metrics and not _resp_map:
                                continue
                            if _record_idx >= len(_run_sample_metrics):
                                logger.warning(
                                    "enable_verifier: skip record_idx=%d (out of "
                                    "evaluation_run_output.sample_metrics, size=%d)",
                                    _record_idx, len(_run_sample_metrics),
                                )
                                continue
                            if _sample_metrics:
                                _run_sample_metrics[_record_idx].update(_sample_metrics)
                            # ``responses[_idx]`` is ``{target_metric: raw_response_str}``.
                            # Surface the ``verifier_score`` raw text under a stable key
                            # so audits can read it alongside the parsed score.
                            _raw_response = None
                            if isinstance(_resp_map, dict):
                                _raw_response = _resp_map.get("verifier_score")
                            if _raw_response:
                                _run_sample_metrics[_record_idx]["verifier_score/response"] = _raw_response
                            # Mirror into inference[run_idx][record_idx].metrics so the
                            # JSON's per-sample inference block carries verifier_score next
                            # to anls/edit_distance/etc. (same pattern as lines 389-393).
                            if not isinstance(_inference_outputs[_run_index][_record_idx].get("metrics"), dict):
                                _inference_outputs[_run_index][_record_idx]["metrics"] = dict()
                            if _sample_metrics:
                                _inference_outputs[_run_index][_record_idx]["metrics"].update(_sample_metrics)
                            if _raw_response:
                                _inference_outputs[_run_index][_record_idx]["metrics"]["verifier_score/response"] = _raw_response
            
                _completed_at = datetime.now()

                # update meta
                _evaluation_run_output.inference_engine = args.inference_engine
                _evaluation_run_output.evaluation_engine = args.evaluation_engine
                _evaluation_run_output.task_name = task_name
                _evaluation_run_output.evaluation_method = evaluation_method
                _evaluation_run_output.run_index = _run_index
                _evaluation_run_output.num_runs = _task_config.meta.num_runs
                # compute runtimes, latency, and throughput
                # runtime_inference: wall-clock for the inference call, captured in infer.py per run.
                # If missing (legacy outputs), record as None — never back-derive from per-sample latencies.
                _num_valid_samples = 0
                for _record in _inference_outputs[0]:
                    if not _record["prediction"]:
                        continue
                    _num_valid_samples += 1
                _runtime_inference = None
                if (
                    isinstance(_inference_runtimes, (list, tuple))
                    and _run_index < len(_inference_runtimes)
                ):
                    _runtime_inference = _inference_runtimes[_run_index]
                _evaluation_run_output.runtime_inference = _runtime_inference
                if (
                    _num_valid_samples > 0
                    and isinstance(_runtime_inference, (int, float))
                    and _runtime_inference > 0
                ):
                    _evaluation_run_output.latency = _runtime_inference / _num_valid_samples
                    _evaluation_run_output.throughput = _num_valid_samples / _runtime_inference
                # runtime_evaluation: wall-clock for the metric computation only
                # (postprocess is timed separately into runtime_postprocess)
                _runtime_evaluation = (_completed_at - _started_at)
                _runtime_evaluation = _runtime_evaluation.seconds + _runtime_evaluation.microseconds * 1e-6
                _evaluation_run_output.runtime_evaluation = _runtime_evaluation
                _evaluation_run_output.runtime_postprocess = _runtime_postprocess_per_run[_run_index]
                _evaluation_run_output.update_statistics(inference_run_outputs=_run_records)

                # build per-run output dict and append to run_outputs
                if hasattr(_evaluation_run_output, "to_dict"):
                    _evaluation_run_output = _evaluation_run_output.to_dict()
                else: 
                    _evaluation_run_output = _evaluation_run_output.__dict__.copy()
                evaluation_output.add_run_output(run_output=_evaluation_run_output)

            _checkpoint = evaluation_output.to_dict()
            evaluation_outputs[task_name] = _checkpoint

            if isinstance(args.output_dirpath, str):
                _output_filepath = os.path.join(args.output_dirpath, "output", output_filename)
                # Drop any base64 multimodal payloads that ``evaluate_task``
                # restoration re-inflated into in-memory messages — engines
                # re-load multimodal items from the dataset on every evaluate
                # path, so keeping them on disk just wastes space.
                _drop_message_multimodal_values_inplace(_inference_outputs)
                write_file(
                    filepath=_output_filepath,
                    obj={
                        "config": _task_config.to_dict(),
                        # Save-time yaml snapshot (post-merge, current run's
                        # yaml). Lets next resume honor explicit ``null`` and
                        # gives an audit trail of yaml history across runs.
                        "yaml": getattr(_task_config, "_raw_yaml", None),
                        "inference": _inference_outputs,
                        "evaluation": _checkpoint,
                    },
                )
                logger.info('Saved output: %s', _output_filepath)
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
                    logger.info('Uploaded output to S3: %s', _remote_output_dirpath)
                except Exception as ex:
                    logger.warning('Failed to upload output to S3: %s', _remote_output_dirpath)

    else:
        evaluation_outputs = dict()
        for task_idx, (task_name, evaluation_method, output_filename) in tqdm(
            enumerate(zip(args.benchmarks, args.evaluation_methods, output_filenames)), initial=0, total=len(args.benchmarks), desc=f'Loading existing result',
        ):
            _output_filepath = os.path.join(args.output_dirpath, "output", output_filename)
            if not os.path.exists(_output_filepath):
                logger.warning('output %s not exist: %s', task_name, _output_filepath)
                continue
            _output = read_file(filepath=_output_filepath)
            evaluation_outputs[task_name] = _output.get("evaluation", None)
    
    # format to submit leaderboard
    for task_name, output_filename in zip(args.benchmarks, output_filenames):
        _output_filepath = os.path.join(args.output_dirpath, "output", output_filename)
        if not os.path.exists(_output_filepath):
            # Fail-graceful: a benchmark whose inference failed upstream has no
            # output file. Warn and skip leaderboard formatting for this task so
            # the other benchmarks' submission_output still gets produced.
            logger.warning(
                'Skip leaderboard formatting %s: output file missing (%s). '
                'Inference likely failed; re-run with --resume=True --benchmarks=%s.',
                task_name, _output_filepath, task_name,
            )
            continue
        logger.info('Load output: %s', _output_filepath)
        _output = read_file(filepath=_output_filepath)
        _inference_data = ensure_per_run_format(_output["inference"])
        if not _inference_data or not _inference_data[0]:
            logger.warning(
                'Skip leaderboard formatting %s: empty inference records in %s',
                task_name, _output_filepath,
            )
            continue
        submission_output = LeaderboardFormatter.format(
            benchmark=task_name,
            records=_inference_data[0],
        )
        if submission_output is not None:
            _submission_output_filepath = os.path.join(args.output_dirpath, "submission_output", output_filename)
            write_file(
                filepath=_submission_output_filepath, 
                obj=submission_output,
            )
            logger.info('Saved submission output: %s', _submission_output_filepath)

    # verbose
    # Route per-task summary through ``tqdm.write`` so its lines render cleanly
    # above the active progress bar instead of being half-overwritten by the
    # next ``\r``-based bar update (the cause of the earlier line truncation).
    for task_idx, (task_name, evaluation_method, output_filename) in tqdm(
        enumerate(zip(args.benchmarks, args.evaluation_methods, output_filenames)), initial=0, total=len(args.benchmarks), desc=f'Loading existing result',
    ):
        evaluation_output = evaluation_outputs.get(task_name, None)
        if evaluation_output is None:
            continue
        elif isinstance(evaluation_output, dict):
            EvaluationOutput(**evaluation_output).verbose(_print=tqdm.write)
        else:
            evaluation_output.verbose(_print=tqdm.write)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)