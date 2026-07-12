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

import copy
import importlib
import logging
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterator, List, Tuple, Union, Optional
import yaml

logger = logging.getLogger(__name__)

from omni_evaluator import EvaluationEngine, NullPredictionPolicy
from omni_evaluator.evaluation.metrics.judge_evaluator import JudgeEvaluator
from omni_evaluator.evaluation.metrics.text_evaluator import TextEvaluator
from omni_evaluator.evaluation.prepare_dataset import load_dataset
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.evaluation import EvaluationRunOutput
from omni_evaluator.schemas.task import TaskConfig, TaskEvaluationJudge
from omni_evaluator.utils.data import generator_factory
from omni_evaluator.utils.io import read_file


DEFAULT_BENCHMARKS = [
    "infovqa_test",
    "ai2d_test",
    "docvqa_test",
    "chartqa_test",
    "textvqa_validation",
    "mmmu", # set which shows high variance in usual before & after RLHF
]

def _build_task_config(
    task_name: str,
    reasoning: Optional[bool] = False,
) -> TaskConfig:
    # Load YAML config for a builtin task and construct a TaskConfig.
    # Args: task_name - benchmark name matching a tasks subpackage, reasoning - select reasoning prompt variant
    # Returns: fully populated TaskConfig for the builtin evaluation engine
    _config_filepath = importlib.resources.files(
        f'omni_evaluator.evaluation.builtin.tasks.{task_name}'
    ).joinpath("config.yaml")
    task_config = read_file(filepath=_config_filepath)
    # Snapshot the yaml dict BEFORE any in-place mutation below — this is what
    # ``SchemaInterface.merge`` consults on resume to honor explicit ``null``
    # vs "field not authored". Attached on the returned instance as transient
    # attr ``_raw_yaml`` (not a dataclass field; carried through resume by
    # JSON-persisting it under ``_output["yaml"]`` in infer.py/evaluate.py).
    _raw_yaml = copy.deepcopy(task_config)
    task_config["task_name"] = task_name

    if (
        isinstance(task_config.get("evaluation", None), dict)
        and isinstance(task_config["evaluation"].get("judges", None), dict)
    ):
        _judges = dict()
        for _metric_name, _judge_kwargs in task_config["evaluation"]["judges"].items():
            _judges[_metric_name] = TaskEvaluationJudge(**_judge_kwargs)
        task_config["evaluation"]["judges"] = _judges

    task_config["evaluation_engine"] = EvaluationEngine.builtin
    # TaskConfig.ensure: mode unwrap (direct/reasoning) in prompts/postprocess/inference
    # + apply_reasoning_defaults (e.g. max_new_tokens floor). Single entry
    # point so yaml fresh build and JSON-hydrate paths both stay consistent.
    _instance = TaskConfig.ensure(
        task_config,
        mode="direct" if not reasoning else "reasoning",
    )
    _instance._raw_yaml = _raw_yaml
    return _instance

def get_data_iterator(
    evaluation_engine: str,
    task_name: str,
    evaluation_method: str,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    subtask_type: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_entity_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    reasoning: Optional[Union[str, bool]] = None,
    num_fewshot: Optional[int] = None,
    fewshot_image_max_size: Optional[int] = None,
    do_cot: Optional[bool] = False,
    cache_dirpath: Optional[str] = None,
    local_dirpath: Optional[str] = None,
    batch_size: Optional[int] = None,
    run_index: Optional[int] = 0,
    debug: Optional[bool] = False,
    dataset_subset_override: Optional[Dict[str, Any]] = None,
) -> Tuple[Callable[[], Iterator], TaskConfig]:
    # Build a fresh-iterator factory and task config for a builtin evaluation task.
    # Args: evaluation_engine - target engine enum, task_name - benchmark name
    # Returns: (factory, TaskConfig) — factory() yields a fresh Record iterator each call;
    #   TaskConfig.num_records is populated. The factory wraps `load_dataset` so per-rank
    #   consumers (see infer.py + split_iterator) each obtain an independent generator.
    task_config = TaskConfig.from_builtin(
        task_name=task_name,
        reasoning=reasoning,
    )
    # CLI --dataset_subset overrides config-level dataset.subset. Empty dict {}
    # clears the config subset (== all samples). None means "do not override".
    if dataset_subset_override is not None:
        # Re-run TaskDataset.__post_init__ via re-construction so validation +
        # normalization (drop None/empty keys) consistently apply.
        from omni_evaluator.schemas.task import TaskDataset
        _ds_dict = task_config.dataset.to_dict()
        _ds_dict["subset"] = dataset_subset_override or None
        task_config.dataset = TaskDataset(**_ds_dict)

    _load_kwargs = dict(
        task_name=task_name,
        task_config=task_config,
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        num_subtitle_cues=num_subtitle_cues,
        local_dirpath=local_dirpath,
        cache_dirpath=cache_dirpath,
        batch_size=batch_size,
        run_index=run_index,
    )
    # Size is needed up front for downstream split. load_dataset returns the
    # generator alongside size; we keep only the size here and let the factory
    # rebuild the generator (HF dataset is cached internally, jsonl line count
    # is cheap, so the duplicate call cost is negligible).
    _, dataset_size = load_dataset(**_load_kwargs)
    task_config.num_records = dataset_size
    if evaluation_method:
        task_config.evaluation.method = evaluation_method

    factory = generator_factory(load_dataset, **_load_kwargs)
    return factory, task_config

def restore_multimodal_items(
    records: List[Dict[str, Any]],
    task_name: str,
    task_config: Union[Dict[str, Any], TaskConfig],
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_entity_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    cache_dirpath: Optional[str] = None,
    local_dirpath: Optional[str] = None,
) -> None:
    """Re-hydrate ``records[i]["messages"]`` multimodal payloads in place.

    Sample-level multimodal cleanup (see
    ``omni_evaluator/evaluate.py:_drop_message_multimodal_values_inplace``)
    strips base64/PIL payloads from ``audio``/``image``/``video`` content
    items on dump — so on resume evaluate the messages carry
    ``value=None`` and the judge/verifier API-render crashes.

    Rather than overwrite the cached message tree with a fresh
    ``load_dataset`` build (which would silently re-render text with the
    *current* CLI ``system_prompt`` / ``task_prompt`` rather than the ones
    used at inference time), we patch **only** the missing multimodal
    ``value`` fields, keyed by (message_idx, content_idx). Text content
    and message ordering stay untouched — so the inference-time prompt is
    preserved and the judge/verifier sees the exact same conversation the
    model actually saw, just with the image/audio bytes re-attached.

    Mutates ``records`` in place.
    """
    _dataset_iterator, _dataset_size = load_dataset(
        task_name=task_name,
        task_config=task_config,
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        num_subtitle_cues=num_subtitle_cues,
        local_dirpath=local_dirpath,
        cache_dirpath=cache_dirpath,
        batch_size=1,
        run_index=0,
    )
    for _record_idx, _fresh_rec in tqdm(
        enumerate(_dataset_iterator),
        initial=0,
        total=_dataset_size,
        desc='Restoring multimodal items',
    ):
        if _record_idx >= len(records):
            break
        _cached_msgs = records[_record_idx].get("messages")
        _fresh_msgs = _fresh_rec.get("messages") if isinstance(_fresh_rec, dict) else getattr(_fresh_rec, "messages", None)
        if not isinstance(_cached_msgs, list) or not _fresh_msgs:
            continue
        # normalize fresh messages to dict form so both sides look the same
        _fresh_dicts = [
            (_m.to_dict() if hasattr(_m, "to_dict") else _m)
            for _m in _fresh_msgs
        ]
        # Zip multimodal items across the whole dialogue (ignoring role /
        # message boundaries / text content) — robust to prompt-only
        # differences between inference-time cache and the fresh rebuild.
        for _cached_content, _fresh_content in zip(
            ChatMessage.iter_multimodal_contents(_cached_msgs),
            ChatMessage.iter_multimodal_contents(_fresh_dicts),
        ):
            if _cached_content.get("value", None) is not None:
                continue                                            # already has payload — skip
            if _cached_content.get("type", None) != _fresh_content.get("type", None):
                continue                                            # sanity: modality must match
            _cached_content["value"] = _fresh_content.get("value", None)


def evaluate_task(
    evaluation_engine: str,
    task_name: str,
    task_config: Union[Dict[str, Any], TaskConfig],
    evaluation_method: str,
    records: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    subtask_type: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_entity_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    reasoning: Optional[Union[str, bool]] = None,
    num_fewshot: Optional[int] = None,
    fewshot_image_max_size: Optional[int] = None,
    cache_dirpath: Optional[str] = None,
    local_dirpath: Optional[str] = None,
    do_async: Optional[bool] = False,
    debug: Optional[bool] = False,
) -> Tuple[EvaluationRunOutput, List[Dict[str, Any]]]:
    # Run text/judge evaluation on inference records and aggregate metrics.
    # Args: records - list of inference result dicts
    # Returns: tuple of (EvaluationRunOutput with aggregated metrics, per-sample metric dicts)
    # update evaluation_config
    _task_config = TaskConfig.from_builtin(
        task_name=task_name,
        reasoning=reasoning,
    )
    task_config.evaluation = _task_config.evaluation

    # restore multimodal items — hoisted into ``restore_multimodal_items`` so
    # the same restoration also runs before ``custom_module.evaluate`` in
    # ``evaluate.py`` (custom evaluate paths otherwise see ``value=None``
    # after the multimodal cleanup and crash on judge/verifier image render).
    restore_multimodal_items(
        records=records,
        task_name=task_name,
        task_config=task_config,
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        num_subtitle_cues=num_subtitle_cues,
        local_dirpath=local_dirpath,
        cache_dirpath=cache_dirpath,
    )

    metrics = dict()
    group_metrics = dict()
    sample_metrics = [dict() for _ in range(0, len(records))]
    if "audio" in task_config.meta.output_modality:
        pass

    elif "image" in task_config.meta.output_modality:
        pass

    elif "text" in task_config.meta.output_modality:
        # TaskEvaluation.text_evaluator carries per-metric kwargs in dict form.
        # Empty when no text metrics are declared; skip the call in that case.
        _text_evaluator = task_config.evaluation.text_evaluator
        if _text_evaluator is not None and _text_evaluator.metrics:
            _evaluation_results = TextEvaluator.evaluate(
                # ``metrics`` is already ``Dict[str, Dict[str, Any]]`` — per-metric
                # yaml kwargs flow straight through without a per-metric schema.
                target_metrics=_text_evaluator.metrics,
                records=records,
                group_field=_text_evaluator.group_field or "category",
                sources=None, # comet
                confidences=None, # calibration_error
                do_normalize=(
                    _text_evaluator.do_normalize
                    if _text_evaluator.do_normalize is not None
                    else task_config.evaluation.do_normalize
                ),
                null_prediction_policy=_text_evaluator.null_prediction_policy or NullPredictionPolicy.miss,
                fallback_value=_text_evaluator.fallback_value,
                # Inline OR cascade — runtime override > yaml task-level > False.
                # (Per-metric yaml ``do_async`` is consulted inside the
                # tree_edit_score dispatch where the kwargs dict is in scope.)
                do_async=(do_async or _text_evaluator.do_async or False),
            )
        else:
            _evaluation_results = None
        if _evaluation_results is not None:
            _metrics = _evaluation_results.pop("scores", None)
            if _metrics:
                metrics.update(_metrics)

            _group_metrics = _evaluation_results.pop("group_metrics", None)
            if (
                isinstance(_group_metrics, dict)
                and len(_group_metrics) > 0
            ):
                for _group_name, _group_metrics_ in _group_metrics.items():
                    if _group_name not in group_metrics:
                        group_metrics[_group_name] = dict()
                    group_metrics[_group_name].update(_group_metrics_)

            _sample_metrics = _evaluation_results.pop("sample_scores", None)
            if (
                isinstance(_sample_metrics, dict)
                and len(_sample_metrics) > 0
            ):
                for _metrics_name, _metric_values in _sample_metrics.items():
                    if len(_metric_values) != len(records):
                        continue
                    for _record_idx, _metric_value in enumerate(_metric_values):
                        sample_metrics[_record_idx][_metrics_name] = _metric_value

    elif "video" in task_config.meta.output_modality:
        pass

    # Judge metrics are sourced from TaskEvaluation.judge_evaluator (canonical
    # new structure). Empty/None when no judge metric is declared.
    _judge_evaluator = task_config.evaluation.judge_evaluator
    judge_metrics: List[str] = (
        list(_judge_evaluator.metrics.keys())
        if (_judge_evaluator is not None and _judge_evaluator.metrics)
        else list()
    )
    for _judge_metric in tqdm(
        judge_metrics,
        initial=0,
        total=len(judge_metrics),
        desc=f'JudgeEvaluator: {judge_metrics}',
    ):
        _judge_logic = _judge_evaluator.metrics[_judge_metric]
        _judge_kwargs = _judge_logic.to_kwargs()
        # `max_rating` (and every other judge kwarg) flows through `_judge_kwargs`
        # — JudgeEvaluator splats it into the leaf record_async/sync calls.
        _judge_results = JudgeEvaluator.evaluate(
            target_metrics=[_judge_metric, ],
            records=records,
            judge_kwargs_list=_judge_kwargs,
            judge_prompt_list=None,
            # Inline OR cascade — runtime > per-metric yaml > task-level yaml > False.
            do_async=(
                do_async
                or _judge_logic.do_async
                or _judge_evaluator.do_async
                or False
            ),
        )
        _metrics = _judge_results.pop("metrics", None)
        metrics.update(_metrics)

        _group_metrics = _judge_results.pop("group_metrics", None)
        if (
            isinstance(_group_metrics, dict)
            and len(_group_metrics) > 0
        ):
            for _group_name, _group_metrics_ in _group_metrics.items():
                if _group_name not in group_metrics:
                    group_metrics[_group_name] = dict()
                group_metrics[_group_name].update(_group_metrics_)

        _sample_metrics = _judge_results.pop("sample_metrics", None)
        if (
            isinstance(_sample_metrics, (list, tuple))
            and len(_sample_metrics) > 0
        ):
            for _record_idx, _sample_metrics_ in enumerate(_sample_metrics):
                sample_metrics[_record_idx].update(_sample_metrics_)

    # omni_evaluator
    # aggregate evaluation_output — outer loop in evaluate.py overwrites
    # most meta fields after this returns, but direct callers (tests) rely on
    # the engine populating evaluation_engine / evaluation_method itself.
    evaluation_run_output = EvaluationRunOutput.from_task(
        None,
        task_name,
        task_config,
        records,
        metrics,
        group_metrics=dict(group_metrics) if len(group_metrics) > 0 else None,
        sample_metrics=sample_metrics,
        num_valid_evaluation=len(records),
        evaluation_engine=evaluation_engine,
        evaluation_method=evaluation_method,
    )
    return evaluation_run_output, sample_metrics
