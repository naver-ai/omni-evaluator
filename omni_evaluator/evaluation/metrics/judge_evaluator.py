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

import ast
import asyncio
from collections import Counter, defaultdict
import copy
import dateparser
import fractions
import importlib
import logging
import numpy as np
import os
from pathlib import Path
import PIL
import re
import regex
import string
import sys
import threading
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import traceback
from typing import Any, Dict, List, Optional, Union, Literal
import unicodedata
import yaml

from omni_evaluator.api import get_api_group, get_client
from omni_evaluator.api.chat_completions import (
    chat_completion_sync, chat_completion_async,
)
from omni_evaluator.evaluation.metrics._interface import EvaluatorInterface
from omni_evaluator.evaluation.metrics.prompts import (
    judge_binary as prompts_judge_binary,
    judge_binary_rubrics as prompts_judge_binary_rubrics,
    judge_pairwise as prompts_judge_pairwise,
    judge_pairwise_rubrics as prompts_judge_pairwise_rubrics,
    judge_rating as prompts_judge_rating,
    judge_rating_rubrics as prompts_judge_rating_rubrics,
)
from omni_evaluator.inference import (
    TIMEOUT, MAX_RETRY, WAIT_BETWEEN_RETRY,
    NUM_MAX_COROUTINES, 
)
from omni_evaluator.postprocess.custom import (
    parse_circled_answer,
)
from omni_evaluator.schemas.chat import (
    Message as ChatMessage,
    AudioContent as ChatAudioContent,
    ImageContent as ChatImageContent,
    TextContent as ChatTextContent,
    VideoContent as ChatVideoContent,
    CONTENT_ACCESSOR_MAP,
)
from omni_evaluator.schemas.generation_options import ApiGenerationOptions
from omni_evaluator.schemas.inference import Record
from omni_evaluator.utils.data import format_task_prompt, normalize_unit
from omni_evaluator.utils.string import is_numeric

logger = logging.getLogger(__name__)


JUDGE_PROMPTS = {
    "judge_binary": {
        "en": prompts_judge_binary.prompt_en,
        "ko": prompts_judge_binary.prompt_ko,
    },
    "judge_binary_rubrics": {
        "en": prompts_judge_binary_rubrics.prompt_en,
        "ko": prompts_judge_binary_rubrics.prompt_ko,
    },
    "judge_binary_example": {
        "en": prompts_judge_binary.prompt_example_en,
        "ko": prompts_judge_binary.prompt_reference_ko,
    },
    "judge_binary_example_rubrics": {
        "en": prompts_judge_binary_rubrics.prompt_reference_en,
        "ko": prompts_judge_binary_rubrics.prompt_reference_ko,
    },
    "judge_pairwise": {
        "en": prompts_judge_pairwise.prompt_en,
        "ko": prompts_judge_pairwise.prompt_ko,
    },
    "judge_pairwise_rubrics": {
        "en": prompts_judge_pairwise_rubrics.prompt_en,
        "ko": prompts_judge_pairwise_rubrics.prompt_ko,
    },
    "judge_rating": {
        "en": prompts_judge_rating.prompt_en,
        "ko": prompts_judge_rating.prompt_ko,
    },
    "judge_rating_rubrics": {
        "en": prompts_judge_rating_rubrics.prompt_en,
        "ko": prompts_judge_rating_rubrics.prompt_ko,
    },
    "judge_rating_example": {
        "en": prompts_judge_rating.prompt_example_en,
        "ko": prompts_judge_rating.prompt_reference_ko,
    },
    "judge_rating_example_rubrics": {
        "en": prompts_judge_rating_rubrics.prompt_reference_en,
        "ko": prompts_judge_rating_rubrics.prompt_reference_ko,
    },
}    

class JudgeEvaluator(EvaluatorInterface):
    metrics = [
        "judge_binary",
        "judge_rating",
        "judge_pairwise",
    ]

    @classmethod
    def evaluate(
        cls,
        target_metrics: List[str],
        records: List[Union[Dict[str, Any], Record]],
        judge_kwargs_list: Union[Dict[str, Any], List[Dict[str, Any]]],
        judge_prompt_list: Optional[Union[str, List[str]]] = None,
        exclude_rubrics: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        do_async: Optional[bool] = False,
    ):
        # NOTE: `max_rating` flows exclusively through entries of
        # `judge_kwargs_list` (each entry is a TaskEvaluationJudge.to_dict()).
        # Callers wanting a runtime override should merge it into the dict, e.g.
        #   if cli_max_rating is not None:
        #       _judge_kwargs["max_rating"] = cli_max_rating
        # This keeps a single source-of-truth and avoids duplicate-kwarg TypeErrors.
        if target_metrics is None:
            target_metrics = cls.metrics
        if isinstance(judge_kwargs_list, dict):
            judge_kwargs_list = [judge_kwargs_list, ] * len(records)
        if isinstance(judge_prompt_list, str):
            judge_prompt_list = [judge_prompt_list, ] * len(records)
            
        # evaluate by total samples
        metrics = defaultdict(list)
        group_metrics = dict()
        sample_metrics = list()
        responses = list()
        for _ in range(0, len(records)):
            sample_metrics.append(dict())
            responses.append(dict())
            
        for _idx, _target_metric in enumerate(target_metrics):
            if _target_metric.startswith("judge_binary"):
                _judge_results = cls.judge_binary(
                    records=records,
                    judge_kwargs_list=judge_kwargs_list,
                    judge_prompt_list=judge_prompt_list,
                    batch_size=batch_size,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                    do_async=do_async,
                )
                (
                    metrics,
                    group_metrics,
                    sample_metrics,
                ) = cls._collect_judge_results(
                    target_metric=_target_metric,
                    metrics=metrics,
                    group_metrics=group_metrics,
                    sample_metrics=sample_metrics,
                    records=records,
                    judge_results=_judge_results,
                    exclude_rubrics=exclude_rubrics,
                )
                for _idx, _result in enumerate(_judge_results):
                    responses[_idx][_target_metric] = _result.get("response", None)

            # ``verifier_score`` is the verifier's metric key and is owned entirely
            # by ``Verifier`` (its own evaluate() loop), not the generic judge
            # dispatch here — so it is intentionally NOT routed to judge_rating.
            if _target_metric.startswith("judge_rating"):
                _judge_results = cls.judge_rating(
                    records=records,
                    judge_kwargs_list=judge_kwargs_list,
                    judge_prompt_list=judge_prompt_list,
                    batch_size=batch_size,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                    do_async=do_async,
                )
                (
                    metrics,
                    group_metrics,
                    sample_metrics,
                ) = cls._collect_judge_results(
                    target_metric=_target_metric,
                    metrics=metrics,
                    group_metrics=group_metrics,
                    sample_metrics=sample_metrics,
                    records=records,
                    judge_results=_judge_results,
                    exclude_rubrics=exclude_rubrics,
                )
                for _idx, _result in enumerate(_judge_results):
                    responses[_idx][_target_metric] = _result.get("response", None)

            if _target_metric.startswith("judge_pairwise"):
                _judge_results = cls.judge_pairwise(
                    records=records,
                    judge_kwargs_list=judge_kwargs_list,
                    judge_prompt_list=judge_prompt_list,
                    batch_size=batch_size,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                    do_async=do_async,
                )
                (
                    metrics,
                    group_metrics,
                    sample_metrics,
                ) = cls._collect_judge_results(
                    target_metric=_target_metric,
                    metrics=metrics,
                    group_metrics=group_metrics,
                    sample_metrics=sample_metrics,
                    records=records,
                    judge_results=_judge_results,
                    exclude_rubrics=exclude_rubrics,
                )
                for _idx, _result in enumerate(_judge_results):
                    responses[_idx][f'{_target_metric}_ab'] = _result.get("response_ab", None)
                    responses[_idx][f'{_target_metric}_ba'] = _result.get("response_ba", None)

        # aggregate
        metrics = dict({
            _metric_name: np.nanmean(_metric_values)
            for _metric_name, _metric_values in metrics.items()
        }) # defaultdict to dict
        for _group_name, _group_metrics in group_metrics.items():
            # ``num_samples`` is the count of records contributing to this
            # group — surfaced alongside the per-metric means so callers can
            # judge statistical weight (matches task-custom group_metrics
            # convention in htmlbench/mia/llavaw/kovisit).
            _group_num_samples = max(
                (len(_v) for _v in _group_metrics.values() if isinstance(_v, list)),
                default=0,
            )
            for _metric_name, _metric_values in _group_metrics.items():
                group_metrics[_group_name][_metric_name] = np.mean(_metric_values)
            group_metrics[_group_name] = dict(group_metrics[_group_name])
            group_metrics[_group_name]["num_samples"] = _group_num_samples
        
        return {
            "metrics": metrics,
            "sample_metrics": sample_metrics,
            "group_metrics": group_metrics if group_metrics else None,
            "responses": responses,
        }
        
    @classmethod
    def _aggregate_judge_results(
        cls,
        results_per_model: List[List[Dict[str, Any]]],
        judge_models: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        num_records = len(results_per_model[0])
        _lengths = [len(_r) for _r in results_per_model]
        if len(set(_lengths)) > 1:
            logger.warning(
                f'judge results length mismatch across models: {_lengths}. '
                f'Missing records will be excluded from aggregation.'
            )
        aggregated = list()
        for _record_idx in range(num_records):
            _record_per_model = [
                results_per_model[_m][_record_idx]
                if _record_idx < len(results_per_model[_m])
                and isinstance(results_per_model[_m][_record_idx], dict)
                else None
                for _m in range(len(results_per_model))
            ]
            _record_results = [_r for _r in _record_per_model if _r is not None]
            if not _record_results:
                aggregated.append({})
                continue
            _per_model = {}
            for _m, _r in enumerate(_record_per_model):
                _model_key = judge_models[_m] if judge_models and _m < len(judge_models) else str(_m)
                if _r is None:
                    _per_model[_model_key] = None
                else:
                    _per_model[_model_key] = {
                        "scores": _r.get("scores"),
                        "choice": dict(_r["choice"]) if isinstance(_r.get("choice"), dict) else None,
                        "reasons": _r.get("reasons"),
                    }
            _merged = {}
            for _key in _record_results[0].keys():
                if _key == "choice":
                    # choice is a defaultdict keyed by winner ("A"/"B"/"tie") — aggregate
                    # by summing counts across models then keeping the majority key
                    _choice_totals = defaultdict(int)
                    for _r in _record_results:
                        _c = _r.get("choice")
                        if _c is not None:
                            for _ck, _cv in _c.items():
                                _choice_totals[_ck] += _cv
                    if _choice_totals:
                        _merged[_key] = dict(_choice_totals)
                    else:
                        _merged[_key] = None
                    continue
                _values = [_r[_key] for _r in _record_results if _r.get(_key) is not None]
                if not _values:
                    _merged[_key] = None
                elif isinstance(_values[0], dict):
                    _merged[_key] = {
                        _rk: float(np.nanmean([
                            _v[_rk] for _v in _values
                            if _rk in _v and isinstance(_v[_rk], (int, float))
                        ])) if any(
                            _rk in _v and isinstance(_v[_rk], (int, float))
                            for _v in _values
                        ) else None
                        for _rk in _values[0].keys()
                    }
                elif isinstance(_values[0], (int, float)):
                    _merged[_key] = float(np.nanmean(_values))
                else:
                    _merged[_key] = None
            _merged["per_model"] = _per_model
            aggregated.append(_merged)
        return aggregated

    @classmethod
    def _collect_judge_results(
        cls,
        target_metric: str,
        metrics: Dict[str, List[Union[int, float]]],
        group_metrics: Dict[str, Dict[str, List[Union[int, float]]]],
        sample_metrics: List[Dict[str, Any]],
        records: List[Union[Dict[str, Any], Record]],
        judge_results: List[Dict[str, Any]],
        exclude_rubrics: Optional[List[str]] = None,
    ):
        # Callers may pass ``exclude_rubrics=None`` (signature default). The inner
        # ``_rubric_name in exclude_rubrics`` check explodes on None — normalize once.
        if exclude_rubrics is None:
            exclude_rubrics = []
        for _record_idx, (_record, _judge_result) in enumerate(zip(records, judge_results)):
            _category_names = _record.get("meta", dict()).get("category", None)
            if not isinstance(_category_names, (tuple, list)):
                _category_names = [_category_names, ]
            for _category_name in _category_names:
                if not _category_name:
                    continue
                if (
                    _category_name
                    and _category_name not in group_metrics
                ):
                    group_metrics[_category_name] = defaultdict(list)
            
            scores = _judge_result.pop("scores", None)
            for _metric_name, _metric_value in _judge_result.items():
                if not isinstance(_metric_value, (int, float, bool)):
                    continue
                _metric_name_ = f'{target_metric}/{_metric_name}'
                metrics[_metric_name_].append(_metric_value)
                sample_metrics[_record_idx][_metric_name_] = _metric_value
                if _category_names:
                    for _category_name in _category_names:
                        if not _category_name:
                            continue
                        group_metrics[_category_name][_metric_name_].append(_metric_value)
            
            if (
                isinstance(scores, dict)
                and len(scores) > 0
            ):
                _rubric_scores = list()
                for _rubric_name, _rubric_score in scores.items():
                    if (
                        _rubric_name in exclude_rubrics
                        or not isinstance(_rubric_score, (int, float))
                    ):
                        continue
                    _metric_name_ = f'{target_metric}/{_rubric_name}'
                    metrics[_metric_name_].append(_rubric_score)
                    sample_metrics[_record_idx][_metric_name_] = _rubric_score
                    if _category_names:
                        for _category_name in _category_names:
                            if not _category_name:
                                continue
                            group_metrics[_category_name][_metric_name_].append(_rubric_score)
                    _rubric_scores.append(_rubric_score)
                if len(_rubric_scores) > 0:
                    _rubric_sum = np.sum(_rubric_scores)
                    _rubric_avg = np.nanmean(_rubric_scores)
                    metrics[f'{target_metric}/rubric_sum'].append(_rubric_sum)
                    metrics[f'{target_metric}/rubric_avg'].append(_rubric_avg)
                    sample_metrics[_record_idx][f'{target_metric}/rubric_sum'] = _rubric_sum
                    sample_metrics[_record_idx][f'{target_metric}/rubric_avg'] = _rubric_avg
                    if _category_names:
                        for _category_name in _category_names:
                            if not _category_name:
                                continue
                            group_metrics[_category_name][f'{target_metric}/rubric_sum'].append(_rubric_sum)
                            group_metrics[_category_name][f'{target_metric}/rubric_avg'].append(_rubric_avg)

            elif isinstance(scores, (int, float, bool)):
                _metric_name = f'{target_metric}'
                metrics[_metric_name].append(scores)
                sample_metrics[_record_idx][_metric_name] = scores
                if _category_names:
                    for _category_name in _category_names:
                        if not _category_name:
                            continue
                        group_metrics[_category_name][_metric_name].append(scores)

            _reasons = _judge_result.pop("reasons", None)
            if _reasons:
                if isinstance(_reasons, dict):
                    for _rubric_name, _reason_value in _reasons.items():
                        sample_metrics[_record_idx][f'{target_metric}/{_rubric_name}/reasons'] = _reason_value
                else:
                    sample_metrics[_record_idx][f'{target_metric}/reasons'] = _reasons

            _per_model = _judge_result.pop("per_model", None)
            if isinstance(_per_model, dict):
                for _model_name, _model_result in _per_model.items():
                    if not isinstance(_model_result, dict):
                        continue
                    _model_scores = _model_result.get("scores")
                    if isinstance(_model_scores, dict):
                        for _rubric_name, _rubric_score in _model_scores.items():
                            if isinstance(_rubric_score, (int, float)):
                                sample_metrics[_record_idx][f'{target_metric}/{_model_name}/{_rubric_name}'] = _rubric_score
                    elif isinstance(_model_scores, (int, float)):
                        sample_metrics[_record_idx][f'{target_metric}/{_model_name}'] = _model_scores
                    _model_choice = _model_result.get("choice")
                    if isinstance(_model_choice, dict):
                        sample_metrics[_record_idx][f'{target_metric}/{_model_name}/choice'] = _model_choice
                    _model_reasons = _model_result.get("reasons")
                    if _model_reasons is not None:
                        if isinstance(_model_reasons, dict):
                            for _rubric_name, _reason_value in _model_reasons.items():
                                if _reason_value is not None:
                                    sample_metrics[_record_idx][f'{target_metric}/{_model_name}/{_rubric_name}/reason'] = _reason_value
                        elif isinstance(_model_reasons, (list, tuple)) and len(_model_reasons) == 2:
                            sample_metrics[_record_idx][f'{target_metric}/{_model_name}/reasons'] = _model_reasons
                        elif isinstance(_model_reasons, str) and _model_reasons:
                            sample_metrics[_record_idx][f'{target_metric}/{_model_name}/reason'] = _model_reasons

        return (
            metrics,
            group_metrics,
            sample_metrics,
        )

    @classmethod
    def judge_rating(
        cls,
        records: List[Union[Dict[str, Any], Record]],
        judge_kwargs_list: List[Dict[str, Any]],
        judge_prompt_list: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        do_async: Optional[bool] = False,
    ) -> List[Dict[str, Any]]:
        if not records or not judge_kwargs_list:
            raise ValueError(
                f"empty records or judge_kwargs_list "
                f"(records={len(records) if records is not None else None}, "
                f"judge_kwargs_list={len(judge_kwargs_list) if judge_kwargs_list is not None else None}); "
                f"caller must filter eligible records before invoking JudgeEvaluator"
            )
        judge_model = judge_kwargs_list[0].get("judge_model", "")
        if isinstance(judge_model, str):
            judge_model = [judge_model]
        if len(judge_model) == 1:
            return cls.judge_rating_single(
                records=records,
                judge_kwargs_list=judge_kwargs_list,
                judge_prompt_list=judge_prompt_list,
                batch_size=batch_size,
                timeout=timeout,
                max_retry=max_retry,
                wait_between_retry=wait_between_retry,
                do_async=do_async,
            )
        results_per_model = [
            cls.judge_rating_single(
                records=records,
                judge_kwargs_list=[{**_kw, "judge_model": _m} for _kw in judge_kwargs_list],
                judge_prompt_list=judge_prompt_list,
                batch_size=batch_size,
                timeout=timeout,
                max_retry=max_retry,
                wait_between_retry=wait_between_retry,
                do_async=do_async,
            )
            for _m in judge_model
        ]
        return cls._aggregate_judge_results(results_per_model, judge_models=judge_model)

    @classmethod
    def judge_rating_single(
        cls,
        records: List[Union[Dict[str, Any], Record]],
        judge_kwargs_list: List[Dict[str, Any]],
        judge_prompt_list: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        do_async: Optional[bool] = False,
    ) -> Dict[str, float]:
        if not isinstance(timeout, (int, float)):
            timeout = TIMEOUT
        if not isinstance(max_retry, int):
            max_retry = MAX_RETRY
        if not isinstance(wait_between_retry, (int, float)):
            wait_between_retry = WAIT_BETWEEN_RETRY

        outputs = None
        if do_async:
            semaphore_size = batch_size
            if (
                not isinstance(semaphore_size, int)
                or semaphore_size < 1
            ):
                semaphore_size = NUM_MAX_COROUTINES
            semaphore_size = min(semaphore_size, NUM_MAX_COROUTINES)
            semaphore = asyncio.Semaphore(semaphore_size)
            
            tasks = list()
            for _record_idx, (_record, _judge_kwargs) in enumerate(
                zip(records, judge_kwargs_list)
            ):
                _judge_message = None
                if (
                    isinstance(judge_prompt_list, (list, tuple))
                    and _record_idx < len(judge_prompt_list)
                ):
                    _judge_message = judge_prompt_list[_record_idx]
                tasks.append(cls.judge_rating_record_async(
                    record=_record,
                    **_judge_kwargs,
                    judge_message=_judge_message,
                    semaphore=semaphore,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                ))
            outputs = asyncio.run(tqdm_asyncio.gather(*tasks, initial=0, total=len(records), desc=f'Evaluating judge_rating'))
        
        else:
            outputs = list()
            for _record_idx, (_record, _judge_kwargs) in tqdm(
                enumerate(zip(records, judge_kwargs_list)),
                initial=0,
                total=len(records),
                desc=f'Evaluating judge_rating',
            ):
                _judge_message = None
                if (
                    isinstance(judge_prompt_list, (list, tuple))
                    and _record_idx < len(judge_prompt_list)
                ):
                    _judge_message = judge_prompt_list[_record_idx]
                _output = cls.judge_rating_record_sync(
                    record=_record,
                    **_judge_kwargs,
                    judge_message=_judge_message,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                )
                outputs.append(_output)

        return outputs
    
    @classmethod
    def judge_rating_record_sync(
        cls, 
        record: Union[Dict[str, Any], Record],
        lang: str,
        judge_model: str,
        judge_message: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        baseline: Optional[str] = None,
        rubrics: Optional[Dict[str, Any]] = None,
        response_format: Optional[type] = None,
        max_tokens: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_rating: Optional[Union[int, float]] = None,
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        api_kwargs = cls._preprocess_judge_rating_record(
            record=record,
            lang=lang,
            judge_model=judge_model,
            judge_message=judge_message,
            judge_prompt=judge_prompt,
            system_prompt=system_prompt,
            instruction=instruction,
            baseline=baseline,
            rubrics=rubrics,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

        _response = chat_completion_sync(
            api_name=judge_model,
            messages=api_kwargs["judge_messages"],
            generation_options=api_kwargs["generation_options"],
            response_format=response_format,
            timeout=timeout,
            max_retry=max_retry,
            wait_between_retry=wait_between_retry,
        )

        output = cls._postprocess_judge_rating_record(
            response=_response,
            rubrics=rubrics,
            score_format=api_kwargs["score_format"],
            reason_format=api_kwargs["reason_format"],
            max_rating=max_rating,
        )
        return output

    @classmethod
    async def judge_rating_record_async(
        cls, 
        record: Union[Dict[str, Any], Record],
        lang: str,
        judge_model: str,
        judge_message: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        baseline: Optional[str] = None,
        rubrics: Optional[Dict[str, Any]] = None,
        response_format: Optional[type] = None,
        max_tokens: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_rating: Optional[Union[int, float]] = None,
        semaphore: Optional[asyncio.locks.Semaphore] = None, 
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        api_kwargs = cls._preprocess_judge_rating_record(
            record=record,
            lang=lang,
            judge_model=judge_model,
            judge_message=judge_message,
            judge_prompt=judge_prompt,
            system_prompt=system_prompt,
            instruction=instruction,
            baseline=baseline,
            rubrics=rubrics,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

        _response = await chat_completion_async(
            api_name=judge_model,
            messages=api_kwargs["judge_messages"],
            generation_options=api_kwargs["generation_options"],
            response_format=response_format,
            semaphore=semaphore,
            timeout=timeout,
            max_retry=max_retry,
            wait_between_retry=wait_between_retry,
        )

        output = cls._postprocess_judge_rating_record(
            response=_response,
            rubrics=rubrics,
            score_format=api_kwargs["score_format"],
            reason_format=api_kwargs["reason_format"],
            max_rating=max_rating,
        )
        return output

    @classmethod
    def _preprocess_judge_rating_record(
        cls,
        record: Union[Dict[str, Any], Record],
        lang: str,
        judge_model: str,
        judge_message: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        baseline: Optional[str] = None,
        rubrics: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, float]:
        judge_messages = list()
        if system_prompt:
            judge_messages.append(ChatMessage(
                role="system",
                content=[ChatTextContent(
                    type="text",
                    value=system_prompt,
                ),]
            ))
        judge_messages.append(ChatMessage(
            role="user",
            content=list(),
        ))
        
        score_format = None
        reason_format = None
        score_rubric = None
        if judge_message:
            judge_messages[-1]["content"].append(ChatTextContent(
                type="text",
                value=judge_message,
            ))
        else:
            score_format = "[SCORE]" 
            reason_format = "[REASON]" 
            score_rubric = ""
            if rubrics:
                score_format = "\n".join([
                    f'{_rubric_name}: [SCORE]'
                    for _rubric_name, _rubric_desc in rubrics.items()
                ])
                reason_format = "\n".join([
                    f'{_rubric_name}: [REASON]'
                    for _rubric_name, _rubric_desc in rubrics.items()
                ])
                score_rubric = "\n".join([
                    f"{_rubric_name}: {_rubric_desc}" 
                    for _rubric_name, _rubric_desc in rubrics.items()
                ])

            if not judge_prompt: 
                if baseline:
                    if rubrics:
                        judge_prompt = JUDGE_PROMPTS["judge_rating_example_rubrics"][lang]
                    else:
                        judge_prompt = JUDGE_PROMPTS["judge_rating_example_"][lang]
                else:
                    if rubrics:
                        judge_prompt = JUDGE_PROMPTS["judge_rating_rubrics"][lang]
                    else:
                        judge_prompt = JUDGE_PROMPTS["judge_rating"][lang]

            dialogue = list()
            for _message in record["messages"]:
                for _content in _message["content"]:
                    _content_cls = CONTENT_ACCESSOR_MAP.get(_content["type"])
                    _value_key = _content_cls.get_key(_content) if _content_cls else None
                    _content_key = _message["role"].title()
                    if _message.get("name", None):
                        _content_key = _message["name"].title()
                    _content_value = _content[_value_key] if _value_key else None
                    if _content["type"] == "text":
                        dialogue.append(
                            f'{_content_key}: {_content_value}',
                        )
                    elif _content["type"] == "audio":
                        judge_messages[-1].content.append(ChatAudioContent(
                            type="audio",
                            value=_content_value,
                        ))
                    elif _content["type"] == "image":
                        judge_messages[-1].content.append(ChatImageContent(
                            type="image",
                            value=_content_value,
                        ))
                    elif _content["type"] == "video":
                        judge_messages[-1].content.append(ChatVideoContent(
                            type="video",
                            value=_content_value,
                        ))

            _judge_prompt = format_task_prompt(
                task_prompt=judge_prompt,
                query="",
                score_format=score_format,
                reason_format=reason_format,
                instruction=instruction,
                dialogue="\n".join(dialogue),
                response=record["prediction"],
                label=record["label"],
                score_rubric=score_rubric,
                example=baseline,
            )
            judge_messages[-1]["content"].append(ChatTextContent(
                type="text",
                value=_judge_prompt,
            ))
            
        if kwargs.get("process_message_kwargs", dict()):
            judge_messages = [
                ChatMessage.preprocess_message(
                    message=_message,
                    **kwargs["process_message_kwargs"],
                )
                for _message in judge_messages
            ]
        
        api_group = get_api_group(api_name=judge_model)
        generation_options = dict()
        if max_tokens:
            generation_options["max_tokens"] = max_tokens
        if temperature:
            generation_options["temperature"] = temperature
        if top_k:
            generation_options["top_k"] = top_k
        if top_p:
            generation_options["top_p"] = top_p
        generation_options.update(kwargs)
        generation_options = ApiGenerationOptions.from_dict(
            api_name=judge_model,
            obj=generation_options,
            api_group=api_group,
        ).to_dict()

        api_kwargs = {
            "judge_messages": judge_messages,
            "generation_options": generation_options,
            "score_format": score_format,
            "reason_format": reason_format,
            "score_rubric": score_rubric,
        }
        return api_kwargs

    @classmethod
    def _postprocess_judge_rating_record(
        cls,
        response: str,
        rubrics: Optional[Dict[str, Any]] = None,
        score_format: Optional[Dict[str, Any]] = None,
        reason_format: Optional[Dict[str, Any]] = None,
        max_rating: Optional[Union[int, float]] = None,
    ) -> Dict[str, float]:
        scores = None
        if not response:
            pass
        elif rubrics and score_format:
            scores, scores_baseline = dict(), dict()
            for _rubric_name in rubrics.keys():
                scores[_rubric_name] = None
                scores_baseline[_rubric_name] = None

            _score = cls._parse_response_score(
                response=response,
                rubrics=rubrics,
                max_rating=max_rating,
            )
            for _rubric_name in rubrics.keys():
                if (
                    _rubric_name in _score
                    and isinstance(_score[_rubric_name], (int, float))
                ):
                    scores[_rubric_name] = _score[_rubric_name]
        else:
            scores = cls._parse_response_score(
                response=response,
                rubrics=None,
                max_rating=max_rating,
            )
            
        reasons = None
        if reason_format:
            try:
                reasons = cls._parse_response_reason(
                    response=response,
                    rubrics=rubrics,
                )
            except Exception as ex:
                logger.debug(f'Error in parsing reasons from judge response: {ex}')

        output = {
            "scores": scores,
            "reasons": reasons,
            "response": response,
        }
        return output

    @classmethod
    def judge_binary(
        cls,
        records: List[Union[Dict[str, Any], Record]],
        judge_kwargs_list: List[Dict[str, Any]],
        judge_prompt_list: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        do_async: Optional[bool] = False,
    ) -> List[Dict[str, Any]]:
        if not records or not judge_kwargs_list:
            raise ValueError(
                f"empty records or judge_kwargs_list "
                f"(records={len(records) if records is not None else None}, "
                f"judge_kwargs_list={len(judge_kwargs_list) if judge_kwargs_list is not None else None}); "
                f"caller must filter eligible records before invoking JudgeEvaluator"
            )
        judge_model = judge_kwargs_list[0].get("judge_model", "")
        if isinstance(judge_model, str):
            judge_model = [judge_model]
        if len(judge_model) == 1:
            return cls.judge_binary_single(
                records=records,
                judge_kwargs_list=judge_kwargs_list,
                judge_prompt_list=judge_prompt_list,
                batch_size=batch_size,
                timeout=timeout,
                max_retry=max_retry,
                wait_between_retry=wait_between_retry,
                do_async=do_async,
            )
        results_per_model = [
            cls.judge_binary_single(
                records=records,
                judge_kwargs_list=[{**_kw, "judge_model": _m} for _kw in judge_kwargs_list],
                judge_prompt_list=judge_prompt_list,
                batch_size=batch_size,
                timeout=timeout,
                max_retry=max_retry,
                wait_between_retry=wait_between_retry,
                do_async=do_async,
            )
            for _m in judge_model
        ]
        return cls._aggregate_judge_results(results_per_model, judge_models=judge_model)

    @classmethod
    def judge_binary_single(
        cls,
        records: List[Union[Dict[str, Any], Record]],
        judge_kwargs_list: List[Dict[str, Any]],
        judge_prompt_list: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        do_async: Optional[bool] = False,
    ) -> Dict[str, float]:
        if not isinstance(timeout, (int, float)):
            timeout = TIMEOUT
        if not isinstance(max_retry, int):
            max_retry = MAX_RETRY
        if not isinstance(wait_between_retry, (int, float)):
            wait_between_retry = WAIT_BETWEEN_RETRY

        outputs = None
        if do_async:
            semaphore_size = batch_size
            if (
                not isinstance(semaphore_size, int)
                or semaphore_size < 1
            ):
                semaphore_size = NUM_MAX_COROUTINES
            semaphore_size = min(semaphore_size, NUM_MAX_COROUTINES)
            semaphore = asyncio.Semaphore(semaphore_size)
            
            tasks = list()
            for _record_idx, (_record, _judge_kwargs) in enumerate(
                zip(records, judge_kwargs_list)
            ):
                _judge_message = None
                if (
                    isinstance(judge_prompt_list, (list, tuple))
                    and _record_idx < len(judge_prompt_list)
                ):
                    _judge_message = judge_prompt_list[_record_idx]
                tasks.append(cls.judge_binary_record_async(
                    record=_record,
                    **_judge_kwargs,
                    judge_message=_judge_message,
                    semaphore=semaphore,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                ))
            outputs = asyncio.run(tqdm_asyncio.gather(*tasks, initial=0, total=len(records), desc=f'Evaluating judge_binary'))
        
        else:
            outputs = list()
            for _record_idx, (_record, _judge_kwargs) in tqdm(
                enumerate(zip(records, judge_kwargs_list)),
                initial=0,
                total=len(records),
                desc=f'Evaluating judge_binary',
            ):
                _judge_message = None
                if (
                    isinstance(judge_prompt_list, (list, tuple))
                    and _record_idx < len(judge_prompt_list)
                ):
                    _judge_message = judge_prompt_list[_record_idx]
                _output = cls.judge_binary_record_sync(
                    record=_record,
                    **_judge_kwargs,
                    judge_message=_judge_message,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                )
                outputs.append(_output)

        return outputs
    
    @classmethod
    def judge_binary_record_sync(
        cls, 
        record: Union[Dict[str, Any], Record],
        lang: str,
        judge_model: str,
        judge_message: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        baseline: Optional[str] = None,
        rubrics: Optional[Dict[str, Any]] = None,
        response_format: Optional[type] = None,
        max_tokens: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_rating: Optional[Union[int, float]] = None,
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        api_kwargs = cls._preprocess_judge_binary_record(
            record=record,
            lang=lang,
            judge_model=judge_model,
            judge_message=judge_message,
            judge_prompt=judge_prompt,
            system_prompt=system_prompt,
            instruction=instruction,
            baseline=baseline,
            rubrics=rubrics,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

        _response = chat_completion_sync(
            api_name=judge_model,
            messages=api_kwargs["judge_messages"],
            generation_options=api_kwargs["generation_options"],
            response_format=response_format,
            timeout=timeout,
            max_retry=max_retry,
            wait_between_retry=wait_between_retry,
        )

        output = cls._postprocess_judge_binary_record(
            response=_response,
            rubrics=rubrics,
            reason_format=api_kwargs["reason_format"],
        )
        return output

    @classmethod
    async def judge_binary_record_async(
        cls, 
        record: Union[Dict[str, Any], Record],
        lang: str,
        judge_model: str,
        judge_message: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        baseline: Optional[str] = None,
        rubrics: Optional[Dict[str, Any]] = None,
        response_format: Optional[type] = None,
        max_tokens: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_rating: Optional[Union[int, float]] = None,
        semaphore: Optional[asyncio.locks.Semaphore] = None, 
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        api_kwargs = cls._preprocess_judge_binary_record(
            record=record,
            lang=lang,
            judge_model=judge_model,
            judge_message=judge_message,
            judge_prompt=judge_prompt,
            system_prompt=system_prompt,
            instruction=instruction,
            baseline=baseline,
            rubrics=rubrics,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

        _response = await chat_completion_async(
            api_name=judge_model,
            messages=api_kwargs["judge_messages"],
            generation_options=api_kwargs["generation_options"],
            response_format=response_format,
            semaphore=semaphore,
            timeout=timeout,
            max_retry=max_retry,
            wait_between_retry=wait_between_retry,
        )

        output = cls._postprocess_judge_binary_record(
            response=_response,
            rubrics=rubrics,
            reason_format=api_kwargs["reason_format"],
        )
        return output

    @classmethod
    def _preprocess_judge_binary_record(
        cls,
        record: Union[Dict[str, Any], Record],
        lang: str,
        judge_model: str,
        judge_message: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        baseline: Optional[str] = None,
        rubrics: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, float]:
        judge_messages = list()
        if system_prompt:
            judge_messages.append(ChatMessage(
                role="system",
                content=[ChatTextContent(
                    type="text",
                    value=system_prompt,
                ),]
            ))
        judge_messages.append(ChatMessage(
            role="user",
            content=list(),
        ))
        
        score_format = None
        reason_format = None
        score_rubric = None
        if judge_message:
            judge_messages[-1]["content"].append(ChatTextContent(
                type="text",
                value=judge_message,
            ))
        else:
            score_format = "[JUDGE]" 
            reason_format = "[REASON]" 
            score_rubric = ""
            if rubrics:
                score_format = "\n".join([
                    f'{_rubric_name}: [SCORE]'
                    for _rubric_name, _rubric_desc in rubrics.items()
                ])
                reason_format = "\n".join([
                    f'{_rubric_name}: [REASON]'
                    for _rubric_name, _rubric_desc in rubrics.items()
                ])
                score_rubric = "\n".join([
                    f"{_rubric_name}: {_rubric_desc}" 
                    for _rubric_name, _rubric_desc in rubrics.items()
                ])

            if not judge_prompt:
                if baseline: 
                    if rubrics:
                        judge_prompt = JUDGE_PROMPTS["judge_binary_example_rubrics"][lang]
                    else:
                        judge_prompt = JUDGE_PROMPTS["judge_binary_example"][lang]
                else:
                    if rubrics:
                        judge_prompt = JUDGE_PROMPTS["judge_binary_rubrics"][lang]
                    else:
                        judge_prompt = JUDGE_PROMPTS["judge_binary"][lang]
            
            dialogue = list()
            if isinstance(kwargs.get("process_message_kwargs", None), dict):
                # add audio_transcription or image_caption to judge
                if (
                    kwargs["process_message_kwargs"].get("audio_content_key", None)
                    and kwargs["process_message_kwargs"]["audio_content_key"] in record["meta"]
                    and record["meta"][kwargs["process_message_kwargs"]["audio_content_key"]]
                ):
                    _audio_content = record["meta"][kwargs["process_message_kwargs"]["audio_content_key"]]
                    judge_messages[-1].content.append(ChatTextContent(
                        type="text",
                        value=f'AudioTranscription: {_audio_content}',
                    ))
                    
                if (
                    kwargs["process_message_kwargs"].get("image_content_key", None)
                    and kwargs["process_message_kwargs"]["image_content_key"] in record["meta"]
                    and record["meta"][kwargs["process_message_kwargs"]["image_content_key"]]
                ):
                    _image_content = record["meta"][kwargs["process_message_kwargs"]["image_content_key"]]
                    judge_messages[-1].content.append(ChatTextContent(
                        type="text",
                        value=f'ImageCaption: {_image_content}',
                    ))                
            
            for _message in record["messages"]:
                for _content in _message["content"]:
                    _content_cls = CONTENT_ACCESSOR_MAP.get(_content["type"])
                    _value_key = _content_cls.get_key(_content) if _content_cls else None
                    _content_key = _message["role"].title()
                    if _message.get("name", None):
                        _content_key = _message["name"].title()
                    _content_value = _content[_value_key] if _value_key else None
                    if _content["type"] == "text":
                        dialogue.append(
                            f'{_content_key}: {_content_value}',
                        )
                    elif _content["type"] == "audio":
                        judge_messages[-1].content.append(ChatAudioContent(
                            type="audio",
                            value=_content_value,
                        ))
                    elif _content["type"] == "image":
                        judge_messages[-1].content.append(ChatImageContent(
                            type="image",
                            value=_content_value,
                        ))
                    elif _content["type"] == "video":
                        judge_messages[-1].content.append(ChatVideoContent(
                            type="video",
                            value=_content_value,
                        ))

            _judge_prompt = format_task_prompt(
                task_prompt=judge_prompt,
                query="",
                score_format=score_format,
                reason_format=reason_format,
                instruction=instruction,
                dialogue="\n".join(dialogue),
                response=record["prediction"],
                label=record["label"],
                score_rubric=score_rubric,
                example=baseline,
            )
            judge_messages[-1]["content"].append(ChatTextContent(
                type="text",
                value=_judge_prompt,
            ))
        
        if kwargs.get("process_message_kwargs", dict()):
            judge_messages = [
                ChatMessage.preprocess_message(
                    message=_message,
                    **kwargs["process_message_kwargs"],
                )
                for _message in judge_messages
            ]
        
        api_group = get_api_group(api_name=judge_model)
        generation_options = dict()
        if max_tokens:
            generation_options["max_tokens"] = max_tokens
        if temperature:
            generation_options["temperature"] = temperature
        if top_k:
            generation_options["top_k"] = top_k
        if top_p:
            generation_options["top_p"] = top_p
        generation_options.update(kwargs)
        generation_options = ApiGenerationOptions.from_dict(
            api_name=judge_model,
            obj=generation_options,
            api_group=api_group,
        ).to_dict()

        api_kwargs = {
            "judge_messages": judge_messages,
            "generation_options": generation_options,
            "score_format": score_format,
            "reason_format": reason_format,
            "score_rubric": score_rubric,
        }
        return api_kwargs
    
    @classmethod
    def _postprocess_judge_binary_record(
        cls,
        response: str,
        rubrics: Optional[Dict[str, Any]] = None,
        reason_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        # win_rate, win_point
        scores = None
        if not response:
            pass
        else:
            scores = cls._parse_response_binary(
                response=response,
                rubrics=rubrics,
                verbose=False,
            )
        
        accuracy = None
        if scores is None:
            pass
        elif isinstance(scores, bool):
            accuracy = 1.0 if scores else 0.0
        elif isinstance(scores, dict): # rubric ratings
            accuracy = sum(scores.values()) / len(rubrics)

        reasons = None
        if reason_format:
            try:
                reasons = cls._parse_response_reason(
                    response=response,
                    rubrics=rubrics,
                )
            except Exception as ex:
                logger.debug(f'Error in parsing reasons from judge response: {ex}')

        output = {
            "accuracy": accuracy,
            "scores": scores,
            "reasons": reasons,
            "response": response,
        }
        return output

    @classmethod
    def judge_pairwise(
        cls,
        records: List[Union[Dict[str, Any], Record]],
        judge_kwargs_list: List[Dict[str, Any]],
        judge_prompt_list: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        do_async: Optional[bool] = False,
    ) -> List[Dict[str, Any]]:
        if not records or not judge_kwargs_list:
            raise ValueError(
                f"empty records or judge_kwargs_list "
                f"(records={len(records) if records is not None else None}, "
                f"judge_kwargs_list={len(judge_kwargs_list) if judge_kwargs_list is not None else None}); "
                f"caller must filter eligible records before invoking JudgeEvaluator"
            )
        judge_model = judge_kwargs_list[0].get("judge_model", "")
        if isinstance(judge_model, str):
            judge_model = [judge_model]
        if len(judge_model) == 1:
            return cls.judge_pairwise_single(
                records=records,
                judge_kwargs_list=judge_kwargs_list,
                judge_prompt_list=judge_prompt_list,
                batch_size=batch_size,
                timeout=timeout,
                max_retry=max_retry,
                wait_between_retry=wait_between_retry,
                do_async=do_async,
            )
        results_per_model = [
            cls.judge_pairwise_single(
                records=records,
                judge_kwargs_list=[{**_kw, "judge_model": _m} for _kw in judge_kwargs_list],
                judge_prompt_list=judge_prompt_list,
                batch_size=batch_size,
                timeout=timeout,
                max_retry=max_retry,
                wait_between_retry=wait_between_retry,
                do_async=do_async,
            )
            for _m in judge_model
        ]
        return cls._aggregate_judge_results(results_per_model, judge_models=judge_model)

    @classmethod
    def judge_pairwise_single(
        cls,
        records: List[Union[Dict[str, Any], Record]],
        judge_kwargs_list: List[Dict[str, Any]],
        judge_prompt_list: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        do_async: Optional[bool] = False,
    ) -> Dict[str, float]:
        if not isinstance(timeout, (int, float)):
            timeout = TIMEOUT
        if not isinstance(max_retry, int):
            max_retry = MAX_RETRY
        if not isinstance(wait_between_retry, (int, float)):
            wait_between_retry = WAIT_BETWEEN_RETRY

        outputs = None
        if do_async:
            semaphore_size = batch_size
            if (
                not isinstance(semaphore_size, int)
                or semaphore_size < 1
            ):
                semaphore_size = NUM_MAX_COROUTINES
            semaphore_size = min(semaphore_size, NUM_MAX_COROUTINES)
            semaphore = asyncio.Semaphore(semaphore_size)
            
            tasks = list()
            for _record_idx, (_record, _judge_kwargs) in enumerate(
                zip(records, judge_kwargs_list)
            ):
                _judge_message = None
                if (
                    isinstance(judge_prompt_list, (list, tuple))
                    and _record_idx < len(judge_prompt_list)
                ):
                    _judge_message = judge_prompt_list[_record_idx]
                tasks.append(cls.judge_pairwise_record_async(
                    record=_record,
                    **_judge_kwargs,
                    judge_message=_judge_message,
                    semaphore=semaphore,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                ))
            outputs = asyncio.run(tqdm_asyncio.gather(*tasks, initial=0, total=len(records), desc=f'Evaluating judge_pairwise'))
        
        else:
            outputs = list()
            for _record_idx, (_record, _judge_kwargs) in tqdm(
                enumerate(zip(records, judge_kwargs_list)),
                initial=0,
                total=len(records),
                desc=f'Evaluating judge_pairwise',
            ):
                _judge_message = None
                if (
                    isinstance(judge_prompt_list, (list, tuple))
                    and _record_idx < len(judge_prompt_list)
                ):
                    _judge_message = judge_prompt_list[_record_idx]
                _output = cls.judge_pairwise_record_sync(
                    record=_record,
                    **_judge_kwargs,
                    judge_message=_judge_message,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                )
                outputs.append(_output)

        return outputs
    
    @classmethod
    def judge_pairwise_record_sync(
        cls,
        record: Union[Dict[str, Any], Record],
        lang: str,
        judge_model: str,
        judge_message: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        baseline: Optional[str] = None,
        rubrics: Optional[Dict[str, Any]] = None,
        response_format: Optional[type] = None,
        max_tokens: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_rating: Optional[Union[int, float]] = None,
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        api_kwargs = cls._preprocess_judge_pairwise_record(
            record=record,
            lang=lang,
            baseline=baseline,
            judge_model=judge_model,
            judge_message=judge_message,
            judge_prompt=judge_prompt,
            system_prompt=system_prompt,
            instruction=instruction,
            rubrics=rubrics,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

        _response_ab = chat_completion_sync(
            api_name=judge_model,
            messages=api_kwargs["judge_messages_ab"],
            generation_options=api_kwargs["generation_options"],
            response_format=response_format,
            timeout=timeout,
            max_retry=max_retry,
            wait_between_retry=wait_between_retry,
        )
        _response_ba = chat_completion_sync(
            api_name=judge_model,
            messages=api_kwargs["judge_messages_ba"],
            generation_options=api_kwargs["generation_options"],
            response_format=response_format,
            timeout=timeout,
            max_retry=max_retry,
            wait_between_retry=wait_between_retry,
        )

        # _judge_pairwise_record_postprocess
        output = cls._postprocess_judge_pairwise_record(
            response_ab=_response_ab,
            response_ba=_response_ba,
            rubrics=rubrics,
            score_format=api_kwargs["score_format"],
            reason_format=api_kwargs["reason_format"],
            max_rating=max_rating,
        )
        return output
    
    @classmethod
    async def judge_pairwise_record_async(
        cls,
        record: Union[Dict[str, Any], Record],
        lang: str,
        judge_model: str,
        judge_message: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        baseline: Optional[str] = None,
        rubrics: Optional[Dict[str, Any]] = None,
        response_format: Optional[type] = None,
        max_tokens: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_rating: Optional[Union[int, float]] = None,
        semaphore: Optional[asyncio.locks.Semaphore] = None, 
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        api_kwargs = cls._preprocess_judge_pairwise_record(
            record=record,
            lang=lang,
            baseline=baseline,
            judge_model=judge_model,
            judge_message=judge_message,
            judge_prompt=judge_prompt,
            system_prompt=system_prompt,
            instruction=instruction,
            rubrics=rubrics,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

        _response_ab = await chat_completion_async(
            api_name=judge_model,
            messages=api_kwargs["judge_messages_ab"],
            generation_options=api_kwargs["generation_options"],
            response_format=response_format,
            semaphore=semaphore,
            timeout=timeout,
            max_retry=max_retry,
            wait_between_retry=wait_between_retry,
        )
        _response_ba = await chat_completion_async(
            api_name=judge_model,
            messages=api_kwargs["judge_messages_ba"],
            generation_options=api_kwargs["generation_options"],
            response_format=response_format,
            semaphore=semaphore,
            timeout=timeout,
            max_retry=max_retry,
            wait_between_retry=wait_between_retry,
        )

        # _judge_pairwise_record_postprocess
        output = cls._postprocess_judge_pairwise_record(
            response_ab=_response_ab,
            response_ba=_response_ba,
            rubrics=rubrics,
            score_format=api_kwargs["score_format"],
            reason_format=api_kwargs["reason_format"],
            max_rating=max_rating,
        )
        return output
    
    @classmethod
    def _preprocess_judge_pairwise_record(
        cls,
        record: Union[Dict[str, Any], Record],
        lang: str,
        judge_model: str,
        judge_message: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        baseline: Optional[str] = None,
        rubrics: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, float]:
        judge_messages = list()
        if system_prompt:
            judge_messages.append(ChatMessage(
                role="system",
                content=[ChatTextContent(
                    type="text",
                    value=system_prompt,
                ),]
            ))
        judge_messages.append(ChatMessage(
            role="user",
            content=list(),
        ))
        
        score_format = "([SCORE_A],[SOCRE_B])" 
        reason_format = "([REASON_A],[REASON_B])" 
        score_rubric = ""
        if rubrics:
            score_format = "\n".join([
                f'{_rubric_name}: ([SCORE_A],[SOCRE_B])'
                for _rubric_name, _rubric_desc in rubrics.items()
            ])
            reason_format = "\n".join([
                f'{_rubric_name}: ([REASON_A],[REASON_B])'
                for _rubric_name, _rubric_desc in rubrics.items()
            ])
            score_rubric = "\n".join([
                f"{_rubric_name}: {_rubric_desc}" 
                for _rubric_name, _rubric_desc in rubrics.items()
            ])

        # reason_format = ""
        if not judge_prompt: 
            if rubrics:
                judge_prompt = JUDGE_PROMPTS["judge_pairwise_rubrics"][lang]
            else:
                judge_prompt = JUDGE_PROMPTS["judge_pairwise"][lang]
        
        dialogue = list()
        for _message in record["messages"]:
            for _content in _message["content"]:
                _content_key = _message["role"].title()
                if _message.get("name", None):
                    _content_key = _message["name"].title()
                _content_value = _content["value"] if "value" in _content else _content[_content["type"]]
                if _content["type"] == "text":
                    dialogue.append(
                        f'{_content_key}: {_content_value}',
                    )
                elif _content["type"] == "audio":
                    judge_messages[-1].content.append(ChatAudioContent(
                        type="audio",
                        value=_content_value,
                    ))
                elif _content["type"] == "image":
                    judge_messages[-1].content.append(ChatImageContent(
                        type="image",
                        value=_content_value,
                    ))
                elif _content["type"] == "video":
                    judge_messages[-1].content.append(ChatVideoContent(
                        type="video",
                        value=_content_value,
                    ))

        _judge_prompt_ab = format_task_prompt(
            task_prompt=judge_prompt,
            query="",
            score_format=score_format,
            reason_format=reason_format,
            instruction=instruction,
            dialogue="\n".join(dialogue),
            response_a=record["prediction"],
            response_b=baseline,
            label=record["label"],
            score_rubric=score_rubric,
        )
        judge_messages_ab = copy.deepcopy(judge_messages)
        judge_messages_ab[-1]["content"].append(ChatTextContent(
            type="text",
            value=_judge_prompt_ab,
        ))
        
        _judge_prompt_ba = format_task_prompt(
            task_prompt=judge_prompt,
            query="",
            score_format=score_format,
            reason_format=reason_format,
            instruction=instruction,
            dialogue="\n".join(dialogue),
            response_a=baseline,
            response_b=record["prediction"],
            label=record["label"],
            score_rubric=score_rubric,
        )
        judge_messages_ba = copy.deepcopy(judge_messages)
        judge_messages_ba[-1]["content"].append({
            "type": "text",
            "value": _judge_prompt_ba,
        })
        
        if kwargs.get("process_message_kwargs", dict()):
            judge_messages_ab = [
                ChatMessage.preprocess_message(
                    message=_message,
                    **kwargs["process_message_kwargs"],
                )
                for _message in judge_messages_ab
            ]
            judge_messages_ba = [
                ChatMessage.preprocess_message(
                    message=_message,
                    **kwargs["process_message_kwargs"],
                )
                for _message in judge_messages_ba
            ]
        
        api_group = get_api_group(api_name=judge_model)
        generation_options = dict()
        if max_tokens:
            generation_options["max_tokens"] = max_tokens
        if temperature:
            generation_options["temperature"] = temperature
        if top_k:
            generation_options["top_k"] = top_k
        if top_p:
            generation_options["top_p"] = top_p
        generation_options.update(kwargs)
        generation_options = ApiGenerationOptions.from_dict(
            api_name=judge_model,
            obj=generation_options,
            api_group=api_group,
        ).to_dict()

        api_kwargs = {
            "judge_messages_ab": judge_messages_ab,
            "judge_messages_ba": judge_messages_ba,
            "generation_options": generation_options,
            "score_format": score_format,
            "reason_format": reason_format,
            "score_rubric": score_rubric,
        }
        return api_kwargs
    
    @classmethod
    def _postprocess_judge_pairwise_record(
        cls,
        response_ab: str,
        response_ba: str,
        rubrics: Optional[Dict[str, Any]] = None,
        score_format: Optional[Dict[str, Any]] = None,
        reason_format: Optional[Dict[str, Any]] = None,
        max_rating: Optional[Union[int, float]] = None,
    ) -> Dict[str, float]:
        # win_rate, win_point
        _choice_ab = None
        if not response_ab:
            pass
        else:
            _choice_ab = cls._parse_response_choice(
                response=response_ab,
            )
        _choice_ba = None
        if not response_ba:
            pass
        else:
            _choice_ba = cls._parse_response_choice(
                response=response_ba,
            )

        choice = cls._aggregate_choice(
            choice_ab=_choice_ab,
            choice_ba=_choice_ba,
        ) # e.g. {"A": 1, "B": 0, "tie": 0, "total": 1}
        win_rate, win_point = 0.0, 0.0
        if choice["total"] < 1:
            pass
        elif "A" in choice:
            win_rate, win_point = 1.0, 1.0
        elif "tie" in choice:
            win_rate, win_point = 0.0, 0.5

        # rubric_scores
        rubric_scores, rubric_scores_baseline = None, None
        if rubrics and score_format:
            rubric_scores, rubric_scores_baseline = dict(), dict()
            for _rubric_name in rubrics.keys():
                rubric_scores[_rubric_name] = None
                rubric_scores_baseline[_rubric_name] = None

            _score_ab = cls._parse_response_score(
                response=response_ab,
                rubrics=rubrics,
                max_rating=max_rating,
            )
            _score_ba = cls._parse_response_score(
                response=response_ba,
                rubrics=rubrics,
                max_rating=max_rating,
            )
            for _rubric_name in rubrics.keys():
                _rubric_value_a, _rubric_value_b = None, None
                _denom_a, _denom_b = 0, 0
                if (
                    _score_ab 
                    and _rubric_name in _score_ab
                    and isinstance(_score_ab[_rubric_name], (list, tuple))
                ):
                    _rubric_value = _score_ab[_rubric_name]
                    if isinstance(_rubric_value[0], (int, float)):
                        if _rubric_value_a is None:
                            _rubric_value_a = _rubric_value[0]
                        else:
                            _rubric_value_a += _rubric_value[0]
                        _denom_a += 1
                        
                    if isinstance(_rubric_value[1], (int, float)):
                        if _rubric_value_b is None:
                            _rubric_value_b = _rubric_value[1]
                        else:
                            _rubric_value_b += _rubric_value[1]
                        _denom_b += 1
                
                if (
                    _score_ba
                    and _rubric_name in _score_ba
                    and isinstance(_score_ba[_rubric_name], (list, tuple))
                ):
                    _rubric_value = _score_ba[_rubric_name]
                    if isinstance(_rubric_value[0], (int, float)):
                        if _rubric_value_b is None:
                            _rubric_value_b = _rubric_value[0]
                        else:
                            _rubric_value_b += _rubric_value[0]
                        _denom_b += 1
                        
                    if isinstance(_rubric_value[1], (int, float)):
                        if _rubric_value_a is None:
                            _rubric_value_a = _rubric_value[1]
                        else:
                            _rubric_value_a += _rubric_value[1]
                        _denom_a += 1
                        
                if (
                    isinstance(_rubric_value_a, (int, float))
                    and _denom_a > 0
                ):
                    rubric_scores[_rubric_name] = _rubric_value_a / _denom_a
                if (
                    isinstance(_rubric_value_b, (int, float))
                    and _denom_b > 0
                ):
                    rubric_scores_baseline[_rubric_name] = _rubric_value_b / _denom_b

        reasons = None
        if reason_format:
            try:
                _reasons_ab = cls._parse_response_reason(
                    response=response_ab,
                    rubrics=rubrics,
                    pairwise=True,
                )
                _reasons_ba = cls._parse_response_reason(
                    response=response_ba,
                    rubrics=rubrics,
                    pairwise=True,
                )
                
                if (
                    rubrics
                    and (
                        isinstance(_reasons_ab, dict) 
                        or isinstance(_reasons_ba, dict)
                    )
                ):
                    reasons = dict()
                    for _rubric_name in rubrics.keys():
                        _reasons_a, _reasons_b = list(), list()
                        
                        _reasons_ab_ = _reasons_ab.get(_rubric_name, dict())
                        if len(_reasons_ab_) > 0:
                            _reasons_a.append(_reasons_ab_[0])
                        if len(_reasons_ab_) > 1:
                            _reasons_b.append(_reasons_ab_[1])
                        
                        _reasons_ba_ = _reasons_ba.get(_rubric_name, dict())
                        if len(_reasons_ba_) > 0:
                            _reasons_b.append(_reasons_ba_[0])
                        if len(_reasons_ba_) > 1:
                            _reasons_a.append(_reasons_ba_[1])
                        
                        reasons[_rubric_name] = (_reasons_a, _reasons_b)
                
                elif (
                    isinstance(_reasons_ab, list) 
                    or isinstance(_reasons_ba, list)
                ):
                    _reasons_a, _reasons_b = list(), list()
                    if len(_reasons_ab_) > 0:
                        _reasons_a.append(_reasons_ab_[0])
                    if len(_reasons_ab_) > 1:
                        _reasons_b.append(_reasons_ab_[1])
                    if len(_reasons_ba_) > 0:
                        _reasons_b.append(_reasons_ba_[0])
                    if len(_reasons_ba_) > 1:
                        _reasons_a.append(_reasons_ba_[1])
                    reasons = (_reasons_a, _reasons_b)
                
                else:
                    reasons = (_reasons_ab, _reasons_ba)
            except Exception as ex:
                logger.debug(f'Error in parsing reasons from judge response: {ex}')

        output = {
            "win_rate": win_rate,
            "win_point": win_point,
            "scores": rubric_scores,
            "rubric_scores_baseline": rubric_scores_baseline,
            "choice": choice,
            "reasons": reasons,
            "response_ab": response_ab,
            "response_ba": response_ba,
        }
        return output
    
    @classmethod
    def _parse_response_choice(
        cls,
        response: str,
        verbose: Optional[bool] = True,
    ):
        if not response:
            return None

        # parse 'choice' from api_responses
        parts = list()
        for _part in response.split("\n"):
            if "Choice:".lower() not in _part.lower():
                continue
            _part = _part.replace("Choice:", "").strip()
            parts.append(_part)
        
        if len(parts) < 1:
            logger.debug(f'Choice is not included in judge response: {response}')
            return None
        else:
            _choice = parts[0].lower()
            if _choice in ["true", "t"]:
                return True
            elif _choice in ["false", "f"]:
                return False
            elif _choice == "a":
                return "A"
            elif _choice == "b":
                return "B"
            elif (
                "a" not in _choice
                and "b" not in _choice
            ):
                return None    
            else:
                a_index = len(_choice)
                if "a" in _choice:
                    a_index = _choice.index("a") 
                b_index = len(_choice)
                if "b" in _choice:
                    b_index = _choice.index("b")
                if a_index < b_index:
                    return "A"
                elif b_index < a_index:
                    return "B"
                else:  # a_index == b_index
                    return None
    
    @classmethod
    def _aggregate_choice(
        cls,
        choice_ab: str,
        choice_ba: str,
    ):
        # aggregate to calculate winrate given two judge results among {"A", "B", None}
        judge_cnt = defaultdict(int)
        if choice_ab is None and choice_ba is None:
            judge_cnt["tie"] += 1
        elif choice_ab is not None and choice_ba is None:
            if choice_ab == "A":
                judge_cnt["A"] += 1
            elif choice_ab == "B":
                judge_cnt["B"] += 1
            else:
                judge_cnt["tie"] += 1
        elif choice_ab is None and choice_ba is not None:
            if choice_ba == "A":
                judge_cnt["B"] += 1
            elif choice_ba == "B":
                judge_cnt["A"] += 1
            else:
                judge_cnt["tie"] += 1
        else:  # judge_ab is not None and judge_ba is not None
            if choice_ab == "A" and choice_ba == "B":
                judge_cnt["A"] += 1
            elif choice_ab == "B" and choice_ba == "A":
                judge_cnt["B"] += 1
            else:
                judge_cnt["tie"] += 1
        judge_cnt["total"] += 1
        return judge_cnt
    
    # Tokens that map to True/False in a judge_binary response.
    # ASCII-alphanumeric tokens are matched with word boundaries to avoid
    # accidental substring hits (e.g. "true" must not match inside "trust").
    # Non-ASCII tokens (Korean, etc.) are matched as substrings since they
    # rarely collide with surrounding text and don't share word boundaries.
    _BINARY_TRUE_TOKENS = [
        "true", "yes", "correct", "right",
        "맞다", "맞음", "맞아", "맞아요", "맞습니다", "옳다", "옳음", "정답", "참",
    ]
    _BINARY_FALSE_TOKENS = [
        "false", "no", "incorrect", "wrong",
        "틀리다", "틀림", "틀려", "틀려요", "틀립니다", "그르다", "오답", "거짓",
    ]

    @classmethod
    def _binary_match(cls, text: str, true_first: bool = True) -> Optional[bool]:
        """Return True / False / None depending on which token group hits *text* first.

        Word-boundary regex is used for ASCII tokens (case-insensitive); plain
        substring for non-ASCII tokens. The earliest match wins (or `true_first`
        breaks ties when both groups match at the same offset).
        """
        def _earliest_hit(tokens):
            best = None
            for tok in tokens:
                pat = rf"\b{re.escape(tok)}\b" if tok.isascii() else re.escape(tok)
                m = re.search(pat, text, re.IGNORECASE)
                if m and (best is None or m.start() < best):
                    best = m.start()
            return best

        t_hit = _earliest_hit(cls._BINARY_TRUE_TOKENS)
        f_hit = _earliest_hit(cls._BINARY_FALSE_TOKENS)
        if t_hit is None and f_hit is None:
            return None
        if t_hit is None:
            return False
        if f_hit is None:
            return True
        # Both matched — earliest position wins; tie-break by true_first
        if t_hit < f_hit:
            return True
        if f_hit < t_hit:
            return False
        return True if true_first else False

    @classmethod
    def _parse_response_binary(
        cls,
        response: str,
        rubrics: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = True,
    ):
        # parse true/false from api responses (multilingual tokens supported).
        output = dict()
        output_candidate = None
        parts = response.split("\n")
        for _part_idx, _part in enumerate(parts):
            if (
                isinstance(rubrics, dict)
                and len(rubrics) > 0
            ):
                for _rubric_name in rubrics.keys():
                    if (
                        _rubric_name.lower() not in _part.lower()
                        or _rubric_name in output
                    ):
                        continue

                    # strip the rubric name so 'coherence: true' → 'true'
                    _rubric_value = _part.replace(_rubric_name.lower(), "")
                    _rubric_value = _rubric_value.replace(":", "").strip()
                    if _rubric_value in ["true", "t"] or "true" in _rubric_value:
                        output[_rubric_name] = True
                    elif _rubric_value in ["false", "f"] or "false" in _rubric_value:
                        output[_rubric_name] = False
                if len(output) == len(rubrics):
                    break
                if (
                    _part_idx == len(parts) - 1
                    and not output
                ):  # when no parsed scores until last part
                    output_candidate = cls._binary_match(_part)
            else:
                matched = cls._binary_match(_part)
                if matched is not None:
                    output = matched
                    break
                if _part_idx == len(parts) - 1:
                    # last-part single-letter fallback (legacy behaviour)
                    if re.search(r"\bt\b", _part, re.IGNORECASE):
                        output = True
                        break
                    if re.search(r"\bf\b", _part, re.IGNORECASE):
                        output = False
                        break

        if (
            isinstance(output, dict)
            and len(output) < 1
        ):
            if output_candidate is not None:
                output = output_candidate
            else:
                output = None
        return output
    
    @classmethod
    def _parse_response_score(
        cls,
        response: str,
        rubrics: Optional[Dict[str, Any]] = None,
        max_rating: Optional[Union[int, float]] = None,
        verbose: Optional[bool] = True,
    ):
        if not response:
            return None
        
        scores = None
        if (
            isinstance(rubrics, dict) 
            and len(rubrics) > 0
        ):
            scores = dict()
            for _rubric_name in rubrics.keys():
                scores[_rubric_name] = [None, None]

        for _part in response.split("\n"):
            if (
                isinstance(rubrics, dict) 
                and len(rubrics) > 0
            ):
                for _rubric_name in scores.keys():
                    if _rubric_name.lower() not in _part.lower():
                        continue
                    try:
                        _rubric_value = _part.lower().replace(_rubric_name.lower(), "")
                        _rubric_value = _rubric_value.replace(":", "").replace("(", "").replace(")", "").strip()
                        _match = re.search(r"\s*[\(\[]?\s*(-?\d+)\s*[\)\]]?\s*$", _rubric_value)
                        _match_ab = re.search(r"\s*[\(\[]?\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]]?\s*$", _rubric_value)
                        if _match_ab: # judge_pairwise
                            _rubric_value_a = _match_ab.group(1)
                            _rubric_value_b = _match_ab.group(2)
                            scores[_rubric_name][0] = is_numeric(x=_rubric_value_a)
                            scores[_rubric_name][1] = is_numeric(x=_rubric_value_b)
                        elif _match: # judge_binary
                            _rubric_value = _match.group(0)
                            scores[_rubric_name][0] = is_numeric(x=_rubric_value)
                    except Exception as ex:
                        break  # failed to parsing score
            else:
                _match = re.search(r"\s*[\(\[]?\s*(-?\d+)\s*[\)\]]?\s*$", _part)
                _match_ab = re.search(r"\s*[\(\[]?\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]]?\s*$", _part)
                if _match_ab: # judge_pairwise
                    _score_a = _match_ab.group(1)
                    _score_b = _match_ab.group(2)
                    scores = (
                        is_numeric(x=_score_a), 
                        is_numeric(x=_score_b),
                    )
                elif _match: # judge_binary
                    _score = _match.group(0)
                    scores = is_numeric(x=_score)
                
        output = scores
        if (
            isinstance(rubrics, dict) 
            and len(rubrics) > 0
        ):
            output = dict()
            for _rubric_name, _rubric_value in scores.items():
                if (
                    isinstance(_rubric_value[0], (int, float))
                    and isinstance(_rubric_value[1], (int, float))
                ):
                    output[_rubric_name] = _rubric_value
                elif isinstance(_rubric_value[0], (int, float)):
                    output[_rubric_name] = _rubric_value[0]
                else:
                    output[_rubric_name] = None
                    
        if isinstance(output, (int, float)):
            if (
                isinstance(max_rating, (int, float))
                and output <= max_rating
            ):
                output = normalize_unit(
                    x=output,
                    unit=max_rating,
                )
        elif isinstance(output, (list, tuple)):
            if isinstance(max_rating, (int, float)):
                # ``_score`` may be None when the judge response was unparseable
                # for that slot (e.g. truncated rationale, malformed numeric).
                # ``None <= int`` raises TypeError, so gate the normalize/compare
                # on numeric-ness and pass non-numeric entries through.
                output = [
                    normalize_unit(
                        x=_score,
                        unit=max_rating,
                    ) if isinstance(_score, (int, float)) and _score <= max_rating else _score
                    for _score in output
                ]
        elif isinstance(output, dict):
            if isinstance(max_rating, (int, float)):
                output = {
                    _key: (
                        normalize_unit(
                            x=_score,
                            unit=max_rating,
                        ) if isinstance(_score, (int, float)) and _score <= max_rating else _score
                    )
                    for _key, _score in output.items()
                }
        return output
    
    @classmethod
    def _split_pairwise_reason(cls, text: str) -> List[Optional[str]]:
        # split "(reason_a, reason_b)" into [reason_a, reason_b] at depth-0 comma
        text = text.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        depth = 0
        for i, ch in enumerate(text):
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == "," and depth == 0:
                a = text[:i].strip()
                b = text[i + 1:].strip()
                return [a or None, b or None]
        return [text.strip() or None, None]

    @classmethod
    def _parse_response_reason(
        cls,
        response: str,
        rubrics: Optional[Dict[str, Any]] = None,
        pairwise: bool = False,
        verbose: Optional[bool] = True,
    ):
        if not response:
            return None
        
        # parse 'reason' from api_responses
        reasons = None
        if (
            isinstance(rubrics, dict)
            and len(rubrics) > 0
        ):
            reasons = {_rubric_name: None for _rubric_name in rubrics.keys()}

        for _part in response.split("\n"):
            if (
                isinstance(rubrics, dict)
                and len(rubrics) > 0
            ):
                for _rubric_name in reasons.keys():
                    if reasons[_rubric_name] is not None:
                        continue
                    if _rubric_name.lower() not in _part.lower():
                        continue
                    try:
                        _value = _part.lower().replace(_rubric_name.lower(), "")
                        _value = _value.replace(":", "", 1).strip()
                        # skip score lines: value is purely numeric (single or pairwise)
                        if re.search(r"^\s*[\(\[]?\s*-?\d+\s*(,\s*-?\d+\s*)?[\)\]]?\s*$", _value):
                            continue
                        _reason = re.sub(
                            rf"(?i){re.escape(_rubric_name)}\s*:", "", _part, count=1
                        ).strip()
                        if _reason:
                            if pairwise:
                                _reason = cls._split_pairwise_reason(_reason)
                            reasons[_rubric_name] = _reason
                    except Exception:
                        break
            else:
                # Accept both ``Reason:`` (default judge prompts) and
                # ``Explanation:`` (VERIFIER_PROMPT and similar) so callers
                # don't have to rewrite their judge prompt just to surface
                # the rationale in sample_metrics.
                if re.search(r"(?i)\[?(reason|explanation)\]?\s*:", _part):
                    _reason = re.sub(
                        r"(?i)^\s*\[?(reason|explanation)\]?\s*:\s*", "", _part,
                    ).strip()
                    if _reason:
                        if pairwise:
                            _reason = cls._split_pairwise_reason(_reason)
                        reasons = _reason
                        break

        return reasons