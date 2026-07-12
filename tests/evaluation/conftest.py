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

"""Common fixtures shared by all evaluation engine tests."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pytest


@pytest.fixture
def fake_inference_record_factory():
    """Builds a list of text-only single-turn record dicts accepted by `evaluate_task(records=...)`."""
    def _factory(
        predictions: List[str],
        labels: List[Union[str, List[str]]],
        *,
        category: Optional[str] = None,
        benchmark: str = "dummy",
    ) -> List[Dict[str, Any]]:
        assert len(predictions) == len(labels), (
            f"predictions/labels length mismatch: {len(predictions)} vs {len(labels)}"
        )
        records: List[Dict[str, Any]] = []
        for idx, (pred, label) in enumerate(zip(predictions, labels)):
            rec: Dict[str, Any] = {
                "benchmark": benchmark,
                "index": str(idx),
                "messages": [
                    {"role": "user", "content": [{"type": "text", "value": "Q?"}]},
                ],
                "prediction": pred,
                "prediction_postprocessed": None,
                # The perplexity path (lm_eval/lmms engine) reads records[idx]["perplexities"]
                # — it is an optional field, so defaulting to None ensures that non-multiple-choice
                # generation paths evaluate based on prediction without raising KeyError.
                "perplexities": None,
                # The lm_eval/lmms engine's group_metrics aggregation reads record["meta"]["category"]
                # → meta dict is required (if category is None, group aggregation is skipped).
                "meta": {"category": category},
                "label": label,
                "tool_calls": None,
                "latency": 0.0,
            }
            if category is not None:
                rec["category"] = category
            records.append(rec)
        return records

    return _factory


@pytest.fixture
def evaluation_task_config_factory():
    """Builds a `TaskConfig` populated with fields read by the evaluation pipeline (`evaluation.target_metrics`, etc.)."""
    def _factory(
        *,
        task_name: str = "dummy",
        evaluation_engine: str = "builtin",
        target_metrics: Optional[List[str]] = None,
        evaluation_method: str = "generation",
        output_modality: Optional[List[str]] = None,
        num_records: int = 1,
        judges: Optional[Dict[str, Any]] = None,
    ):
        from omni_evaluator import DatasetSource
        from omni_evaluator.schemas.task import (
            TaskConfig, TaskDataset, TaskEvaluation, TaskMeta,
        )

        if target_metrics is None:
            target_metrics = ["exact_match"]
        if output_modality is None:
            output_modality = ["text"]

        eval_kwargs: Dict[str, Any] = dict(
            method=evaluation_method,
            target_metrics=list(target_metrics),
            display_metrics=list(target_metrics),
            do_normalize=False,
        )
        if judges is not None:
            eval_kwargs["judges"] = judges

        return TaskConfig(
            task_name=task_name,
            evaluation_engine=evaluation_engine,
            num_records=num_records,
            meta=TaskMeta(
                benchmark_name=task_name,
                output_modality=output_modality,
            ),
            dataset=TaskDataset(source=DatasetSource.local),
            evaluation=TaskEvaluation(**eval_kwargs),
        )

    return _factory


@pytest.fixture
def fake_dataset_iterator_factory():
    """Fake for mocking `load_dataset(...)` — builds a callable that returns `(iterator, dataset_size)`."""
    def _factory(messages_list: Optional[List[List[Any]]] = None) -> Callable:
        if messages_list is None:
            messages_list = []
        dataset_size = len(messages_list)

        def _fake_load_dataset(*args, **kwargs):
            def _gen():
                for msgs in messages_list:
                    yield {"messages": msgs}
            return _gen(), dataset_size

        return _fake_load_dataset

    return _factory


@pytest.fixture
def evaluation_config_dir():
    """Returns the path to the `tests/configs/evaluation/` directory — location of live smoke yamls."""
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "configs" / "evaluation"
