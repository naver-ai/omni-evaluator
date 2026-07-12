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

"""Unit tests for builtin `engine.py` — mocks `load_dataset` / `JudgeEvaluator` / `TaskConfig.from_builtin` and runs `TextEvaluator` for real."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from omni_evaluator.evaluation.builtin import engine
from omni_evaluator.schemas.evaluation import EvaluationRunOutput
from tests.evaluation.test_engine_common import EvaluationEngineCommonTests


@pytest.mark.eval_engine("builtin")
@pytest.mark.timeout(60)
class TestBuiltinEvaluationEngine(EvaluationEngineCommonTests):
    """Unit tests for builtin `engine.get_data_iterator` / `engine.evaluate_task`."""

    # ── base fixture overrides ───────────────────────────────────

    @pytest.fixture
    def engine_module(self):
        return engine

    # ── builtin-specific fixtures ────────────────────────────────

    @pytest.fixture
    def patch_boundary(
        self, monkeypatch, evaluation_task_config_factory, fake_dataset_iterator_factory,
    ):
        """Patches the three mock boundaries of the builtin engine (`load_dataset` / `JudgeEvaluator.evaluate` / `TaskConfig.from_builtin`) and returns a call log dict."""
        _log: Dict[str, List[Any]] = {
            "judge_calls": [],
            "load_dataset_calls": [],
            "from_builtin_calls": [],
        }

        def _patch(
            *,
            task_config,
            judge_eval_return=None,
            dataset_messages=None,
        ):
            # 1) load_dataset: block dataset download / multimodal restore step
            def _fake_load(*args, **kwargs):
                _log["load_dataset_calls"].append(kwargs)
                base = fake_dataset_iterator_factory(messages_list=dataset_messages)
                return base(*args, **kwargs)
            monkeypatch.setattr(engine, "load_dataset", _fake_load)

            # 2) TaskConfig.from_builtin: bypass yaml loading.
            #    `evaluate_task` calls this once more to update evaluation config,
            #    but since the fixture's task_config already has the evaluation field
            #    populated, returning it as-is works fine.
            def _fake_from_builtin(task_name=None, reasoning=False, **_):
                _log["from_builtin_calls"].append({"task_name": task_name, "reasoning": reasoning})
                return task_config
            monkeypatch.setattr(
                engine.TaskConfig, "from_builtin", classmethod(lambda cls, **kw: _fake_from_builtin(**kw)),
            )

            # 3) JudgeEvaluator.evaluate: block LLM judge calls.
            #    If target_metrics has no judge_* entries, this mock must not be called
            #    (verified via call counter).
            def _fake_judge(**kwargs):
                _log["judge_calls"].append(kwargs)
                if judge_eval_return is not None:
                    return judge_eval_return
                records = kwargs.get("records") or []
                return {
                    "metrics": {},
                    "group_metrics": {},
                    "sample_metrics": [dict() for _ in records],
                }
            monkeypatch.setattr(
                engine.JudgeEvaluator, "evaluate", classmethod(lambda cls, **kw: _fake_judge(**kw)),
            )

            return _log

        return _patch

    # ── builtin-specific dynamic tests ──────────────────────────

    def test_returns_run_output(
        self,
        engine_module, patch_boundary,
        fake_inference_record_factory, evaluation_task_config_factory,
    ):
        """`evaluate_task` returns an `(EvaluationRunOutput, sample_metrics)` tuple."""
        records = fake_inference_record_factory(
            predictions=["yes", "no"],
            labels=["yes", "yes"],
        )
        task_config = evaluation_task_config_factory(
            task_name="dummy",
            target_metrics=["exact_match"],
            num_records=len(records),
        )
        patch_boundary(task_config=task_config)

        result, sample_metrics = engine_module.evaluate_task(
            evaluation_engine="builtin",
            task_name="dummy",
            task_config=task_config,
            evaluation_method="generation",
            records=records,
            debug=True,
        )

        assert isinstance(result, EvaluationRunOutput)
        assert isinstance(sample_metrics, list)
        assert len(sample_metrics) == len(records)

    def test_exact_match(
        self,
        engine_module, patch_boundary,
        fake_inference_record_factory, evaluation_task_config_factory,
    ):
        """Scores `exact_match` deterministically using the real `TextEvaluator` — verifies both orchestration and metric accuracy simultaneously."""
        records = fake_inference_record_factory(
            predictions=["yes", "no"],
            labels=["yes", "yes"],
        )
        task_config = evaluation_task_config_factory(
            task_name="dummy",
            target_metrics=["exact_match"],
            num_records=len(records),
        )
        patch_boundary(task_config=task_config)

        result, sample_metrics = engine_module.evaluate_task(
            evaluation_engine="builtin",
            task_name="dummy",
            task_config=task_config,
            evaluation_method="generation",
            records=records,
            debug=True,
        )

        assert "exact_match" in result.metrics, (
            f"expected 'exact_match' in metrics, got {list(result.metrics.keys())}"
        )
        assert sample_metrics[0].get("exact_match") == pytest.approx(1.0)
        assert sample_metrics[1].get("exact_match") == pytest.approx(0.0)

    def test_skips_judge(
        self,
        engine_module, patch_boundary,
        fake_inference_record_factory, evaluation_task_config_factory,
    ):
        """`JudgeEvaluator.evaluate` is not called when `target_metrics` has no `judge_*` entries."""
        records = fake_inference_record_factory(
            predictions=["yes"],
            labels=["yes"],
        )
        task_config = evaluation_task_config_factory(
            task_name="dummy",
            target_metrics=["exact_match"],
            num_records=len(records),
        )
        log = patch_boundary(task_config=task_config)

        engine_module.evaluate_task(
            evaluation_engine="builtin",
            task_name="dummy",
            task_config=task_config,
            evaluation_method="generation",
            records=records,
            debug=True,
        )

        assert log["judge_calls"] == [], (
            f"JudgeEvaluator should not be called when target_metrics has no judge_*, "
            f"got {len(log['judge_calls'])} calls"
        )

    def test_empty_predictions(
        self,
        engine_module, patch_boundary,
        fake_inference_record_factory, evaluation_task_config_factory,
    ):
        """Empty predictions are counted in `num_empty_predictions` and `coverage_inference` decreases."""
        records = fake_inference_record_factory(
            predictions=["yes", "", "no"],  # 1 empty prediction
            labels=["yes", "yes", "yes"],
        )
        task_config = evaluation_task_config_factory(
            task_name="dummy",
            target_metrics=["exact_match"],
            num_records=len(records),
        )
        patch_boundary(task_config=task_config)

        result, _ = engine_module.evaluate_task(
            evaluation_engine="builtin",
            task_name="dummy",
            task_config=task_config,
            evaluation_method="generation",
            records=records,
            debug=True,
        )

        assert result.num_empty_predictions == 1
        assert result.coverage_inference == pytest.approx(2.0 / 3.0)

    def test_output_schema(
        self,
        engine_module, patch_boundary,
        fake_inference_record_factory, evaluation_task_config_factory,
    ):
        """All required fields of `EvaluationRunOutput` (`task_name`, `evaluation_engine`, `evaluation_method`, `num_samples`, `metrics`, `sample_metrics`) are populated."""
        records = fake_inference_record_factory(
            predictions=["yes"],
            labels=["yes"],
        )
        task_config = evaluation_task_config_factory(
            task_name="dummy",
            target_metrics=["exact_match"],
            num_records=len(records),
        )
        patch_boundary(task_config=task_config)

        result, sample_metrics = engine_module.evaluate_task(
            evaluation_engine="builtin",
            task_name="dummy",
            task_config=task_config,
            evaluation_method="generation",
            records=records,
            debug=True,
        )

        assert result.task_name == "dummy"
        assert result.evaluation_engine == "builtin"
        assert result.evaluation_method == "generation"
        assert result.num_samples == len(records)
        assert isinstance(result.metrics, dict) and len(result.metrics) > 0
        assert isinstance(result.sample_metrics, list)
        assert len(result.sample_metrics) == len(records)
