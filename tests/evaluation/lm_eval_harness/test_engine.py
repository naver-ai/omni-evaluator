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

"""Validates the entry-point export contract, live smoke, and helpers (`sample_to_record`, `process_results_multichoice`) of `lm_eval_harness/engine.py`."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip(
    "lm_eval",
    reason='install with `pip install -e ".[lm_eval]"`',
)

from omni_evaluator.evaluation.lm_eval_harness import engine
from omni_evaluator.evaluation.lm_eval_harness.engine import (
    process_results_multichoice,
    sample_to_record,
)
from omni_evaluator.schemas.inference import Record
from tests.evaluation.test_engine_common import EvaluationEngineCommonTests


_LM_EVAL_YAML = "lm_eval_harness.yaml"


# ─────────────────────────────────────────────────────────────────────
#  Entry point: static contract + live smoke
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.eval_engine("lm_eval_harness")
@pytest.mark.requires_extra("lm_eval")
@pytest.mark.timeout(60)
class TestLmEvalHarnessEvaluationEngine(EvaluationEngineCommonTests):
    """lm_eval_harness `engine.get_data_iterator` / `engine.evaluate_task` export contract and live smoke."""

    @pytest.fixture
    def engine_module(self):
        return engine

    # ── live smoke (slow + requires_extra) ─────────────────────

    @pytest.mark.slow
    @pytest.mark.requires_hf_token
    @pytest.mark.timeout(600)  # Sequential verification of all benchmarks exceeds the class default of 60s (dataset collection per benchmark)
    def test_live_smoke(
        self, evaluation_config_dir, fake_inference_record_factory,
    ):
        """Sequentially verifies one `get_data_iterator` → `evaluate_task` cycle for every benchmark in `lm_eval_harness.yaml`."""
        import yaml

        with open(evaluation_config_dir / _LM_EVAL_YAML) as f:
            cfg = yaml.safe_load(f)
        benchmarks = [b.strip() for b in cfg["benchmarks"].split(",")]

        for benchmark in benchmarks:
            # `get_data_iterator` has no `evaluation_method` argument (§1.3 table)
            iterator, task_config = engine.get_data_iterator(
                evaluation_engine="lm_eval_harness",
                task_name=benchmark,
                run_index=0,
            )
            assert task_config.num_records > 0, f"{benchmark}: num_records must be > 0"
            records_iter = list(iterator)
            assert len(records_iter) >= 1, f"{benchmark}: iterator yielded no records"

            print(
                f"\n  ── [lm_eval_harness live smoke / {benchmark}] ──"
                f"\n    num_records  : {task_config.num_records}"
                f"\n    yielded      : {len(records_iter)}"
            )

            fake_records = fake_inference_record_factory(
                predictions=["dummy"] * len(records_iter),
                labels=["dummy"] * len(records_iter),
                benchmark=benchmark,
            )
            result, sample_metrics = engine.evaluate_task(
                evaluation_engine="lm_eval_harness",
                task_name=benchmark,
                task_config=task_config,
                evaluation_method="generation",
                records=fake_records,
                debug=True,
            )
            assert isinstance(result.metrics, dict) and len(result.metrics) > 0, (
                f"{benchmark}: metrics dict must be non-empty"
            )
            print(f"    metrics      : {list(result.metrics.keys())}")


# ─────────────────────────────────────────────────────────────────────
#  Helper: sample_to_record  (§1.7 — pure helper unit test)
# ─────────────────────────────────────────────────────────────────────


def _make_instance(
    query="What is 2+2?",
    repeats=1,
    request_type="generate_until",
    task_name="arc_easy",
):
    return SimpleNamespace(
        arguments=(query,),
        repeats=repeats,
        request_type=request_type,
        task_name=task_name,
    )


def _make_task(
    doc_to_target=None,
    doc_to_choice=None,
    generation_kwargs=None,
    num_fewshot=None,
):
    cfg = SimpleNamespace(
        generation_kwargs=generation_kwargs,
        num_fewshot=num_fewshot,
        doc_to_choice=doc_to_choice,
    )
    return SimpleNamespace(
        config=cfg,
        doc_to_target=doc_to_target or (lambda d: "fallback_target"),
        doc_to_choice=doc_to_choice,
    )


@pytest.mark.eval_engine("lm_eval_harness")
@pytest.mark.timeout(10)
def test_sample_to_record():
    """`sample_to_record` returns a Record with the core identity fields (`benchmark`, `index`, `messages`) populated."""
    rec = sample_to_record(
        task_name="arc_easy",
        doc_id=42,
        doc={"question": "Q?", "answer": "A"},
        instance=_make_instance(),
        task=_make_task(),
    )
    assert isinstance(rec, Record)
    assert rec.benchmark == "arc_easy"
    assert rec.index == 42
    assert rec.messages  # non-empty


# ─────────────────────────────────────────────────────────────────────
#  Helper: process_results_multichoice  (§1.7 — pure helper unit test)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.eval_engine("lm_eval_harness")
@pytest.mark.timeout(10)
def test_process_results_multichoice():
    """`process_results_multichoice` returns a `{metric_name: float}` dict."""
    out = process_results_multichoice(
        self=None,
        doc={"answer": "A"},
        results=["A"],
        doc_to_text=None,
        doc_to_target=lambda d: "A",
        doc_to_choice=lambda d: ["A", "B", "C", "D"],
    )
    assert isinstance(out, dict)
    assert "acc" in out
    assert isinstance(out["acc"], (int, float))
