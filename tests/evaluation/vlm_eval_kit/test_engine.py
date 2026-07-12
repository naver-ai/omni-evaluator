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

"""Validates the entry-point export contract, live smoke (including `dataset_name`/`benchmark_config` variant args), and helper (`sample_to_record`) for `vlm_eval_kit/engine.py`."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "vlmeval",
    reason='install with `pip install -e ".[vlmeval]"`',
)

from omni_evaluator.evaluation.vlm_eval_kit import engine
from omni_evaluator.evaluation.vlm_eval_kit.engine import sample_to_record
from omni_evaluator.schemas.inference import Record
from tests.evaluation.test_engine_common import EvaluationEngineCommonTests


_VLM_EVAL_KIT_YAML = "vlm_eval_kit.yaml"


# ─────────────────────────────────────────────────────────────────────
#  Entry point: static contract + live smoke
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.eval_engine("vlm_eval_kit")
@pytest.mark.requires_extra("vlmeval")
@pytest.mark.timeout(60)
class TestVlmEvalKitEvaluationEngine(EvaluationEngineCommonTests):
    """vlm_eval_kit `engine.get_data_iterator` / `engine.evaluate_task` export contract and live smoke."""

    @pytest.fixture
    def engine_module(self):
        return engine
    # ── live smoke (slow + requires_extra) ─────────────────────

    @pytest.mark.slow
    @pytest.mark.requires_hf_token
    @pytest.mark.timeout(600)  # Sequential validation of all benchmarks exceeds the class default of 60s (dataset collection per benchmark)
    def test_live_smoke(
        self, evaluation_config_dir, fake_inference_record_factory,
    ):
        """Sequentially validates one `get_data_iterator` → `evaluate_task` cycle for every benchmark in `vlm_eval_kit.yaml`."""
        import yaml

        with open(evaluation_config_dir / _VLM_EVAL_KIT_YAML) as f:
            cfg = yaml.safe_load(f)
        benchmarks = [b.strip() for b in cfg["benchmarks"].split(",")]

        for benchmark in benchmarks:
            # get_data_iterator: still has task_name, debug (§1.4)
            iterator, task_config = engine.get_data_iterator(
                evaluation_engine="vlm_eval_kit",
                task_name=benchmark,
                run_index=0,
                debug=True,
            )
            assert task_config.num_records > 0, f"{benchmark}: num_records must be > 0"
            records_iter = list(iterator)
            assert len(records_iter) >= 1, f"{benchmark}: iterator yielded no records"

            print(
                f"\n  ── [vlm_eval_kit live smoke / {benchmark}] ──"
                f"\n    num_records  : {task_config.num_records}"
                f"\n    yielded      : {len(records_iter)}"
            )

            fake_records = fake_inference_record_factory(
                predictions=["dummy"] * len(records_iter),
                labels=["dummy"] * len(records_iter),
                benchmark=benchmark,
            )
            # evaluate_task: dataset_name / benchmark_config variant arg names
            result, sample_metrics = engine.evaluate_task(
                evaluation_engine="vlm_eval_kit",
                dataset_name=benchmark,
                evaluation_method="generation",
                benchmark_config=task_config,
                records=fake_records,
            )
            assert isinstance(result.metrics, dict) and len(result.metrics) > 0, (
                f"{benchmark}: metrics dict must be non-empty"
            )
            print(f"    metrics      : {list(result.metrics.keys())}")


# ─────────────────────────────────────────────────────────────────────
#  Helper: sample_to_record  (§1.7 — pure helper unit test)
# ─────────────────────────────────────────────────────────────────────


def _make_dataset(TYPE="VQA", dataset_name="ai2d_test", img_root=None):
    return SimpleNamespace(
        TYPE=TYPE,
        dataset_name=dataset_name,
        img_root=img_root,
    )


def _make_row(
    question="What is shown?",
    answer="A",
    index="row_0",
    category=None,
    **extra,
):
    row = {
        "question": question,
        "answer": answer,
        "index": index,
        "category": category,
    }
    row.update(extra)
    return row


@pytest.mark.eval_engine("vlm_eval_kit")
@pytest.mark.timeout(10)
def test_sample_to_record():
    """`sample_to_record` returns a Record and the core identity fields (`benchmark`, `index`, `messages`) are populated."""
    rec = sample_to_record(
        dataset_name="ai2d_test",
        row=_make_row(),
        user_content=[{"type": "text", "value": "Describe the image."}],
        dataset=_make_dataset(TYPE="VQA"),
    )
    assert isinstance(rec, Record)
    assert rec.benchmark == "ai2d_test"
    assert rec.index == "row_0"
    assert rec.messages  # non-empty
