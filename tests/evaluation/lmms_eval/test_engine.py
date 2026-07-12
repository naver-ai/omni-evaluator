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

"""Verifies the entry-point export contract, live smoke, and helper (`sample_to_record`) of `lmms_eval/engine.py`."""
from __future__ import annotations

from types import SimpleNamespace

import PIL.Image
import pytest

pytest.importorskip(
    "lmms_eval",
    reason='install with `pip install -e ".[lmms_eval]"`',
)

from omni_evaluator.evaluation.lmms_eval import engine
from omni_evaluator.evaluation.lmms_eval.engine import sample_to_record
from omni_evaluator.inference import NUM_DEBUG_SAMPLES  # does not accept debug=True
from omni_evaluator.schemas.inference import Record
from tests.evaluation.test_engine_common import EvaluationEngineCommonTests


_LMMS_EVAL_YAML = "lmms_eval.yaml"


# ─────────────────────────────────────────────────────────────────────
#  Entry point: static contract + live smoke
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.eval_engine("lmms_eval")
@pytest.mark.requires_extra("lmms_eval")
@pytest.mark.timeout(60)
class TestLmmsEvalEvaluationEngine(EvaluationEngineCommonTests):
    """lmms_eval `engine.get_data_iterator` / `engine.evaluate_task` export contract and live smoke."""

    @pytest.fixture
    def engine_module(self):
        return engine

    # ── live smoke (slow + requires_extra) ─────────────────────

    @pytest.mark.slow
    @pytest.mark.requires_hf_token
    @pytest.mark.timeout(600)  # real dataset loading (get_data_iterator) exceeds the class default 60s → raised
    def test_live_smoke(
        self, evaluation_config_dir, fake_inference_record_factory,
    ):
        """Sequentially verifies 1 cycle of `get_data_iterator` → `evaluate_task` for every benchmark in `lmms_eval.yaml`."""
        import itertools

        import yaml

        with open(evaluation_config_dir / _LMMS_EVAL_YAML) as f:
            cfg = yaml.safe_load(f)
        benchmarks = [b.strip() for b in cfg["benchmarks"].split(",")]

        for benchmark in benchmarks:
            # get_data_iterator: no debug argument → truncate with islice (§1.4)
            iterator, task_config = engine.get_data_iterator(
                evaluation_engine="lmms_eval",
                task_name=benchmark,
                run_index=0,
            )
            assert task_config.num_records > 0, f"{benchmark}: num_records must be > 0"
            # yield only up to NUM_DEBUG_SAMPLES via islice
            truncated = list(itertools.islice(iterator, NUM_DEBUG_SAMPLES))
            assert len(truncated) >= 1, f"{benchmark}: iterator yielded no records"

            print(
                f"\n  ── [lmms_eval live smoke / {benchmark}] ──"
                f"\n    num_records  : {task_config.num_records}"
                f"\n    yielded      : {len(truncated)}"
            )

            fake_records = fake_inference_record_factory(
                predictions=["dummy"] * len(truncated),
                labels=["dummy"] * len(truncated),
                benchmark=benchmark,
            )
            result, sample_metrics = engine.evaluate_task(
                evaluation_engine="lmms_eval",
                task_name=benchmark,
                evaluation_method="generation",
                task_config=task_config,
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
    query="What is in the image?",
    doc_to_visual=None,
    task_name="ai2d",
    repeats=1,
    request_type="generate_until",
):
    # lmms_eval Instance.arguments = (query, generation_options, doc_to_visual,
    # doc_id, task_name, task_split) - the default branch only uses arguments[0], [2]
    args = (query, None, doc_to_visual or (lambda d: []), "doc_0", task_name, "test")
    return SimpleNamespace(
        arguments=args,
        repeats=repeats,
        request_type=request_type,
        task_name=task_name,
    )


def _make_task(
    output_type="generate_until",
    doc_to_target=None,
    doc_to_choice=None,
    generation_kwargs=None,
    num_fewshot=None,
    metadata=None,
    doc_to_messages=None,  # if set → ConfigurableMessagesTask branch
):
    cfg = SimpleNamespace(
        output_type=output_type,
        generation_kwargs=generation_kwargs,
        num_fewshot=num_fewshot,
        metadata=metadata,
        doc_to_choice=doc_to_choice,
    )
    task = SimpleNamespace(
        config=cfg,
        doc_to_target=doc_to_target or (lambda d: "fallback_target"),
        doc_to_choice=doc_to_choice,
    )
    if doc_to_messages is not None:
        task.doc_to_messages = doc_to_messages
    return task


@pytest.mark.eval_engine("lmms_eval")
@pytest.mark.timeout(10)
def test_sample_to_record():
    """`sample_to_record` returns a Record and the core identity fields (`benchmark`, `index`, `messages`) are populated."""
    rec = sample_to_record(
        task_name="ai2d",
        doc_id=7,
        doc={"question": "Q?", "answer": "A"},
        instance=_make_instance(),
        task=_make_task(),
    )
    assert isinstance(rec, Record)
    assert rec.benchmark == "ai2d"
    assert rec.index == 7
    assert rec.messages  # non-empty
