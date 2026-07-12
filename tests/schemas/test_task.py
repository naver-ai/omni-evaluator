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

"""Unit tests for omni_evaluator/schemas/task.py — verifies __post_init__ conversions and from_engine dispatch routing for TaskConfig and TaskEvaluation."""

import sys
import types

import pytest

from omni_evaluator import (
    DatasetSource,
    EvaluationMethod,
    Modality,
    SubtaskType,
    TaskType,
)
from omni_evaluator.schemas.task import (
    TaskConfig,
    TaskDataset,
    TaskEvaluation,
    TaskEvaluationJudge,
    TaskEvaluationPostprocess,
    TaskInference,
    TaskInferenceGenerationOptions,
    TaskMeta,
    TaskPrompts,
    TaskVerifier,
)

from .test_schemas_common import _SchemaMixin


# ============================================================================
# One class per schema — common mixin contract + schema-specific branches.
# ============================================================================


class TestTaskMeta(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return TaskMeta(benchmark_name="b")

    def test_enum_coercion(self):
        """Raw str for task_type / subtask_type (single or list) is normalized to an enum list."""
        m = TaskMeta(
            benchmark_name="b",
            task_type="visual_question_answering",          # non-list → wrapped in list
            subtask_type=["document", "chart"],
        )
        assert m.task_type == [TaskType.visual_question_answering]
        assert m.subtask_type == [SubtaskType.document, SubtaskType.chart]

    def test_none_passthrough(self):
        """task_type / subtask_type remains None without conversion when None."""
        m = TaskMeta(benchmark_name="b")
        assert m.task_type is None and m.subtask_type is None


class TestTaskDataset(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return TaskDataset(source=DatasetSource.local)


class TestTaskPrompts(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return TaskPrompts()


class TestTaskInferenceGenerationOptions(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return TaskInferenceGenerationOptions()


class TestTaskInference(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return TaskInference()

    def test_options_conversion(self):
        """generation_options is converted to TaskInferenceGenerationOptions when a dict; remains None when None."""
        converted = TaskInference(generation_options={"max_new_tokens": 16})
        assert isinstance(converted.generation_options, TaskInferenceGenerationOptions)
        assert converted.generation_options.max_new_tokens == 16
        assert TaskInference().generation_options is None


class TestTaskEvaluationJudge(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return TaskEvaluationJudge(lang="en", judge_model="m")


class TestTaskEvaluationPostprocess(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return TaskEvaluationPostprocess()


class TestTaskEvaluation(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return TaskEvaluation(method="generation", target_metrics=["acc"])

    def test_postprocess_conversion(self):
        """postprocess is converted to TaskEvaluationPostprocess when a dict."""
        ev = TaskEvaluation(
            method="generation",
            target_metrics=["exact_match"],
            postprocess={"pipeline": ["strip"], "allow_api": True},
        )
        assert isinstance(ev.postprocess, TaskEvaluationPostprocess)
        assert ev.postprocess.pipeline == ["strip"] and ev.postprocess.allow_api is True


# ============================================================================
# TaskConfig.from_engine — stub fixture for dispatch routing verification
# (module-level, consumed by TestTaskConfig).
# ============================================================================


@pytest.fixture
def stub_builtin_engine(monkeypatch):
    """Replaces `omni_evaluator.evaluation.builtin._build_task_config` with a stub to verify dispatch routing only."""
    calls = {}

    def _fake_build(task_name, reasoning=False):
        calls["task_name"] = task_name
        calls["reasoning"] = reasoning
        return f"built:{task_name}"

    mod = types.ModuleType("omni_evaluator.evaluation.builtin")
    mod._build_task_config = _fake_build
    monkeypatch.setitem(sys.modules, "omni_evaluator.evaluation.builtin", mod)
    return calls


class TestTaskConfig(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return TaskConfig(
            task_name="t",
            evaluation_engine="builtin",
            meta=TaskMeta(benchmark_name="b"),
            dataset=TaskDataset(source=DatasetSource.local),
        )

    def test_nested_conversion(self, sample_task_config_dict):
        """Dict values for meta/dataset/evaluation/prompts/inference are all converted to nested dataclasses."""
        cfg = TaskConfig.from_kwargs(
            **{
                **sample_task_config_dict,
                "prompts": {"system_prompt": "sys"},
                "inference": {"num_ocr_tokens": 5},
            }
        )
        assert isinstance(cfg.meta, TaskMeta) and cfg.meta.benchmark_name == "dummy"
        assert isinstance(cfg.dataset, TaskDataset) and cfg.dataset.source == DatasetSource.local
        assert isinstance(cfg.evaluation, TaskEvaluation)
        assert cfg.evaluation.method == EvaluationMethod.generation
        assert isinstance(cfg.prompts, TaskPrompts) and cfg.prompts.system_prompt == "sys"
        assert isinstance(cfg.inference, TaskInference) and cfg.inference.num_ocr_tokens == 5

    def test_verifier_conversion(self, sample_task_config_dict):
        """A dict ``verifier:`` block hydrates to TaskVerifier (set fields kept,
        unset -> None for per-field VerifierArgs fallback, unknown keys dropped);
        an absent block leaves verifier=None."""
        cfg = TaskConfig.from_kwargs(
            **{**sample_task_config_dict,
               "verifier": {"engine": "huggingface", "num_concurrency": 4, "prompt": "DROP_ME"}}
        )
        assert isinstance(cfg.verifier, TaskVerifier)
        assert cfg.verifier.engine == "huggingface" and cfg.verifier.num_concurrency == 4
        assert cfg.verifier.api_name is None and cfg.verifier.temperature is None  # unset -> args fallback
        assert not hasattr(cfg.verifier, "prompt")                                  # unknown key dropped
        assert TaskConfig.from_kwargs(**sample_task_config_dict).verifier is None    # absent block

    def test_instance_passthrough(self):
        """Already-instantiated nested fields pass through without re-conversion, and evaluation=None remains None."""
        meta = TaskMeta(benchmark_name="b", output_modality=[Modality.text])
        dataset = TaskDataset(source=DatasetSource.local)
        cfg = TaskConfig(
            task_name="t",
            evaluation_engine="builtin",
            meta=meta,
            dataset=dataset,
            evaluation=None,
        )
        assert cfg.meta is meta and cfg.dataset is dataset
        assert cfg.evaluation is None
        assert isinstance(cfg.prompts, TaskPrompts)        # default_factory result
        assert isinstance(cfg.inference, TaskInference)

    def test_dispatch_builtin(self, stub_builtin_engine):
        """evaluation_engine="builtin" → from_builtin delegates to builtin._build_task_config, and the str engine name is normalized to an enum."""
        result = TaskConfig.from_engine("builtin", task_name="ai2d", reasoning=True)
        assert result == "built:ai2d"
        assert stub_builtin_engine == {"task_name": "ai2d", "reasoning": True}

    def test_dispatch_unknown(self):
        """Unknown engine_name → ValueError raised at EvaluationEngine construction."""
        with pytest.raises(ValueError):
            TaskConfig.from_engine("nonexistent", task_name="x")

    def test_round_trip(self, sample_task_config_dict):
        """dict → TaskConfig → to_dict preserves core fields (including nested flattening)."""
        cfg = TaskConfig.from_kwargs(**sample_task_config_dict)
        d = cfg.to_dict()
        assert d["task_name"] == "dummy"
        assert d["meta"]["benchmark_name"] == "dummy"
        assert d["dataset"]["source"] == DatasetSource.local
        assert d["evaluation"]["target_metrics"] == ["exact_match"]
