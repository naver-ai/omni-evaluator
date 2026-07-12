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

"""Unit tests for omni_evaluator/schemas/evaluation.py — RunOutput numpy/statistics branches + EvaluationOutput validate/aggregate/merge."""

import numpy as np
import pytest

from omni_evaluator.schemas.evaluation import (
    EvaluationOutput,
    EvaluationRunOutput,
    EvaluationStatistics,
)

from .test_schemas_common import _SchemaMixin


class _FakeTokenizer:
    """Minimal stand-in for tokenizer branching — tokenizes by whitespace."""

    def tokenize(self, text):
        return text.split()


def _run_output(**overrides):
    """Minimal run_output dict for aggregate/add_run_output input — replace per-branch values via overrides.

    statistics defaults to an empty dict — intentional, because a non-empty value breaks the source statistics aggregation loop (see xfail below).
    """
    base = {
        "coverage_inference": 1.0,
        "coverage_evaluation": 1.0,
        "latency": 1.0,
        "throughput": 1.0,
        "runtime_inference": 1.0,
        "runtime_evaluation": 1.0,
        "runtime_postprocess": 1.0,
        "statistics": {},
        "metric_keys": ["acc"],
        "metrics": {"acc": 1.0},
        "group_metrics": {},
        "num_samples": 2,
    }
    base.update(overrides)
    return base


# ============================================================================
# Per schema — common mixin contract + specialised branches.
# ============================================================================


class TestEvaluationStatistics(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return EvaluationStatistics(avg_num_chars=10)


class TestEvaluationRunOutput(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return EvaluationRunOutput(task_name="t")

    def test_defaults(self):
        """All fields can be created with defaults — run_index/num_runs are 1, the rest are None."""
        output = EvaluationRunOutput()
        assert output.run_index == 1
        assert output.num_runs == 1
        assert output.metrics is None
        assert output.statistics is None
        assert output.outputs is None

    def test_metrics_numpy_cast(self):
        """__post_init__ converts np.floating→float and np.integer→int in the metrics dict, and preserves non-numpy values."""
        output = EvaluationRunOutput(
            metrics={"score": np.float64(0.9), "count": np.int64(3), "name": "exact_match"}
        )
        assert type(output.metrics["score"]) is float
        assert type(output.metrics["count"]) is int
        assert output.metrics["name"] == "exact_match"

    def test_metrics_non_dict_noop(self):
        """If metrics is not a dict (including None), __post_init__ does nothing."""
        assert EvaluationRunOutput(metrics=None).metrics is None

    def test_stats_str_prediction(self):
        """If prediction is a str, averages char/word counts as a single prediction (no token key since tokenizer=None)."""
        output = EvaluationRunOutput()
        output.update_statistics([{"prediction": "hello world"}], tokenizer=None)
        assert output.statistics.avg_num_chars == 11
        assert output.statistics.avg_num_words == 2
        assert output.statistics.avg_num_tokens is None

    def test_stats_with_tokenizer(self):
        """If a tokenizer is provided, avg_num_tokens is populated with the tokenize result length."""
        output = EvaluationRunOutput()
        output.update_statistics([{"prediction": "a b c"}], tokenizer=_FakeTokenizer())
        assert output.statistics.avg_num_tokens == 3
        assert output.statistics.avg_num_words == 3

    def test_stats_list_prediction(self):
        """If prediction is a list, each element is unpacked; dict elements are unwrapped via truthy `value`."""
        output = EvaluationRunOutput()
        output.update_statistics([{"prediction": [{"value": "abcd"}, "xy"]}], tokenizer=None)
        # "abcd"(4) + "xy"(2) → average 3 chars, 1 word each → average 1
        assert output.statistics.avg_num_chars == 3
        assert output.statistics.avg_num_words == 1

    def test_stats_dict_prediction(self):
        """If prediction is a dict, each value is unpacked; dict values are unwrapped via truthy `value`."""
        output = EvaluationRunOutput()
        output.update_statistics(
            [{"prediction": {"k1": {"value": "abcd"}, "k2": "xy"}}], tokenizer=None
        )
        assert output.statistics.avg_num_chars == 3
        assert output.statistics.avg_num_words == 1

    def test_stats_non_str_skipped(self):
        """Non-str predictions are excluded from counts; if nothing remains, the record is skipped entirely and all statistics are None."""
        output = EvaluationRunOutput()
        output.update_statistics([{"prediction": 123}, {"prediction": []}], tokenizer=None)
        assert output.statistics == EvaluationStatistics()
        assert output.statistics.avg_num_chars is None


class TestEvaluationOutput(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return EvaluationOutput(task_name="t")

    # ── validate ─────────────────────────────────────────────────────────────
    @pytest.fixture
    def valid_kwargs(self):
        """Minimum field set required to pass validate — reused by invalidating one key at a time per branch."""
        return dict(
            inference_engine="hf",
            evaluation_engine="builtin",
            task_name="t",
            metrics={"acc": 1.0},
            group_metrics={"catA": {"acc": 1.0}},
        )

    def test_validate_accepts_instance_and_dict(self, valid_kwargs):
        """Both instance and dict input forms — required fields + non-empty metrics/group_metrics → True."""
        assert EvaluationOutput.validate(EvaluationOutput(**valid_kwargs)) is True
        assert EvaluationOutput.validate(dict(valid_kwargs)) is True

    def test_validate_rejects_non_output(self):
        """Neither instance nor dict (None/str/int) fails coercion → False."""
        assert EvaluationOutput.validate(None) is False
        assert EvaluationOutput.validate("nope") is False
        assert EvaluationOutput.validate(123) is False

    def test_validate_rejects_missing_identity(self, valid_kwargs):
        """If either engine or task_name is empty, returns False (same identity branch)."""
        assert EvaluationOutput.validate({**valid_kwargs, "task_name": None}) is False
        assert EvaluationOutput.validate({**valid_kwargs, "inference_engine": None}) is False
        assert EvaluationOutput.validate({**valid_kwargs, "evaluation_engine": None}) is False

    def test_validate_rejects_bad_metrics(self, valid_kwargs):
        """Returns False if metrics is an empty dict or not a dict."""
        assert EvaluationOutput.validate({**valid_kwargs, "metrics": {}}) is False
        assert EvaluationOutput.validate({**valid_kwargs, "metrics": None}) is False

    def test_validate_rejects_bad_group_metrics(self, valid_kwargs):
        """Returns False if group_metrics is an empty dict or not a dict."""
        assert EvaluationOutput.validate({**valid_kwargs, "group_metrics": {}}) is False
        assert EvaluationOutput.validate({**valid_kwargs, "group_metrics": None}) is False

    # ── aggregate_run_outputs ──────────────────────────────────────────────────
    def test_aggregate_mean_std(self):
        """Multiple runs without sample_metrics produce only mean+std for metrics/coverage/group, without -any/-all suffixes."""
        runs = [
            _run_output(coverage_inference=1.0, metrics={"acc": 1.0}, group_metrics={"catA": {"acc": 1.0}}),
            _run_output(coverage_inference=0.0, metrics={"acc": 0.0}, group_metrics={"catA": {"acc": 0.0}}),
        ]
        out = EvaluationOutput.aggregate_run_outputs(runs)
        assert out["coverage_inference"] == 0.5
        assert out["metrics"]["acc"] == 0.5
        assert out["metrics"]["acc-std"] == pytest.approx(0.5)
        assert "acc-any" not in out["metrics"]
        assert out["group_metrics"]["catA"]["acc"] == 0.5
        assert set(out["metric_keys"]) == {"acc"}

    def test_aggregate_any_all_from_sample_metrics(self):
        """If sample_metrics is present, additionally aggregates per-sample any(at least one correct)/all(all correct) ratios."""
        runs = [
            _run_output(metrics={"acc": 0.5}, sample_metrics=[{"acc": 1}, {"acc": 0}]),
            _run_output(metrics={"acc": 1.0}, sample_metrics=[{"acc": 1}, {"acc": 1}]),
        ]
        out = EvaluationOutput.aggregate_run_outputs(runs)
        assert out["metrics"]["acc"] == pytest.approx(0.75)
        assert out["metrics"]["acc-any"] == 1.0   # both samples are correct in at least one run
        assert out["metrics"]["acc-all"] == 0.5   # only sample0 is correct in all runs
        assert out["metrics"]["acc-std"] == pytest.approx(0.25)

    @pytest.mark.xfail(
        raises=TypeError,
        strict=True,
        reason="source bug: the statistics aggregation loop indexes an accumulation list with a str key "
        "(`statistics[name][name]=...`, evaluation.py:277-279). Runs with non-empty statistics cannot be aggregated. "
        "Fix will be signaled as XPASS",
    )
    def test_aggregate_nonempty_statistics_is_broken(self):
        """Pins that aggregating runs with populated statistics is broken — TypeError on list[str] assignment."""
        runs = [
            _run_output(statistics={"avg_num_chars": 10.0}),
            _run_output(statistics={"avg_num_chars": 20.0}),
        ]
        EvaluationOutput.aggregate_run_outputs(runs)

    # ── add_run_output ─────────────────────────────────────────────────────────
    def test_add_first_run_copies(self):
        """The first run copies fields directly without aggregation."""
        out = EvaluationOutput(task_name="t")
        out.add_run_output(_run_output(metrics={"acc": 1.0}))
        assert len(out.run_outputs) == 1
        assert out.metrics == {"acc": 1.0}
        assert out.coverage_inference == 1.0
        assert out.num_samples == 2

    def test_add_second_run_aggregates(self):
        """From the second run onward, re-aggregates accumulated runs to update the overall result."""
        out = EvaluationOutput(task_name="t")
        out.add_run_output(_run_output(coverage_inference=1.0, metrics={"acc": 1.0}))
        out.add_run_output(_run_output(coverage_inference=0.0, metrics={"acc": 0.0}))
        assert len(out.run_outputs) == 2
        assert out.metrics["acc"] == 0.5
        assert out.metrics["acc-std"] == pytest.approx(0.5)
        assert out.coverage_inference == 0.5
