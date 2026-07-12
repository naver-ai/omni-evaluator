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

"""Unit-tests the non-LLM deterministic surface of JudgeEvaluator (response parsers, postprocess, aggregation)."""
from __future__ import annotations

from collections import defaultdict

import pytest

from omni_evaluator.evaluation.metrics.judge_evaluator import JudgeEvaluator

from ._judge_cases import (
    AGGREGATE_CHOICE,
    BINARY,
    BINARY_RUBRIC,
    CHOICE,
    POSTPROCESS_BINARY,
    POSTPROCESS_PAIRWISE,
    POSTPROCESS_RATING,
    REASON,
    REASON_RUBRIC,
    SCORE_NONE,
    SCORE_RUBRIC,
    SCORE_VALUE,
    SPLIT_PAIRWISE_REASON,
)

pytestmark = pytest.mark.eval_engine("builtin")


# ── _parse_response_choice ───────────────────────────────────────────────────
@pytest.mark.parametrize("response, expected", CHOICE)
def test_parse_response_choice(response, expected):
    """'Choice:' line → A/B/True/False/None."""
    assert JudgeEvaluator._parse_response_choice(response) == expected


# ── _aggregate_choice ────────────────────────────────────────────────────────
@pytest.mark.parametrize("choice_ab, choice_ba, expected", AGGREGATE_CHOICE)
def test_aggregate_choice(choice_ab, choice_ba, expected):
    """AB/BA two judgments → winner count dict."""
    assert dict(JudgeEvaluator._aggregate_choice(choice_ab, choice_ba)) == expected


# ── _parse_response_binary ───────────────────────────────────────────────────
@pytest.mark.parametrize("response, expected", BINARY)
def test_parse_response_binary(response, expected):
    """Path without rubrics: true/false mapping, returns None if not found."""
    assert JudgeEvaluator._parse_response_binary(response) == expected


# ── _parse_response_score ────────────────────────────────────────────────────
@pytest.mark.parametrize("response, max_rating, expected", SCORE_VALUE)
def test_parse_response_score(response, max_rating, expected):
    """Single/paired score (float); normalizes when max_rating is given."""
    assert JudgeEvaluator._parse_response_score(
        response, max_rating=max_rating
    ) == pytest.approx(expected)


@pytest.mark.parametrize("response", SCORE_NONE)
def test_parse_response_score_unparseable(response):
    """Returns None when no number is found."""
    assert JudgeEvaluator._parse_response_score(response) is None


# ── _split_pairwise_reason ───────────────────────────────────────────────────
@pytest.mark.parametrize("text, expected", SPLIT_PAIRWISE_REASON)
def test_split_pairwise_reason(text, expected):
    """Splits only at depth-0 commas; ignores commas inside nested brackets."""
    assert JudgeEvaluator._split_pairwise_reason(text) == expected


# ── _parse_response_reason ───────────────────────────────────────────────────
@pytest.mark.parametrize("response, expected", REASON)
def test_parse_response_reason(response, expected):
    """Extracts text after '[REASON]:' / 'Reason:' tag, returns None if absent."""
    assert JudgeEvaluator._parse_response_reason(response) == expected


# ─────────────────────────────────────────────────────────────────────────────
#  rubric-mode branch (parser returns per-rubric dict when it receives a rubrics dict)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("response, rubrics, expected", SCORE_RUBRIC)
def test_parse_response_score_rubric(response, rubrics, expected):
    """Per-rubric 'name: <num>' lines → {rubric: float}."""
    assert JudgeEvaluator._parse_response_score(response, rubrics=rubrics) == pytest.approx(expected)


@pytest.mark.parametrize("response, rubrics, expected", REASON_RUBRIC)
def test_parse_response_reason_rubric(response, rubrics, expected):
    """Per-rubric 'name: <reason>' lines → {rubric: reason}."""
    assert JudgeEvaluator._parse_response_reason(response, rubrics=rubrics) == expected


@pytest.mark.parametrize("response, rubrics, expected", BINARY_RUBRIC)
def test_parse_response_binary_rubric(response, rubrics, expected):
    """rubric-mode binary must return a {rubric: bool} dict."""
    assert JudgeEvaluator._parse_response_binary(response, rubrics=rubrics) == expected


# ─────────────────────────────────────────────────────────────────────────────
#  (C) _postprocess_judge_*_record — assembles parser results into a result dict (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("response, max_rating, expected_scores", POSTPROCESS_RATING)
def test_postprocess_rating(response, max_rating, expected_scores):
    """rating postprocess: fills scores and echoes response."""
    out = JudgeEvaluator._postprocess_judge_rating_record(response, max_rating=max_rating)
    assert out["scores"] == pytest.approx(expected_scores)
    assert out["response"] == response


@pytest.mark.parametrize("response, expected_scores, expected_acc", POSTPROCESS_BINARY)
def test_postprocess_binary(response, expected_scores, expected_acc):
    """binary postprocess: scores(bool) + accuracy(1.0/0.0/None)."""
    out = JudgeEvaluator._postprocess_judge_binary_record(response)
    assert out["scores"] == expected_scores
    assert out["accuracy"] == expected_acc


@pytest.mark.parametrize("ab, ba, win_rate, win_point", POSTPROCESS_PAIRWISE)
def test_postprocess_pairwise(ab, ba, win_rate, win_point):
    """pairwise postprocess: aggregates AB/BA choices → win_rate / win_point."""
    out = JudgeEvaluator._postprocess_judge_pairwise_record(ab, ba)
    assert out["win_rate"] == pytest.approx(win_rate)
    assert out["win_point"] == pytest.approx(win_point)


# ─────────────────────────────────────────────────────────────────────────────
#  (C) _aggregate_judge_results — aggregates scores across models (np.nanmean)
# ─────────────────────────────────────────────────────────────────────────────

def _result(scores):
    return {"scores": scores, "choice": None, "reasons": None}


def test_aggregate_results_scalar_mean_over_models():
    """Scalar scores from multiple judge models are combined by mean (nanmean)."""
    aggregated = JudgeEvaluator._aggregate_judge_results(
        [[_result(8.0)], [_result(6.0)]]
    )
    assert aggregated[0]["scores"] == pytest.approx(7.0)


def test_aggregate_results_rubric_dict_mean_over_models():
    """rubric dict scores are averaged across models per rubric."""
    aggregated = JudgeEvaluator._aggregate_judge_results(
        [[_result({"c": 8.0})], [_result({"c": 6.0})]]
    )
    assert aggregated[0]["scores"]["c"] == pytest.approx(7.0)


# ─────────────────────────────────────────────────────────────────────────────
#  (C) _collect_judge_results — judge_results → metrics / sample_metrics dict
# ─────────────────────────────────────────────────────────────────────────────

def _collect(judge_results, num_records=1):
    metrics, group_metrics = defaultdict(list), {}
    sample_metrics = [{} for _ in range(num_records)]
    records = [{"meta": {"category": None}} for _ in range(num_records)]
    return JudgeEvaluator._collect_judge_results(
        target_metric="judge_rating",
        metrics=metrics,
        group_metrics=group_metrics,
        sample_metrics=sample_metrics,
        records=records,
        judge_results=judge_results,
        exclude_rubrics=[],
    )


def test_collect_scalar_score():
    """Scalar scores are recorded in metrics['judge_rating'] and sample_metrics."""
    metrics, _group, sample = _collect([_result(8.0)])
    assert metrics["judge_rating"] == [8.0]
    assert sample[0]["judge_rating"] == pytest.approx(8.0)


def test_collect_rubric_scores_sum_and_avg():
    """rubric dict scores → per-rubric keys + rubric_sum / rubric_avg derived metrics."""
    metrics, _group, _sample = _collect([_result({"coherence": 8.0, "fluency": 6.0})])
    assert metrics["judge_rating/coherence"] == [8.0]
    assert metrics["judge_rating/fluency"] == [6.0]
    assert metrics["judge_rating/rubric_sum"][0] == pytest.approx(14.0)
    assert metrics["judge_rating/rubric_avg"][0] == pytest.approx(7.0)
