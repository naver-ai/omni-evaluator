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

"""Input/expected-value case table for the `JudgeEvaluator` response parser."""
from __future__ import annotations

import pytest


# ── _parse_response_choice(response) → "A" | "B" | True | False | None ───────
CHOICE = [
    pytest.param("Choice: A", "A", id="choice_A"),
    pytest.param("Choice: B", "B", id="choice_B"),
    pytest.param("Choice: true", True, id="choice_true"),
    pytest.param("Choice: false", False, id="choice_false"),
    pytest.param("reason...\nChoice: A wins", "A", id="a_before_b_in_text"),
    pytest.param("no choice line", None, id="no_choice_marker"),
    pytest.param("", None, id="empty_response"),
]


# ── _aggregate_choice(choice_ab, choice_ba) → defaultdict(int) ───────────────
AGGREGATE_CHOICE = [
    pytest.param(None, None, {"tie": 1, "total": 1}, id="both_none_tie"),
    pytest.param("A", None, {"A": 1, "total": 1}, id="ab_only_A"),
    pytest.param(None, "B", {"A": 1, "total": 1}, id="ba_only_B_maps_to_A"),
    pytest.param("A", "B", {"A": 1, "total": 1}, id="consistent_A"),
    pytest.param("B", "A", {"B": 1, "total": 1}, id="consistent_B"),
    pytest.param("A", "A", {"tie": 1, "total": 1}, id="inconsistent_tie"),
]


# ── _parse_response_binary(response, rubrics=None) → bool | None ─────────────
BINARY = [
    pytest.param("The answer is true", True, id="true"),
    pytest.param("definitely false", False, id="false"),
    pytest.param("garbage", None, id="unparseable_none"),
]


# ── _parse_response_score(response, max_rating=None) → float | tuple | None ──
SCORE_VALUE = [
    pytest.param("Score: 8", None, 8.0, id="single_score"),
    pytest.param("(8, 7)", None, (8.0, 7.0), id="pairwise_scores"),
    pytest.param("Score: 8", 10, 0.8, id="single_score_normalized_by_max_rating"),
]
SCORE_NONE = [
    pytest.param("no number here", id="no_digit"),
    pytest.param("", id="empty"),
]


# ── _split_pairwise_reason(text) → [a, b] (depth-0 comma split) ──────────────
SPLIT_PAIRWISE_REASON = [
    pytest.param("(good, bad)", ["good", "bad"], id="parenthesized_pair"),
    pytest.param("good, bad", ["good", "bad"], id="bare_pair"),
    pytest.param("only one", ["only one", None], id="no_comma_second_none"),
    pytest.param("(a (x,y), b)", ["a (x,y)", "b"], id="nested_paren_not_split"),
]


# ── _parse_response_reason(response, rubrics=None) → str | None ──────────────
REASON = [
    pytest.param("[REASON]: it is good", "it is good", id="bracketed_tag"),
    pytest.param("Reason: clear", "clear", id="plain_tag"),
    pytest.param("no reason tag", None, id="no_tag_none"),
]


# ── rubric-mode branching (when parsers receive a rubrics dict) ───────────────
SCORE_RUBRIC = [
    pytest.param("coherence: 8", {"coherence": "_"}, {"coherence": 8.0}, id="single_rubric"),
    pytest.param(
        "coherence: 8\nfluency: 6",
        {"coherence": "_", "fluency": "_"},
        {"coherence": 8.0, "fluency": 6.0},
        id="two_rubrics",
    ),
]

REASON_RUBRIC = [
    pytest.param(
        "coherence: well written",
        {"coherence": "_"},
        {"coherence": "well written"},
        id="single_rubric_reason",
    ),
]


# ── rubric-mode _parse_response_binary — *intended* (correct) behavior ───────
# ⚠️ Expected values are the intended correct answers; currently FAILing due to a source bug (evidence that the bug exists).
BINARY_RUBRIC = [
    pytest.param("coherence: true", {"coherence": "_"}, {"coherence": True},
                 id="single_true"),
    pytest.param("coherence: false", {"coherence": "_"}, {"coherence": False},
                 id="single_false"),
    pytest.param("coherence: true\nfluency: false", {"coherence": "_", "fluency": "_"},
                 {"coherence": True, "fluency": False}, id="multi_rubric"),
]


# ── _postprocess_judge_rating_record(response, max_rating=None) ──────────────
POSTPROCESS_RATING = [
    pytest.param("Score: 8", None, 8.0, id="single_score"),
    pytest.param("Score: 8", 10, 0.8, id="normalized_by_max_rating"),
    pytest.param("", None, None, id="empty_response_score_none"),
]


# ── _postprocess_judge_binary_record(response) ──────────────────────────────
POSTPROCESS_BINARY = [
    pytest.param("the answer is true", True, 1.0, id="true_acc_1"),
    pytest.param("definitely false", False, 0.0, id="false_acc_0"),
    pytest.param("", None, None, id="empty_none"),
]


# ── _postprocess_judge_pairwise_record(response_ab, response_ba) ────────────
POSTPROCESS_PAIRWISE = [
    pytest.param("Choice: A", "Choice: B", 1.0, 1.0, id="consistent_A_wins"),
    pytest.param("Choice: B", "Choice: A", 0.0, 0.0, id="consistent_B_wins"),
    pytest.param("Choice: A", "Choice: A", 0.0, 0.5, id="inconsistent_tie_half_point"),
]
