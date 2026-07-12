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

"""Input/expected-value case table for TextEvaluator metric unit tests."""
from __future__ import annotations

import pytest


# ── compute_exact_match(label, prediction) → float ───────────────────────────
EXACT_MATCH = [
    pytest.param("yes", "yes", 1.0, id="string_eq"),
    pytest.param("yes", "no", 0.0, id="string_neq"),
    pytest.param("Cat.", "cat", 1.0, id="case_and_dot_insensitive"),   # "cat"=="cat"
    pytest.param("3.0", "3", 1.0, id="numeric_eq_3.0_vs_3"),            # float 3.0==3.0
    pytest.param("1", "①", 1.0, id="circled_number_① _eq_1"),     # ①→"1"→1.0
]


# ── compute_exact_match tolerance injection/no-injection (label, prediction, kwargs) → float ──
# Default relative_tolerance=None → absolute error 1e-6 (strict). 5% relative tolerance must be explicitly injected to apply.
EXACT_MATCH_TOLERANCE = [
    pytest.param("100", "103", {}, 0.0, id="no_tol_strict_default"),                       # no injection → 1e-6, |−3|≮1e-6
    pytest.param("100", "103", {"relative_tolerance": 0.05}, 1.0, id="rel_tol_within"),    # tol=5, |−3|<5
    pytest.param("100", "106", {"relative_tolerance": 0.05}, 0.0, id="rel_tol_outside"),   # tol=5, |−6|≮5
]


# ── compute_substring_match(label, prediction, normalize=None) → float ────────
# ⚠️ normalize="squad" default path is broken with AttributeError — pinned as xfail in test.
SUBSTRING_MATCH = [
    pytest.param("b c", "a b c d", 1.0, id="contiguous_hit"),
    pytest.param("x y", "a b c d", 0.0, id="not_a_subsequence"),
    pytest.param("a b c d", "a b c d", 1.0, id="full_match"),
]


# ── compute_string_match(label, prediction) → float ──────────────────────────
STRING_MATCH = [
    pytest.param("b c", "a b c d", 1.0, id="substring_hit"),
    pytest.param("z", "a b c d", 0.0, id="absent"),
    pytest.param("a b", "a\nb c", 1.0, id="newline_collapsed_hit"),
    pytest.param("", "anything", 1.0, id="empty_label_is_substring_of_all"),
]


# ── compute_ned(label, prediction, uncased=True) → float ─────────────────────
NED = [
    pytest.param("abc", "abc", 0.0, id="identical_0"),
    pytest.param("abc", "abxc", 0.25, id="one_insert_dist1_over_maxlen4"),
    pytest.param("abc", "abd", 1 / 3, id="one_sub_dist1_over_3"),
]


# ── compute_levenshtein_distance(label, prediction, uncased=True) → float ─────
LEVENSHTEIN = [
    pytest.param("abc", "abc", 0.0, id="identical_0"),
    pytest.param("kitten", "sitting", 3 / 7, id="classic_dist3_over_maxlen7"),
]


# ── compute_jaccard_distance(labels, prediction, n=1) → float ────────────────
JACCARD = [
    pytest.param(["a b c"], "a b c", 0.0, id="identical_sets"),
    pytest.param(["a b"], "a b c", 1 / 3, id="subset_1_minus_2_over_3"),
    pytest.param(["x y", "a b c"], "a b c", 0.0, id="min_over_labels_picks_exact"),
]


# ── compute_anls(labels, prediction, threshold=0.5) → float ──────────────────
ANLS = [
    pytest.param(["abc"], "abc", 1.0, id="exact_nl0"),
    pytest.param(["abc"], "abd", 1 - 1 / 3, id="nl_0.333_below_threshold"),
    pytest.param(["ab"], "xb", 0, id="nl_0.5_at_threshold_returns_0"),
    pytest.param(["abcd"], "wxyd", 0, id="nl_0.75_above_threshold_returns_0"),
]


# ── compute_f1(labels, prediction, normalize="squad", aggregate="max") ────────
F1 = [
    pytest.param(["cat dog bird"], "cat dog",
                 {"precision": 1.0, "recall": 2 / 3, "f1": 0.8}, id="recall_partial_2common_of_3gt"),
    pytest.param(["cat dog"], "cat fish",
                 {"precision": 0.5, "recall": 0.5, "f1": 0.5}, id="one_common_half_each"),
    pytest.param(["cat"], "dog",
                 {"precision": 0, "recall": 0, "f1": 0}, id="no_overlap"),
    pytest.param(["cat"], "",
                 {"precision": 0, "recall": 0, "f1": 0}, id="empty_prediction"),
    pytest.param(["cat fish", "cat dog"], "cat dog",
                 {"precision": 1.0, "recall": 1.0, "f1": 1.0}, id="max_aggregate_picks_best_label"),
]


# ── compute_binary_f1(labels: List[List[str]], predictions: List[str]) ────────
BINARY_F1 = [
    pytest.param(
        [["yes"], ["no"], ["no"], ["yes"]],
        ["Yes, it is.", "No it isn't.", "Yes definitely", "not sure"],
        {"f1": 0.5, "accuracy": 0.5, "precision": 0.5, "recall": 0.5},
        id="one_each_TP_TN_FP_FN",
    ),
    pytest.param(
        [["yes"], ["no"]],
        ["yes", "no"],
        {"f1": 1.0, "accuracy": 1.0, "precision": 1.0, "recall": 1.0},
        id="perfect_classification",
    ),
]


# ── compute_vqaeval(labels, prediction) → float ──────────────────────────────
VQAEVAL = [
    pytest.param(["cat"], "cat", 1 / 3, id="one_match_over_3"),
    pytest.param(["cat", "cat"], "cat", 2 / 3, id="two_matches_over_3"),
    pytest.param(["cat", "cat", "cat"], "cat", 1.0, id="three_matches_caps_at_1"),
    pytest.param(["dog"], "cat", 0, id="no_match_returns_0"),
    pytest.param(["a cat"], "cat", 1 / 3, id="article_normalized_then_match"),
]


# ── compute_mmau_string_match(label, prediction, option_contents) → float ─────
MMAU = [
    pytest.param("cat", "the cat", ["cat", "dog"], 1.0, id="answer_subset_no_incorrect_token"),
    pytest.param("cat", "dog", ["cat", "dog"], 0.0, id="answer_token_absent"),
    pytest.param("cat", "cat dog", ["cat", "dog"], 0.0, id="prediction_has_incorrect_token"),
    pytest.param("(A) cat", "cat", ["(A) cat", "(B) dog"], 1.0, id="option_prefix_stripped"),
]


# ── normalize(text, method="squad") → str ────────────────────────────────────
NORMALIZE_SQUAD = [
    pytest.param("The A Cat.", "cat", id="articles_and_punct_and_case_stripped"),
    pytest.param("Hello, World!", "hello world", id="punct_removed_lowercased"),
]


# ── pure predicate helpers ───────────────────────────────────────────────────
IS_NUMERIC_STRING = [
    pytest.param("3.14", True, id="float"),
    pytest.param("1e5", True, id="scientific"),
    pytest.param("abc", False, id="word"),
    pytest.param("", False, id="empty"),
]
IS_CIRCLED_NUMBER = [
    pytest.param("①", True, id="circled_one"),
    pytest.param("1", False, id="plain_digit"),
]
IS_DATE_STRING = [
    pytest.param("2020-01-01", True, id="iso_date"),
    pytest.param("xyzzy", False, id="nonsense_not_date"),
]


# ── compute_tree_edit (direct mode — no stage-1 LLM conversion) ──────────────
# (labels, predictions, groups, expected_sample_scores)
TREE_EDIT = [
    pytest.param(
        [["<html><body><p>hi</p></body></html>"]],
        ["<html><body><p>hi</p></body></html>"],
        None,
        [pytest.approx(1.0)],
        id="identical",
    ),
    pytest.param(
        [["<p>hi</p>"]], [""], None,
        [None],
        id="empty_pred_returns_none",
    ),
    pytest.param(
        [["<p>WRONG</p>", "<p>hi</p>"]], ["<p>hi</p>"], None,
        [pytest.approx(1.0)],
        id="multi_label_max",
    ),
]


# ── compute_temporal_iou ─────────────────────────────────────────────────────
# (label, prediction, expected_dict)
TEMPORAL_IOU = [
    pytest.param(
        "10.0 - 20.0", "10.0 - 20.0",
        {"iou": 1.0, "r@0.3": 1.0, "r@0.5": 1.0, "r@0.7": 1.0},
        id="perfect_overlap",
    ),
    pytest.param(
        "10.0 - 20.0", "15.0 - 25.0",
        # inter=5, union=15 → iou≈0.333
        {"iou": pytest.approx(1 / 3, rel=1e-3), "r@0.3": 1.0, "r@0.5": 0.0, "r@0.7": 0.0},
        id="partial_overlap",
    ),
    pytest.param(
        "bad", "bad",
        {"iou": 0.0, "r@0.3": 0.0, "r@0.5": 0.0, "r@0.7": 0.0},
        id="unparseable",
    ),
]
