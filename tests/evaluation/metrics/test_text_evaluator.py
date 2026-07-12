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

"""Unit-verifies the purely deterministic metric functions of TextEvaluator."""
from __future__ import annotations

import pytest

from omni_evaluator.evaluation.metrics.text_evaluator import TextEvaluator

from ._text_cases import (
    ANLS,
    BINARY_F1,
    EXACT_MATCH,
    EXACT_MATCH_TOLERANCE,
    F1,
    IS_CIRCLED_NUMBER,
    IS_DATE_STRING,
    IS_NUMERIC_STRING,
    JACCARD,
    LEVENSHTEIN,
    MMAU,
    NED,
    NORMALIZE_SQUAD,
    STRING_MATCH,
    SUBSTRING_MATCH,
    TEMPORAL_IOU,
    TREE_EDIT,
    VQAEVAL,
)

pytestmark = pytest.mark.eval_engine("builtin")


# ── compute_exact_match ──────────────────────────────────────────────────────
@pytest.mark.parametrize("label, prediction, expected", EXACT_MATCH)
def test_exact_match(label, prediction, expected):
    """circled-number / case and period ignored / numeric equivalence → 0.0|1.0 (default=strict)."""
    assert TextEvaluator.compute_exact_match(label=label, prediction=prediction) == expected


@pytest.mark.parametrize("label, prediction, kwargs, expected", EXACT_MATCH_TOLERANCE)
def test_exact_match_tolerance(label, prediction, kwargs, expected):
    """numeric tolerance: relative_tolerance not injected (default 1e-6, strict) vs injected (5%) branch."""
    assert (
        TextEvaluator.compute_exact_match(label=label, prediction=prediction, **kwargs)
        == expected
    )


# ── compute_substring_match ──────────────────────────────────────────────────
@pytest.mark.parametrize("label, prediction, expected", SUBSTRING_MATCH)
def test_substring_match(label, prediction, expected):
    """Token contiguous subsequence matching (normalize off) → 0.0|1.0."""
    assert (
        TextEvaluator.compute_substring_match(
            label=label, prediction=prediction, normalize=None
        )
        == expected
    )


@pytest.mark.xfail(
    reason="source bug: compute_substring_match's default normalize='squad' calls "
    "non-existent cls.normalize_squad, raising AttributeError (method name is _normalize_squad)",
    raises=AttributeError,
    strict=True,
)
def test_substring_match_squad_default_is_broken():
    """Pins that the default normalize='squad' path is broken — strict xfail will notify when source is fixed."""
    TextEvaluator.compute_substring_match(label="b c", prediction="a b c d")


# ── compute_string_match ─────────────────────────────────────────────────────
@pytest.mark.parametrize("label, prediction, expected", STRING_MATCH)
def test_string_match(label, prediction, expected):
    """Substring containment after strip + newline normalization → 0.0|1.0."""
    assert TextEvaluator.compute_string_match(label=label, prediction=prediction) == expected


# ── compute_ned ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("label, prediction, expected", NED)
def test_ned(label, prediction, expected):
    """Normalized edit distance = Levenshtein / max(len)."""
    assert TextEvaluator.compute_ned(label=label, prediction=prediction) == pytest.approx(expected)


# ── compute_levenshtein_distance ─────────────────────────────────────────────
@pytest.mark.parametrize("label, prediction, expected", LEVENSHTEIN)
def test_levenshtein_distance(label, prediction, expected):
    """DP edit distance / max(len(label), len(prediction))."""
    assert TextEvaluator.compute_levenshtein_distance(
        label=label, prediction=prediction
    ) == pytest.approx(expected)


# ── compute_jaccard_distance ─────────────────────────────────────────────────
@pytest.mark.parametrize("labels, prediction, expected", JACCARD)
def test_jaccard_distance(labels, prediction, expected):
    """Token-set 1 - |∩|/|∪|, minimum distance over labels."""
    assert TextEvaluator.compute_jaccard_distance(
        labels=labels, prediction=prediction
    ) == pytest.approx(expected)


# ── compute_anls ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("labels, prediction, expected", ANLS)
def test_anls(labels, prediction, expected):
    """min ned over labels; if threshold < 0.5 then 1-nl, else 0."""
    assert TextEvaluator.compute_anls(
        labels=labels, prediction=prediction
    ) == pytest.approx(expected)


# ── compute_f1 ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("labels, prediction, expected", F1)
def test_f1(labels, prediction, expected):
    """squad-normalized token P/R/F1 dict; multiple labels aggregated by per-item max."""
    assert TextEvaluator.compute_f1(labels=labels, prediction=prediction) == pytest.approx(expected)


# ── compute_binary_f1 ────────────────────────────────────────────────────────
@pytest.mark.parametrize("labels, predictions, expected", BINARY_F1)
def test_binary_f1(labels, predictions, expected):
    """yes/no classification P/R/F1/accuracy dict (no/not tokens → neg)."""
    assert TextEvaluator.compute_binary_f1(
        labels=labels, predictions=predictions
    ) == pytest.approx(expected)


# ── compute_vqaeval ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("labels, prediction, expected", VQAEVAL)
def test_vqaeval(labels, prediction, expected):
    """VQA accuracy = min(matches/3, 1); gt equivalence count after normalization."""
    assert TextEvaluator.compute_vqaeval(
        labels=labels, prediction=prediction
    ) == pytest.approx(expected)


# ── compute_mmau_string_match ────────────────────────────────────────────────
@pytest.mark.parametrize("label, prediction, option_contents, expected", MMAU)
def test_mmau_string_match(label, prediction, option_contents, expected):
    """answer token contained AND wrong-answer tokens not contained → 1.0."""
    assert TextEvaluator.compute_mmau_string_match(
        label=label, prediction=prediction, option_contents=option_contents
    ) == pytest.approx(expected)


def test_mmau_string_match_empty_prediction():
    """Empty prediction has no tokens and immediately returns False (early return)."""
    assert (
        TextEvaluator.compute_mmau_string_match(
            label="cat", prediction="", option_contents=["cat", "dog"]
        )
        is False
    )


# ── normalize (squad) ────────────────────────────────────────────────────────
@pytest.mark.parametrize("text, expected", NORMALIZE_SQUAD)
def test_normalize_squad(text, expected):
    """squad: lowercase + punctuation removal + article removal + whitespace collapse."""
    assert TextEvaluator.normalize(text=text, method="squad") == expected


# ── pure predicate helpers ───────────────────────────────────────────────────
@pytest.mark.parametrize("text, expected", IS_NUMERIC_STRING)
def test_is_numeric_string(text, expected):
    """Whether float() parsing is possible."""
    assert TextEvaluator._is_numeric_string(text) is expected


@pytest.mark.parametrize("text, expected", IS_CIRCLED_NUMBER)
def test_is_circled_number(text, expected):
    """Whether the value exists in the vqa circled-number mapping."""
    assert TextEvaluator._is_circled_number(text) is expected


@pytest.mark.parametrize("text, expected", IS_DATE_STRING)
def test_is_date_string(text, expected):
    """True if dateparser can parse the value as a date."""
    assert TextEvaluator._is_date_string(text) is expected


# ── compute_tree_edit ────────────────────────────────────────────────────────
@pytest.mark.parametrize("labels, predictions, groups, expected_sample", TREE_EDIT)
def test_tree_edit(labels, predictions, groups, expected_sample):
    """Direct mode (no api_name/source_format) — multi-label max, empty-pred → None."""
    _, sample = TextEvaluator.compute_tree_edit(
        labels=labels, predictions=predictions, groups=groups,
    )
    assert sample["tree_edit_score"] == expected_sample


# ── compute_temporal_iou ─────────────────────────────────────────────────────
@pytest.mark.parametrize("label, prediction, expected", TEMPORAL_IOU)
def test_temporal_iou(label, prediction, expected):
    """Interval IoU + threshold-recall flags; unparseable → all zero."""
    assert TextEvaluator.compute_temporal_iou(label=label, prediction=prediction) == expected
