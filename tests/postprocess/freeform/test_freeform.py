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

"""Unit tests for omni_evaluator/postprocess/freeform."""

import pytest

from omni_evaluator.postprocess.freeform import FreeformProcessor

from ._cases import (
    FREEFORM_API_NULLISH_RETURNS,
    FREEFORM_CLEAN_SPAN,
    FREEFORM_EN_COMPLEX,
    FREEFORM_EN_CUE,
    FREEFORM_EN_KO_MIXED_KO_WINS,
    FREEFORM_KO_COMPLEX,
    FREEFORM_KO_CUE,
    FREEFORM_NO_CUE,
)


# `tests/postprocess/CLAUDE.md §8` — regex catastrophic-backtracking guard.
pytestmark = pytest.mark.timeout(1)
_API_TARGET = "omni_evaluator.postprocess.freeform.chat_completion_sync"


# ============================================================================
# CASE STUDY — branch tests parametrized with case data from `_cases.py`.
# ============================================================================


@pytest.mark.parametrize("prediction,expected", FREEFORM_EN_CUE)
def test_extract_en_cue(prediction, expected):
    """English cue ('answer is X', 'final answer: X', quoted answer) → token extraction."""
    assert FreeformProcessor.extract(prediction=prediction, query="") == expected


@pytest.mark.parametrize("prediction,expected", FREEFORM_KO_CUE)
def test_extract_ko_cue(prediction, expected):
    """Korean cue ('정답은 X 입니다', '답은 X 입니다') → token extraction."""
    assert FreeformProcessor.extract(prediction=prediction, query="") == expected


@pytest.mark.parametrize("prediction", FREEFORM_NO_CUE)
def test_extract_no_cue_returns_none(prediction):
    """Returns None when neither EN nor KO cue is present."""
    assert FreeformProcessor.extract(prediction=prediction, query="") is None


@pytest.mark.parametrize("prediction,expected", FREEFORM_EN_COMPLEX)
def test_extract_en_complex(prediction, expected):
    """EN complex patterns: long CoT / bracket list / decimal / multiline / quoted Korean token."""
    assert FreeformProcessor.extract(prediction=prediction, query="") == expected


@pytest.mark.parametrize("prediction,expected", FREEFORM_KO_COMPLEX)
def test_extract_ko_complex(prediction, expected):
    """KO complex patterns: long Korean CoT / bracket list / decimal."""
    assert FreeformProcessor.extract(prediction=prediction, query="") == expected


@pytest.mark.parametrize("prediction,expected", FREEFORM_EN_KO_MIXED_KO_WINS)
def test_extract_en_ko_mixed_ko_wins(prediction, expected):
    """When EN cue and KO cue appear in the same text, KO overwrites EN (`_extract_freeform` loop behavior)."""
    assert FreeformProcessor.extract(prediction=prediction, query="") == expected


@pytest.mark.parametrize("raw,expected", FREEFORM_CLEAN_SPAN)
def test_clean_span(raw, expected):
    """`_clean_span` — strips outer whitespace / paired quotes / trailing punctuation."""
    assert FreeformProcessor._clean_span(raw) == expected


@pytest.mark.parametrize("api_return", FREEFORM_API_NULLISH_RETURNS)
def test_extract_api_path_empty_or_na_returns_none(patch_api_chat_completion, api_return):
    """Normalizes to None when API returns empty string / whitespace / 'N/A' (case-insensitive)."""
    patch_api_chat_completion(target=_API_TARGET, return_value=api_return)
    assert FreeformProcessor.extract(
        prediction="reasoning...",
        query="Q?",
        api_name="gpt-4o-mini",
    ) is None


# ============================================================================
# Standalone demos — single inline input / mock-based verification.
# ============================================================================


def test_extract_single_char_passthrough():
    """Length-1 prediction → returned as-is without cue search."""
    assert FreeformProcessor.extract(prediction="X", query="") == "X"


def test_extract_non_string_passthrough():
    """non-str prediction → passes through unchanged (None stays None)."""
    assert FreeformProcessor.extract(prediction=None, query="") is None


def test_extract_api_path_returns_chat_completion_output(patch_api_chat_completion):
    """When `api_name` is specified → `chat_completion_sync` is called and its return value is passed through."""
    calls = patch_api_chat_completion(target=_API_TARGET, return_value="Paris")
    result = FreeformProcessor.extract(
        prediction="some reasoning text...",
        query="What is the capital of France?",
        api_name="gpt-4o-mini",
    )
    assert result == "Paris"
    assert len(calls) == 1
    assert calls[0]["api_name"] == "gpt-4o-mini"


def test_extract_without_api_skips_chat_completion(patch_api_chat_completion):
    """Without `api_name` → only regex path is used; fake `chat_completion_sync` is never called."""
    calls = patch_api_chat_completion(target=_API_TARGET, return_value="should_not_appear")
    FreeformProcessor.extract(
        prediction="The answer is 14.",
        query="What is the missing number?",
    )
    assert calls == []
