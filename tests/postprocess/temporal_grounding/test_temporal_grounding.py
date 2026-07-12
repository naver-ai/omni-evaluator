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

"""Unit tests for omni_evaluator/postprocess/temporal_grounding/__init__.py."""

import pytest

from omni_evaluator.postprocess.temporal_grounding import TemporalGroundingProcessor

from ._cases import (
    TG_BARE_FALLBACK,
    TG_EMPTY,
    TG_INVARIANT_LATER_VALID,
    TG_INVARIANT_REJECTED,
    TG_LAST_WINS,
    TG_NFKC_FULLWIDTH,
    TG_NON_STRING,
    TG_NO_ANSWER,
    TG_PAIR_PROSE,
    TG_THINK_STRIPPED,
    TG_TIMESTAMP_HH_MM_SS,
    TG_TIMESTAMP_MM_SS,
)


# `tests/postprocess/CLAUDE.md §8` — regex catastrophic-backtracking guard.
pytestmark = pytest.mark.timeout(1)


# ============================================================================
# Separator-pair (primary) — last-wins regex over prose / EN-KO / variants.
# ============================================================================


@pytest.mark.parametrize("text", TG_PAIR_PROSE)
def test_pair_in_prose(text):
    """`{num}{sep}{num}` pair (any of -, –, —, to, ~, until, ',') inside prose → canonical `[s, e]`."""
    assert TemporalGroundingProcessor.extract(prediction=text) == "[24.3, 30.4]"


@pytest.mark.parametrize("text,expected", TG_LAST_WINS)
def test_last_pair_wins(text, expected):
    """When multiple valid pairs exist, the LAST is the model's final answer."""
    assert TemporalGroundingProcessor.extract(prediction=text) == expected


# ============================================================================
# Token form — `_to_seconds` parses numeric / MM:SS / HH:MM:SS.
# ============================================================================


@pytest.mark.parametrize("text,expected", TG_TIMESTAMP_MM_SS)
def test_mm_ss_token(text, expected):
    """`MM:SS(.ms)` tokens normalize to seconds via `_to_seconds`."""
    assert TemporalGroundingProcessor.extract(prediction=text) == expected


@pytest.mark.parametrize("text,expected", TG_TIMESTAMP_HH_MM_SS)
def test_hh_mm_ss_token(text, expected):
    """`HH:MM:SS(.ms)` tokens normalize to seconds via `_to_seconds`."""
    assert TemporalGroundingProcessor.extract(prediction=text) == expected


# ============================================================================
# Bare-token fallback — used only when no separator pair regex matches.
# ============================================================================


@pytest.mark.parametrize("text,expected", TG_BARE_FALLBACK)
def test_bare_fallback(text, expected):
    """No separator pair → first two timestamp/number tokens are taken as `(s, e)`."""
    assert TemporalGroundingProcessor.extract(prediction=text) == expected


# ============================================================================
# No answer / non-string / empty — early returns.
# ============================================================================


@pytest.mark.parametrize("text", TG_NO_ANSWER)
def test_no_answer_returns_none(text):
    """Refusal text or single token (< 2 numbers) → None."""
    assert TemporalGroundingProcessor.extract(prediction=text) is None


@pytest.mark.parametrize("value", TG_NON_STRING)
def test_non_string_passthrough(value):
    """Non-string input returned unchanged (no extraction)."""
    assert TemporalGroundingProcessor.extract(prediction=value) == value


@pytest.mark.parametrize("text", TG_EMPTY)
def test_empty_returns_none(text):
    """Empty / whitespace-only string → None."""
    assert TemporalGroundingProcessor.extract(prediction=text) is None


# ============================================================================
# Invariant `0 ≤ s < e ≤ 1e6` — invalid pairs are silently dropped.
# ============================================================================


@pytest.mark.parametrize("text", TG_INVARIANT_REJECTED)
def test_invariant_rejected(text):
    """reversed / negative / zero-duration pairs are dropped (no later valid pair → None)."""
    assert TemporalGroundingProcessor.extract(prediction=text) is None


@pytest.mark.parametrize("text,expected", TG_INVARIANT_LATER_VALID)
def test_invariant_skips_to_next_valid(text, expected):
    """When an invariant-violating pair precedes a valid one, last-wins picks the valid one."""
    assert TemporalGroundingProcessor.extract(prediction=text) == expected


# ============================================================================
# NFKC normalize + `<think>` strip — text preprocessing.
# ============================================================================


@pytest.mark.parametrize("text", TG_NFKC_FULLWIDTH)
def test_nfkc_normalizes_fullwidth(text):
    """Fullwidth digits / dash are halfwidth-normalized before regex."""
    assert TemporalGroundingProcessor.extract(prediction=text) == "[24.3, 30.4]"


@pytest.mark.parametrize("text", TG_THINK_STRIPPED)
def test_think_trace_stripped(text):
    """Pair inside `<think>...</think>` is ignored; post-trace answer wins."""
    assert TemporalGroundingProcessor.extract(prediction=text) == "[24.3, 30.4]"


# ============================================================================
# api_name — LLM fallback is intentionally unimplemented.
# ============================================================================


def test_api_name_raises_not_implemented():
    """Non-empty api_name raises NotImplementedError to prevent silent fallthrough."""
    with pytest.raises(NotImplementedError):
        TemporalGroundingProcessor.extract(prediction="24.3 - 30.4", api_name="gpt-4o-mini")
