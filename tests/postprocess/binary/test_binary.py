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

"""Unit tests for omni_evaluator/postprocess/binary/__init__.py."""

import pytest

from omni_evaluator.postprocess.binary import BinaryProcessor

from ._cases import (
    BINARY_LONG_NEGATIVE,
    BINARY_LONG_NO_CUE,
    BINARY_LONG_POSITIVE,
    BINARY_MIXED_COMPLEX,
    BINARY_NEGATIVE,
    BINARY_NO_CUE,
    BINARY_NON_STRING,
    BINARY_POSITIVE,
)


# `tests/postprocess/CLAUDE.md §8` — regex catastrophic-backtracking guard.
pytestmark = pytest.mark.timeout(1)


# ============================================================================
# CASE STUDY — branch tests parametrized with case data from `_cases.py`.
# ============================================================================


@pytest.mark.parametrize("text", BINARY_POSITIVE)
def test_positive_cue_returns_true(text):
    """positive cue (KO·EN) → True."""
    assert BinaryProcessor.extract(prediction=text) is True


@pytest.mark.parametrize("text", BINARY_NEGATIVE)
def test_negative_cue_returns_false(text):
    """negative cue (KO·EN) → False."""
    assert BinaryProcessor.extract(prediction=text) is False


@pytest.mark.parametrize("text", BINARY_NO_CUE)
def test_no_cue_returns_none(text):
    """Returns None when no cue is present."""
    assert BinaryProcessor.extract(prediction=text) is None


@pytest.mark.parametrize("text", BINARY_LONG_POSITIVE)
def test_long_reasoning_positive(text):
    """Returns True when a positive cue follows hundreds of characters of CoT — regex completes within 1 second."""
    assert BinaryProcessor.extract(prediction=text) is True


@pytest.mark.parametrize("text", BINARY_LONG_NEGATIVE)
def test_long_reasoning_negative(text):
    """Returns False when a negative cue follows hundreds of characters of CoT."""
    assert BinaryProcessor.extract(prediction=text) is False


@pytest.mark.parametrize("text", BINARY_LONG_NO_CUE)
def test_long_no_cue_returns_none(text):
    """Returns None when hundreds of characters contain no cue — backtracking regression detection."""
    assert BinaryProcessor.extract(prediction=text) is None


@pytest.mark.parametrize("text,expected", BINARY_MIXED_COMPLEX)
def test_mixed_complex(text, expected):
    """NEGATIVE-first rule is maintained for complex cases where an opposing cue is embedded in the middle or EN+KO is mixed."""
    assert BinaryProcessor.extract(prediction=text) is expected


@pytest.mark.parametrize("value", BINARY_NON_STRING)
def test_is_binary_rejects_non_string(value):
    """non-str input is rejected before pattern search."""
    assert BinaryProcessor.is_binary(query=value) is False


# ============================================================================
# Standalone demonstrations — single inline input / multiple asserts in one function.
# ============================================================================


def test_is_binary_rejects_empty_string():
    """A string of length 0 is not recognized as a binary query."""
    assert BinaryProcessor.is_binary(query="") is False


def test_is_binary_detects_either_polarity():
    """Classified as binary even if only one side — positive or negative — matches a pattern."""
    assert BinaryProcessor.is_binary(query="The answer is yes.") is True
    assert BinaryProcessor.is_binary(query="The answer is no.") is True
    assert BinaryProcessor.is_binary(query="hello world") is False


def test_is_binary_nfkc_normalizes_fullwidth():
    """Fullwidth characters are recognized as cues after NFKC normalization ('ｙｅｓ' → 'yes')."""
    assert BinaryProcessor.is_binary(query="ｙｅｓ") is True
