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

"""Unit tests for omni_evaluator/postprocess/multichoice."""

import pytest

from omni_evaluator.postprocess.multichoice import MultichoiceProcessor

from ._cases import (
    MC_CIRCLED_COMPLEX,
    MC_CIRCLED_PREDICTIONS,
    MC_LETTER_COMPLEX,
    MC_LETTER_PREDICTIONS,
    MC_NUMBER_COMPLEX,
    MC_NUMBER_PREDICTIONS,
    MC_PASSTHROUGH_SINGLE,
)


# `tests/postprocess/CLAUDE.md §8` — regex catastrophic-backtracking guard.
pytestmark = pytest.mark.timeout(1)
_API_TARGET = "omni_evaluator.postprocess.multichoice.chat_completion_sync"


# ============================================================================
# CASE STUDY — branch tests parametrized with case data from `_cases.py`.
# ============================================================================


@pytest.mark.parametrize("prediction,expected", MC_LETTER_PREDICTIONS)
def test_extract_letter_scheme(prediction, expected):
    """letter scheme: extracts A-J from various cue/wrapper styles."""
    assert MultichoiceProcessor.extract(
        prediction=prediction,
        query="",
        options=["A", "B", "C", "D"],
    ) == expected


@pytest.mark.parametrize("prediction,expected", MC_NUMBER_PREDICTIONS)
def test_extract_number_scheme(prediction, expected):
    """number scheme: int options → extracts 1-9."""
    assert MultichoiceProcessor.extract(
        prediction=prediction,
        query="",
        options=[1, 2, 3, 4],
    ) == expected


@pytest.mark.parametrize("prediction,expected", MC_CIRCLED_PREDICTIONS)
def test_extract_circled_scheme(prediction, expected):
    """circled_number scheme: circled number options → extracts ①-⑨."""
    assert MultichoiceProcessor.extract(
        prediction=prediction,
        query="",
        options=["①", "②", "③", "④"],
    ) == expected


@pytest.mark.parametrize("prediction,expected", MC_LETTER_COMPLEX)
def test_extract_letter_complex(prediction, expected):
    """Long CoT / mid-answer errors / multiple boxed / tail-line / markdown / Korean reasoning and other production patterns."""
    assert MultichoiceProcessor.extract(
        prediction=prediction,
        query="",
        options=["A", "B", "C", "D"],
    ) == expected


@pytest.mark.parametrize("prediction,expected", MC_NUMBER_COMPLEX)
def test_extract_number_complex(prediction, expected):
    """number scheme — extracts the answer after long reasoning."""
    assert MultichoiceProcessor.extract(
        prediction=prediction,
        query="",
        options=[1, 2, 3, 4],
    ) == expected


@pytest.mark.parametrize("prediction,expected", MC_CIRCLED_COMPLEX)
def test_extract_circled_complex(prediction, expected):
    """circled scheme — long Korean reasoning."""
    assert MultichoiceProcessor.extract(
        prediction=prediction,
        query="",
        options=["①", "②", "③", "④"],
    ) == expected


@pytest.mark.parametrize("prediction", MC_PASSTHROUGH_SINGLE)
def test_extract_single_char_passthrough(prediction):
    """length-1 prediction → returned as-is without scheme detection/extraction."""
    assert MultichoiceProcessor.extract(prediction=prediction, query="") == prediction


# ============================================================================
# Standalone demos — single inline input / mock-based verification.
# ============================================================================


def test_extract_non_string_passthrough():
    """non-str prediction → passes through as-is (None stays None)."""
    assert MultichoiceProcessor.extract(prediction=None, query="") is None


def test_extract_strips_think_block():
    """Text inside `<think>...</think>` is ignored during candidate matching."""
    prediction = "<think>I think A is the answer</think> The answer is B."
    assert MultichoiceProcessor.extract(
        prediction=prediction,
        query="",
        options=["A", "B", "C", "D"],
    ) == "B"


def test_extract_no_candidate_no_tail_returns_none():
    """Candidate extraction failure + tail-line fallback also fails → None."""
    assert MultichoiceProcessor.extract(
        prediction="This is a paragraph without any letter cue or numbered choice.",
        query="",
        options=["A", "B", "C", "D"],
    ) is None


def test_extract_api_path_returns_chat_completion_output(patch_api_chat_completion):
    """`api_name` specified → `chat_completion_sync` is called and its return value is passed through."""
    calls = patch_api_chat_completion(target=_API_TARGET, return_value="B")
    result = MultichoiceProcessor.extract(
        prediction="reasoning text where the answer is not obvious",
        query="Q?",
        options=["A", "B", "C", "D"],
        api_name="gpt-4o-mini",
    )
    assert result == "B"
    assert len(calls) == 1
    assert calls[0]["api_name"] == "gpt-4o-mini"


def test_extract_without_api_skips_chat_completion(patch_api_chat_completion):
    """`api_name` not specified → only the regex path is used; the fake `chat_completion_sync` is not called."""
    calls = patch_api_chat_completion(target=_API_TARGET, return_value="should_not_appear")
    MultichoiceProcessor.extract(
        prediction="The answer is A.",
        query="",
        options=["A", "B", "C", "D"],
    )
    assert calls == []


def test_is_multichoice_options_letters():
    """letter options → 'letter' scheme."""
    scheme, choices = MultichoiceProcessor.is_multichoice(
        query="", options=["A", "B", "C", "D"], return_choices=True,
    )
    assert scheme == "letter"
    assert list(choices) == ["A", "B", "C", "D"]


def test_is_multichoice_options_int_list():
    """int options → 'number' scheme, choices converted to str."""
    scheme, choices = MultichoiceProcessor.is_multichoice(
        query="", options=[1, 2, 3, 4], return_choices=True,
    )
    assert scheme == "number"
    assert list(choices) == ["1", "2", "3", "4"]


def test_is_multichoice_options_circled():
    """circled number options → 'circled_number' scheme."""
    scheme, choices = MultichoiceProcessor.is_multichoice(
        query="", options=["①", "②", "③"], return_choices=True,
    )
    assert scheme == "circled_number"
    assert list(choices) == ["①", "②", "③"]


def test_is_multichoice_option_contents_letters():
    """option_contents (alternative input for options) is handled identically."""
    scheme, choices = MultichoiceProcessor.is_multichoice(
        query="", option_contents=["A", "B"], return_choices=True,
    )
    assert scheme == "letter"
    assert list(choices) == ["A", "B"]


def test_is_multichoice_bool_return():
    """`return_choices=False` → returns a single bool."""
    assert MultichoiceProcessor.is_multichoice(
        query="", options=["A", "B"], return_choices=False,
    ) is True
