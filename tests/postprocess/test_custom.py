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

"""Unit tests for omni_evaluator/postprocess/custom.py.

Verifies only the key branches of the four functions (parse_boxed_format /
parse_last_pattern / parse_circled_answer / parse_think). Variants that fall
into the same branch are grouped within one test using multiple asserts
(`tests/CLAUDE.md §8`).
"""

import pytest

from omni_evaluator.postprocess.custom import (
    parse_boxed_format,
    parse_circled_answer,
    parse_last_pattern,
    parse_think,
)


# `tests/postprocess/CLAUDE.md §8` — regex catastrophic-backtracking guard.
pytestmark = pytest.mark.timeout(1)


# ---------------------------------------------------------------------------
# parse_boxed_format
# ---------------------------------------------------------------------------


def test_parse_boxed_format_extracts_last_match():
    r"""`\boxed{X}` → X. When multiple matches exist, the last one (= final answer) captured by rfind wins.

    Reasoning models may emit `\boxed{}` during intermediate steps and re-emit the
    final answer at the end, so the trailing box is treated as canonical. Nested
    braces are handled by a depth-aware scan that balances closing braces so
    `\frac{1}{2}` is not truncated.
    """
    assert parse_boxed_format(r"text \boxed{42} more") == "42"
    assert parse_boxed_format(r"\boxed{first} mid \boxed{second}") == "second"
    assert parse_boxed_format(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_parse_boxed_format_no_match_returns_none():
    r"""Returns None when the `\boxed{...}` pattern is absent."""
    assert parse_boxed_format("no boxed here") is None
    assert parse_boxed_format(r"\boxed missing braces") is None


# ---------------------------------------------------------------------------
# parse_last_pattern  (prefix="<a>", suffix="</a>" fixed)
# ---------------------------------------------------------------------------


def test_parse_last_pattern_flags():
    """Three branches: return_last / return_first / return_all."""
    text = "<a>1</a> <a>2</a>"
    assert parse_last_pattern(text, "<a>", "</a>", return_last=True, return_all=False) == "2"
    assert parse_last_pattern(text, "<a>", "</a>", return_last=False, return_all=False) == "1"
    assert parse_last_pattern(text, "<a>", "</a>", return_last=True, return_all=True) == ["1", "2"]


def test_parse_last_pattern_empty_delimiter_raises():
    """Raises ValueError at the entry point when prefix or suffix is empty."""
    with pytest.raises(ValueError, match="non-empty"):
        parse_last_pattern("x", prefix="", suffix="</a>")


def test_parse_last_pattern_no_match_returns_none():
    """Returns None when the prefix is not found anywhere in the text (auto-appended suffix does not create a match)."""
    assert parse_last_pattern("plain text", prefix="<a>", suffix="</a>") is None


# ---------------------------------------------------------------------------
# parse_circled_answer
# ---------------------------------------------------------------------------


def test_parse_circled_answer_maps_to_decimal():
    """U+2460..U+2473 (①..⑳) → decimal string."""
    assert parse_circled_answer("①") == "1"
    assert parse_circled_answer("⑳") == "20"


def test_parse_circled_answer_invalid_raises():
    """Raises KeyError because a non-circled-number character is not in the map."""
    with pytest.raises(KeyError):
        parse_circled_answer("X")


# ---------------------------------------------------------------------------
# parse_think
# ---------------------------------------------------------------------------


def test_parse_think_splits_reasoning_and_prediction():
    """think block + EOT → reasoning and final prediction are separated."""
    result = parse_think(
        prediction="<think>let me think</think>final answer<|im_end|>",
        think_start_pattern="<think>",
        think_end_pattern="</think>",
        eot_token="<|im_end|>",
    )
    assert result["reasoning_content"] == "let me think"
    assert result["prediction"] == "final answer"


def test_parse_think_no_end_pattern_returns_none_reasoning():
    """When `</think>` is absent, reasoning is None and prediction passes through unchanged."""
    result = parse_think(prediction="just plain text without think block")
    assert result["reasoning_content"] is None
    assert result["prediction"] == "just plain text without think block"
