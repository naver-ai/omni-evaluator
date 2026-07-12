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

"""Unit tests for omni_evaluator/postprocess/code."""

import pytest

from omni_evaluator.postprocess.code import CodeProcessor

from ._cases import CODE_EMPTY_OR_WHITESPACE, CODE_EXPLICIT_BLOCK


# `tests/postprocess/CLAUDE.md §8` — regex catastrophic-backtracking guard.
pytestmark = pytest.mark.timeout(1)


# ============================================================================
# CASE STUDY — branch tests parametrized with case data from `_cases.py`.
# ============================================================================


@pytest.mark.parametrize("prediction,language,expected", CODE_EXPLICIT_BLOCK)
def test_extract_with_explicit_code_block(prediction, language, expected):
    """All lang alternatives of `PATTERN__CODE_BLOCK` (python/java/javascript/sh/cpp/json) + no-tag."""
    result = CodeProcessor.extract(prediction=prediction, language=language)
    assert result == expected


@pytest.mark.parametrize("prediction", CODE_EMPTY_OR_WHITESPACE)
def test_extract_empty_or_whitespace_returns_none(prediction):
    """Empty or whitespace-only extraction result → None."""
    result = CodeProcessor.extract(prediction=prediction, language="python")
    assert result is None


# ============================================================================
# Standalone demos — single inline input / raises / multiple asserts.
# ============================================================================


def test_extract_code_without_codeblock():
    """prediction without fence → wrapper extracts the entire text as code."""
    result = CodeProcessor.extract(
        prediction="def foo(): return 42",
        language="python",
    )
    assert result == "def foo(): return 42"


def test_returns_none_when_query_is_none():
    """Even with extract_continuation=True, continuation logic is skipped when query is None."""
    result = CodeProcessor.extract(
        prediction="def foo(): return 42",
        language="python",
        extract_continuation=True,
        query=None,
    )
    assert result == "def foo(): return 42"


def test_extract_continuation_false_ignores_query():
    """With extract_continuation=False, code is returned as-is without entering continuation logic even if query is a str."""
    result = CodeProcessor.extract(
        prediction="def foo(): return 42",
        language="python",
        query="def foo():",          # query is present but ignored because flag is False.
        extract_continuation=False,
    )
    assert result == "def foo(): return 42"


def test_extract_continuation_strips_overlap():
    """When there is meaningful overlap with query, only the tail continuation is retained."""
    query = "def fibonacci(n):\n    if n < 2:\n        return n"
    prediction = (
        "def fibonacci(n):\n    if n < 2:\n        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)"
    )
    result = CodeProcessor.extract(
        prediction=prediction,
        language="python",
        query=query,
        extract_continuation=True,
    )
    assert result is not None
    assert "fibonacci(n - 1)" in result
    assert not result.lstrip().startswith("def fibonacci(n):")


def test_extract_continuation_unsupported_language():
    """extract_continuation=True only allows languages registered in PATTERN__CODE_START (currently python)."""
    with pytest.raises(ValueError, match="Code_start_pattern"):
        CodeProcessor.extract(
            prediction="public class Foo {}",
            language="java",
            query="public class Foo",
            extract_continuation=True,
        )


def test_has_code_block():
    """`_has_code_block` only checks for the presence of a fence in raw text."""
    assert CodeProcessor._has_code_block("```python\nprint('hi')\n```") is True
    assert CodeProcessor._has_code_block("just plain text") is False
