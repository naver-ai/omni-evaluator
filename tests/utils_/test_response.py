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

"""Unit tests for utils/response.py — provider-agnostic usable-response detection.

``is_valid_response`` is the single fail-fast rule shared by every API provider
engine (openai chat/responses, anthropic, google). It runs on the normalized
``parse_response`` output, so these tests exercise that contract (prediction /
generated_text / tool_calls / function_call / error_message) rather than any
provider's raw response shape.
"""
from omni_evaluator.utils.response import is_valid_response


def test_valid_with_text_content():
    """Usable when any text field carries content — string, candidate list, or structured dict."""
    assert is_valid_response({"prediction": "hello"}) is True
    assert is_valid_response({"prediction": None, "generated_text": ["hi"]}) is True
    assert is_valid_response({"prediction": {"answer": 1}}) is True  # non-empty structured answer


def test_valid_with_tool_or_function_call():
    """A tool_call or legacy function_call with no text is usable (agentic turn)."""
    assert is_valid_response({"prediction": None, "tool_calls": [{"id": "c1"}]}) is True
    assert is_valid_response({"prediction": "", "function_call": {"name": "f"}}) is True


def test_invalid_when_no_content():
    """No text and no tool/function call → unusable (fail-fast). Whitespace-only and a
    candidate list of empty strings both count as no content."""
    assert is_valid_response({
        "prediction": "", "generated_text": [""], "tool_calls": None,
        "function_call": None, "finish_reason": "length",
    }) is False
    assert is_valid_response({"prediction": "   \n"}) is False


def test_invalid_on_error_message():
    """A parse-level error_message makes the response unusable regardless of other fields."""
    assert is_valid_response({"error_message": "unexpected response format"}) is False


def test_invalid_when_non_mapping():
    """A non-mapping output (e.g. None from a failed request) is unusable."""
    assert is_valid_response(None) is False
