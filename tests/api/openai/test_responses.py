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

"""Unit tests for api/openai/responses.py — Responses API (structured output) path."""
from __future__ import annotations

import json

import pydantic
import pytest

pytest.importorskip("openai")

from omni_evaluator.api.openai.responses import response_create_sync


class _Answer(pydantic.BaseModel):
    answer: str


class _RawResponse:
    """Minimal SDK-response stand-in exposing only `.json()`, for crafting content-contract
    edge cases the fixture factory doesn't build (empty output, reasoning-only incomplete)."""

    def __init__(self, data: dict):
        self._data = data

    def json(self) -> str:
        return json.dumps(self._data)


# ── response parsing ─────────────────────────────────────────────────────────


def test_success_returns_parsed_text(fake_openai_client_factory):
    """output[type=message].content[type=output_text].text → prediction."""
    client, calls = fake_openai_client_factory(response_kind="responses")
    out = response_create_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=1, wait_between_retry=0,
    )
    assert isinstance(out, dict)
    assert out["prediction"] == "hello"
    assert out["finish_reason"] == "completed"  # status is mapped to finish_reason


def test_reasoning_summary_extracted(fake_openai_client_factory, fake_responses_response_factory):
    """type=reasoning summary in output is concatenated and stored as reasoning_content."""
    response = fake_responses_response_factory(text="ok", reasoning="step1")
    client, _ = fake_openai_client_factory(response=response, response_kind="responses")
    out = response_create_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=1, wait_between_retry=0,
    )
    assert out["reasoning_content"] == "step1"


# ── Responses API-specific kwargs ────────────────────────────────────────────


def test_response_format_serialized_as_json_schema(fake_openai_client_factory):
    """response_format pydantic is serialized as `text.format` json_schema for the Responses API."""
    client, calls = fake_openai_client_factory(response_kind="responses")
    response_create_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        response_format=_Answer,
        max_retry=1, wait_between_retry=0,
    )
    fmt = (calls[0].get("text") or {}).get("format")
    assert isinstance(fmt, dict)
    assert fmt["type"] == "json_schema"
    assert "answer" in fmt["schema"]["properties"]   # verify _Answer fields are present


def test_instructions_kwarg_passthrough(fake_openai_client_factory):
    """`instructions` is passed through to create() as-is."""
    client, calls = fake_openai_client_factory(response_kind="responses")
    response_create_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        instructions="be concise",
        max_retry=1, wait_between_retry=0,
    )
    assert calls[0].get("instructions") == "be concise"


def test_input_serialized_for_responses_template(fake_openai_client_factory):
    """messages → passed as create(input=...) (different key from chat's `messages=`)."""
    client, calls = fake_openai_client_factory(response_kind="responses")
    response_create_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=1, wait_between_retry=0,
    )
    assert "input" in calls[0]
    assert "messages" not in calls[0]


# ── content-contract fail-fast guard (is_valid_response) ──────────────


def test_empty_output_returns_none_not_retried(fake_openai_client_factory):
    """200-OK with empty `output` (safety block / refusal) → None, and not retried (1 call)."""
    client, calls = fake_openai_client_factory(
        response=_RawResponse({"output": [], "status": "completed"}),
        response_kind="responses",
    )
    out = response_create_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=5, wait_between_retry=0,
    )
    assert out is None
    assert len(calls) == 1  # not retried despite max_retry=5


def test_reasoning_only_incomplete_returns_none_not_retried(fake_openai_client_factory):
    """Reasoning model that exhausted the budget (only a reasoning item, status=incomplete,
    incomplete_details.reason=max_output_tokens) → None, and not retried."""
    client, calls = fake_openai_client_factory(
        response=_RawResponse({
            "output": [{"type": "reasoning", "summary": []}],
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        }),
        response_kind="responses",
    )
    out = response_create_sync(
        client=client, api_name="gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=5, wait_between_retry=0,
    )
    assert out is None
    assert len(calls) == 1


def test_function_call_only_is_usable(fake_openai_client_factory):
    """A response carrying only a function_call item (no output_text) is NOT flagged unusable —
    tool-use turns must survive. Returns a dict, single call."""
    client, calls = fake_openai_client_factory(
        response=_RawResponse({
            "output": [{"type": "function_call", "name": "f", "arguments": "{}", "call_id": "c1"}],
            "status": "completed",
        }),
        response_kind="responses",
    )
    out = response_create_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=1, wait_between_retry=0,
    )
    assert isinstance(out, dict)
    assert len(calls) == 1
