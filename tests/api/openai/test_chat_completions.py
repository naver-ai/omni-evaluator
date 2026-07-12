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

"""Unit tests for api/openai/chat_completions.py — response parsing, retries, message serialization, generation_options normalization."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("openai")

from omni_evaluator.api.openai.chat_completions import chat_completion_sync


class _RawResponse:
    """Minimal SDK-response stand-in exposing only `.json()`, for crafting content-contract
    edge cases the fixture factories don't build (empty choices, refusal, contentless message)."""

    def __init__(self, data: dict):
        self._data = data

    def json(self) -> str:
        return json.dumps(self._data)


# ── response parsing / retries ────────────────────────────────────────────────────────


def test_success_returns_parsed_dict(fake_openai_client_factory):
    """Successful response → returns dict with `prediction` / `latency` / `generated_text` populated."""
    client, calls = fake_openai_client_factory()
    out = chat_completion_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=1, wait_between_retry=0,
    )
    assert isinstance(out, dict)
    assert out["prediction"] == "hello"
    assert "latency" in out
    assert out["finish_reason"] == "stop"
    assert len(calls) == 1


def test_all_retries_fail_returns_none(fake_openai_client_factory):
    """create() raises every time → tries max_retry times then returns None."""
    client, calls = fake_openai_client_factory(side_effect=[RuntimeError("boom")] * 3)
    out = chat_completion_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=3, wait_between_retry=0,
    )
    assert out is None
    assert len(calls) == 3


def test_transient_failure_then_success(fake_openai_client_factory):
    """First attempt raises → second attempt succeeds → returns dict."""
    client, calls = fake_openai_client_factory(side_effect=[RuntimeError("once")])
    out = chat_completion_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=3, wait_between_retry=0,
    )
    assert isinstance(out, dict)
    assert out["prediction"] == "hello"
    assert len(calls) == 2


# ── message serialization / generation_options ───────────────────────────────────────


def test_message_serialized_to_openai_template(fake_openai_client_factory):
    """Both ChatMessage and dict inputs are passed to create() in openai template shape (role + content)."""
    client, calls = fake_openai_client_factory()
    chat_completion_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=1, wait_between_retry=0,
    )
    msgs = calls[0]["messages"]
    assert isinstance(msgs, list) and len(msgs) == 1
    assert msgs[0].get("role") == "user"
    # if content is text, either string or [{"type":"text","text":...}] — both accepted
    content = msgs[0].get("content")
    assert content is not None


def test_generation_options_max_new_tokens_to_max_tokens(fake_openai_client_factory):
    """`max_new_tokens` (our schema) → `max_tokens` (openai schema) key mapping."""
    client, calls = fake_openai_client_factory()
    chat_completion_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        generation_options={"max_new_tokens": 123, "temperature": 0.5},
        max_retry=1, wait_between_retry=0,
    )
    kwargs = calls[0]
    assert kwargs.get("max_tokens") == 123
    assert kwargs.get("temperature") == 0.5
    assert "max_new_tokens" not in kwargs  # original key must be gone after conversion


def test_model_arg_propagated(fake_openai_client_factory):
    """api_name is passed as `model=` to create() — final hop verification of provider routing."""
    client, calls = fake_openai_client_factory()
    chat_completion_sync(
        client=client, api_name="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=1, wait_between_retry=0,
    )
    assert calls[0]["model"] == "gpt-4o-mini"


# ── content-contract fail-fast guard (is_valid_response) ──────────────


def test_empty_choices_returns_none_not_retried(fake_openai_client_factory):
    """200-OK with empty `choices` (safety block / refusal) → None, and not retried (1 call)."""
    client, calls = fake_openai_client_factory(response=_RawResponse({"choices": []}))
    out = chat_completion_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=5, wait_between_retry=0,
    )
    assert out is None
    assert len(calls) == 1  # not retried despite max_retry=5


def test_contentless_length_finish_returns_none_not_retried(fake_openai_client_factory):
    """Reasoning model that burned the output budget (content=None, finish_reason=length) →
    None, and not retried (identical request cannot help)."""
    client, calls = fake_openai_client_factory(response=_RawResponse({
        "choices": [{"message": {"role": "assistant", "content": None},
                     "finish_reason": "length", "index": 0}],
    }))
    out = chat_completion_sync(
        client=client, api_name="gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=5, wait_between_retry=0,
    )
    assert out is None
    assert len(calls) == 1


def test_refusal_returns_none_not_retried(fake_openai_client_factory):
    """A `message.refusal` (content-policy refusal) with no content → None, and not retried."""
    client, calls = fake_openai_client_factory(response=_RawResponse({
        "choices": [{"message": {"role": "assistant", "content": None,
                                 "refusal": "I can't help with that"},
                     "finish_reason": "stop", "index": 0}],
    }))
    out = chat_completion_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=5, wait_between_retry=0,
    )
    assert out is None
    assert len(calls) == 1


def test_tool_call_only_is_usable(fake_openai_client_factory):
    """A response carrying only a tool call (no text) is NOT flagged unusable — agentic
    tool-use turns must survive. Returns a dict, single call."""
    client, calls = fake_openai_client_factory(response=_RawResponse({
        "choices": [{"message": {"role": "assistant", "content": None,
                                 "tool_calls": [{"id": "call_1", "type": "function",
                                                 "function": {"name": "f", "arguments": "{}"}}]},
                     "finish_reason": "tool_calls", "index": 0}],
    }))
    out = chat_completion_sync(
        client=client, api_name="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=1, wait_between_retry=0,
    )
    assert isinstance(out, dict)
    assert out["tool_calls"]
    assert len(calls) == 1
