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

"""Unit tests for api/anthropic/chat_completions.py — response parsing, retries, generation_options normalization, system message extraction, timeout forwarding."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("anthropic")

from omni_evaluator.api.anthropic.chat_completions import chat_completion_sync
from omni_evaluator.schemas.chat import (
    Message as ChatMessage,
    TextContent,
)


def _user_msg(text: str = "hi"):
    return ChatMessage(role="user", content=[TextContent(value=text)])


class _RawResponse:
    """Minimal SDK-response stand-in exposing only `.json()`, for crafting content-contract
    edge cases the fixture factory doesn't build (empty content, thinking-only)."""

    def __init__(self, data: dict):
        self._data = data

    def json(self) -> str:
        return json.dumps(self._data)


# ── response parsing / retries ────────────────────────────────────────────────


def test_success_returns_parsed_dict(fake_anthropic_client_factory):
    """Successful response → dict populated with `prediction` / `latency` / `finish_reason`."""
    client, calls, _ = fake_anthropic_client_factory()
    out = chat_completion_sync(
        client=client, api_name="claude-3-5-sonnet",
        messages=[_user_msg()],
        max_retry=1, wait_between_retry=0,
    )
    assert isinstance(out, dict)
    assert out["prediction"] == "hello"
    assert out["finish_reason"] == "end_turn"
    assert "latency" in out
    assert len(calls) == 1


def test_thinking_block_becomes_reasoning(
    fake_anthropic_client_factory, fake_anthropic_response_factory,
):
    """`content[type=thinking].thinking` is stored as reasoning_content."""
    response = fake_anthropic_response_factory(text="ok", thinking="step1")
    client, _, _ = fake_anthropic_client_factory(response=response)
    out = chat_completion_sync(
        client=client, api_name="claude-3-7-sonnet",
        messages=[_user_msg()],
        max_retry=1, wait_between_retry=0,
    )
    assert out["reasoning_content"] == "step1"


def test_all_retries_fail_returns_none(fake_anthropic_client_factory):
    """create() raises every time → returns None after max_retry attempts."""
    client, calls, _ = fake_anthropic_client_factory(
        side_effect=[RuntimeError("boom")] * 3,
    )
    out = chat_completion_sync(
        client=client, api_name="claude-3-5-sonnet",
        messages=[_user_msg()],
        max_retry=3, wait_between_retry=0,
    )
    assert out is None
    assert len(calls) == 3


# ── message serialization / system extraction ──────────────────────────────────


def test_system_message_extracted_to_top_level(fake_anthropic_client_factory):
    """ChatMessage(role=system) is removed from `messages=` and passed as top-level `system=`."""
    client, calls, _ = fake_anthropic_client_factory()
    chat_completion_sync(
        client=client, api_name="claude-3-5-sonnet",
        messages=[
            ChatMessage(role="system", content=[TextContent(value="be brief")]),
            _user_msg("hi"),
        ],
        max_retry=1, wait_between_retry=0,
    )
    kwargs = calls[0]
    assert kwargs["system"] == "be brief"
    # only user messages should remain in messages=
    assert all(m.get("role") != "system" for m in kwargs["messages"])


def test_no_system_message_passes_empty_list(fake_anthropic_client_factory):
    """When no system message is present, `system=[]` is passed (explicit default in production code)."""
    client, calls, _ = fake_anthropic_client_factory()
    chat_completion_sync(
        client=client, api_name="claude-3-5-sonnet",
        messages=[_user_msg()],
        max_retry=1, wait_between_retry=0,
    )
    assert calls[0]["system"] == [] or calls[0]["system"] in (None, "")


# ── generation_options ───────────────────────────────────────────────────────


def test_generation_options_stop_words_to_stop_sequences(fake_anthropic_client_factory):
    """`stop_words` (our schema) → `stop_sequences` (Anthropic schema) key mapping."""
    client, calls, _ = fake_anthropic_client_factory()
    chat_completion_sync(
        client=client, api_name="claude-3-5-sonnet",
        messages=[_user_msg()],
        generation_options={"max_new_tokens": 50, "stop_words": ["END"]},
        max_retry=1, wait_between_retry=0,
    )
    assert calls[0].get("stop_sequences") == ["END"]
    assert calls[0].get("max_tokens") == 50
    assert "stop_words" not in calls[0]
    assert "max_new_tokens" not in calls[0]


# ── timeout boundary ─────────────────────────────────────────────────────────


def test_timeout_passed_via_with_options(fake_anthropic_client_factory):
    """timeout is forwarded via `client.with_options(timeout=...)` chained call."""
    client, _, with_options_calls = fake_anthropic_client_factory()
    chat_completion_sync(
        client=client, api_name="claude-3-5-sonnet",
        messages=[_user_msg()],
        timeout=42, max_retry=1, wait_between_retry=0,
    )
    assert with_options_calls and with_options_calls[0].get("timeout") == 42


# ── content-contract fail-fast guard (is_valid_response) ──────────────


def test_empty_content_returns_none_not_retried(fake_anthropic_client_factory):
    """200-OK with empty `content` blocks (safety block / refusal) → None, and not retried (1 call)."""
    client, calls, _ = fake_anthropic_client_factory(
        response=_RawResponse({"content": [], "stop_reason": "max_tokens"}),
    )
    out = chat_completion_sync(
        client=client, api_name="claude-3-5-sonnet",
        messages=[_user_msg()],
        max_retry=5, wait_between_retry=0,
    )
    assert out is None
    assert len(calls) == 1  # not retried despite max_retry=5


def test_thinking_only_returns_none_not_retried(fake_anthropic_client_factory):
    """Reasoning model that burned the budget on thinking (only a `thinking` block, no text/tool,
    stop_reason=max_tokens) → None, and not retried."""
    client, calls, _ = fake_anthropic_client_factory(
        response=_RawResponse({
            "content": [{"type": "thinking", "thinking": "hmm..."}],
            "stop_reason": "max_tokens",
        }),
    )
    out = chat_completion_sync(
        client=client, api_name="claude-opus-4-8",
        messages=[_user_msg()],
        max_retry=5, wait_between_retry=0,
    )
    assert out is None
    assert len(calls) == 1


def test_tool_use_only_is_usable(fake_anthropic_client_factory):
    """A response carrying only a `tool_use` block (no text) is NOT flagged unusable — agentic
    tool-use turns must survive. Returns a dict, single call."""
    client, calls, _ = fake_anthropic_client_factory(
        response=_RawResponse({
            "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
            "stop_reason": "tool_use",
        }),
    )
    out = chat_completion_sync(
        client=client, api_name="claude-3-5-sonnet",
        messages=[_user_msg()],
        max_retry=1, wait_between_retry=0,
    )
    assert isinstance(out, dict)
    assert out["tool_calls"]
    assert len(calls) == 1
