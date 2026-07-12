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

"""Fixtures exclusive to tests/api/openai/ — OpenAI SDK response object shape."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Optional

import pytest


pytest.importorskip("openai", reason="Provider-level tests are meaningless without the openai SDK installed")


class _FakeOpenaiResponse:
    """Mock for openai SDK ChatCompletion / Response — mimics the `.json()` interface."""

    def __init__(self, data: dict):
        self._data = data

    def json(self) -> str:
        return json.dumps(self._data)


@pytest.fixture
def fake_chat_response_factory():
    """Chat Completions API response shape builder — populates the `choices[*].message.content` path."""
    def _factory(
        content: str = "hello",
        finish_reason: str = "stop",
        reasoning: Optional[str] = None,
        tool_calls=None,
    ):
        message = {"content": content, "role": "assistant"}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        return _FakeOpenaiResponse({
            "choices": [{
                "message": message,
                "finish_reason": finish_reason,
                "index": 0,
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
    return _factory


@pytest.fixture
def fake_responses_response_factory():
    """Responses API response shape builder — populates the `output[*].type=="message"` / `"reasoning"` paths."""
    def _factory(text: str = "hello", reasoning: Optional[str] = None, status: str = "completed"):
        output = []
        if reasoning is not None:
            output.append({
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning}],
            })
        output.append({
            "type": "message",
            "content": [{"type": "output_text", "text": text}],
        })
        return _FakeOpenaiResponse({
            "output": output,
            "status": status,
        })
    return _factory


@pytest.fixture
def fake_openai_client_factory(fake_chat_response_factory, fake_responses_response_factory):
    """openai client mock — exposes `.chat.completions.create` / `.responses.create` + records calls."""
    def _factory(*, response=None, response_kind: str = "chat", side_effect=None):
        calls: list[dict] = []

        def _make_default():
            return (
                fake_chat_response_factory()
                if response_kind == "chat"
                else fake_responses_response_factory()
            )

        def _create(**kwargs):
            calls.append(kwargs)
            if side_effect is not None:
                exc = side_effect.pop(0) if side_effect else None
                if isinstance(exc, BaseException):
                    raise exc
                if exc is not None:
                    return exc
            return response or _make_default()

        chat = SimpleNamespace(completions=SimpleNamespace(create=_create))
        responses = SimpleNamespace(create=_create)
        return SimpleNamespace(chat=chat, responses=responses), calls
    return _factory
