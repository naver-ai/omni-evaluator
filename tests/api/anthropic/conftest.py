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

"""Fixtures dedicated to tests/api/anthropic/ — Anthropic SDK response object shape."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Optional

import pytest


pytest.importorskip("anthropic", reason="No point running provider-level tests without the anthropic SDK installed")


class _FakeAnthropicResponse:
    """Anthropic Message response mock — exposes only `.json()`."""

    def __init__(self, data: dict):
        self._data = data

    def json(self) -> str:
        return json.dumps(self._data)


@pytest.fixture
def fake_anthropic_response_factory():
    """Messages API response shape — text/thinking/tool_use block structure inside content."""
    def _factory(
        text: str = "hello",
        thinking: Optional[str] = None,
        stop_reason: str = "end_turn",
        tool_use=None,
    ):
        content = []
        if thinking is not None:
            content.append({"type": "thinking", "thinking": thinking})
        content.append({"type": "text", "text": text})
        if tool_use is not None:
            content.append({"type": "tool_use", **tool_use})
        return _FakeAnthropicResponse({
            "content": content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
    return _factory


@pytest.fixture
def fake_anthropic_client_factory(fake_anthropic_response_factory):
    """Anthropic chained client mock: `with_options(...).messages.create(...)`."""
    def _factory(*, response=None, side_effect=None):
        calls: list[dict] = []
        with_options_calls: list[dict] = []

        def _create(**kwargs):
            calls.append(kwargs)
            if side_effect is not None:
                exc = side_effect.pop(0) if side_effect else None
                if isinstance(exc, BaseException):
                    raise exc
            return response or fake_anthropic_response_factory()

        inner = SimpleNamespace(messages=SimpleNamespace(create=_create))

        def _with_options(**kw):
            with_options_calls.append(kw)
            return inner

        client = SimpleNamespace(with_options=_with_options)
        return client, calls, with_options_calls
    return _factory
