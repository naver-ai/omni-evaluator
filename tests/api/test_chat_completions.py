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

"""Unit tests for api/chat_completions.py — provider-agnostic dispatch and return shape."""
from __future__ import annotations

from types import SimpleNamespace

import pydantic
import pytest

# cc_mod is for §9.1 late-binding — held as a module alias only to
# monkeypatch internal dispatch symbols (e.g. `chat_completion_sync_openai`).
# The function under test is imported directly below.
from omni_evaluator.api import chat_completions as cc_mod
from omni_evaluator.api.chat_completions import batch_chat_completion_sync, chat_completion_sync
from omni_evaluator.schemas.inference import ApiInferenceOutput


def _ok_dict(prediction: str = "hello"):
    return {
        "prediction": prediction,
        "reasoning_content": None,
        "tool_calls": None,
        "function_call": None,
        "annotations": [],
        "finish_reason": "stop",
        "latency": 0.01,
    }


@pytest.fixture
def patch_dispatch(monkeypatch):
    """Replaces the 4 provider branches and get_client with fakes and returns a captured-kwargs list."""
    calls: dict[str, list[dict]] = {
        "openai_chat": [],
        "openai_resp": [],
        "anthropic": [],
        "google": [],
    }

    def _make(key, response=_ok_dict):
        def _fake(**kwargs):
            calls[key].append(kwargs)
            return response()
        return _fake

    async def _make_async(key, response=_ok_dict):
        # placeholder; tests below use sync top-level only
        ...

    monkeypatch.setattr(cc_mod, "chat_completion_sync_openai", _make("openai_chat"))
    monkeypatch.setattr(cc_mod, "response_create_sync_openai", _make("openai_resp"))
    monkeypatch.setattr(cc_mod, "chat_completion_sync_anthropic", _make("anthropic"))
    monkeypatch.setattr(cc_mod, "chat_completion_sync_google", _make("google"))
    monkeypatch.setattr(
        cc_mod, "get_client",
        lambda **kw: SimpleNamespace(close=lambda: None),
    )
    return calls


_DUMMY_MSGS = [{"role": "user", "content": "hi"}]


# ── chat_completion_sync dispatch ────────────────────────────────────────────


def test_dispatch_openai_chat(patch_dispatch):
    """gpt-* with no response_format/reasoning → openai chat completions path."""
    chat_completion_sync(api_name="gpt-4o", messages=_DUMMY_MSGS)
    assert len(patch_dispatch["openai_chat"]) == 1
    assert patch_dispatch["openai_resp"] == []


def test_dispatch_openai_responses_with_format(patch_dispatch):
    """With a pydantic response_format, takes the openai responses API path."""
    class Out(pydantic.BaseModel):
        answer: str

    chat_completion_sync(api_name="gpt-4o", messages=_DUMMY_MSGS, response_format=Out)
    assert len(patch_dispatch["openai_resp"]) == 1
    assert patch_dispatch["openai_chat"] == []


def test_dispatch_openai_responses_with_reasoning_effort(patch_dispatch):
    """When reasoning_options contains `reasoning_effort`, takes the responses path."""
    chat_completion_sync(
        api_name="gpt-4o", messages=_DUMMY_MSGS,
        reasoning_options={"reasoning_effort": "medium"},
    )
    assert len(patch_dispatch["openai_resp"]) == 1


def test_dispatch_anthropic(patch_dispatch):
    """claude-* → anthropic chat path."""
    chat_completion_sync(api_name="claude-3-5-sonnet", messages=_DUMMY_MSGS)
    assert len(patch_dispatch["anthropic"]) == 1


def test_dispatch_google(patch_dispatch):
    """gemini-* → google chat path."""
    chat_completion_sync(api_name="gemini-2.0-flash", messages=_DUMMY_MSGS)
    assert len(patch_dispatch["google"]) == 1


# ── return shape ─────────────────────────────────────────────────────────────


def test_return_string_by_default(patch_dispatch):
    """return_dict=False (default) → returns only the prediction str."""
    out = chat_completion_sync(api_name="gpt-4o", messages=_DUMMY_MSGS)
    assert out == "hello"


def test_return_dict_wraps_apiinferenceoutput(patch_dispatch):
    """return_dict=True → wraps result in ApiInferenceOutput."""
    out = chat_completion_sync(
        api_name="gpt-4o", messages=_DUMMY_MSGS, return_dict=True,
    )
    assert isinstance(out, ApiInferenceOutput)
    assert out.prediction == "hello"


def test_provider_none_returns_none(monkeypatch):
    """When the provider returns None (all retries failed) → top-level also returns None."""
    monkeypatch.setattr(cc_mod, "chat_completion_sync_openai", lambda **kw: None)
    monkeypatch.setattr(
        cc_mod, "get_client",
        lambda **kw: SimpleNamespace(close=lambda: None),
    )
    out = chat_completion_sync(api_name="gpt-4o", messages=_DUMMY_MSGS)
    assert out is None


def test_provider_none_with_return_dict_empty_output(monkeypatch):
    """Provider returns None + return_dict=True → empty ApiInferenceOutput (prediction=None)."""
    monkeypatch.setattr(cc_mod, "chat_completion_sync_openai", lambda **kw: None)
    monkeypatch.setattr(
        cc_mod, "get_client",
        lambda **kw: SimpleNamespace(close=lambda: None),
    )
    out = chat_completion_sync(
        api_name="gpt-4o", messages=_DUMMY_MSGS, return_dict=True,
    )
    assert isinstance(out, ApiInferenceOutput)
    assert out.prediction is None


# ── batch_chat_completion_sync ───────────────────────────────────────────────


def test_batch_iterates_per_message(patch_dispatch):
    """batch calls the provider once per messages_list entry and returns a list of ApiInferenceOutput."""
    msgs_list = [_DUMMY_MSGS, _DUMMY_MSGS, _DUMMY_MSGS]
    out = batch_chat_completion_sync(api_name="gpt-4o", messages_list=msgs_list)
    assert len(patch_dispatch["openai_chat"]) == 3
    assert isinstance(out, list) and len(out) == 3
    assert all(isinstance(r, ApiInferenceOutput) for r in out)
    assert all(r.prediction == "hello" for r in out)


def test_batch_rejects_perplexity_evaluation(patch_dispatch):
    """`evaluation_method='perplexity'` is not supported by the API → AssertionError."""
    with pytest.raises(AssertionError, match="perplexity"):
        batch_chat_completion_sync(
            api_name="gpt-4o",
            messages_list=[_DUMMY_MSGS],
            evaluation_method="perplexity",
        )
