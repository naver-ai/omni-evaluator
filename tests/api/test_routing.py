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

"""Unit tests for api/routing.py — provider branching, client selection, capability dict."""
from __future__ import annotations

import pytest

from omni_evaluator import ApiGroup
from omni_evaluator.api.routing import get_api_group, get_client, get_engine_features


# ── get_api_group ────────────────────────────────────────────────────────────


def test_api_group_openai():
    """`gpt-`, `chatgpt-`, `o1`, `o3`, `o4` prefixes all map to openai."""
    for name in ["gpt-4o", "chatgpt-4o-audio", "o1-preview", "o3-mini", "o4-mini"]:
        assert get_api_group(name) == ApiGroup.openai


def test_api_group_anthropic():
    """`claude` prefix maps to anthropic."""
    for name in ["claude-3-5-sonnet", "claude-opus-4-20250514"]:
        assert get_api_group(name) == ApiGroup.anthropic


def test_api_group_google():
    """`gemini` prefix maps to google."""
    for name in ["gemini-2.0-flash", "gemini-2.5-pro"]:
        assert get_api_group(name) == ApiGroup.google


def test_api_group_unknown_raises():
    """Unknown prefix → ValueError — blocks regression of silently falling through to a default."""
    with pytest.raises(ValueError, match="unsupported api_name"):
        get_api_group("llama-3")


def test_api_group_case_insensitive():
    """Uppercase input also matches via lower-case normalization."""
    assert get_api_group("GPT-4O") == ApiGroup.openai
    assert get_api_group("Claude-3") == ApiGroup.anthropic


# ── get_client ───────────────────────────────────────────────────────────────


# L1 unit tests that only verify routing (model → client type) — no real API calls.
# get_client constructs via SDK(api_key=os.getenv(...)), so a dummy key is injected
# to make tests deterministic regardless of real keys / .env.test presence (without one,
# the openai/google SDK errors at construction time). Live connectivity is covered by
# *_smoke_* (requires_env) tests in test_engine.py.
def test_get_client_openai_sync(monkeypatch):
    """openai model → openai.OpenAI instance (sync)."""
    pytest.importorskip("openai")
    import openai
    monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key")
    client = get_client("gpt-4o", do_async=False)
    assert isinstance(client, openai.OpenAI)


def test_get_client_openai_async(monkeypatch):
    """openai model + do_async=True → openai.AsyncOpenAI."""
    pytest.importorskip("openai")
    import openai
    monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key")
    client = get_client("gpt-4o", do_async=True)
    assert isinstance(client, openai.AsyncOpenAI)


def test_get_client_anthropic_sync(monkeypatch):
    """anthropic model → anthropic.Anthropic."""
    pytest.importorskip("anthropic")
    import anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy-key")
    client = get_client("claude-3-5-sonnet", do_async=False)
    assert isinstance(client, anthropic.Anthropic)


def test_get_client_anthropic_async(monkeypatch):
    """anthropic model + do_async=True → anthropic.AsyncAnthropic."""
    pytest.importorskip("anthropic")
    import anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy-key")
    client = get_client("claude-3-5-sonnet", do_async=True)
    assert isinstance(client, anthropic.AsyncAnthropic)


def test_get_client_google_sync(monkeypatch):
    """google model → genai.Client (sync)."""
    pytest.importorskip("google.genai")
    from google import genai
    monkeypatch.setenv("GOOGLE_API_KEY", "test-dummy-key")
    client = get_client("gemini-2.0-flash", do_async=False)
    assert isinstance(client, genai.Client)


# ── get_engine_features ──────────────────────────────────────────────────────


def test_engine_features_keys_present():
    """Capability dict for all providers contains the standard `support_*` keys."""
    expected_keys = {
        "support_text_understanding",
        "support_image_understanding",
        "support_audio_understanding",
        "support_video_understanding",
        "support_text_generation",
        "support_reasoning",
    }
    for name in ["gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"]:
        features = get_engine_features(name)
        assert isinstance(features, dict)
        assert expected_keys.issubset(features.keys()), (
            f"{name} missing keys: {expected_keys - features.keys()}"
        )
