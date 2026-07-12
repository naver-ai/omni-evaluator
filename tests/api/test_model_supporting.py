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

"""Unit tests for api/model_supporting.py — yaml capability matrix."""
from __future__ import annotations

import pytest

from omni_evaluator.api.model_supporting import _SUPPORTING, get_model_supporting


_BASE_CAPABILITY_KEYS = {
    "support_text_understanding",
    "support_image_understanding",
    "support_audio_understanding",
    "support_video_understanding",
    "support_text_generation",
    "support_image_generation",
    "support_audio_generation",
    "support_video_generation",
    "support_reasoning",
    "support_compute_perplexity",
}


# ── _SUPPORTING (yaml load) ──────────────────────────────────────────────────


def test_yaml_provider_keys():
    """All 3 providers — openai / anthropic / google — are registered at the yaml top level."""
    assert {"openai", "anthropic", "google"}.issubset(_SUPPORTING.keys())


# ── get_model_supporting ─────────────────────────────────────────────────────


def test_default_only_returns_baseline():
    """An api_name that matches no override returns the provider default as-is."""
    caps = get_model_supporting("openai", "gpt-4o-mini")
    assert _BASE_CAPABILITY_KEYS.issubset(caps.keys())
    assert caps["support_reasoning"] is False  # default
    assert caps["support_text_generation"] is True


def test_override_prefix_match():
    """`o1` override is prefix-matched, overwriting `support_reasoning=True`."""
    caps = get_model_supporting("openai", "o1-preview")
    assert caps["support_reasoning"] is True
    # default values are also preserved
    assert caps["support_text_generation"] is True


def test_override_does_not_leak_across_providers():
    """anthropic / google overrides do not affect openai — provider scope is isolated."""
    openai_caps = get_model_supporting("openai", "claude-3-5-sonnet")
    # `claude` does not match any openai override, so the default is returned as-is
    assert openai_caps["support_reasoning"] is False


def test_unknown_provider_returns_empty():
    """An unregistered provider returns an empty dict — safe default with no KeyError leak."""
    assert get_model_supporting("unknown-provider", "anything") == {}


def test_gemini3_reasoning_registered():
    """gemini-3 family (thinking models) must resolve to support_reasoning=True, not the conservative
    default. Regression: gemini-3.1-pro-preview fell to default=False, so --thinking_budget was
    silently ignored and unmanaged thinking exhausted the output budget (empty-parts responses)."""
    for name in ("gemini-3.1-pro-preview", "gemini-3.1-flash-lite"):
        assert get_model_supporting("google", name)["support_reasoning"] is True
