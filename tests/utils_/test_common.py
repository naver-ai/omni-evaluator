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

"""Unit tests for utils/common.py — engine/task listing, dynamic module loading, URL/healthcheck, set_seed."""
import random

import numpy as np
import pytest

from omni_evaluator.utils.common import (
    get_custom_module,
    healthcheck,
    list_evaluation_engines,
    list_inference_engines,
    list_tasks,
    remove_stop_words,
    set_seed,
    validate_url,
)


# ─────────────────────────────────────────────────────────────
# get_custom_module — dynamically imports a custom module by task_name or module_path
# ─────────────────────────────────────────────────────────────

def test_get_custom_module_resolves():
    """Returns the module when module_path is importable."""
    module = get_custom_module("builtin", module_path="omni_evaluator.utils.torch")
    assert module is not None
    assert hasattr(module, "resolve_torch_dtype")


def test_get_custom_module_missing_returns_none():
    """Returns None when no custom module exists for a builtin task."""
    assert get_custom_module("builtin", task_name="__no_such_task__") is None


def test_get_custom_module_requires_target():
    """Raises ValueError when neither task_name nor module_path is provided."""
    with pytest.raises(ValueError):
        get_custom_module("builtin")


# ─────────────────────────────────────────────────────────────
# healthcheck — returns True if HTTP GET succeeds with 200 after retries, False otherwise
# ─────────────────────────────────────────────────────────────

class _FakeResponse:
    """Minimal response seen by healthcheck — exposes only status_code."""

    def __init__(self, status_code: int):
        self.status_code = status_code


def test_healthcheck_success(stub_requests_get, monkeypatch):
    """Returns True without retrying when a 200 response is received."""
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    calls = stub_requests_get(lambda url, **kw: _FakeResponse(200))
    assert healthcheck("http://host", max_retries=3, interval=0) is True
    assert len(calls) == 1


def test_healthcheck_failure(stub_requests_get, monkeypatch):
    """Returns False after exhausting max_retries when non-200 responses persist."""
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    calls = stub_requests_get(lambda url, **kw: _FakeResponse(503))
    assert healthcheck("http://host", max_retries=3, interval=0) is False
    assert len(calls) == 3


def test_healthcheck_exception(stub_requests_get, monkeypatch):
    """Retries and ultimately returns False even when requests raise an exception."""
    import requests

    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    def _raise(url, **kw):
        raise requests.RequestException("boom")

    stub_requests_get(_raise)
    assert healthcheck("http://host", max_retries=2, interval=0) is False


# ─────────────────────────────────────────────────────────────
# list_evaluation_engines — set of registered evaluation harness names
# ─────────────────────────────────────────────────────────────

def test_list_evaluation_engines():
    """Returns all registered evaluation engine values."""
    assert set(list_evaluation_engines()) == {
        "builtin",
        "lmms_eval",
        "lm_eval_harness",
        "vlm_eval_kit",
    }


# ─────────────────────────────────────────────────────────────
# list_inference_engines — set of registered inference engine names
# ─────────────────────────────────────────────────────────────

def test_list_inference_engines():
    """Returns all registered inference engine values."""
    assert set(list_inference_engines()) == {
        "huggingface",
        "llama_cpp",
        "vllm",
        "sglang",
        "api/openai",
        "api/anthropic",
        "api/google",
    }


# ─────────────────────────────────────────────────────────────
# list_tasks — task list for an evaluation_engine (importlib discovery)
# ─────────────────────────────────────────────────────────────

def test_list_tasks_builtin():
    """The builtin harness scans the tasks directory via importlib and returns task names."""
    tasks = list_tasks("builtin")
    assert isinstance(tasks, list) and len(tasks) > 0
    assert "_example_huggingface" in tasks


def test_list_tasks_invalid_raises():
    """Raises ValueError for an unknown evaluation_engine."""
    with pytest.raises(ValueError):
        list_tasks("__no_such_engine__")


# ─────────────────────────────────────────────────────────────
# remove_stop_words — truncates text at a stop word
# ─────────────────────────────────────────────────────────────

def test_remove_stop_words():
    """Truncates text after a stop word; preserves original text when stop_words is None."""
    assert remove_stop_words("hello<eot> world", ["<eot>"]) == "hello"
    assert remove_stop_words("hello world", None) == "hello world"


# ─────────────────────────────────────────────────────────────
# set_seed — fixes random / numpy / torch seeds in one call
# ─────────────────────────────────────────────────────────────

def test_set_seed_reproducible():
    """The same seed produces identical values from random, numpy, and torch (3-lib regression)."""
    import torch

    set_seed(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_seed(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second


def test_set_seed_returns_random_when_none():
    """Generates and returns a random 32-bit seed when seed=None."""
    seed = set_seed(None)
    assert isinstance(seed, int) and 0 <= seed < 2**32


# ─────────────────────────────────────────────────────────────
# validate_url — protocol validation and correction option
# ─────────────────────────────────────────────────────────────

def test_validate_url():
    """Passes through URLs with a protocol; prepends the default protocol when correction is enabled."""
    assert validate_url("http://host:8080") == "http://host:8080"
    assert validate_url("host:8080", correction=True) == "http://host:8080"


def test_validate_url_raises():
    """Raises ValueError when protocol is missing without correction, or when host is missing."""
    with pytest.raises(ValueError):
        validate_url("host:8080", correction=False)
    with pytest.raises(ValueError):
        validate_url("http:///path", correction=True)
