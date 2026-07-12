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

"""Unit tests for utils/torch.py — dtype resolution, compute capability, CUDA indexes, tensor comparison."""
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from omni_evaluator.utils.torch import (  # noqa: E402
    find_first_difference,
    get_compute_capability,
    get_cuda_indexes,
    resolve_torch_dtype,
)


# ─────────────────────────────────────────────────────────────
# find_first_difference — first mismatch index or length of the shorter sequence
# ─────────────────────────────────────────────────────────────

def test_find_first_difference():
    """Returns the first mismatch index, or the length of the shorter sequence if identical."""
    a = torch.tensor([1, 2, 3, 9])
    b = torch.tensor([1, 2, 5, 9])
    assert find_first_difference(a, b) == 2
    assert find_first_difference(a, a) == len(a)


# ─────────────────────────────────────────────────────────────
# get_compute_capability — major.minor float when CUDA is available, else None
# ─────────────────────────────────────────────────────────────

def test_get_compute_capability(monkeypatch):
    """Returns major.minor as a float when CUDA is available, or None when unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (8, 0))
    assert get_compute_capability() == 8.0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_compute_capability() is None


# ─────────────────────────────────────────────────────────────
# get_cuda_indexes — [0..n-1] when CUDA is available, else None
# ─────────────────────────────────────────────────────────────

def test_get_cuda_indexes(monkeypatch):
    """Returns [0..n-1] when CUDA is available, or None when unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert get_cuda_indexes() == [0, 1]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_cuda_indexes() is None


# ─────────────────────────────────────────────────────────────
# resolve_torch_dtype — passthrough / 'auto' / aliases / capability fallback / error
# ─────────────────────────────────────────────────────────────

def test_resolve_dtype_passthrough():
    """Returns the value as-is if it is already a torch.dtype."""
    assert resolve_torch_dtype(torch.float32) is torch.float32


def test_resolve_dtype_auto():
    """Passes the 'auto' string through unchanged."""
    assert resolve_torch_dtype("auto") == "auto"


def test_resolve_dtype_aliases():
    """Multiple aliases for fp32 / bf16 / fp16 each map to the correct torch.dtype."""
    assert resolve_torch_dtype("float32") is torch.float32
    assert resolve_torch_dtype("32") is torch.float32
    assert resolve_torch_dtype("bf16") is torch.bfloat16
    assert resolve_torch_dtype("fp16") is torch.float16
    assert resolve_torch_dtype(16) is torch.float16


def test_resolve_dtype_default(monkeypatch):
    """Selects a default dtype based on capability when the input is not a str/torch.dtype (no cc → fp16)."""
    monkeypatch.setattr(
        "omni_evaluator.utils.torch.get_compute_capability", lambda *a, **k: None
    )
    assert resolve_torch_dtype(None) is torch.float16


def test_resolve_dtype_invalid_raises():
    """Raises ValueError for an unknown dtype string."""
    with pytest.raises(ValueError):
        resolve_torch_dtype("float8")
