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

"""Unit tests for utils/resource.py — CPU/GPU memory queries, CUDA device count, resource splitting."""
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from omni_evaluator.utils.resource import (  # noqa: E402
    get_available_cpu_memory,
    get_dynamic_max_memory,
    get_num_cuda_devices,
    get_total_cpu_memory,
    split_resources,
)


# ─────────────────────────────────────────────────────────────
# get_available_cpu_memory — the larger of min(MIN, 80%) vs 25%
# ─────────────────────────────────────────────────────────────

def test_get_available_cpu_memory(monkeypatch):
    """The larger of min(_MIN_CPU_MEMORY, 80%) and 25% — with a large total, 25% is the lower bound."""
    import psutil
    import types as _types

    total = 1024**4  # 1 TiB → 25% is larger than _MIN_CPU_MEMORY(200GiB)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: _types.SimpleNamespace(total=total)
    )
    assert get_available_cpu_memory() == int(total * 0.25)


# ─────────────────────────────────────────────────────────────
# get_dynamic_max_memory — per-device available memory + world_size split
# ─────────────────────────────────────────────────────────────

def test_get_dynamic_max_memory(monkeypatch):
    """Collects per-device available memory and CPU memory divided by world_size into a dict."""
    import psutil
    import types as _types

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda idx: (10, 20))
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: _types.SimpleNamespace(total=80)
    )
    memory = get_dynamic_max_memory([0], world_size=2)
    assert memory[0] == 10
    assert memory["cpu"] == 40


def test_get_dynamic_max_memory_all_skipped_raises(monkeypatch):
    """Raises ValueError when all devices fall below the skip_device_under threshold."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda idx: (1, 100))
    with pytest.raises(ValueError):
        get_dynamic_max_memory([0], skip_device_under=0.5)


# ─────────────────────────────────────────────────────────────
# get_num_cuda_devices — delegates to torch.cuda.device_count
# ─────────────────────────────────────────────────────────────

def test_get_num_cuda_devices(monkeypatch):
    """Delegates directly to torch.cuda.device_count()."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert get_num_cuda_devices() == 4


# ─────────────────────────────────────────────────────────────
# get_total_cpu_memory — returns psutil.virtual_memory().total
# ─────────────────────────────────────────────────────────────

def test_get_total_cpu_memory(monkeypatch):
    """Returns psutil.virtual_memory().total."""
    import psutil
    import types as _types

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: _types.SimpleNamespace(total=8 * 1024**3)
    )
    assert get_total_cpu_memory() == 8 * 1024**3


# ─────────────────────────────────────────────────────────────
# split_resources — distributes num_resources across workers (+1 to leading ranks)
# ─────────────────────────────────────────────────────────────

def test_split_resources():
    """Remainder is distributed +1 from the front ranks; sum is preserved as num_resources."""
    assert split_resources(10, 2) == [5, 5]
    assert split_resources(7, 3) == [3, 2, 2]
    assert sum(split_resources(7, 3)) == 7
