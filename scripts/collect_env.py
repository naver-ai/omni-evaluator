#!/usr/bin/env python3

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

"""Collect runtime environment info for PR descriptions.

Reproducibility on this repo depends on CUDA / GPU driver, torch wheel build,
and inference / evaluation extras (vllm, sglang, lmms_eval, lm_eval, vlmeval,
audio_bench, lighteval) that can silently drift across hosts. PR reviewers
need to know which host the changes were validated on. Run this script and
paste the output into the PR body.

Usage:
    python scripts/collect_env.py                              # print to stdout
    python scripts/collect_env.py | xclip -selection clipboard # Linux
    python scripts/collect_env.py | pbcopy                     # macOS
"""
from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
import subprocess
import sys
from typing import List, Optional


# Packages whose version is load-bearing for reproducibility on this repo.
# Grouped to keep the report scannable.
_PACKAGE_GROUPS: List[tuple] = [
    ("Core", [
        "torch", "transformers", "accelerate", "tokenizers",
        "huggingface-hub", "datasets",
    ]),
    ("Inference engines", [
        "vllm", "sglang",
    ]),
    ("Evaluation harness extras", [
        "lmms-eval", "lm-eval", "vlmeval", "audio-bench", "lighteval",
    ]),
    ("Multimedia / utility", [
        "pillow", "numpy", "pydantic", "pyyaml",
        "librosa", "soundfile", "torchcodec",
    ]),
]


def _run(cmd: List[str], timeout: int = 5) -> Optional[str]:
    """Best-effort `subprocess.check_output` — returns None on any failure."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout)
        return out.decode("utf-8", errors="replace").strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError):
        return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _gpu_info() -> Optional[str]:
    if shutil.which("nvidia-smi") is None:
        return None
    return _run([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader",
    ])


def _cuda_release() -> Optional[str]:
    """Prefer `nvcc --version` (compile-time CUDA); fall back to driver report."""
    nvcc = _run(["nvcc", "--version"])
    if nvcc:
        for line in nvcc.splitlines():
            if "release" in line.lower():
                return line.strip()
        return nvcc.splitlines()[-1].strip()
    return _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])


def main() -> None:
    out: List[str] = []
    out.append("## Environment")
    out.append("")
    out.append("```")
    out.append(f"Python   : {sys.version.split()[0]}  ({sys.executable})")
    out.append(f"OS       : {platform.platform()}")

    gpu = _gpu_info()
    if gpu:
        gpu_lines = gpu.splitlines()
        out.append(f"GPU      : {gpu_lines[0]}")
        for extra in gpu_lines[1:]:
            out.append(f"           {extra}")
    else:
        out.append("GPU      : (no nvidia-smi)")

    cuda = _cuda_release()
    if cuda:
        out.append(f"CUDA     : {cuda}")

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    sha = _run(["git", "rev-parse", "--short", "HEAD"])
    if branch or sha:
        out.append(f"Git      : branch={branch or '?'}  sha={sha or '?'}")

    out.append("")
    width = max(len(name) for _, names in _PACKAGE_GROUPS for name in names)
    for label, names in _PACKAGE_GROUPS:
        out.append(f"[{label}]")
        for name in names:
            v = _pkg_version(name)
            out.append(f"  {name:<{width}}  {v if v else '(not installed)'}")
        out.append("")
    out.append("```")

    print("\n".join(out))


if __name__ == "__main__":
    main()
