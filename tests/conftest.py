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

"""Registers global markers, loads `.env.test`, and configures environment-requirement skip hooks."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest
from dotenv import load_dotenv


# ── .env.test loading ──
# The mere existence of the file is opt-in for live API tests — if the file is absent, requires_env tests are unconditionally skipped.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_TEST_PATH = _REPO_ROOT / ".env.test"
_ENV_TEST_LOADED: bool = _ENV_TEST_PATH.exists()
if _ENV_TEST_LOADED:
    load_dotenv(_ENV_TEST_PATH, override=False)


_DOMAIN_MARKERS = [
    'inference_engine(name): "api" | "vllm" | "sglang" | "hf"',
    'eval_engine(name): "builtin" | "lm_eval_harness" | "lmms_eval" | "vlm_eval_kit"',
    'model_size(name): "small" | "medium" | "large" | "xl"',
]
_REQUIREMENT_MARKERS = [
    'requires_gpu: skip if CUDA is unavailable',
    'requires_multi_gpu(n=2): skip if fewer than n GPUs are available',
    'requires_multi_node: skip when not run on a multi-node environment',
    'requires_env(*env_var_names): skip unless .env.test is present AND every given env var is set '
    '(pass the env var name directly, e.g. requires_env("OPENAI_API_KEY"))',
    'requires_hf_token: skip if HF_TOKEN env var is missing',
    'requires_extra(*pkg_names): skip if any listed importable Python package is not installed. '
    'Arg is the import name, not the pyproject extras key '
    '(e.g. requires_extra("lmms_eval") for the [lmms_eval] extra).',
    'available_gpu(*names): skip if the current GPU model is not one of the listed names',
]
_RUNTIME_MARKERS = [
    'slow: 1~10 min; skip by default, opt-in with --runslow',
    'very_slow: 10 min+; skip by default, opt-in with --runveryslow',
]


def pytest_configure(config):
    for m in (
        _DOMAIN_MARKERS
        + _REQUIREMENT_MARKERS
        + _RUNTIME_MARKERS
    ):
        config.addinivalue_line("markers", m)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run tests marked 'slow' (1~10 min each)",
    )
    parser.addoption(
        "--runveryslow", action="store_true", default=False,
        help="run tests marked 'very_slow' (10 min+ each)",
    )
    parser.addoption(
        "--pipeline-config", action="append", default=[], metavar="PATH",
        help="opt-in pipeline config yaml(s) to run INSTEAD of the always-on default/ set. "
             "Repeatable; value = path to a yaml (absolute, or relative to the working dir), "
             "or 'all' for every config under tests/configs/pipeline/optional/. "
             "When given, default/ configs are NOT run (tests/pipelines/CLAUDE.md §1.2).",
    )


# ── vLLM live test healthcheck gate ──
# Handled as SKIP + Warning to avoid mistaking server absence for a code regression.
def skip_if_vllm_down(url: str, *, token: str | None = None) -> None:
    """Skips the test and leaves a warning if the vLLM server at `url` is not responding."""
    from omni_evaluator.inference.vllm.engine import healthcheck

    if token is None:
        token = os.getenv("VLLM_API_KEY")
    if not healthcheck(url=url, max_retries=1, interval=0, token=token):
        msg = f"vLLM server not reachable at {url!r} — skipping live test"
        warnings.warn(msg, stacklevel=2)
        pytest.skip(msg)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _num_cuda_devices() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _import_spec(pkg_name: str):
    import importlib.util
    try:
        return importlib.util.find_spec(pkg_name)
    except Exception:
        return None


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--runslow")
    run_very_slow = config.getoption("--runveryslow")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(pytest.mark.skip(reason="slow test; pass --runslow to enable"))
        if "very_slow" in item.keywords and not run_very_slow:
            item.add_marker(pytest.mark.skip(reason="very slow test; pass --runveryslow to enable"))

        if "requires_gpu" in item.keywords and not _has_cuda():
            item.add_marker(pytest.mark.skip(reason="CUDA unavailable"))

        for marker in item.iter_markers(name="requires_multi_gpu"):
            n = marker.kwargs.get("n", marker.args[0] if marker.args else 2)
            if _num_cuda_devices() < int(n):
                item.add_marker(pytest.mark.skip(reason=f"requires at least {n} GPUs"))

        for marker in item.iter_markers(name="requires_env"):
            if not _ENV_TEST_LOADED:
                item.add_marker(pytest.mark.skip(
                    reason=f"live API tests gated by .env.test (missing at {_ENV_TEST_PATH})"
                ))
                continue
            env_vars = list(marker.args)
            missing = [v for v in env_vars if not os.getenv(v)]
            if missing:
                item.add_marker(pytest.mark.skip(
                    reason=f"missing env var(s) in .env.test: {', '.join(missing)}"
                ))

        if "requires_hf_token" in item.keywords and not os.getenv("HF_TOKEN"):
            item.add_marker(pytest.mark.skip(reason="HF_TOKEN not set"))

        for marker in item.iter_markers(name="requires_extra"):
            pkgs = list(marker.args)
            missing = [p for p in pkgs if _import_spec(p) is None]
            if missing:
                item.add_marker(pytest.mark.skip(
                    reason=f"missing optional package(s): {', '.join(missing)} "
                    f"(install via pyproject extras)"
                ))
