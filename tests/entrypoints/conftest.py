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

"""Common fixtures and e2e argv builder for entrypoints/ — for subprocess CLI execution and result verification."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

import pytest
from omegaconf import OmegaConf

# entrypoints-only config copies — matrix cells read yaml from here (eliminates cross-area coupling).
_ENTRYPOINTS_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "entrypoints"
INFERENCE_CONFIG_DIR = _ENTRYPOINTS_CONFIG_DIR / "inference"
EVALUATION_CONFIG_DIR = _ENTRYPOINTS_CONFIG_DIR / "evaluation"

# Passing `--local_dirpath` resolves multimodal items to local files, bypassing SSRF blocking.
# No default value since the path differs per node — builtin e2e is skipped if `LOCAL_DATA_DIR` is not set.
_LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR")


@pytest.fixture
def run_cli():
    """Runs `python -m omni_evaluator ...` as a subprocess and returns CompletedProcess."""
    def _run(*args, **kwargs):
        return subprocess.run(
            [sys.executable, "-m", "omni_evaluator", *args],
            capture_output=True,
            text=True,
            **kwargs,
        )
    return _run


@pytest.fixture(scope="session")
def local_data_dir():
    """Local root for builtin datasets — skipped if `LOCAL_DATA_DIR` is not set or does not exist."""
    if not _LOCAL_DATA_DIR:
        pytest.skip("builtin e2e needs LOCAL_DATA_DIR set to the dataset root")
    if not Path(_LOCAL_DATA_DIR).is_dir():
        pytest.skip(f"LOCAL_DATA_DIR does not exist: {_LOCAL_DATA_DIR}")
    return _LOCAL_DATA_DIR


def build_evaluate_argv(
    config_path: Path,
    output_dirpath: Path,
    *,
    benchmark: str | None = None,
    local_dirpath: str | None = None,
    extra: Sequence[str] = (),
) -> List[str]:
    """Converts an engine config (yaml) to `evaluate` subcommand argv.

    Caps the record count with `--debug` and runs only a single benchmark.
    """
    cfg = OmegaConf.load(config_path)
    bm = benchmark or str(cfg.benchmarks).split(",")[0].strip()

    argv = [
        "evaluate",
        f"--inference_engine={cfg.inference_engine}",
        f"--evaluation_engine={cfg.evaluation_engine}",
        f"--benchmarks={bm}",
        f"--max_new_tokens={cfg.get('max_new_tokens', 4096)}",
        f"--output_dirpath={output_dirpath}",
        f"--cache_dirpath={output_dirpath / 'cache'}",
        "--exp_name=entrypoint_e2e",
        "--debug",
    ]

    # Engine identifier — api uses api_name, others (vllm/hf) use model_name_or_path
    if cfg.get("api_name"):
        argv.append(f"--api_name={cfg.api_name}")
    if cfg.get("model_name_or_path"):
        argv.append(f"--model_name_or_path={cfg.model_name_or_path}")
    if cfg.get("url"):
        argv.append(f"--url={cfg.url}")

    # vllm: url/version/key are node-specific → injected from env
    if str(cfg.inference_engine) == "vllm":
        for env_name, flag in (
            ("VLLM_URL", "--url"),
            ("VLLM_API_VERSION", "--vllm_api_version"),
            ("VLLM_API_KEY", "--vllm_api_key"),
        ):
            if os.getenv(env_name):
                argv.append(f"{flag}={os.environ[env_name]}")

    # Take only the first evaluation_method to match the single benchmark
    if cfg.get("evaluation_methods"):
        argv.append(f"--evaluation_methods={str(cfg.evaluation_methods).split(',')[0].strip()}")

    # Value-type options (CLI arg name = config key. Merged: semaphore_size→inference_concurrency,
    # timeout_sync/async→request_timeout)
    for key in ("torch_dtype", "device_map", "inference_concurrency", "request_timeout", "max_retry"):
        if cfg.get(key) is not None:
            argv.append(f"--{key}={cfg[key]}")

    # store_true-type options
    if cfg.get("trust_remote_code"):
        argv.append("--trust_remote_code")
    if cfg.get("do_async"):
        argv.append("--do_async")

    if local_dirpath:
        argv.append(f"--local_dirpath={local_dirpath}")

    argv.extend(extra)
    return argv


@pytest.fixture(scope="module")
def inferred_dir(tmp_path_factory, local_data_dir):
    """Inference output dir from a single `evaluate --skip_evaluation` run using hf(Qwen2.5-Omni)+builtin."""
    out = tmp_path_factory.mktemp("infer_out")
    argv = build_evaluate_argv(
        INFERENCE_CONFIG_DIR / "huggingface.yaml",
        out,
        benchmark="ai2d_test",
        local_dirpath=local_data_dir,
        extra=("--skip_evaluation",),
    )
    result = subprocess.run(
        [sys.executable, "-m", "omni_evaluator", *argv],
        capture_output=True,
        text=True,
    )
    return result, out
