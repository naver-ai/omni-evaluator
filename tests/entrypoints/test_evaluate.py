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

"""`python -m omni_evaluator evaluate` live e2e — inference/evaluation engine cross matrix + stage separation."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tests.conftest import skip_if_vllm_down
from tests.entrypoints.conftest import (
    EVALUATION_CONFIG_DIR,
    INFERENCE_CONFIG_DIR,
    build_evaluate_argv,
)


def _run_evaluate(argv) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "omni_evaluator", *argv],
        capture_output=True,
        text=True,
    )


def _load_output(root: Path) -> dict:
    """Find and load a single benchmark output JSON under the tmp root — exactly one must exist."""
    matches = list(root.glob("**/output/*.json"))
    assert len(matches) == 1, f"expected 1 output file, found {matches}"
    return json.loads(matches[0].read_text())


def _has_prediction(output: dict) -> bool:
    """Returns True if at least one non-empty prediction exists in the inference output."""
    return any(
        record.get("prediction")
        for run_records in output["inference"]
        for record in run_records
    )


@pytest.mark.slow
@pytest.mark.parametrize("config_name", [
    pytest.param(
        INFERENCE_CONFIG_DIR / "api_openai.yaml",
        marks=[pytest.mark.requires_env("OPENAI_API_KEY"), pytest.mark.inference_engine("api")],
        id="api",
    ),
    pytest.param(
        INFERENCE_CONFIG_DIR /"vllm.yaml",
        marks=[pytest.mark.requires_env("VLLM_API_KEY", "VLLM_URL", "VLLM_API_VERSION"), pytest.mark.inference_engine("vllm")],
        id="vllm",
    ),
    pytest.param(
        INFERENCE_CONFIG_DIR /"huggingface.yaml",
        marks=[pytest.mark.requires_gpu, pytest.mark.inference_engine("hf")],
        id="hf",
    ),
])
def test_inference_engine(config_name, local_data_dir, tmp_path):
    """Per inference engine, `evaluate` (evaluation=builtin) runs end-to-end and produces inference+evaluation output."""
    cfg = OmegaConf.load(config_name)
    if str(cfg.inference_engine) == "vllm":
        skip_if_vllm_down(os.environ["VLLM_URL"])
    argv = build_evaluate_argv(
        config_name,
        tmp_path,
        benchmark="ai2d_test",
        local_dirpath=local_data_dir,
    )
    result = _run_evaluate(argv)
    assert result.returncode == 0, result.stderr[-2000:]
    output = _load_output(tmp_path)
    assert _has_prediction(output)
    assert isinstance(output.get("evaluation"), dict)


@pytest.mark.slow
@pytest.mark.parametrize("config_name", [
    pytest.param(
        EVALUATION_CONFIG_DIR / "lmms_eval.yaml",
        marks=[pytest.mark.requires_gpu, pytest.mark.requires_extra("lmms_eval"), pytest.mark.eval_engine("lmms_eval")],
        id="lmms_eval",
    ),
    pytest.param(
        EVALUATION_CONFIG_DIR / "lm_eval_harness.yaml",
        marks=[pytest.mark.requires_gpu, pytest.mark.requires_extra("lm_eval"), pytest.mark.eval_engine("lm_eval_harness")],
        id="lm_eval_harness",
    ),
    pytest.param(
        EVALUATION_CONFIG_DIR / "vlm_eval_kit.yaml",
        marks=[pytest.mark.requires_gpu, pytest.mark.requires_extra("vlmeval"), pytest.mark.eval_engine("vlm_eval_kit")],
        id="vlm_eval_kit",
    ),
])
def test_evaluation_engine(config_name, tmp_path):
    """Per evaluation harness, `evaluate` (inference=hf) runs end-to-end and produces evaluation output."""
    argv = build_evaluate_argv(config_name, tmp_path)
    result = _run_evaluate(argv)
    assert result.returncode == 0, result.stderr[-2000:]
    output = _load_output(tmp_path)
    assert isinstance(output.get("evaluation"), dict)


# ── stage separation (hf Qwen2.5-Omni + builtin) — --skip_* branches of infer.py / evaluate.py ──

@pytest.mark.slow
@pytest.mark.requires_gpu
def test_skip_evaluation(inferred_dir):
    """`evaluate --skip_evaluation` runs inference only and does not produce an evaluation key."""
    result, out = inferred_dir
    assert result.returncode == 0, result.stderr[-2000:]
    output = _load_output(out)
    assert _has_prediction(output)
    assert "evaluation" not in output


@pytest.mark.slow
@pytest.mark.requires_gpu
def test_skip_inference(inferred_dir, local_data_dir, tmp_path):
    """`evaluate --skip_inference` on top of existing inference output fills in only the evaluation."""
    _, infer_out = inferred_dir
    work = tmp_path / "eval_in"
    shutil.copytree(infer_out, work)

    argv = build_evaluate_argv(
        INFERENCE_CONFIG_DIR / "huggingface.yaml",
        work,
        benchmark="ai2d_test",
        local_dirpath=local_data_dir,
        extra=("--skip_inference",),
    )
    result = _run_evaluate(argv)
    assert result.returncode == 0, result.stderr[-2000:]
    output = _load_output(work)
    assert isinstance(output.get("evaluation"), dict)
