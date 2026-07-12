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

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from omni_evaluator.postprocess import get_postprocess_functions
from omni_evaluator.schemas.task import TaskConfig


DEBUG_SAMPLES = 20
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "pipeline"
ENGINE_EXTRAS = {
    "lmms_eval": "lmms_eval",
    "lm_eval_harness": "lm_eval",
    "vlm_eval_kit": "vlmeval",
}


# Expand config yaml list into per-benchmark pytest params.
def pytest_generate_tests(metafunc):
    if "benchmark" not in metafunc.fixturenames:
        return

    params = []
    for path in _selected_configs(metafunc.config):
        cfg = OmegaConf.load(path)
        engine = str(cfg.get("evaluation_engine"))
        marks = _engine_marks(engine)
        for benchmark in map(str.strip, str(cfg.benchmarks).split(",")):
            params.append(pytest.param(path, benchmark, marks=marks, id=f"{path.stem}-{benchmark}"))

    metafunc.parametrize(
        "pipeline_config,benchmark",
        params,
        indirect=["pipeline_config"],
        scope="module",
    )


# Default runs all configs under default/; if options are given, run only those paths.
def _selected_configs(config) -> list[Path]:
    requested = config.getoption("--pipeline-config")
    if not requested:
        return sorted((CONFIG_DIR / "default").glob("*.yaml"))
    if "all" in requested:
        return sorted((CONFIG_DIR / "optional").glob("*.yaml"))

    paths = []
    for raw in requested:
        path = Path(raw)  # absolute path or relative path from the working directory
        if not path.is_file():
            raise pytest.UsageError(f"--pipeline-config path not found: {raw}")
        paths.append(path)
    return paths


# builtin runs unconditionally; external engines run only in environments with extras installed.
def _engine_marks(engine: str) -> list[Any]:
    if engine == "builtin":
        return []
    marks = [pytest.mark.eval_engine(engine)]
    if pkg := ENGINE_EXTRAS.get(engine):
        marks.append(pytest.mark.requires_extra(pkg))
    return marks


# Expensive pipeline runs are performed only once per config at module scope.
@pytest.fixture(scope="session")
def local_data_dir():
    root = os.getenv("LOCAL_DATA_DIR")
    if not root:
        pytest.skip("pipeline e2e needs LOCAL_DATA_DIR set to the builtin dataset root")
    if not Path(root).is_dir():
        pytest.skip(f"LOCAL_DATA_DIR does not exist: {root}")
    return root


@pytest.fixture(scope="module")
def pipeline_config(request) -> Path:
    return request.param


@pytest.fixture(scope="module")
def pipeline_run(pipeline_config, request, tmp_path_factory):
    from omni_evaluator.evaluate import main as evaluate_main

    cfg = OmegaConf.load(pipeline_config)
    engine = str(cfg.get("evaluation_engine"))
    local_dir = request.getfixturevalue("local_data_dir") if engine == "builtin" else None
    output_root = tmp_path_factory.mktemp("pipeline_out")
    args = _build_args(cfg, output_root, local_dir)

    with _debug_samples(engine, DEBUG_SAMPLES):
        evaluate_main(args)

    return PipelineRun(args, output_root)


# Build a Namespace via the same parser/validation path as the production entrypoint.
def _build_args(cfg, output_root: Path, local_dir: str | None):
    from omni_evaluator.args import get_parser

    argv = []
    for key, value in OmegaConf.to_container(cfg, resolve=True).items():
        if isinstance(value, bool):
            argv += [f"--{key}"] if value else []
        else:
            argv.append(f"--{key}={value}")

    argv += [
        f"--output_dirpath={output_root}",
        f"--cache_dirpath={output_root / 'cache'}",
        "--debug",
    ]
    if local_dir and str(cfg.get("evaluation_engine")) == "builtin":
        argv.append(f"--local_dirpath={local_dir}")

    parser, validations = get_parser(parser=argparse.ArgumentParser(), argv=argv)
    args = parser.parse_args(argv)
    for validate in validations:
        args = validate(args=args)
    return args


# ⚠️ import-time constants must be patched on both the source module and the consuming module for the change to take effect.
@contextmanager
def _debug_samples(engine: str, n: int):
    import omni_evaluator.inference as inference
    import omni_evaluator.inference.huggingface.engine as hf_engine

    modules = [inference, hf_engine]
    if engine == "lmms_eval":
        import omni_evaluator.evaluation.lmms_eval.engine as lmms_engine

        modules.append(lmms_engine)

    saved = [module.NUM_DEBUG_SAMPLES for module in modules]
    try:
        for module in modules:
            module.NUM_DEBUG_SAMPLES = n
        yield
    finally:
        for module, value in zip(modules, saved):
            module.NUM_DEBUG_SAMPLES = value


# Encapsulates the per-benchmark JSON output locations left by evaluate.main.
class PipelineRun:
    def __init__(self, args, output_root: Path):
        self.args = args
        self.output_root = output_root
        self.engine = str(args.evaluation_engine)

    def output_path(self, benchmark: str) -> Path | None:
        return self._find("output", benchmark)

    def submission_path(self, benchmark: str) -> Path | None:
        return self._find("submission_output", benchmark)

    def output_json(self, benchmark: str) -> dict[str, Any]:
        path = self.output_path(benchmark)
        assert path is not None, f"output json not found for {benchmark} under {self.output_root}"
        return json.loads(path.read_text())

    def _find(self, subdir: str, benchmark: str) -> Path | None:
        matches = list(self.output_root.glob(f"**/{subdir}/{benchmark}__*.json"))
        return matches[0] if len(matches) == 1 else None


# Extraction helpers to decouple test bodies from output structure details.
def _records(output: dict[str, Any]) -> list[dict[str, Any]]:
    return output["inference"][0]


def _metrics(output: dict[str, Any]) -> dict[str, Any]:
    evaluation = output.get("evaluation")
    assert isinstance(evaluation, dict), "evaluation must be a dict"

    runs = evaluation.get("run_outputs") or []
    if runs and isinstance(runs[0], dict) and runs[0].get("metrics"):
        return runs[0]["metrics"]
    return evaluation.get("metrics") or {}


@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.timeout(2400)
class TestPipeline:
    def test_pipeline_contract(self, pipeline_run: PipelineRun, benchmark: str):
        # output JSON always exists and has the essential top-level keys.
        output_path = pipeline_run.output_path(benchmark)
        assert output_path is not None and output_path.exists()

        output = pipeline_run.output_json(benchmark)
        for key in ("config", "inference", "evaluation"):
            assert key in output, f"{benchmark}: output missing '{key}'"

        records = _records(output)

        # Verify prediction, postprocess, metric, and optional submission.
        self._assert_predictions(records, benchmark)
        self._assert_postprocess(output, records, pipeline_run.engine, benchmark)
        self._assert_metrics(_metrics(output), benchmark)
        self._assert_submission_if_present(pipeline_run, benchmark)

    def _assert_predictions(self, records, benchmark):
        # Every record must have a non-empty string prediction.
        for record in records:
            prediction = record.get("prediction")
            assert isinstance(prediction, str) and prediction, f"empty prediction in {benchmark}"

    def _assert_postprocess(self, output, records, engine, benchmark):
        # If a task has postprocess functions configured, the applied results must also appear in output.
        task_config = TaskConfig(**output["config"]) if isinstance(output["config"], dict) else output["config"]
        # get_postprocess_functions returns a (variants dict, default variant key) tuple.
        functions, _variant = get_postprocess_functions(
            evaluation_engine=engine,
            task_name=benchmark,
            task_config=task_config,
        )
        assert isinstance(functions, dict)
        if functions:
            assert any("prediction_postprocessed" in record for record in records), (
                f"{benchmark}: postprocess functions exist but none applied"
            )

    def _assert_metrics(self, metrics, benchmark):
        # metrics must be native finite numbers; ratio-type values must be in [0, 1].
        assert isinstance(metrics, dict) and metrics, f"{benchmark}: empty metrics"
        for name, value in metrics.items():
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"{benchmark}.{name} must be a native number, got {type(value)}"
            )
            assert type(value).__module__ == "builtins", (
                f"{benchmark}.{name} leaked a non-native type: {type(value)}"
            )
            assert math.isfinite(value), f"{benchmark}.{name} is not finite: {value}"
            if any(token in name.lower() for token in ("acc", "match", "score", "rate")):
                assert 0.0 <= float(value) <= 1.0, f"{benchmark}.{name} out of [0,1]: {value}"

    def _assert_submission_if_present(self, pipeline_run, benchmark):
        # Check JSON shape only when a submission formatter has produced the file.
        if path := pipeline_run.submission_path(benchmark):
            assert isinstance(json.loads(path.read_text()), (dict, list))
