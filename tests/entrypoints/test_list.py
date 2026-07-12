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

"""`python -m omni_evaluator list ...` subcommand output path and error path verification."""
import pytest


def test_inference_engines(run_cli):
    """`list --inference_engines` → exit 0 + non-empty list output."""
    result = run_cli("list", "--inference_engines")
    assert result.returncode == 0
    assert "api/google" in result.stdout


def test_evaluation_engines(run_cli):
    """`list --evaluation_engines` → exit 0 + non-empty list output."""
    result = run_cli("list", "--evaluation_engines")
    assert result.returncode == 0
    assert "builtin" in result.stdout


def test_tasks(run_cli):
    """`list --tasks --evaluation_engine builtin` → exit 0 + task list output for that engine."""
    result = run_cli("list", "--tasks", "--evaluation_engine", "builtin")
    assert result.returncode == 0
    assert "ai2d_test" in result.stdout


def test_no_flag(run_cli):
    """`list` with no selection flag prints subcommand help and exits 1."""
    result = run_cli("list")
    assert result.returncode == 1


def test_tasks_missing_engine(run_cli):
    """`--tasks` without `--evaluation_engine` → argparse error, exit 2."""
    result = run_cli("list", "--tasks")
    assert result.returncode == 2
