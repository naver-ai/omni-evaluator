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

"""Validates `python -m omni_evaluator [sub] --help` output."""
import pytest


def test_top_level(run_cli):
    """Top-level `--help` exits with code 0 and prints usage to stdout."""
    result = run_cli("--help")
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_evaluate(run_cli):
    """`evaluate --help` exits with code 0 via the full parser including evaluation args.

    The `evaluate` stub subparser has add_help=False, but the entrypoint re-parses
    remaining args through the full parser, so `--help` must be handled there
    (regression safety net).
    """
    result = run_cli("evaluate", "--help")
    assert result.returncode == 0
