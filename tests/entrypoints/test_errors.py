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

"""`__main__.main` error/exit code contract — exit 1 (entrypoint direct exit) vs exit 2 (argparse exit)."""
import pytest


def test_no_args(run_cli):
    """Calling with no arguments prints help and exits with code 1 (entrypoint direct exit)."""
    result = run_cli()
    assert result.returncode == 1


def test_unknown_subcommand(run_cli):
    """Unknown subcommand → argparse exits with code 2 on invalid choice."""
    result = run_cli("bogus")
    assert result.returncode == 2


def test_evaluate_invalid_choice(run_cli):
    """`evaluate` with an invalid enum value (inference_engine) → argparse exits with code 2.

    The parser blocks before execution — ensures that step toggles/engine typos
    do not silently pass through.
    """
    result = run_cli("evaluate", "--inference_engine=nonsense")
    assert result.returncode == 2
