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

"""launch_server.py smoke — pure command builder + app configuration.

Server runtime (async dispatch / subprocess / uvicorn) is out of scope. Only the
branch coverage of shell-injection-safe argument serialization (`_build_command`)
and FastAPI app route registration are verified.

launch_server is a standalone script at the repo root, so it pulls in fastapi/uvicorn
at import time — skip in environments where they are not installed. Since it is an
independent entrypoint not covered by any other test's import, the import smoke itself
is meaningful (exception noted in tests/CLAUDE.md §8.3).
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from launch_server import _build_command, app


# ── _build_command ────────────────────────────────────────────────
def test_str_arguments():
    """String arguments are split via shlex and appended after BASE (BASE only when empty string)."""
    assert _build_command("--model x --debug") == ["python", "evaluate.py", "--model", "x", "--debug"]
    assert _build_command("") == ["python", "evaluate.py"]


def test_dict_arguments():
    """dict arguments: None omitted, bool True becomes flag only, others become --k=v, False dropped."""
    cmd = _build_command({"model": "x", "debug": True, "verbose": False, "seed": None})
    assert cmd == ["python", "evaluate.py", "--model=x", "--debug"]


def test_unsupported_type_falls_back_to_base():
    """Arguments that are neither str nor dict return BASE only."""
    assert _build_command(123) == ["python", "evaluate.py"]


# ── app ───────────────────────────────────────────────────────────
def test_routes_registered():
    """Expected endpoints are registered in the FastAPI app (routing behavior not verified)."""
    paths = {route.path for route in app.routes}
    assert {"/add_job", "/get_state", "/remove_job", "/jobs"} <= paths
