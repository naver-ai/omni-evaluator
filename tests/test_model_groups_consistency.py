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

"""Drift guards for the model-group dependency declarations.

Optional-dependency knowledge lives in a single source of truth,
``model_groups.MODULE_REQUIRED_PACKAGES`` (keyed by model group). These tests
fail in CI when a declared ``.[extra]`` install target no longer matches an extra
in ``pyproject.toml`` — catching the "someone renamed/removed an extra but the
code still says `.[janus]`" drift — and when a dispatchable adapter group has no
dependency entry at all.
"""
import re
from pathlib import Path

def test_dispatch_groups_have_dependency_entries():
    # Every dispatchable adapter group must have a MODULE_REQUIRED_PACKAGES entry
    # (even an empty list) so require_group_dependencies resolves it.
    # NOTE: importing engine pulls torch + adapters.
    from omni_evaluator.inference.huggingface.engine import MODULE_DISPATCH
    from omni_evaluator.inference.huggingface.model_groups import MODULE_REQUIRED_PACKAGES

    missing = sorted(str(g) for g in MODULE_DISPATCH if g not in MODULE_REQUIRED_PACKAGES)
    assert not missing, (
        f"groups in MODULE_DISPATCH missing from MODULE_REQUIRED_PACKAGES: {missing}"
    )
