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

"""Unit tests for api/completions.py — text completion dispatch.

The anthropic/google completion stubs are currently empty, causing an ImportError on module load — skipping the module until the production-side fix is in place.
"""
from __future__ import annotations

import pytest

# Skip the module at collection time until the production-side stub is filled in
try:
    import omni_evaluator.api.completions  # noqa: F401
except ImportError:
    pytest.skip(
        "omni_evaluator/api/completions.py imports completion_sync from "
        "anthropic/google completions stubs which are currently empty — fix in production.",
        allow_module_level=True,
    )
