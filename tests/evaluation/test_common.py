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

"""Validates the `get_system_prompt()` routing logic in `evaluation/common.py`."""
from __future__ import annotations


def test_miss_returns_none():
    """`get_system_prompt`: returns None if either lookup fails."""
    from omni_evaluator.evaluation.common import get_system_prompt

    assert get_system_prompt("missing", {"other": "key"}) is None       # task not found
    assert get_system_prompt("k", {"k": "__not_a_prompt_key__"}) is None  # value not found
    assert get_system_prompt("any", {}) is None                          # empty map


def test_resolves():
    """`get_system_prompt`: returns the value from SYSTEM_PROMPTS when both lookups hit."""
    from omni_evaluator.evaluation.common import get_system_prompt
    from omni_evaluator.inference.prompts import SYSTEM_PROMPTS

    assert SYSTEM_PROMPTS, "SYSTEM_PROMPTS must not be empty (regression guard)"
    sample_key = next(iter(SYSTEM_PROMPTS))
    expected = SYSTEM_PROMPTS[sample_key]

    result = get_system_prompt(
        task_name="dummy",
        system_prompt_map={"dummy": sample_key},
    )
    assert result == expected
