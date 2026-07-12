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

"""Common fixtures for postprocess tests."""

from typing import Any, Callable, Dict, List

import pytest


@pytest.fixture
def patch_api_chat_completion(monkeypatch):
    """Replaces the local `chat_completion_sync` binding in the consumer module with a fake.

    # ⚠️ The consumer module's binding must be patched, not the original module, so that already-imported names are updated.
    """

    def _setup(target: str, return_value: Any = "B") -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []

        def _fake(**kwargs):
            calls.append(kwargs)
            return return_value

        monkeypatch.setattr(target, _fake)
        return calls

    return _setup


@pytest.fixture
def task_config_factory() -> Callable[..., Dict[str, Any]]:
    """Minimal `TaskConfig`-compatible dict builder accepted by `get_postprocess_functions`."""

    def _build(
        *,
        pipeline=None,
        version=None,
        api_name=None,
        allow_api=None,
        evaluation_engine: str = "builtin",
        task_name: str = "test_task",
    ) -> Dict[str, Any]:
        return {
            "task_name": task_name,
            "evaluation_engine": evaluation_engine,
            "meta": {"benchmark_name": task_name},
            "dataset": {"source": "local"},
            "evaluation": {
                "method": "generation",
                "target_metrics": [],
                "postprocess": {
                    "pipeline": pipeline,
                    "version": version,
                    "api_name": api_name,
                    "allow_api": allow_api,
                },
            },
        }

    return _build
