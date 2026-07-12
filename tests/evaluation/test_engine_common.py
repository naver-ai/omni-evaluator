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

"""Base class for the `get_data_iterator` / `evaluate_task` export contract inherited by all evaluation harnesses."""
from __future__ import annotations

import inspect
import typing
from typing import Any, Tuple

import pytest

from omni_evaluator.schemas.evaluation import EvaluationRunOutput
from omni_evaluator.schemas.task import TaskConfig


def _origin_is_tuple(annotation) -> bool:
    """Check whether the annotation is a `Tuple[...]` / `tuple[...]` annotation."""
    origin = typing.get_origin(annotation)
    return origin in (tuple, Tuple)


def _first_tuple_arg(annotation):
    """Return the first type element X of `Tuple[X, Y]`. Returns None if not a Tuple."""
    if not _origin_is_tuple(annotation):
        return None
    args = typing.get_args(annotation)
    if not args:
        return None
    return args[0]


def _last_tuple_arg(annotation):
    """Return the last type element Y of `Tuple[X, Y]`. Returns None if not a Tuple."""
    if not _origin_is_tuple(annotation):
        return None
    args = typing.get_args(annotation)
    if not args:
        return None
    return args[-1]


class EvaluationEngineCommonTests:
    """Common export contract for 5 harnesses — subclasses only need to override the `engine_module` fixture."""

    # ── fixture to be overridden by subclasses ────────────────────

    @pytest.fixture
    def engine_module(self):
        raise NotImplementedError(
            "subclass must override `engine_module` fixture to return "
            "the harness engine module (omni_evaluator.evaluation.<harness>.engine)"
        )


    # ── static contract tests ────────────────────────────────────

    def test_exports_iterator(self, engine_module):
        """The `engine` module exposes `get_data_iterator` as a callable attribute."""
        assert hasattr(engine_module, "get_data_iterator"), (
            f"{engine_module.__name__} must export `get_data_iterator`"
        )
        assert callable(engine_module.get_data_iterator), (
            f"{engine_module.__name__}.get_data_iterator must be callable"
        )

    def test_exports_evaluate(self, engine_module):
        """The `engine` module exposes `evaluate_task` as a callable attribute."""
        assert hasattr(engine_module, "evaluate_task"), (
            f"{engine_module.__name__} must export `evaluate_task`"
        )
        assert callable(engine_module.evaluate_task), (
            f"{engine_module.__name__}.evaluate_task must be callable"
        )
