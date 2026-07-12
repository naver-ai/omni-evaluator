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

"""Unit tests for inference/sglang/engine.py — placeholder."""
import pytest


def test_imports():
    """`omni_evaluator.inference.sglang.engine` module can be imported."""
    from omni_evaluator.inference.sglang import engine  # noqa: F401


@pytest.mark.inference_engine("sglang")
@pytest.mark.requires_gpu
def test_initializes():
    """SGLang engine instantiation + basic entry point smoke — TODO (follow-up session)."""
    pytest.skip("TODO: implement SGLang engine init smoke")
