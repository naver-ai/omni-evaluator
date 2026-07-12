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

"""Unit tests for inference/vllm/completions.py — placeholder."""
import pytest


def test_imports():
    """The `omni_evaluator.inference.vllm.completions` module can be imported."""
    from omni_evaluator.inference.vllm import completions  # noqa: F401


@pytest.mark.inference_engine("vllm")
def test_smoke():
    """vLLM completions function smoke — TODO (follow-up session)."""
    pytest.skip("TODO: implement vLLM completions smoke")
