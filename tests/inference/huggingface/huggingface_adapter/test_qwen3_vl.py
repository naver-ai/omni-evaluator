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

"""Verifies loads/generate_text/generate_with_image smoke tests for the Qwen3-VL HF adapter."""
from __future__ import annotations

import pytest

from omni_evaluator.inference.huggingface.adapters.qwen3_vl import Qwen3VlModule
from tests.inference.huggingface.test_huggingface_adapter_common import (
    HuggingfaceAdapterCommonTests,
)


@pytest.mark.inference_engine("hf")
@pytest.mark.model_size("medium")  # 4B → 3-13B (DESIGN 10.1)
@pytest.mark.requires_gpu
@pytest.mark.requires_hf_token
@pytest.mark.slow
class TestQwen3Vl(HuggingfaceAdapterCommonTests):
    MODULE_CLS = Qwen3VlModule
    DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
    SUPPORTED_MODEL_IDS = [
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ]
