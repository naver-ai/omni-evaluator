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

"""Validates loads/generate_text smoke for the Qwen2-Audio HF adapter (audio-only adapter)."""
from __future__ import annotations

import pytest

from omni_evaluator.inference.huggingface.adapters.qwen2_audio import Qwen2AudioModule
from tests.inference.huggingface.test_huggingface_adapter_common import (
    HuggingfaceAdapterCommonTests,
)


@pytest.mark.inference_engine("hf")
@pytest.mark.model_size("medium")  # 7B → 3-13B (DESIGN 10.1)
@pytest.mark.requires_gpu
@pytest.mark.requires_hf_token
@pytest.mark.slow
class TestQwen2Audio(HuggingfaceAdapterCommonTests):
    MODULE_CLS = Qwen2AudioModule
    DEFAULT_MODEL_ID = "Qwen/Qwen2-Audio-7B"
    SUPPORTED_MODEL_IDS = [
        "Qwen/Qwen2-Audio-7B",
    ]

    def test_generates_text(self, module, audio_message_bytes):
        """Verifies that the text understanding path returns a string prediction with audio-accompanied input (base text-only override, audio-required adapter)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_message_bytes},
                    {"type": "text", "text": "Answer in one word."},
                ],
            },
        ]
        output = module.generate_text(
            messages=messages,
            generation_options={"max_new_tokens": 8, "do_sample": False},
        )
        assert isinstance(output.prediction, str)
