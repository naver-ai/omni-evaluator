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

"""Validates loads/generate_text/generate_with_image smoke for the HyperCLOVAX-SEED-Vision V2 HF adapter."""
from __future__ import annotations

import pytest

from omni_evaluator.inference.huggingface.adapters.hyperclovax_vision_v2 import (
    HyperclovaxSeedVisionV2Module,
)
from tests.inference.huggingface.test_huggingface_adapter_common import (
    HuggingfaceAdapterCommonTests,
)


@pytest.mark.inference_engine("hf")
@pytest.mark.model_size("medium")  # 3B → 3-13B (DESIGN 10.1)
@pytest.mark.requires_gpu
@pytest.mark.requires_hf_token
@pytest.mark.slow
class TestHyperclovaxSeedVisionV2(HuggingfaceAdapterCommonTests):
    MODULE_CLS = HyperclovaxSeedVisionV2Module
    DEFAULT_MODEL_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"

    @pytest.fixture(scope="class")
    def module(self, model_id, hf_cache_dir, tmp_path_factory):
        """Loads the adapter with single GPU forced (`device_map={"": 0}`). This model causes device mismatch under multi-GPU sharding."""
        if self.MODULE_CLS is None:
            pytest.fail(
                f"{type(self).__name__} must set MODULE_CLS class variable"
            )
        temp_dir = tmp_path_factory.mktemp(f"{self.MODULE_CLS.__name__}_module")
        try:
            return self.MODULE_CLS(
                model_name_or_path=model_id,
                torch_dtype="float16",
                device_map={"": 0},
                cache_dir=str(hf_cache_dir),
                temp_dirpath=str(temp_dir),
                # The v2 adapter does not hard-code trust_remote_code in __init__, so it is passed explicitly in the constructor.
                trust_remote_code=True,
            )
        except (ImportError, NameError) as ex:
            pytest.skip(
                f"Cannot import {self.MODULE_CLS.__name__} dependencies in this venv "
                f"({type(ex).__name__}: {ex})"
            )

    def test_generates_with_image(self, module):
        """Verifies that the vision path flows end-to-end and generates tokens with image+text input, asserted via `generated_text` (base override).

        The v2 post-processing makes `.prediction` an empty string for the public (non-reasoning) 3B model, so `generated_text` is used for the assertion instead.
        """
        from PIL import Image

        image = Image.new("RGB", (224, 224), color="red")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe the image in one word."},
                ],
            },
        ]
        output = module.generate_text(
            messages=messages,
            generation_options={"max_new_tokens": 8, "do_sample": False},
        )
        assert isinstance(output.generated_text, str)
        assert len(output.generated_text) > 0
