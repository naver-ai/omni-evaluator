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

"""Validates loads/generate_text/generate_with_image smoke for the DeepSeek-Janus-Pro HF adapter."""
from __future__ import annotations

import pytest

from omni_evaluator.inference.huggingface.adapters.janus_pro import JanusProModule
from tests.inference.huggingface.test_huggingface_adapter_common import (
    HuggingfaceAdapterCommonTests,
)


@pytest.mark.inference_engine("hf")
@pytest.mark.model_size("small")  # 1B → <3B (DESIGN 10.1)
@pytest.mark.requires_gpu
@pytest.mark.requires_hf_token
@pytest.mark.requires_extra("janus")  # janus package required (inherits JanusModule)
@pytest.mark.slow
class TestJanusPro(HuggingfaceAdapterCommonTests):
    MODULE_CLS = JanusProModule
    DEFAULT_MODEL_ID = "deepseek-ai/Janus-Pro-1B"
    SUPPORTED_MODEL_IDS = [
        "deepseek-ai/Janus-Pro-1B",
        "deepseek-ai/Janus-Pro-7B",
    ]
    @pytest.fixture(scope="class")
    def module(self, model_id, hf_cache_dir, tmp_path_factory):
        """Loads the adapter with single GPU forced (`device_map={"": 0}`). Janus is a multi-GPU sharding-unsafe model."""
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
            )
        except (ImportError, NameError) as ex:
            pytest.skip(
                f"Cannot import {self.MODULE_CLS.__name__} dependencies in this venv "
                f"({type(ex).__name__}: {ex})"
            )
