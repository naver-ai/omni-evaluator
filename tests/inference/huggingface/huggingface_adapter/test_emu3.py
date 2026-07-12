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

"""Validates loads/generate_with_image smoke and text→image generation paths of the Emu3 HF adapter.

The text-only understanding path is disabled via a no-op override because Emu3 requires image input,
and image+text understanding is verified by the inherited `test_generates_with_image`. Requires transformers==4.44.
"""
from __future__ import annotations

from typing import List

import pytest

from omni_evaluator.inference.huggingface.adapters.emu3 import Emu3Module
from tests.inference.huggingface.test_huggingface_adapter_common import (
    HuggingfaceAdapterCommonTests,
    _release_module_gpu_memory,
)


@pytest.mark.inference_engine("hf")
@pytest.mark.model_size("medium")  # ~8B → 3-13B (DESIGN 10.1)
@pytest.mark.requires_gpu
@pytest.mark.requires_hf_token
@pytest.mark.requires_extra("emu3")  # emu3 package required (external)
@pytest.mark.slow
class TestEmu3(HuggingfaceAdapterCommonTests):
    """Emu3 understanding + generation smoke (inherits `HuggingfaceAdapterCommonTests`).

    Fixture source map:

        ── this class ──
        _allow_conditional_flash_attn_import, module

        ── base ──
        model_id, _skip_on_transformers_version, _release_gpu_after_class

        ── tests/inference/huggingface/conftest.py ──
        hf_cache_dir

        ── pytest builtin ──
        tmp_path_factory
    """

    MODULE_CLS = Emu3Module
    DEFAULT_MODEL_ID = "BAAI/Emu3-Chat"
    SUPPORTED_MODEL_IDS: List[str] = [
        "BAAI/Emu3-Chat",
        "BAAI/Emu3-Gen",
        "BAAI/Emu3-Stage1",
    ]

    # ── environment bypass / model loading override ─────────────────────────

    @pytest.fixture(scope="class")
    def _allow_conditional_flash_attn_import(self):
        """Bypasses transformers 4.44's `check_imports` so it does not force installation of conditional flash_attn imports."""
        import transformers.dynamic_module_utils as dmu

        _orig = dmu.get_imports
        dmu.get_imports = lambda filename: [
            imp for imp in _orig(filename) if imp != "flash_attn"
        ]
        try:
            yield
        finally:
            dmu.get_imports = _orig

    @pytest.fixture(scope="class")
    def module(
        self, _allow_conditional_flash_attn_import, model_id, hf_cache_dir, tmp_path_factory
    ):
        """Loads the adapter with `device_map="balanced_low_0"` after bypassing flash_attn. Skips on ImportError/NameError."""
        temp_dir = tmp_path_factory.mktemp("Emu3Module_module")
        try:
            adapter = Emu3Module(
                model_name_or_path=model_id,
                torch_dtype="float16",
                device_map="balanced_low_0",
                cache_dir=str(hf_cache_dir),
                temp_dirpath=str(temp_dir),
            )
        except (ImportError, NameError) as ex:
            pytest.skip(
                f"Cannot import Emu3Module dependencies in this venv "
                f"({type(ex).__name__}: {ex})"
            )
        try:
            yield adapter
        finally:
            _release_module_gpu_memory(adapter)

    # ── base test override / additions ─────────────────────────

    def test_generates_text(self):
        """Disables the base text-only smoke as a no-op (Emu3 requires image input)."""
        pass

    @pytest.mark.timeout(1800)
    def test_generates_image(self, module):
        """Verifies that the text→image generation path (`generate_image`) returns a list of PIL images (skips on GPU OOM)."""
        import torch
        from PIL import Image

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "a red apple on a wooden table"}],
            },
        ]
        try:
            output = module.generate_image(messages=messages)
        except torch.cuda.OutOfMemoryError as ex:
            torch.cuda.empty_cache()
            pytest.skip(f"GPU OOM during image generation: {ex}")

        assert isinstance(output.generated_images, list)
        assert len(output.generated_images) > 0
        assert all(isinstance(im, Image.Image) for im in output.generated_images)
