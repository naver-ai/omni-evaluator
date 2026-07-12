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

"""HF adapter common smoke tests — any adapter's `test_<adapter>.py` that inherits `HuggingfaceAdapterCommonTests` gets the same regression safety net.

generate-* tests for modalities not supported by `MODULE_CLS.ENGINE_FEATURES` are removed from collection by `__init_subclass__` (not skipped — never collected at all).
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import List, Optional, Type

import pytest

import omni_evaluator.inference.huggingface.adapters as adapters_pkg
from tests.inference.huggingface._adapter_registry import HF_EXAMPLE_MODELS


_HF_MODEL_ID_ENV = "HF_MODEL_ID"

# ENGINE_FEATURES flag → modality test method name mapping (tests for False flags are shadowed with None).
_MODALITY_TEST_BY_FEATURE = {
    "support_text_understanding": "test_generates_text",
    "support_image_understanding": "test_generates_with_image",
    "support_audio_understanding": "test_generates_with_audio",
    "support_video_understanding": "test_generates_with_video",
}

# List of heavy attributes to sever during class-scope module teardown to reclaim GPU memory.
_HEAVY_MODULE_ATTRS = ("model", "processor", "tokenizer", "image_processor")


def _release_module_gpu_memory(module=None) -> None:
    """Severs references to the adapter's heavy attributes and reclaims GPU memory/cache (no-op if torch is not installed)."""
    if module is not None:
        for _attr in _HEAVY_MODULE_ATTRS:
            try:
                if getattr(module, _attr, None) is not None:
                    setattr(module, _attr, None)
            except Exception:
                pass

    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ── modality mixin ─────────────────────────────────────────────
# Each mixin holds exactly one modality generate test; whether it is collected is determined by ENGINE_FEATURES.


class _TextUnderstandingMixin:
    def test_generates_text(self, module):
        """Calling `generate_text()` with text only returns a non-empty string prediction."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Say 'hello' and nothing else."}],
            },
        ]
        output = module.generate_text(
            messages=messages,
            generation_options={"max_new_tokens": 8, "do_sample": False},
        )
        assert isinstance(output.prediction, str)
        assert len(output.prediction) > 0


class _ImageUnderstandingMixin:
    def test_generates_with_image(self, module):
        """Calling `generate_text()` with image+text returns a non-empty string prediction."""
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
        assert isinstance(output.prediction, str)
        assert len(output.prediction) > 0


class _AudioUnderstandingMixin:
    def test_generates_with_audio(self, module, audio_message_bytes):
        """Calling `generate_text()` with audio+text returns a string prediction (empty string allowed)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_message_bytes},
                    {"type": "text", "text": "Describe what you hear in one word."},
                ],
            },
        ]
        output = module.generate_text(
            messages=messages,
            generation_options={"max_new_tokens": 8, "do_sample": False},
        )
        assert isinstance(output.prediction, str)


class _VideoUnderstandingMixin:
    def test_generates_with_video(self, module, video_message_path):
        """Calling `generate_text()` with video+text returns a string prediction (empty string allowed; skip on GPU OOM)."""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_message_path,
                        # Reduce frame resolution via max_pixels to lower visual token cost (nframes excluded to avoid conflict with model config).
                        "max_pixels": 64 * 64,
                    },
                    {"type": "text", "text": "Describe what you see in one word."},
                ],
            },
        ]
        try:
            output = module.generate_text(
                messages=messages,
                generation_options={"max_new_tokens": 8, "do_sample": False},
            )
        except torch.cuda.OutOfMemoryError as ex:
            torch.cuda.empty_cache()
            pytest.skip(
                f"GPU OOM during video path — single V100/A6000 32GB limit "
                f"(80GB GPU or multi-GPU device_map='auto' required): {ex}"
            )
        assert isinstance(output.prediction, str)


class HuggingfaceAdapterCommonTests(
    _TextUnderstandingMixin,
    _ImageUnderstandingMixin,
    _AudioUnderstandingMixin,
    _VideoUnderstandingMixin,
):
    """HF adapter common smoke test base class.

    Class variables (overridden by subclasses):

        MODULE_CLS          : adapter module class                        [required]
        DEFAULT_MODEL_ID    : default (smallest) model ID                 [required]
        SUPPORTED_MODEL_IDS : allowed model ID whitelist                  [optional]

    Fixture source map:

        ── this class ──
        model_id, module

        ── tests/inference/huggingface/conftest.py ──
        hf_cache_dir, audio_message_bytes, video_message_path

        ── pytest builtin ──
        tmp_path_factory, request
    """

    MODULE_CLS: Optional[Type] = None
    DEFAULT_MODEL_ID: Optional[str] = None
    SUPPORTED_MODEL_IDS: Optional[List[str]] = None

    def __init_subclass__(cls, **kwargs):
        """On subclass creation, reads ENGINE_FEATURES and removes generate tests for unsupported modalities from collection."""
        super().__init_subclass__(**kwargs)
        module_cls = getattr(cls, "MODULE_CLS", None)
        features = getattr(module_cls, "ENGINE_FEATURES", None)
        if features is None:
            return
        for flag, method_name in _MODALITY_TEST_BY_FEATURE.items():
            if not features[flag] and method_name not in cls.__dict__:
                setattr(cls, method_name, None)

    # ── model selection / loading ──────────────────────────────────────

    @pytest.fixture(scope="class")
    def model_id(self) -> str:
        """`HF_MODEL_ID` env takes priority; falls back to `DEFAULT_MODEL_ID`; skips if outside `SUPPORTED_MODEL_IDS`."""
        if self.DEFAULT_MODEL_ID is None:
            pytest.fail(
                f"{type(self).__name__} must set DEFAULT_MODEL_ID class variable"
            )
        chosen = os.getenv(_HF_MODEL_ID_ENV) or self.DEFAULT_MODEL_ID
        if (
            self.SUPPORTED_MODEL_IDS is not None
            and chosen not in self.SUPPORTED_MODEL_IDS
        ):
            pytest.skip(
                f"{chosen!r} not in SUPPORTED_MODEL_IDS for "
                f"{type(self).__name__}: {self.SUPPORTED_MODEL_IDS}"
            )
        return chosen

    @pytest.fixture(scope="class", autouse=True)
    def _release_gpu_after_class(self):
        """Universal teardown that reclaims GPU memory/cache after all tests in the class complete."""
        yield
        _release_module_gpu_memory(None)

    @pytest.fixture(scope="class", autouse=True)
    def _skip_on_transformers_version(self, model_id):
        """Skips the entire class if the registry has a transformers version requirement for this model and it is out of range."""
        info = HF_EXAMPLE_MODELS.get(model_id)
        if info is not None:
            info.check_transformers_version(on_fail="skip")

    @pytest.fixture(scope="class")
    def module(self, model_id, hf_cache_dir, tmp_path_factory):
        """Adapter instance (class-scope, loaded once). Skips on ImportError/NameError; reclaims GPU memory in teardown."""
        if self.MODULE_CLS is None:
            pytest.fail(
                f"{type(self).__name__} must set MODULE_CLS class variable"
            )
        temp_dir = tmp_path_factory.mktemp(f"{self.MODULE_CLS.__name__}_module")
        try:
            adapter = self.MODULE_CLS(
                model_name_or_path=model_id,
                torch_dtype="float16",
                device_map="auto",
                cache_dir=str(hf_cache_dir),
                temp_dirpath=str(temp_dir),
            )
        except (ImportError, NameError) as ex:
            pytest.skip(
                f"Cannot import {self.MODULE_CLS.__name__} dependencies in this venv "
                f"({type(ex).__name__}: {ex})"
            )
        try:
            yield adapter
        finally:
            _release_module_gpu_memory(adapter)

    # ── common tests ──────────────────────────────────────────

    def test_loads(self, module):
        """After adapter instantiation, `model`/`tokenizer` are populated, and `processor` is also populated if non-text modalities are declared."""
        assert module.model is not None
        assert module.tokenizer is not None

        features = self.MODULE_CLS.ENGINE_FEATURES
        needs_processor = any(
            features[flag]
            for flag in (
                "support_image_understanding",
                "support_audio_understanding",
                "support_video_understanding",
            )
        )
        if needs_processor:
            assert module.processor is not None, (
                f"{type(module).__name__} ENGINE_FEATURES declares non-text understanding "
                f"modalities so processor is required"
            )


# ── adapter source sweep ────────────────────────────────────────────────


@pytest.mark.inference_engine("hf")
class TestAdapterReturnSweep:
    """Static (AST) sweep over every adapter source file in the adapters folder.

    Source-level guard that each adapter's engine-facing method yields a
    HuggingfaceInferenceOutput: engine.HuggingfaceInferencer.__call__ binds the return
    value to `output` and treats it as that dataclass (sets .latency, reads .prediction),
    so a regression to a bare dict / None / tuple would break the engine. AST-based, so it
    needs no GPU or optional deps and covers all adapters at once — unlike the live smokes
    above, which skip without a GPU.
    """

    _OUTPUT_CLS = "HuggingfaceInferenceOutput"
    # Methods whose return __call__ consumes as the output dataclass, plus the builders a
    # target may delegate to via `return self.<m>(...)`.
    _TARGET_METHODS = frozenset({"generate_text", "generate_audio", "generate_image", "compute_perplexity"})
    _BUILDER_METHODS = _TARGET_METHODS | {"_generate", "generate_video"}
    _ADAPTER_FILES = sorted(
        p for p in Path(adapters_pkg.__file__).parent.glob("*.py") if not p.name.startswith("_")
    )

    # ── AST helpers ─────────────────────────────────────────────────

    def _owned(self, funcdef, node_type):
        """Nodes of node_type lexically inside funcdef, excluding nested function/lambda scopes."""
        found, stack = [], list(funcdef.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, node_type):
                found.append(node)
            stack.extend(ast.iter_child_nodes(node))
        return found

    def _builds_output(self, value) -> bool:
        """True if value constructs a HuggingfaceInferenceOutput or delegates to a builder method."""
        if not isinstance(value, ast.Call):
            return False
        func = value.func
        return (isinstance(func, ast.Name) and func.id == self._OUTPUT_CLS) or (
            isinstance(func, ast.Attribute) and func.attr in self._BUILDER_METHODS
        )

    def _bad_return(self, method, ret) -> Optional[str]:
        """One-line reason this return is not provably a HuggingfaceInferenceOutput, or None if it is."""
        value = ret.value
        if self._builds_output(value):
            return None
        if isinstance(value, ast.Name):
            # `return <name>`: every non-None binding of the name must build the dataclass; a
            # `<name> = None` initializer that is always reassigned before the return is ignored.
            bound = [
                assign.value
                for assign in self._owned(method, ast.Assign)
                for target in assign.targets
                if isinstance(target, ast.Name) and target.id == value.id
            ]
            real = [v for v in bound if not (isinstance(v, ast.Constant) and v.value is None)]
            if real and all(self._builds_output(v) for v in real):
                return None
        is_none = value is None or (isinstance(value, ast.Constant) and value.value is None)
        kind = "None" if is_none else type(value).__name__
        return f"line {ret.lineno} {method.name}() returns {kind}, not {self._OUTPUT_CLS}"

    # ── tests ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("adapter_path", _ADAPTER_FILES, ids=lambda p: p.stem)
    def test_returns_output(self, adapter_path):
        """Every return in an adapter's generate_*/compute_perplexity yields a HuggingfaceInferenceOutput (or delegates to one)."""
        tree = ast.parse(adapter_path.read_text(), filename=str(adapter_path))
        violations = []
        for method in ast.walk(tree):
            if not (isinstance(method, ast.FunctionDef) and method.name in self._TARGET_METHODS):
                continue
            returns = self._owned(method, ast.Return)
            if not returns and not self._owned(method, ast.Raise):
                # No return is fine only for a raise-only override (a disabled modality).
                violations.append(f"line {method.lineno} {method.name}() neither returns nor raises")
            for ret in returns:
                reason = self._bad_return(method, ret)
                if reason is not None:
                    violations.append(reason)
        assert not violations, "non-{} returns in {}:\n  {}".format(
            self._OUTPUT_CLS, adapter_path.name, "\n  ".join(violations)
        )

    def test_not_vacuous(self):
        """The sweep locates many target methods overall — guards against a rename silently emptying it."""
        total = sum(
            sum(
                isinstance(n, ast.FunctionDef) and n.name in self._TARGET_METHODS
                for n in ast.walk(ast.parse(p.read_text(), filename=str(p)))
            )
            for p in self._ADAPTER_FILES
        )
        assert total >= 20, f"expected many target methods across adapters, found {total}"
