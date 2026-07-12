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

"""Unit tests for inference/huggingface/engine.py + Qwen2.5-Omni-3B live smoke."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from omni_evaluator.inference.huggingface import engine
from omni_evaluator.schemas.inference import HuggingfaceInferenceOutput
from tests.inference._smoke_inputs import (
    LIVE_SMOKE_IMAGE_QUERY,
    LIVE_SMOKE_MAX_NEW_TOKENS,
    LIVE_SMOKE_TEXT_QUERY,
    LIVE_SMOKE_VIDEO_QUERY,
)
from tests.inference.test_engine_common import EngineMainCommonTests

# ── Live smoke knobs (HF only) ───────────────────────────────────────────
_LIVE_SMOKE_DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-3B"
_LIVE_SMOKE_TIMEOUT_S = 600  # Generous timeout including model loading + inference + cleanup

# Default ENGINE_FEATURES mock value read by the perplexity path of engine.main.
_FAKE_ENGINE_FEATURES = {
    "support_audio_understanding": True,
    "support_image_understanding": True,
    "support_text_understanding": True,
    "support_video_understanding": True,
    "support_audio_generation": False,
    "support_text_generation": True,
    "support_image_generation": False,
    "support_video_generation": False,
    "support_compute_perplexity": True,
}


def _log_smoke_response(
    model_name_or_path: str, modality: str, query: str, pred, latency,
) -> None:
    """Prints the actual input/output of the live smoke to stdout (visible with `pytest -s`)."""
    latency_str = f"{latency:.3f}s" if isinstance(latency, (int, float)) else str(latency)
    print(
        f"\n  ── [HF live smoke / {modality}] ──"
        f"\n    model    : {model_name_or_path}"
        f"\n    query    : {query!r}"
        f"\n    pred     : {pred!r}"
        f"\n    latency  : {latency_str}"
    )


@pytest.mark.inference_engine("hf")
@pytest.mark.timeout(60)
class TestHfEngineMain(EngineMainCommonTests):
    """HF `engine.main()` unit tests + Qwen2.5-Omni-3B live smoke.

    Fixture source map:

        ── this class ──
        engine_module, main_kwargs_factory, patch_boundary, fake_inference_output

        ── tests/inference/conftest.py ──
        record_factory, image_record_factory, video_record_factory, task_config_factory

        ── pytest builtin ──
        monkeypatch
    """

    # ── baseline kwargs / fake output ─────────────────────────

    @pytest.fixture
    def engine_module(self):
        return engine

    @pytest.fixture
    def main_kwargs_factory(self):
        """Baseline kwargs builder for HF `engine.main()`."""
        def _factory(records, task_config, **overrides):
            base = dict(
                rank=0, world_size=1, run_index=0, num_runs=1,
                inference_engine="hf", evaluation_engine="builtin",
                benchmark="dummy", evaluation_method="generation",
                benchmark_idx=0, num_benchmarks=1,
                benchmark_dataset=records, task_config=task_config,
                default_generation_options={"max_new_tokens": 8},
                model_name_or_path="placeholder/model",
                debug=True, verbose=False,
            )
            base.update(overrides)
            return base
        return _factory

    @pytest.fixture
    def fake_inference_output(self):
        """HF response dict builder (includes prediction/reasoning_content/perplexities/tool_calls/latency)."""
        def _factory(prediction: str = "x") -> Dict[str, Any]:
            return {
                "prediction": prediction,
                "reasoning_content": None,
                "perplexities": None,
                "tool_calls": None,
                "latency": 0.0,
            }
        return _factory

    # ── boundary abstraction (base fixture override) ────────────────

    @pytest.fixture
    def patch_boundary(self, monkeypatch, fake_inference_output):
        """Monkeypatches `HuggingfaceInferencer` class with a fake and returns the call log."""
        def _patch(output_fn=None):
            if output_fn is None:
                def output_fn(idx, messages):
                    return fake_inference_output(prediction=f"pred-{idx}")
            calls: List[Dict[str, Any]] = []

            class _FakeInferencer:
                def __init__(self, *args, **kwargs):
                    self.module = SimpleNamespace(ENGINE_FEATURES=_FAKE_ENGINE_FEATURES)
                    self.init_kwargs = kwargs

                def __call__(self, messages, **kw):
                    idx = len(calls)
                    calls.append({"messages": messages, "kwargs": kw})
                    return output_fn(idx, messages)

            monkeypatch.setattr(engine, "HuggingfaceInferencer", _FakeInferencer)
            return calls
        return _patch

    # ── live smoke ──────────────────────────────────────────
    # Override class-level timeout(60) at method-level due to model loading time.

    def _run_hf_smoke(
        self,
        modality: str,
        records,
        query: str,
        main_kwargs_factory,
        task_config_factory,
    ) -> None:
        model_id = _LIVE_SMOKE_DEFAULT_MODEL
        out = engine.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=len(records)),
            model_name_or_path=model_id,
            default_generation_options={
                "max_new_tokens": LIVE_SMOKE_MAX_NEW_TOKENS,
                "do_sample": False,
            },
        ))
        assert isinstance(out, list) and len(out) == 1
        pred = out[0].get("prediction")
        _log_smoke_response(
            model_name_or_path=model_id,
            modality=modality, query=query, pred=pred,
            latency=out[0].get("latency"),
        )
        assert isinstance(pred, str) and len(pred) > 0

    @pytest.mark.skip(reason="Qwen2.5-Omni-3B model loading cost is high and image/video processing source bugs are unresolved — reactivate after dedicated environment is prepared")
    @pytest.mark.timeout(_LIVE_SMOKE_TIMEOUT_S)
    @pytest.mark.model_size("medium")
    @pytest.mark.requires_gpu
    @pytest.mark.requires_hf_token
    @pytest.mark.slow
    def test_default_smoke_text(
        self, main_kwargs_factory, record_factory, task_config_factory,
    ):
        """Text-only smoke with the default HF model — verifies the per-record loop runs to completion on an actual GPU model."""
        self._run_hf_smoke(
            modality="text",
            records=record_factory(n=1, query=LIVE_SMOKE_TEXT_QUERY),
            query=LIVE_SMOKE_TEXT_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )

    @pytest.mark.skip(reason="Qwen2.5-Omni-3B model loading cost is high and image/video processing source bugs are unresolved — reactivate after dedicated environment is prepared")
    @pytest.mark.timeout(_LIVE_SMOKE_TIMEOUT_S)
    @pytest.mark.model_size("medium")
    @pytest.mark.requires_gpu
    @pytest.mark.requires_hf_token
    @pytest.mark.slow
    def test_default_smoke_image(
        self, main_kwargs_factory, image_record_url_factory, task_config_factory,
    ):
        """Image+text smoke with the default HF model — verifies image fetch and processor conversion pass through to the boundary correctly."""
        self._run_hf_smoke(
            modality="image",
            records=image_record_url_factory(n=1, query=LIVE_SMOKE_IMAGE_QUERY),
            query=LIVE_SMOKE_IMAGE_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )

    @pytest.mark.skip(reason="Qwen2.5-Omni-3B model loading cost is high and image/video processing source bugs are unresolved — reactivate after dedicated environment is prepared")
    @pytest.mark.timeout(_LIVE_SMOKE_TIMEOUT_S)
    @pytest.mark.model_size("medium")
    @pytest.mark.requires_gpu
    @pytest.mark.requires_hf_token
    @pytest.mark.slow
    def test_default_smoke_video(
        self, main_kwargs_factory, video_record_url_factory, task_config_factory,
    ):
        """Video+text smoke with the default HF model — verifies the video frame stack reaches the boundary correctly."""
        self._run_hf_smoke(
            modality="video",
            records=video_record_url_factory(n=1, query=LIVE_SMOKE_VIDEO_QUERY),
            query=LIVE_SMOKE_VIDEO_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )


# ── HuggingfaceInferencer.__call__ ───────────────────────────────────────
# Output contract of the per-record inferencer, verified without loading a model
# (object.__new__ skips __init__; a fake module supplies the generate paths).
# The dict-like access engine.main relies on (output["prediction"]) is the
# SchemaInterface contract covered in tests/schemas — not re-verified here.


class _FakeModule:
    """Adapter stand-in returning a real HuggingfaceInferenceOutput so __call__'s post-processing runs against the true dataclass."""

    model = None  # read by HuggingfaceInferencer.__del__ during GC

    def __init__(self, features=None):
        self.ENGINE_FEATURES = dict(features or _FAKE_ENGINE_FEATURES)

    def generate_text(self, messages, generation_options, **kw):
        return HuggingfaceInferenceOutput(prediction="hi ")

    def compute_perplexity(self, messages, options, **kw):
        return HuggingfaceInferenceOutput(prediction=0)


@pytest.mark.inference_engine("hf")
class TestHuggingfaceInferencerCall:
    """`HuggingfaceInferencer.__call__` return contract and branch coverage.

    Fixture source map:

        ── this class ──
        inferencer
    """

    @pytest.fixture
    def inferencer(self):
        """Builds a HuggingfaceInferencer around a fake module without running __init__ (no model load)."""
        def _build(features=None):
            inf = object.__new__(engine.HuggingfaceInferencer)
            inf.module = _FakeModule(features)
            inf.model_group = None  # not used by the code paths under test
            inf.reasoning = False
            return inf
        return _build

    def test_generation_returns_output(self, inferencer):
        """generation/text branch returns a HuggingfaceInferenceOutput with latency set and prediction rstripped."""
        out = inferencer()(
            messages=[{"role": "user", "content": [{"type": "text", "text": "q"}]}],
            generation_options={"max_new_tokens": 4},
            evaluation_method="generation",
            output_modality=["text"],
        )
        assert isinstance(out, HuggingfaceInferenceOutput)
        assert out.prediction == "hi"  # "hi " → rstrip
        assert isinstance(out.latency, float)

    def test_perplexity_returns_output(self, inferencer):
        """perplexity branch returns a HuggingfaceInferenceOutput (generation post-processing skipped)."""
        out = inferencer()(
            messages=[{"role": "user", "content": [{"type": "text", "text": "q"}]}],
            options=["A", "B"],
            evaluation_method="perplexity",
        )
        assert isinstance(out, HuggingfaceInferenceOutput)
        assert isinstance(out.latency, float)

    def test_no_modality_branch_never_returns_none(self, inferencer):
        """generation that matches no output branch must raise, not silently return None."""
        inf = inferencer(dict(_FAKE_ENGINE_FEATURES, support_text_generation=False))
        with pytest.raises((AttributeError, RuntimeError)):
            inf(
                messages=[{"role": "user", "content": [{"type": "text", "text": "q"}]}],
                generation_options={"max_new_tokens": 4},
                evaluation_method="generation",
                output_modality=None,
            )
