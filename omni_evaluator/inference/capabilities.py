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

from typing import Dict

from omni_evaluator.enums.engine import InferenceEngine
from omni_evaluator.schemas.inference import InferenceEngineFeatures


# ── Engine-level capability table ──────────────────────────────────────────
# Protocol-level support flags per inference engine. Used to gate benchmarks
# whose evaluation_method the engine cannot satisfy (e.g. perplexity on API
# endpoints that do not expose prompt_logprobs). HuggingFace modules add a
# module-level override on top of this (see each module's ENGINE_FEATURES) —
# the effective capability is the AND of engine-level and module-level flags.
#
# Defined as a leaf module (no transitive postprocess/schemas/evaluation
# imports) so callers can import it without triggering omni_evaluator package
# init's heavy chain. This is what avoids the
#   schemas/evaluation → postprocess → schemas/evaluation
# circular import that surfaces when inference/__init__.py touches schemas.
INFERENCE_ENGINE_FEATURES: Dict[InferenceEngine, InferenceEngineFeatures] = {
    # huggingface: per-module compute_perplexity is the real source of truth;
    # engine-level stays True so the module-level override decides.
    InferenceEngine.huggingface: InferenceEngineFeatures(
        support_audio_understanding=True,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=True,
        support_audio_generation=True,
        support_text_generation=True,
        support_compute_perplexity=True,
    ),
    # vllm: prompt_logprobs available → ppl path exists
    # (see inference/vllm/completions.py:327-399).
    InferenceEngine.vllm: InferenceEngineFeatures(
        support_audio_understanding=True,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=True,
        support_audio_generation=False,
        support_text_generation=True,
        support_compute_perplexity=True,
    ),
    # sglang: same prompt_logprobs pattern as vllm.
    InferenceEngine.sglang: InferenceEngineFeatures(
        support_audio_understanding=True,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=True,
        support_audio_generation=False,
        support_text_generation=True,
        support_compute_perplexity=True,
    ),
    # OpenAI / Anthropic / Google chat APIs do not expose prompt-logprobs,
    # so perplexity-mode tasks must be skipped (not silently fall back to
    # generation, which would yield empty perplexities lists).
    InferenceEngine.api__openai: InferenceEngineFeatures(
        support_audio_understanding=True,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=False,
        support_audio_generation=False,
        support_text_generation=True,
        support_compute_perplexity=False,
    ),
    InferenceEngine.api__anthropic: InferenceEngineFeatures(
        support_audio_understanding=False,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=False,
        support_audio_generation=False,
        support_text_generation=True,
        support_compute_perplexity=False,
    ),
    InferenceEngine.api__google: InferenceEngineFeatures(
        support_audio_understanding=True,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=True,
        support_audio_generation=False,
        support_text_generation=True,
        support_compute_perplexity=False,
    ),
}
