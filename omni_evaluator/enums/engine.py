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

from enum import Enum


class ApiGroup(str, Enum):
    openai: str = "openai"
    anthropic: str = "anthropic"
    google: str = "google"


class EvaluationEngine(str, Enum):
    builtin: str = "builtin"
    lmms_eval: str = "lmms_eval"
    lm_eval_harness: str = "lm_eval_harness"
    vlm_eval_kit: str = "vlm_eval_kit"


class EvaluationMethod(str, Enum):
    generation: str = "generation"
    perplexity: str = "perplexity"


class InferenceEngine(str, Enum):
    huggingface: str = "huggingface"
    llama_cpp: str = "llama_cpp"
    vllm: str = "vllm"
    sglang: str = "sglang"
    api__openai: str = "api/openai"
    api__anthropic: str = "api/anthropic"
    api__google: str = "api/google"


class HuggingfaceModelGroup(str, Enum):
    ax4_vl: str = "ax4_vl"
    deepseek: str = "deepseek"
    kanana1_5_v: str = "kanana1_5_v"
    deepseek_vl: str = "deepseek_vl"
    emu: str = "emu"
    emu3: str = "emu3"
    gemma: str = "gemma"
    hyperclovax: str = "hyperclovax"
    hyperclovax_seed: str = "hyperclovax_seed"
    hyperclovax_seed_vision: str = "hyperclovax_seed_vision"
    hyperclovax_seed_vision_v2: str = "hyperclovax_seed_vision_v2"
    hyperclovax_vision: str = "hyperclovax_vision"
    intern_lm: str = "intern_lm"
    intern_vl: str = "intern_vl"
    janus: str = "janus"
    janus_pro: str = "janus_pro"
    llama: str = "llama"
    llama_3: str = "llama_3"
    llama_vision: str = "llama_vision"
    llava: str = "llava"
    llava_hf: str = "llava_hf"
    llava_onevision_hf: str = "llava_onevision_hf"
    mini_cpm: str = "mini_cpm"
    mini_cpm_o: str = "mini_cpm_o"
    paligemma: str = "paligemma"
    phi: str = "phi"
    phi4: str = "phi4"
    phi4_multimodal: str = "phi4_multimodal"
    qwen: str = "qwen"
    qwen2: str = "qwen2"
    qwen2_audio: str = "qwen2_audio"
    qwen2_audio_instruct: str = "qwen2_audio_instruct"
    qwen2_omni: str = "qwen2_omni"
    qwen2_vl: str = "qwen2_vl"
    qwen3: str = "qwen3"
    qwen3_omni: str = "qwen3_omni"
    qwen3_vl: str = "qwen3_vl"
    qwen_audio: str = "qwen_audio"
    qwen_omni: str = "qwen_omni"
    qwen_vl: str = "qwen_vl"
    stable_diffusion: str = "stable_diffusion"
    stable_diffusion_v1: str = "stable_diffusion_v1"
    vaetki_vl: str = "vaetki_vl"
    voxtral: str = "voxtral"
    whisper: str = "whisper"
    whisper_v3: str = "whisper_v3"
    x_omni: str = "x_omni"


class T2IGeneratorType(str, Enum):
    ta_tok: str = "ta_tok"
    hyperclova_vdm: str = "hyperclova_vdm"
