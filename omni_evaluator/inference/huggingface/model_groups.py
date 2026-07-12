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

"""Lightweight model-group resolution and dependency specs for the HuggingFace inference path.

This module intentionally imports only ``os`` and the enum so a caller can resolve
a model group (or its optional dependencies) without importing engine.py — which
pulls in torch and the ~30 adapter modules (~5.8 s).
"""
import os
from typing import List, Optional, Tuple

from omni_evaluator.enums.engine import HuggingfaceModelGroup

# (import_name, install_target, feature) — the SINGLE source of truth for each
# model group's optional dependencies. The runtime engine path
# (require_group_dependencies, called from engine.py before model load) reads from
# here, so a model's deps are declared in exactly one place. A drift-guard test
# (tests/test_model_groups_consistency.py) cross-checks the ``.[extra]`` targets
# against pyproject's [project.optional-dependencies].
# Groups present in this dict are supported by HuggingfaceInferencer; groups absent
# are unsupported.
MODULE_REQUIRED_PACKAGES: dict = {
    HuggingfaceModelGroup.ax4_vl:                    [],
    HuggingfaceModelGroup.kanana1_5_v:               [],
    HuggingfaceModelGroup.deepseek_vl:               [("deepseek_vl", ".[deepseek_vl]", "DeepSeek-VL")],
    HuggingfaceModelGroup.emu3:                      [("emu3", ".[emu3]", "Emu3")],
    HuggingfaceModelGroup.hyperclovax:               [],
    HuggingfaceModelGroup.hyperclovax_seed:          [],
    HuggingfaceModelGroup.hyperclovax_seed_vision:   [],
    HuggingfaceModelGroup.hyperclovax_seed_vision_v2: [],
    HuggingfaceModelGroup.janus:                     [("janus", ".[janus]", "Janus")],
    HuggingfaceModelGroup.janus_pro:                 [("janus", ".[janus]", "Janus-Pro")],
    HuggingfaceModelGroup.llava:                     [],
    HuggingfaceModelGroup.llava_onevision_hf:        [],
    HuggingfaceModelGroup.mini_cpm_o:                [],
    HuggingfaceModelGroup.phi4_multimodal:           [("qwen_omni_utils", "qwen-omni-utils", "Phi4-Multimodal")],
    HuggingfaceModelGroup.qwen2:                     [],
    HuggingfaceModelGroup.qwen2_audio_instruct:      [],
    HuggingfaceModelGroup.qwen2_audio:               [],
    HuggingfaceModelGroup.qwen2_omni:                [("qwen_omni_utils", "qwen-omni-utils", "Qwen2-Omni")],
    HuggingfaceModelGroup.qwen2_vl:                  [("qwen_vl_utils", "qwen-vl-utils", "Qwen2-VL")],
    HuggingfaceModelGroup.qwen3:                     [],
    HuggingfaceModelGroup.qwen3_omni:                [("qwen_omni_utils", "qwen-omni-utils", "Qwen3-Omni")],
    HuggingfaceModelGroup.qwen3_vl:                  [("qwen_vl_utils", "qwen-vl-utils", "Qwen3-VL")],
    HuggingfaceModelGroup.stable_diffusion_v1:       [],
    HuggingfaceModelGroup.vaetki_vl:                 [("qwen_vl_utils", "qwen-vl-utils", "VAETKI-VL")],
    HuggingfaceModelGroup.voxtral:                   [("mistral_common", '"mistral-common[audio]"', "Voxtral"),
                                                      ("qwen_omni_utils", "qwen-omni-utils", "Voxtral")],
    HuggingfaceModelGroup.whisper_v3:                [],
    HuggingfaceModelGroup.x_omni:                    [],
}


def require_group_dependencies(group: HuggingfaceModelGroup) -> None:
    """Raise ImportError (with an install hint) for any optional package the
    given model group needs but that is not importable.

    Single source of truth: :data:`MODULE_REQUIRED_PACKAGES`. Called by the
    runtime engine path (engine.py, before model load), so dependency
    declarations are never duplicated on the adapter classes.
    ``require_package`` uses ``find_spec`` only — no import, no download — so this
    stays cheap and side-effect free.
    """
    from omni_evaluator.utils.optional_import import require_package

    for spec in MODULE_REQUIRED_PACKAGES.get(group, []) or []:
        name = spec[0]
        extras = spec[1] if len(spec) > 1 else None
        feature = spec[2] if len(spec) > 2 else None
        require_package(name, extras=extras, feature=feature)


def get_model_group(model_name_or_path: str) -> HuggingfaceModelGroup:
    """Map a model path/name to HuggingfaceModelGroup using pure string rules.

    Extracted here (rather than on HuggingfaceInferencer) so callers can resolve a
    model group without importing engine.py and its ~30 adapter modules.
    """
    import logging
    p = model_name_or_path.lower()

    # a.x requires exact variant match — raise early to prevent spurious fallthrough
    if "a.x" in p:
        if "4" in p and "vl" in p:
            return HuggingfaceModelGroup.ax4_vl
        raise ValueError(f'Not implemented model_name_or_path: {model_name_or_path}')

    # Ordered rules: first match wins. More specific rules must come before general ones.
    _RULES = [
        # kanana family
        (lambda p: "kanana" in p,                                                                        HuggingfaceModelGroup.kanana1_5_v),
        # emu family
        (lambda p: "emu" in p and "3" in p,                                                              HuggingfaceModelGroup.emu3),
        (lambda p: "emu" in p,                                                                           HuggingfaceModelGroup.emu),
        # llava family (hf variants before bare llava)
        (lambda p: "llava" in p and ("huggingface" in p or p.endswith("-hf")) and "onevision" in p,      HuggingfaceModelGroup.llava_onevision_hf),
        (lambda p: "llava" in p and ("huggingface" in p or p.endswith("-hf")),                           HuggingfaceModelGroup.llava_hf),
        (lambda p: "llava" in p,                                                                         HuggingfaceModelGroup.llava),
        # aya → deepseek (must precede generic deepseek check)
        (lambda p: "aya" in p,                                                                           HuggingfaceModelGroup.deepseek),
        # deepseek/janus family (janus before deepseek-vl before deepseek)
        (lambda p: "deepseek" in p and "janus" in p and "pro" in p,                                     HuggingfaceModelGroup.janus_pro),
        (lambda p: "deepseek" in p and "janus" in p,                                                    HuggingfaceModelGroup.janus),
        (lambda p: "deepseek" in p and "vl" in p,                                                       HuggingfaceModelGroup.deepseek_vl),
        (lambda p: "deepseek" in p,                                                                      HuggingfaceModelGroup.deepseek),
        # gemma family
        (lambda p: "gemma" in p and "pali" in p,                                                        HuggingfaceModelGroup.paligemma),
        (lambda p: "gemma" in p,                                                                         HuggingfaceModelGroup.gemma),
        # qwen2/2.5 family (before qwen3 and generic qwen)
        (lambda p: ("qwen2" in p or "qwen2.5" in p) and "audio" in p and "instruct" in p,               HuggingfaceModelGroup.qwen2_audio_instruct),
        (lambda p: ("qwen2" in p or "qwen2.5" in p) and "audio" in p,                                   HuggingfaceModelGroup.qwen2_audio),
        (lambda p: ("qwen2" in p or "qwen2.5" in p) and "omni" in p,                                    HuggingfaceModelGroup.qwen2_omni),
        (lambda p: ("qwen2" in p or "qwen2.5" in p) and "vl" in p,                                      HuggingfaceModelGroup.qwen2_vl),
        (lambda p: "qwen2" in p or "qwen2.5" in p,                                                      HuggingfaceModelGroup.qwen2),
        # qwen3 family (before generic qwen)
        (lambda p: "qwen3" in p and "omni" in p,                                                        HuggingfaceModelGroup.qwen3_omni),
        (lambda p: "qwen3" in p and "vl" in p,                                                          HuggingfaceModelGroup.qwen3_vl),
        (lambda p: "qwen3" in p,                                                                         HuggingfaceModelGroup.qwen3),
        # generic qwen family
        (lambda p: "qwen" in p and "audio" in p,                                                        HuggingfaceModelGroup.qwen_audio),
        (lambda p: "qwen" in p and "omni" in p,                                                         HuggingfaceModelGroup.qwen_omni),
        (lambda p: "qwen" in p and "vl" in p,                                                           HuggingfaceModelGroup.qwen_vl),
        (lambda p: "qwen" in p,                                                                          HuggingfaceModelGroup.qwen),
        # intern family
        (lambda p: "intern" in p and "vl" in p,                                                         HuggingfaceModelGroup.intern_vl),
        (lambda p: "intern" in p,                                                                        HuggingfaceModelGroup.intern_lm),
        # llama family
        (lambda p: "llama" in p and "vision" in p,                                                      HuggingfaceModelGroup.llama_vision),
        (lambda p: "llama" in p and "llama-3" in p,                                                     HuggingfaceModelGroup.llama_3),
        (lambda p: "llama" in p,                                                                         HuggingfaceModelGroup.llama),
        # minicpm family
        (lambda p: "minicpm" in p and ("minicpm-o" in p or "minicpm_o" in p),                           HuggingfaceModelGroup.mini_cpm_o),
        (lambda p: "minicpm" in p,                                                                       HuggingfaceModelGroup.mini_cpm),
        # phi family
        (lambda p: "phi" in p and "4" in p and "multimodal" in p,                                       HuggingfaceModelGroup.phi4_multimodal),
        (lambda p: "phi" in p and "4" in p,                                                              HuggingfaceModelGroup.phi4),
        (lambda p: "phi" in p,                                                                           HuggingfaceModelGroup.phi),
        # hyperclovax family (most specific first)
        (lambda p: "hyperclovax" in p and "seed" in p and "vision" in p and ("32b" in p or "4b" in p),  HuggingfaceModelGroup.hyperclovax_seed_vision_v2),
        (lambda p: "hyperclovax" in p and "seed" in p and "vision" in p,                                 HuggingfaceModelGroup.hyperclovax_seed_vision),
        (lambda p: "hyperclovax" in p and "seed" in p,                                                   HuggingfaceModelGroup.hyperclovax_seed),
        (lambda p: "hyperclovax" in p and "vision" in p,                                                 HuggingfaceModelGroup.hyperclovax_vision),
        (lambda p: "hyperclovax" in p,                                                                   HuggingfaceModelGroup.hyperclovax),
        # stable diffusion family
        (lambda p: "stable" in p and "diffusion" in p and "1" in p,                                     HuggingfaceModelGroup.stable_diffusion_v1),
        (lambda p: "stable" in p and "diffusion" in p,                                                   HuggingfaceModelGroup.stable_diffusion),
        # vaetki
        (lambda p: "vaetki" in p,                                                                        HuggingfaceModelGroup.vaetki_vl),
        # audio/speech models
        (lambda p: "voxtral" in p,                                                                       HuggingfaceModelGroup.voxtral),
        (lambda p: "whisper" in p and "v3" in p,                                                        HuggingfaceModelGroup.whisper_v3),
        (lambda p: "whisper" in p,                                                                       HuggingfaceModelGroup.whisper),
        # omni
        (lambda p: "x-omni" in p,                                                                       HuggingfaceModelGroup.x_omni),
    ]

    for predicate, group in _RULES:
        if predicate(p):
            return group

    raise ValueError(f'Not implemented model_name_or_path: {model_name_or_path}')
