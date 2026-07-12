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

"""Registry that centrally manages per-model transformers version compatibility ranges for HF adapter tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import pytest


@dataclass(frozen=True)
class _HfExamplesInfo:
    """Transformers version compatibility range for a single model (min/max both inclusive, None means unconstrained)."""

    min_transformers_version: Optional[str] = None
    max_transformers_version: Optional[str] = None
    transformers_version_reason: Optional[str] = None

    def check_transformers_version(
        self,
        *,
        on_fail: Literal["error", "skip", "return"] = "skip",
    ) -> Optional[str]:
        """If the installed transformers version is outside the required range, performs the `on_fail` action (skip/error/return); returns None if within range."""
        if (
            self.min_transformers_version is None
            and self.max_transformers_version is None
        ):
            return None
        try:
            import transformers
            from packaging.version import Version
        except Exception:
            return None

        current = transformers.__version__
        cur = Version(Version(current).base_version)
        if self.min_transformers_version and cur < Version(
            self.min_transformers_version
        ):
            bound = f">={self.min_transformers_version}"
        elif self.max_transformers_version and cur > Version(
            self.max_transformers_version
        ):
            bound = f"<={self.max_transformers_version}"
        else:
            return None

        msg = f"transformers=={current} installed; this model requires transformers{bound}"
        if self.transformers_version_reason:
            msg += f" ({self.transformers_version_reason})"

        if on_fail == "error":
            raise RuntimeError(msg)
        elif on_fail == "skip":
            pytest.skip(msg)
        return msg


# ── Per-model-family metadata ────────────────────────────────────────────
# You must hard code the min/max transformers versions for each model family here, because the remote config (config.json) is not guaranteed to be available at test time (e.g. if the model is private or deleted).
_HCX_VISION_INFO = _HfExamplesInfo(
    min_transformers_version="4.52",
    max_transformers_version="4.57.1",
    transformers_version_reason=(
        "HCX-SEED-Vision remote config requires transformers 4.52.x "
        "(in 5.x, HCXVisionConfig.text_config changed to get_text_config(), making it incompatible). "
        "config.json transformers_version=4.52.4"
    ),
)
_EMU3_INFO = _HfExamplesInfo(
    min_transformers_version="4.44",
    max_transformers_version="4.44.2",
    transformers_version_reason=(
        "Emu3 remote model code (modeling_emu3.py) depends on legacy transformers API"
        "(is_torch_fx_available, DynamicCache.seen_tokens) — removed in 5.x. "
        "transformers 4.44 only (setup-env references/emu3.md: tf4.44 + gradio + hub==0.36.2)."
    ),
)
_JANUS_PRO_INFO = _HfExamplesInfo(
    min_transformers_version="4.44",
    max_transformers_version="4.44.2",
    transformers_version_reason=(
        "Janus package is compatible with transformers 4.44 "
        "(setup-env references/janus.md: .omni_janus pins transformers==4.44.0)."
    ),
)
_DEEPSEEK_VL_INFO = _HfExamplesInfo(
    min_transformers_version="4.44",
    max_transformers_version="4.44.2",
    transformers_version_reason=(
        "DeepSeekVlModule(JanusModule) → Janus package's transformers 4.44 dependency "
        "(setup-env references/janus.md)."
    ),
)


# ── Model name → metadata ───────────────────────────────────────────
# Only models with version constraints are registered (unregistered → unconstrained).
HF_EXAMPLE_MODELS: Dict[str, _HfExamplesInfo] = {
    # HyperCLOVAX-SEED-Vision (shared checkpoint for v1 / v2 adapters)
    "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B": _HCX_VISION_INFO,
    # Emu3 (tf4.44 only)
    "BAAI/Emu3-Chat": _EMU3_INFO,
    "BAAI/Emu3-Gen": _EMU3_INFO,
    "BAAI/Emu3-Stage1": _EMU3_INFO,
    # Janus-Pro (tf4.44 only)
    "deepseek-ai/Janus-Pro-1B": _JANUS_PRO_INFO,
    "deepseek-ai/Janus-Pro-7B": _JANUS_PRO_INFO,
    # DeepSeek-VL (tf4.44 only)
    "deepseek-ai/deepseek-vl-1.3b-chat": _DEEPSEEK_VL_INFO,
    "deepseek-ai/deepseek-vl-7b-chat": _DEEPSEEK_VL_INFO,
}
