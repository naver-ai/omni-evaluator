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

"""Input constants and config paths shared by all inference engine live smokes. Engine-specific constants live in each `test_engine.py`."""
from __future__ import annotations

from pathlib import Path


# ── Shared paths ────────────────────────────────────────────────
# tests/configs/inference/ — collection of endpoint configs such as api_*.yaml / vllm_v*.yaml.
INFERENCE_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "inference"


# ── Live smoke knobs ────────────────────────────────────────────
# One-liner query per modality — validation only checks for "non-empty string".

LIVE_SMOKE_TEXT_QUERY = "Reply with the single word: pong"
LIVE_SMOKE_IMAGE_QUERY = "Describe this image in one word."
LIVE_SMOKE_VIDEO_QUERY = "Describe this video in one word."

# Token limit — one word from smoke is enough. All engines use the same limit.
LIVE_SMOKE_MAX_NEW_TOKENS = 8

# Google-only token limit — thinking models (gemini-2.5-flash) return empty responses after exhausting reasoning tokens, so extra headroom is reserved.
LIVE_SMOKE_GOOGLE_MAX_NEW_TOKENS = 512
