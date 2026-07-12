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

from typing import Any, Dict, Optional

from omni_evaluator.utils.common import healthcheck as _healthcheck


# Engine-level capability table — defined in a leaf module to avoid the
# schemas/evaluation → postprocess → schemas/evaluation circular import that
# surfaces when this package's __init__ touches schemas. Callers import:
#   from omni_evaluator.inference.capabilities import INFERENCE_ENGINE_FEATURES
# (do NOT re-export here — keeping this __init__ lean is what breaks the cycle.)


# config: common
DEFAULT_NUM_RUNS: int = 1
NUM_DEBUG_SAMPLES: int = 3
DEFAULT_MULTIPLE_CHOICE_OPTIONS = ["A", "B", "C", "D"]

# config: request
TIMEOUT: int = 60
SOCKET_TIMEOUT: int = 900
MAX_RETRY: int = 5
WAIT_BETWEEN_RETRY: int = 10

# config: async
NUM_MAX_COROUTINES: int = 64


def healthcheck(
    url: str,
    max_retries: int = 20,
    interval: int = 30,
    token: Optional[str] = None,
):
    healthcheck_url = f'{url}/health'
    return _healthcheck(
        url=healthcheck_url,
        max_retries=max_retries,
        interval=interval,
        token=token,
    )


def apply_chat_template_kwargs(
    generation_options: Dict[str, Any],
    reasoning: Optional[bool] = False,
    add_generation_prompt: Optional[bool] = None,
) -> Dict[str, Any]:
    if not reasoning:
        if not generation_options.get("chat_template_kwargs"):
            generation_options["chat_template_kwargs"] = dict()
        if "skip_reasoning" not in generation_options["chat_template_kwargs"]:
            generation_options["chat_template_kwargs"]["skip_reasoning"] = True
    if add_generation_prompt:
        if not generation_options.get("chat_template_kwargs"):
            generation_options["chat_template_kwargs"] = dict()
        generation_options["chat_template_kwargs"]["add_generation_prompt"] = True
    return generation_options