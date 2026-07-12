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

from typing import Dict, Optional

from omni_evaluator.inference.prompts import SYSTEM_PROMPTS


def get_system_prompt(
    task_name: str,
    system_prompt_map: Dict[str, str],
) -> Optional[str]:
    system_prompt = None
    if (
        task_name in system_prompt_map
        and system_prompt_map[task_name] in SYSTEM_PROMPTS
    ):
        system_prompt = SYSTEM_PROMPTS[system_prompt_map[task_name]]
    return system_prompt
