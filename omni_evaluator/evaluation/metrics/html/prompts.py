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

"""Stage-1 conversion prompts for the HTML tree-edit metric.

``PROMPTS`` maps ``source_format`` → ``PromptSet``. yaml only carries the
``source_format`` (e.g. ``source_format: "latex"``); the prompt strings themselves
stay here as module constants.

``equation_*`` entries are optional. When per-sample ``group == "equation"``
and ``equation_user_prompt_template`` is set, the equation prompt overrides
the default one for that sample (legacy latex semantics).
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class PromptSet:
    system_prompt: str
    user_prompt_template: str  # ``{pred}`` placeholder, .format(pred=…)
    equation_system_prompt: Optional[str] = None
    equation_user_prompt_template: Optional[str] = None


_SYSTEM_PROMPT_LATEX = "\nYou are a helpful and precise assistant for making given latex's grammer to html's grammer.\n"
_USER_PROMPT_TEMPLATE_LATEX = """
Just output the html's grammer. Do not contain additional content. Return only the full code in <html></html> tags.
Given latex is below.

{pred}
"""
_EQUATION_SYSTEM_PROMPT_LATEX = "\nYou are a helpful and precise assistant for making given latex's grammer to mathml's grammer.\n"
_EQUATION_USER_PROMPT_TEMPLATE_LATEX = """
Just output the mathml's grammer. Do not contain additional content. Return only the full code in <math></math> tags.
Given latex is below.

{pred}
"""

_SYSTEM_PROMPT_MARKDOWN = "\nYou are a helpful and precise assistant for making given markdown's grammer to html's grammer.\n"
_USER_PROMPT_TEMPLATE_MARKDOWN = """
Just output the html's grammer. Do not use rowspan and colspan. Do not contain additional content. Return only the full code in <html></html> tags.
Given markdown is below.

{pred}
"""


PROMPTS: Dict[str, PromptSet] = {
    "latex": PromptSet(
        system_prompt=_SYSTEM_PROMPT_LATEX,
        user_prompt_template=_USER_PROMPT_TEMPLATE_LATEX,
        equation_system_prompt=_EQUATION_SYSTEM_PROMPT_LATEX,
        equation_user_prompt_template=_EQUATION_USER_PROMPT_TEMPLATE_LATEX,
    ),
    "markdown": PromptSet(
        system_prompt=_SYSTEM_PROMPT_MARKDOWN,
        user_prompt_template=_USER_PROMPT_TEMPLATE_MARKDOWN,
    ),
}
