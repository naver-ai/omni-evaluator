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

import random
import string
from typing import Dict, Any, Optional

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)
from omni_evaluator.evaluation.resources.prompts.multichoice.en import (
    voice_sound as multichoice_instructions,
)


OPTION_CONTENTS = [
    "cough",
    "laughter",
    "sigh",
    "sneeze",
    "sniff",
    "throat clearing",
]
OPTIONS = list(string.ascii_uppercase)[:len(OPTION_CONTENTS)]


def sample_to_record(
    task_name: str,
    task_config: Dict[str, Any],
    sample_idx: int,
    sample: Dict[str, Any],
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_entity_tokens: Optional[int] = None,
    run_index: Optional[int] = 0,
    **kwargs,
):
    _answer = sample["answer"]
    labels = list()
    for _option, _option_content in zip(OPTIONS, OPTION_CONTENTS):
        if _answer.lower() == _option_content.lower():
            labels.append(_option)
            labels.append(_option_content)
            break

    # paper-strict alignment (partial): removes the "Choices:" keyword wrap + splits
    # choices by newline (lmms-eval audio MCQ pattern). The random pick from
    # multichoice_instructions + explicitly listing choices is kept as our
    # generation-based scoring intent (lmms-eval vocalsound uses multiple-choice
    # scoring via the datasets.choices feature, so choices don't go in the prompt —
    # a different task type).
    choices = "\n".join([
        f"{_option}. {_option_content}"
        for _option, _option_content in zip(OPTIONS, OPTION_CONTENTS)
    ])
    query = f"{random.choice(multichoice_instructions)}\n{choices}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "value": sample["audio"]["bytes"],
                },
                {
                    "type": "text",
                    "value": query,
                },
            ],
        }
    ]

    sample_meta = {
        "spk_id": sample.get("spk_id"),
        "age_group": sample.get("age_group"),
        "answer": _answer,
    }

    record = default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample_idx,
            "messages": messages,
            "label": labels,
            "options": OPTIONS,
            "option_contents": OPTION_CONTENTS,
            "meta": sample_meta,
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )
    return record
