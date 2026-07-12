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

import re
from typing import Dict, Any, Optional

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)


# Matches `(A) some content` style choices and answer.
_CHOICE_PATTERN = r"\(([A-Z])\)\s*([^()]+)"


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
    options, option_contents = list(), list()
    for _option, _option_content in re.findall(_CHOICE_PATTERN, sample["choices"]):
        options.append(_option)
        option_contents.append(_option_content.strip())

    labels = list()
    for _option, _option_content in re.findall(_CHOICE_PATTERN, sample["answer"]):
        labels.append(_option)
        labels.append(_option_content.strip())
    labels.append(sample["answer"])

    # paper-strict (aligned with lmms-eval muchomusic/utils.py:13):
    #   `{instruction}\n{choices}` only — removes the "Question:" / "Choices:" keyword wrap.
    query = f'{sample["instruction"]}\n{sample["choices"]}'

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "value": sample["context"]["bytes"],
                },
                {
                    "type": "text",
                    "value": query,
                },
            ],
        }
    ]

    sample_meta = {
        "instruction": sample["instruction"],
        "choices": sample["choices"],
        "answer": sample["answer"],
    }

    record = default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample_idx,
            "messages": messages,
            "label": labels,
            "options": options,
            "option_contents": option_contents,
            "meta": sample_meta,
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )
    return record
