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

import json
import os
import string
from typing import Dict, Any, Optional

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)
from omni_evaluator.utils.data import extract_options
from omni_evaluator.utils.multimodal import to_audio_bytes


OPTIONS = list(string.ascii_uppercase)


def _parse_other_attributes(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


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
    _other_attributes = _parse_other_attributes(sample.get("other_attributes"))

    # `choices` may live in either the top-level row or `other_attributes`.
    _choices = sample.get("choices") or _other_attributes.get("choices") or []
    if isinstance(_choices, str):
        _choices = [_choices]

    options, option_contents = extract_options(text=", ".join(_choices))
    if not options:
        option_contents = list(_choices)
        options = OPTIONS[:len(option_contents)]

    label = sample.get("answer") or _other_attributes.get("answer")
    label_contents = list()
    if isinstance(label, str):
        _, label_contents = extract_options(text=label)
    # Expand label to [content, full] forms only. The option index (e.g. "A")
    # is intentionally excluded because this task uses `mmau_string_match`,
    # which does token-subset matching: a one-character label like "A" tokenizes
    # to {"a"} and would be a subset of almost any prediction containing that
    # letter — producing false positives. The same caveat applies to any task
    # that uses `string_match` (substring match) or `substring_match`.
    labels = list()
    if label_contents and label_contents[0]:
        labels.append(label_contents[0])
    labels.append(label)
    instruction_text = sample.get("instruction") or _other_attributes.get("instruction") or sample.get("question")

    # paper-strict (aligned with lmms-eval mmau/utils.py:22):
    #   `{instruction}\n{choices}` only — removes the "Question:" / "Choices:" keyword wrap.
    #   choices are also split by newline instead of by space.
    choices_text = "\n".join([
        f"{_option}. {_option_content}"
        for _option, _option_content in zip(options, option_contents)
    ])
    query = f'{instruction_text}\n{choices_text}'
    audio_bytes = to_audio_bytes(sample["context"])
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "value": audio_bytes,
                },
                {
                    "type": "text",
                    "value": query,
                },
            ],
        }
    ]

    sample_meta = {
        "id": sample.get("id"),
        "instruction": instruction_text,
        "answer": label,
        "choices": option_contents,
        "task": sample.get("task") or _other_attributes.get("task"),
        "category": sample.get("category") or _other_attributes.get("category"),
        "sub-category": sample.get("sub-category") or _other_attributes.get("sub-category"),
        "difficulty": sample.get("difficulty") or _other_attributes.get("difficulty"),
    }

    record = default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample.get("id", sample_idx),
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
