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

import string
from typing import Dict, Any, Optional

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)
from omni_evaluator.utils.data import extract_options


OPTIONS = list(string.ascii_uppercase)


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
    option_contents = list(sample["choices"]) if sample.get("choices") else []
    options = OPTIONS[:len(option_contents)] if option_contents else []
    label = sample["answer"]
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

    # paper-strict (aligned with lmms-eval mmau/utils.py:22):
    #   `{question}\n{choices}` only — removes the "Question:" / "Choices:" keyword wrap.
    #   choices are also split by newline instead of by space.
    choices_text = "\n".join([
        f"{_option}. {_option_content}"
        for _option, _option_content in zip(options, option_contents)
    ])
    query = f'{sample["question"]}\n{choices_text}'

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "value": sample["audio_id"],
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
        "audio_id": sample["audio_id"],
        "question": sample["question"],
        "answer": sample["answer"],
        "choices": option_contents,
        "dataset": sample.get("dataset"),
        "task": sample.get("task"),
        "split": sample.get("split"),
        "category": sample.get("category"),
        "sub-category": sample.get("sub-category"),
        "difficulty": sample.get("difficulty"),
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
