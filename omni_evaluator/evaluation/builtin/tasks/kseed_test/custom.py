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
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence, Callable, Iterable

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record
)
from omni_evaluator.schemas.chat import (
    OcrToken, EntityToken,
    Message as ChatMessage, 
    AudioContent as ChatAudioContent,
    ImageContent as ChatImageContent,
    TextContent as ChatTextContent,
    VideoContent as ChatVideoContent,
)
from omni_evaluator.schemas.inference import Record
from omni_evaluator.schemas.task import TaskConfig, TaskInference, TaskInferenceGenerationOptions
from omni_evaluator.utils.data import format_task_prompt


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
    if "choice_a" in sample:
        if sample["choice_a"] is None:
            sample["choice_a"] = "None"
        options.append("A")
        option_contents.append(sample["choice_a"])
    if "choice_b" in sample:
        if sample["choice_b"] is None:
            sample["choice_b"] = "None"
        options.append("B")
        option_contents.append(sample["choice_b"])
    if (
        "choice_c" in sample
        and sample["choice_c"] is not None
    ):
        options.append("C")
        option_contents.append(sample["choice_c"])
    if (
        "choice_d" in sample
        and sample["choice_d"] is not None
    ):
        options.append("D")
        option_contents.append(sample["choice_d"])

    query = sample["question"]
    
    labels = sample["answer"]
    if isinstance(labels, str):
        labels = [labels, ]

    if (
        options 
        and option_contents
        and sample["answer"] in options
    ):
        query = f'{query}'
        for _option, _option_content in zip(options, option_contents):
            query += f'\n{_option}. {_option_content}'
        _option_index = options.index(sample["answer"])
        _option_content = option_contents[_option_index]
        labels.append(_option_content)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": query,
                },
            ]
        }
    ]
    
    sample_meta = {
        "category": sample["data_type"],
        "data_id": sample["data_id"],
        "data_type": sample["data_type"],
        "question_id": sample["question_id"],
        "question_type_id": sample["question_type_id"],
        "segment": sample["segment"],
        "question": sample["question"],
        "answer": sample["answer"],
        "choice_a": sample["choice_a"],
        "choice_b": sample["choice_b"],
        "choice_c": sample["choice_c"],
        "choice_d": sample["choice_d"],
    }

    record = default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample.get("index", sample_idx),
            "messages": messages,
            "options": options,
            "option_contents": option_contents,
            "label": labels,
            "meta": sample_meta,
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )
    return record