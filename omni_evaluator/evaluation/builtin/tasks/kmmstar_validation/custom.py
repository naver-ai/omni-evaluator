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

import ast
import json
import re
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
from omni_evaluator.utils.data import format_task_prompt, extract_options


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
    messages = list()
    _query = sample["question"]
    _user_contents = list()
    _prev_end_idx = 0
    for _match in re.finditer("<image>", _query):
        _cur_query = _query[_prev_end_idx:_match.start()]
        _cur_query = _cur_query.strip()
        if _cur_query:
            _user_contents.append(dict(
                type="text",
                text=_cur_query,
            ))
        _user_contents.append(dict(
            type="image",
            image=sample["image"],
        ))
        _prev_end_idx = _match.end()
    _user_contents.append(dict(
        type="text",
        text=_query[_prev_end_idx:].strip(),
    ))
    messages.append(dict(
        role="user", 
        content=_user_contents,
    ))

    labels = sample["answer"]
    if isinstance(labels, str):
        labels = [labels, ]
    
    options, option_contents = extract_options(text=sample["question"])
    if len(set(options)) < len(option_contents):
        options = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:len(option_contents)]

    if (
        options 
        and option_contents
        and sample["answer"] in options
    ):
        _option_index = options.index(sample["answer"])
        _option_content = option_contents[_option_index]
        labels.append(_option_content)

    sample_meta = ast.literal_eval(sample["meta_info"])
    sample_meta.update({
        "category": sample["category"],
        "l2_category": sample["l2_category"],
        "index": sample["index"],
        "question": sample["question"],
        "answer": sample["answer"],
    })

    record = default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample["index"],
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