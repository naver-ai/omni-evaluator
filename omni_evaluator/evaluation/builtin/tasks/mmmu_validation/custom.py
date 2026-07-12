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

"""MMMU validation dataset preparation.

The HF MMMU/MMMU schema (`id, question, options, explanation, answer, img_type,
question_type, topic_difficulty, subfield, image_1, ..., image_7`) embeds image
placeholders in the question text using the literal pattern `<image N>` (1-indexed,
single space). Each placeholder is expanded into a `ChatImageContent` referencing
the corresponding `image_N` PIL column. For multiple-choice questions the option
list is appended as `A. ... / B. ...` text. `question_type` is forwarded via meta
so that the downstream `mmmu_accuracy` metric can route MC vs open evaluation.
"""
import ast
import logging
import re
import string
from typing import Any, Dict, List, Optional, Union

from omni_evaluator.schemas.chat import (
    Message as ChatMessage,
    TextContent as ChatTextContent,
    ImageContent as ChatImageContent,
)
from omni_evaluator.schemas.inference import Record

logger = logging.getLogger(__name__)

IMAGE_PLACEHOLDER = re.compile(r"<image\s+(\d+)>")


def sample_to_record(
    task_name: str,
    task_config,
    sample_idx: int,
    sample: Dict[str, Any],
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    task_prompt_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[Record, List[Record]]:
    question_raw = sample.get("question", "") or ""
    # Normalize HF raw value ("multiple-choice") to ours convention ("multiple_choice").
    # Aligns with SubtaskType enum and lets postprocess.conditional_on key use the same form.
    question_type = (sample.get("question_type", "") or "").replace("-", "_")
    answer_raw = sample.get("answer", "")
    sample_id = sample.get("id", str(sample_idx))

    option_contents_raw = sample.get("options") or []
    if isinstance(option_contents_raw, str):
        try:
            option_contents_raw = ast.literal_eval(option_contents_raw)
        except (ValueError, SyntaxError):
            option_contents_raw = []
    option_contents = [str(c) for c in option_contents_raw]
    options = list(string.ascii_uppercase)[:len(option_contents)]

    user_contents: List = []
    cursor = 0
    for match in IMAGE_PLACEHOLDER.finditer(question_raw):
        text_part = question_raw[cursor:match.start()]
        if text_part and text_part.strip():
            user_contents.append(ChatTextContent(type="text", value=text_part.strip()))
        img_idx = match.group(1)
        img = sample.get(f"image_{img_idx}")
        if img is not None:
            user_contents.append(ChatImageContent(type="image", value=img))
        cursor = match.end()
    tail = question_raw[cursor:]
    if tail and tail.strip():
        user_contents.append(ChatTextContent(type="text", value=tail.strip()))

    if options and option_contents and question_type.startswith("multiple"):
        # paper-strict (aligned with the default in lmms-eval mmmu/utils.py:61 construct_prompt):
        #   `f"{question}\n{options}\n\n{mc_prompt}"` — no "Options:" keyword wrap.
        #   (the qwen3_vl variant uses a wrap, but the default is raw options.)
        choices = "\n".join(f"{letter}. {content}" for letter, content in zip(options, option_contents))
        user_contents.append(ChatTextContent(type="text", value=choices))

    if not user_contents:
        cleaned = IMAGE_PLACEHOLDER.sub("", question_raw).strip()
        if cleaned:
            user_contents.append(ChatTextContent(type="text", value=cleaned))

    messages = [ChatMessage(role="user", content=user_contents)]

    label: List[str] = [str(answer_raw)] if answer_raw not in (None, "") else []

    return Record(
        benchmark=task_name,
        index=sample_idx,
        prompt=None,
        messages=messages,
        generation_options=None,
        label=label,
        options=options if options else None,
        option_contents=option_contents if option_contents else None,
        prediction=None,
        latency=None,
        metrics=None,
        meta={
            "question_id": sample_id,
            "question_type": question_type,
            # Standard ours convention: 'category' is subject sub-domain (e.g., "Calculus"),
            # 'subcategory' is image type list (e.g., ["Mathematical Notations"]).
            "category": sample.get("subfield"),
            "subcategory": sample.get("img_type"),
            "topic_difficulty": sample.get("topic_difficulty"),
        },
    )
