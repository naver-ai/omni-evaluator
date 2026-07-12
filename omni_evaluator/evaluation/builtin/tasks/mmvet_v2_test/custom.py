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

"""MMVet-v2 dataset preparation.

The scoring is delegated to :class:`omni_evaluator.evaluation.metrics.judge_evaluator.JudgeEvaluator`
via the ``judge_rating`` target metric configured in ``config.yaml`` — no task-local
judging code lives here. Only :func:`sample_to_record` remains because the official
MM-Vet v2 question format uses ``<IMG>`` / ``<image_N>`` placeholders that need to be
expanded into ordered text/image chat content.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Union

from omni_evaluator.schemas.chat import (
    Message as ChatMessage,
    TextContent as ChatTextContent,
    ImageContent as ChatImageContent,
)
from omni_evaluator.schemas.inference import Record

logger = logging.getLogger(__name__)


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
    """Convert a whyu/mm-vet-v2 HuggingFace sample to a Record."""
    question_raw = sample.get("question", "")
    answer = sample.get("answer", "")
    sample_id = sample.get("id", str(sample_idx))
    capability = sample.get("capability", [])

    # Keep text/image order as in question, aligned with official MM-Vet v2 inference.
    user_contents = []
    segments = question_raw.split("<IMG>")
    image_pat = re.compile(r"<image_(\d+)>")

    for segment in segments:
        if segment is None:
            continue

        cursor = 0
        for match in image_pat.finditer(segment):
            text_part = segment[cursor:match.start()]
            if text_part and text_part.strip():
                user_contents.append(ChatTextContent(type="text", value=text_part.strip()))

            img_idx_str = match.group(1)
            img = sample.get(f"image_{img_idx_str}")
            if img is not None:
                user_contents.append(ChatImageContent(type="image", value=img))

            cursor = match.end()

        tail = segment[cursor:]
        if tail and tail.strip():
            user_contents.append(ChatTextContent(type="text", value=tail.strip()))

    # Fallback in case malformed question produced no content.
    if not user_contents:
        question_clean = question_raw.replace("<IMG>", "").strip()
        question_clean = re.sub(r"<image_\d+>", "", question_clean).strip()
        if question_clean:
            user_contents.append(ChatTextContent(type="text", value=question_clean))

    messages = [
        ChatMessage(role="user", content=user_contents),
    ]

    return Record(
        benchmark=task_name,
        index=sample_idx,
        prompt=None,
        messages=messages,
        generation_options=None,
        label=answer,
        options=None,
        option_contents=None,
        prediction=None,
        latency=None,
        metrics=None,
        meta={
            "question_id": sample_id,
            "question": question_raw,
            "capability": capability,
        },
    )
