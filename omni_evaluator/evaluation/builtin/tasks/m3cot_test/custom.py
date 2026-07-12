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

"""M3CoT dataset preparation.

Source: ``LightChen2333/M3CoT`` (HuggingFace, ACL 2024). Multi-domain, multi-step,
multimodal CoT benchmark with multichoice answer letters. Postprocessing reuses
the shared ``multichoice`` extractor; this module only shapes a sample into a chat
Record by composing question + lettered choices (and optional context).
"""
import logging
import string
from typing import Any, Dict, List, Optional, Union

from omni_evaluator.schemas.chat import (
    Message as ChatMessage,
    TextContent as ChatTextContent,
    ImageContent as ChatImageContent,
)
from omni_evaluator.schemas.inference import Record

logger = logging.getLogger(__name__)


def _format_question_text(
    question: str,
    choices: List[str],
    context: Optional[str],
) -> str:
    lines = []
    if isinstance(context, str) and context.strip():
        lines.append(f"Context: {context.strip()}")
    if isinstance(question, str) and question.strip():
        lines.append(f"Question: {question.strip()}")
    if isinstance(choices, list) and len(choices) > 0:
        lines.append("Choices:")
        for letter, choice in zip(string.ascii_uppercase, choices):
            lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


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
    """Convert a LightChen2333/M3CoT HuggingFace sample to a Record."""
    question = sample.get("question", "")
    choices = sample.get("choices", None)
    if not isinstance(choices, list):
        choices = []
    context = sample.get("context", None)
    answer = sample.get("answer", "")
    image = sample.get("image", None)
    sample_id = sample.get("id", str(sample_idx))
    domain = sample.get("domain", None)
    topic = sample.get("topic", None)
    rationale = sample.get("rationale", None)

    options = [string.ascii_uppercase[i] for i in range(len(choices))] or ["A", "B", "C", "D"]

    question_text = _format_question_text(
        question=question,
        choices=choices,
        context=context,
    )

    user_contents = []
    if image is not None:
        user_contents.append(ChatImageContent(type="image", value=image))
    if question_text:
        user_contents.append(ChatTextContent(type="text", value=question_text))

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
        options=options,
        option_contents=choices,
        prediction=None,
        latency=None,
        metrics=None,
        meta={
            "question_id": sample_id,
            "question": question,
            "choices": choices,
            "context": context,
            "rationale": rationale,
            "domain": domain,
            "topic": topic,
            "category": domain,
        },
    )
