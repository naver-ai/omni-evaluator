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

"""MMVet (v1) dataset preparation.

Scoring is delegated to JudgeEvaluator via the ``judge_rating`` target metric
configured in ``config.yaml``. This module only defines ``sample_to_record`` to
shape ``whyu/mm-vet`` HuggingFace samples into chat-message Records.
"""
import logging
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
    """Convert a whyu/mm-vet HuggingFace sample to a Record."""
    question = sample.get("question", "")
    answer = sample.get("answer", "")
    capability = sample.get("capability", None)
    if not isinstance(capability, list):
        capability = []
    question_id = sample.get("question_id", str(sample_idx))
    image = sample.get("image", None)

    user_contents = []
    if image is not None:
        user_contents.append(ChatImageContent(type="image", value=image))
    if question:
        user_contents.append(ChatTextContent(type="text", value=question))

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
            "question_id": question_id,
            "question": question,
            "capability": capability,
            "category": capability,
        },
    )
