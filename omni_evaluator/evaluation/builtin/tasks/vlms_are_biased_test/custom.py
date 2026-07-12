# Reference from https://huggingface.co/datasets/anvo25/vlms-are-biased

# Modifications Copyright (c) 2026-present NAVER Cloud Corp.
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

"""VLMs-are-Biased dataset preparation.

Source: ``anvo25/vlms-are-biased`` (HuggingFace), split ``main``. Single-image
questions probing visual bias. The ``image`` column is an embedded HF Image
(PIL / bytes) — so this task loads straight from the Hub (no S3 staging).
``prompt`` is the full question already carrying the requested answer format
(e.g. "Answer in curly brackets, e.g., {Yes} or {No}"), and ``ground_truth`` is
the gold short answer.

``sample_to_record`` attaches the decoded image plus the raw prompt; the custom
``curly`` postprocess pulls the answer out of the last ``{...}`` before
``exact_match`` scoring.
"""
import re
from typing import Any, Dict, List, Optional, Union

from omni_evaluator.schemas.chat import (
    Message as ChatMessage,
    ImageContent as ChatImageContent,
    TextContent as ChatTextContent,
)
from omni_evaluator.schemas.inference import Record
from omni_evaluator.utils.multimodal import to_pil_image


def sample_to_record(
    task_name: str,
    task_config,
    sample_idx: int,
    sample: Dict[str, Any],
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    **kwargs,
) -> Union[Record, List[Record]]:
    prompt = sample.get("prompt", "") or ""
    ground_truth = sample.get("ground_truth", None)
    image = sample.get("image", None)
    sample_id = sample.get("ID", str(sample_idx))

    user_contents = []
    if image is not None:
        user_contents.append(ChatImageContent(type="image", value=to_pil_image(image=image)))
    if prompt:
        user_contents.append(ChatTextContent(type="text", value=prompt))

    messages = [ChatMessage(role="user", content=user_contents)]

    return Record(
        benchmark=task_name,
        index=sample_id,
        prompt=None,
        messages=messages,
        generation_options=None,
        label=[ground_truth] if ground_truth is not None else None,
        options=None,
        option_contents=None,
        prediction=None,
        latency=None,
        metrics=None,
        meta={
            "question_id": sample_id,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "topic": sample.get("topic", None),
            "sub_topic": sample.get("sub_topic", None),
            "type_of_question": sample.get("type_of_question", None),
            "expected_bias": sample.get("expected_bias", None),
            "category": sample.get("topic", None),
        },
    )


_CURLY_PATTERN = re.compile(r"\{([^{}]*)\}")


def curly(
    prediction: str,
    verbose: Optional[bool] = False,
    **kwargs,
) -> Optional[str]:
    """Extract the content of the LAST ``{...}`` in the prediction.

    Returns the inner text (stripped) of the final curly-bracket group, or
    ``None`` when there is none — leaving the raw prediction untouched so a
    model that ignored the format instruction is still scored on its text.
    """
    if not isinstance(prediction, str):
        return None
    _matches = _CURLY_PATTERN.findall(prediction)
    if not _matches:
        return None
    return _matches[-1].strip()
