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

import random
from typing import Dict, Any, Optional

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)
from omni_evaluator.evaluation.resources.prompts.transcription.en import (
    transcribe_en as asr_instructions,
)


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
    query = random.choice(asr_instructions)
    label = sample["transcription"]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "value": sample["audio"]["bytes"],
                },
                {
                    "type": "text",
                    "value": query,
                },
            ],
        }
    ]

    sample_meta = {
        "id": sample["id"],
        "num_samples": sample["num_samples"],
        "gender": sample["gender"],
        "language": sample["language"],
        "lang_id": sample["lang_id"],
        "lang_group_id": sample["lang_group_id"],
        "transcription": sample["transcription"],
        "raw_transcription": sample["raw_transcription"],
    }

    record = default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample["id"],
            "messages": messages,
            "label": [label],
            "options": None,
            "option_contents": None,
            "meta": sample_meta,
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )
    return record
