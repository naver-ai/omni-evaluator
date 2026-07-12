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

from typing import Dict, Any, Optional

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)


INSTRUCTION = "Which of the following acoustic scenes best describes the audio?"
OPTION_CONTENTS = [
    "Bus",
    "Cafe",
    "Car",
    "CrowdedIndoor",
    "Elevator",
    "Kitchen",
    "Park",
    "ResidentialArea",
    "Restaurant",
    "Restroom",
    "Street",
    "Subway",
    "SubwayStation",
]
IDX2LABEL = {_idx: _label for _idx, _label in enumerate(OPTION_CONTENTS)}


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
    # Content-only choices (no letters); label is the scene name and is scored
    # by content `mmau_string_match`. The model is asked to answer the scene
    # NAME (not a letter) so the content match works — see config task_prompt.
    # (CochlScene is acoustic-scene classification; string_match on the class
    # name is the natural fit, not letter exact_match.)
    _label_raw = sample["label"]
    _label_text = IDX2LABEL.get(_label_raw, _label_raw)
    labels = [_label_text]

    query = f'{INSTRUCTION}\n' + "\n".join(OPTION_CONTENTS)

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
        "label_id": _label_raw,
        "label": _label_text,
        "file_name": sample.get("file_name"),
    }

    record = default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample_idx,
            "messages": messages,
            "label": labels,
            "options": None,
            "option_contents": OPTION_CONTENTS,
            "meta": sample_meta,
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )
    return record
