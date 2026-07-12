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
import string
from typing import Dict, Any, Optional

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)
from omni_evaluator.evaluation.resources.prompts.multichoice.en import (
    music_genre as multichoice_instructions,
)


LABEL_MAP = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock",
}
OPTION_CONTENTS = [LABEL_MAP[_idx] for _idx in sorted(LABEL_MAP.keys())]
OPTIONS = list(string.ascii_uppercase)[:len(OPTION_CONTENTS)]


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
    # Label-column name compat across dataset mirrors: old 'label' / sanchit-gandhi parquet 'genre'.
    _label_id = sample["label"] if "label" in sample else sample["genre"]
    _label_text = LABEL_MAP[_label_id]
    _option_letter = OPTIONS[OPTION_CONTENTS.index(_label_text)]
    labels = [_option_letter, _label_text]

    # paper-strict (lmms-eval audio MCQ pattern — aligned with muchomusic/utils.py:13):
    #   `{instruction}\n{choices}` only. Removes the "Question:" / "Choices:" keyword wrap.
    #   choices are also split by newline instead of by space.
    choices = "\n".join([
        f"{_option}. {_option_content}"
        for _option, _option_content in zip(OPTIONS, OPTION_CONTENTS)
    ])
    query = random.choice(multichoice_instructions)
    query = f"{query}\n{choices}"

    _audio_value = sample["audio"]["bytes"]
    if not _audio_value:
        _audio_value = sample["audio"]["path"]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "value": _audio_value,
                },
                {
                    "type": "text",
                    "value": query,
                },
            ],
        }
    ]

    sample_meta = {
        "label_id": _label_id,
        "label": _label_text,
    }

    record = default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample_idx,
            "messages": messages,
            "label": labels,
            "options": OPTIONS,
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
