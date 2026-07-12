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

import re
import string
from typing import Dict, Any, List, Optional

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)
from omni_evaluator.utils.data import extract_options


OPTIONS_FALLBACK = list(string.ascii_uppercase)


# Cascading letter-extraction patterns tried in order. First match wins.
# All patterns emit the letter in group 1 (or group 0 for the last-letter
# fallback). The last one is permissive by design — it picks up any standalone
# A-J anywhere in the output — so it is kept local to this MMAU-Pro task
# rather than promoted to the generic multichoice postprocess.
_LETTER_EXTRACTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"answer is \(?([A-J])\)?"),          # e.g. "the answer is (A)"
    re.compile(r".*[aA]nswer:\s*([A-J])"),           # e.g. "Answer: B"
    re.compile(r"\b[A-J]\b(?!.*\b[A-J]\b)", re.DOTALL),  # last standalone letter
]


def _extract_letter(text: str) -> Optional[str]:
    """Try each pattern in order; return the first captured letter."""
    for pattern in _LETTER_EXTRACTION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        return match.group(1) if match.groups() else match.group(0)
    return None


def postprocess(
    prediction: str,
    query: str = "",
    options: Optional[List[str]] = None,
    option_contents: Optional[List[str]] = None,
    **kwargs,
) -> Optional[str]:
    """Extract the model's chosen option letter, then map it back to the
    corresponding option TEXT.

    The downstream metrics (``exact_match`` / ``string_match``) score against
    the content-form label, so returning the raw letter would always miss.
    Mapping letter → content here keeps the metric layer generic.

    Free-form samples (no options provided) return ``None`` — the raw
    prediction is passed through untouched by the upstream pipeline.
    """
    if not isinstance(prediction, str):
        return prediction
    if not (options and option_contents):
        return None

    letter = _extract_letter(prediction)
    if letter is None or letter not in options:
        return None

    idx = options.index(letter)
    if idx >= len(option_contents):
        return None
    return option_contents[idx]


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
    option_contents = list(sample.get("choices") or [])
    options = OPTIONS_FALLBACK[:len(option_contents)] if option_contents else []
    label = sample["answer"]
    label_contents = list()
    if isinstance(label, str):
        _, label_contents = extract_options(text=label)
    # Expand label to [content, full] forms only. The option index (e.g. "A")
    # is intentionally excluded because this task uses `string_match`,
    # which does substring matching (`label in prediction`): a one-character
    # label like "A" would be a substring of almost any non-empty prediction,
    # producing false positives. The same caveat applies to any task that
    # uses `mmau_string_match` or `substring_match`.
    labels = list()
    if label_contents and label_contents[0]:
        labels.append(label_contents[0])
    labels.append(label)

    # paper-strict (aligned with lmms-eval mmau/utils.py:22):
    #   `{question}\n{choices}` only — removes our "Question:" / "Options:" keyword wrap
    #   and the forced "Answer: " suffix. The standard instruction
    #   ("Answer with the option's letter from the given choices directly.") is
    #   appended afterward via the post_prompt in the yaml task_prompt.
    #   choices are also split by newline like lmms-eval (space split -> newline).
    if options and option_contents:
        choices_text = "\n".join([
            f"{_option}. {_option_content}"
            for _option, _option_content in zip(options, option_contents)
        ])
        query = f'{sample["question"]}\n{choices_text}'
    else:
        query = sample["question"]

    # MMAU-Pro samples may have multiple audio files.
    _audio_paths = sample.get("audio_path") or []
    if isinstance(_audio_paths, str):
        _audio_paths = [_audio_paths]
    content = [
        {"type": "audio", "value": _p}
        for _p in _audio_paths
    ]
    content.append({"type": "text", "value": query})

    messages = [{"role": "user", "content": content}]

    sample_meta = {
        "id": sample.get("id"),
        "audio_path": _audio_paths,
        "question": sample["question"],
        "answer": label,
        "choices": option_contents,
        "category": sample.get("category"),
        "sub-cat": sample.get("sub-cat"),
        "length_type": sample.get("length_type"),
        "perceptual_skills": sample.get("perceptual_skills"),
        "reasoning_skills": sample.get("reasoning_skills"),
        "task_classification": sample.get("task_classification"),
        "task_identifier": sample.get("task_identifier"),
    }

    record = default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample.get("id", sample_idx),
            "messages": messages,
            "label": labels,
            "options": options if options else None,
            "option_contents": option_contents if option_contents else None,
            "meta": sample_meta,
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )
    return record
