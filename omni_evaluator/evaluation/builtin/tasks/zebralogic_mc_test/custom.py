# Reference from https://huggingface.co/datasets/WildEval/ZebraLogic
# Reference from https://github.com/WildEval/ZeroEval (Apache-2.0)

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

"""ZebraLogic (multiple-choice mode) dataset preparation.

Source: ``WildEval/ZebraLogic`` (HuggingFace), config ``mc_mode``. Text-only
logic-grid puzzles posed as multiple-choice questions. Each row carries the
full puzzle constraints (``puzzle``), a single question stem (``question``),
the candidate answers (``choices``, a list of strings) and the gold answer
(``answer``) as one of the exact choice strings.

``question`` does NOT embed the lettered options, so this module renders them
into the query and maps the gold choice string to its option letter. The gold
label is the letter (matching the shared ``multichoice`` postprocess +
``exact_match`` convention used by e.g. m3cot).
"""
import string
from typing import Any, Dict, List, Optional, Union

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)
from omni_evaluator.schemas.inference import Record


def _format_query(puzzle: str, question: str, options: List[str], choices: List[str]) -> str:
    lines: List[str] = []
    if isinstance(puzzle, str) and puzzle.strip():
        lines.append(puzzle.strip())
    if isinstance(question, str) and question.strip():
        lines.append(f"Question: {question.strip()}")
    if choices:
        lines.append("Options:")
        for _letter, _choice in zip(options, choices):
            lines.append(f"{_letter}. {_choice}")
    return "\n".join(lines)


def sample_to_record(
    task_name: str,
    task_config,
    sample_idx: int,
    sample: Dict[str, Any],
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_entity_tokens: Optional[int] = None,
    run_index: Optional[int] = 0,
    **kwargs,
) -> Union[Record, List[Record]]:
    puzzle = sample.get("puzzle", "") or ""
    question = sample.get("question", "") or ""
    choices = sample.get("choices", None)
    if not isinstance(choices, list):
        choices = []
    answer = sample.get("answer", None)
    sample_id = sample.get("id", str(sample_idx))

    options = [string.ascii_uppercase[i] for i in range(len(choices))]

    # Map the gold choice string to its option letter. Fall back to the raw
    # answer string when it is already a letter or is not found among choices.
    label = None
    if isinstance(answer, str):
        if answer in choices:
            label = options[choices.index(answer)]
        elif answer.strip().upper() in options:
            label = answer.strip().upper()
        else:
            label = answer

    query = _format_query(puzzle=puzzle, question=question, options=options, choices=choices)

    return default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample_id,
            "query": query,
            "options": options,
            "option_contents": choices,
            "label": [label] if label is not None else None,
            "meta": {
                "question_id": sample_id,
                "question": question,
                "choices": choices,
                "answer": answer,
                "category": "mc",
            },
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )
