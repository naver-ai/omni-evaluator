# Reference from https://huggingface.co/datasets/OpenEvals/IMO-AnswerBench

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

"""IMO-AnswerBench dataset preparation.

Source: ``OpenEvals/IMO-AnswerBench`` (HuggingFace). 400 short-answer olympiad
problems; text-only. Columns (note the spaces): ``Problem ID``, ``Problem``,
``Short Answer``, ``Category``, ``Subcategory``, ``Source``.

Same ``boxed`` + ``exact_match`` pattern as the other math tasks. Answers span
numbers, expressions, intervals and functions, so string exact_match is a
conservative lower bound (see config.yaml); enclosing ``$`` delimiters are
stripped from the label to improve matching.
"""
from typing import Any, Dict, List, Optional, Union

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)
from omni_evaluator.schemas.inference import Record


def _clean_answer(answer: Any) -> Optional[str]:
    if answer is None:
        return None
    _a = str(answer).strip()
    if _a.startswith("$") and _a.endswith("$") and len(_a) >= 2:
        _a = _a[1:-1].strip()
    return _a


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
    problem = sample.get("Problem", "") or ""
    answer = _clean_answer(sample.get("Short Answer", None))
    problem_id = sample.get("Problem ID", sample_idx)
    category = sample.get("Category", None)

    return default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": problem_id,
            "query": problem,
            "label": [answer] if answer is not None else None,
            "meta": {
                "question_id": problem_id,
                "question": problem,
                "answer": answer,
                "category": category,
                "subcategory": sample.get("Subcategory", None),
                "source": sample.get("Source", None),
            },
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )
