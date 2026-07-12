# Reference from https://huggingface.co/datasets/MathArena/hmmt_nov_2025

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

"""HMMT November 2025 dataset preparation.

Source: ``MathArena/hmmt_nov_2025`` (HuggingFace). 30 competition math problems
(Harvard-MIT Math Tournament, Nov 2025); text-only. Columns: ``problem_idx``,
``problem``, ``answer`` (a short numeric/symbolic final answer).

Same pattern as ``polymath_en_test``: the model is asked to put its final answer
in ``\\boxed{}``, the shared ``boxed`` postprocess extracts it, and ``exact_match``
(with normalization) scores it against the gold answer. Enclosing ``$`` math
delimiters are stripped from the label so the boxed-extracted prediction matches.
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
    # Drop enclosing inline-math ``$...$`` delimiters so the boxed-extracted
    # prediction (which carries no ``$``) can match.
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
    problem = sample.get("problem", "") or ""
    answer = _clean_answer(sample.get("answer", None))
    problem_idx = sample.get("problem_idx", sample_idx)

    return default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": problem_idx,
            "query": problem,
            "label": [answer] if answer is not None else None,
            "meta": {
                "question_id": problem_idx,
                "question": problem,
                "answer": answer,
                "category": "hmmt_nov_2025",
            },
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )
