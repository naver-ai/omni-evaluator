# Reference from https://huggingface.co/datasets/Qwen/PolyMath
# Reference from https://github.com/QwenLM/PolyMath (Apache-2.0)

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

"""PolyMath (English) dataset preparation.

Source: ``Qwen/PolyMath`` (HuggingFace), config ``en``. Multilingual math
reasoning; text-only. Each row has ``id``, ``question`` and ``answer``. The
gold ``answer`` is a short (often LaTeX) expression such as ``$\\frac{\\pi}{3}$``.

The prompt asks the model to place its final answer in ``\\boxed{}``; the shared
``boxed`` postprocess extracts it and ``exact_match`` (with normalization)
scores it against the gold answer. Surrounding ``$`` math delimiters are
stripped from the label so the boxed-extracted prediction can match.
"""
from typing import Any, Dict, List, Optional, Union

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)
from omni_evaluator.schemas.inference import Record


def _clean_answer(answer: Any) -> Optional[str]:
    if not isinstance(answer, str):
        return None
    _a = answer.strip()
    # Drop enclosing inline-math ``$...$`` delimiters (kept in the raw dataset)
    # so the boxed-extracted prediction, which carries no ``$``, can match.
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
    question = sample.get("question", "") or ""
    answer = _clean_answer(sample.get("answer", None))
    sample_id = sample.get("id", str(sample_idx))
    difficulty = sample.get("dataset_split", None)

    return default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample_id,
            "query": question,
            "label": [answer] if answer is not None else None,
            "meta": {
                "question_id": sample_id,
                "question": question,
                "answer": answer,
                "difficulty": difficulty,
                "category": difficulty,
            },
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )
