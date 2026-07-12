# Reference from https://huggingface.co/datasets/cruxeval-org/cruxeval
# Reference from https://github.com/facebookresearch/cruxeval (MIT)

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

"""CRUXEval-O (output prediction) dataset preparation.

Source: ``cruxeval-org/cruxeval`` (HuggingFace). Given a Python function ``f``
and an input, the model predicts the OUTPUT of ``f(input)``. The gold answer is
in the ``output`` column as a Python literal — no code execution is needed to
score (unlike code-generation benchmarks), so this fits the standard
huggingface_hub + ``exact_match`` path.

Because the output is a Python literal, formatting differences (e.g. tuple
spacing ``(4, 1)`` vs ``(4,1)``) would break a raw string compare. So both the
gold label and the model's extracted answer are canonicalized via
``ast.literal_eval`` -> ``repr`` before ``exact_match``.

The prompt follows CRUXEval's official direct output-prediction format: the code
plus ``assert f(input) == ??`` wrapped in ``[PYTHON]``/``[ANSWER]`` tags; the
``cruxeval_output`` postprocess extracts the literal the model completes.
"""
import ast
import re
from typing import Any, Dict, List, Optional, Union

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)
from omni_evaluator.schemas.inference import Record

_PROMPT_TEMPLATE = (
    "You are given a Python function and an assertion containing an input to the "
    "function. Complete the assertion with a literal (no unsimplified expressions, "
    "no function calls) containing the output when executing the provided code on "
    "the given input, even if the function is incorrect or incomplete. Do NOT "
    "output any extra information. Provide the full assertion with the correct "
    "output in [ANSWER] and [/ANSWER] tags, following the examples.\n\n"
    "[PYTHON]\n{code}\nassert f({input}) == ??\n[/PYTHON]\n[ANSWER]\n"
)


def _canon(text: Any) -> Optional[str]:
    """Canonicalize a Python-literal string to ``repr(value)``; fall back to the
    stripped original when it is not a parseable literal."""
    if text is None:
        return None
    _t = str(text).strip()
    try:
        return repr(ast.literal_eval(_t))
    except Exception:
        return _t


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
    code = sample.get("code", "") or ""
    _input = sample.get("input", "")
    output = sample.get("output", None)
    sample_id = sample.get("id", str(sample_idx))

    query = _PROMPT_TEMPLATE.format(code=code, input=_input)
    label = _canon(output)

    return default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample_id,
            "query": query,
            "label": [label] if label is not None else None,
            "meta": {
                "question_id": sample_id,
                "code": code,
                "input": _input,
                "output": output,
                "category": "output_prediction",
            },
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )


_ANSWER_BLOCK = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.DOTALL)


def cruxeval_output(
    prediction: str,
    verbose: Optional[bool] = False,
    **kwargs,
) -> Optional[str]:
    """Extract the predicted output literal from a CRUXEval-O completion.

    Handles ``[ANSWER] assert f(...) == <literal> [/ANSWER]`` (the intended
    format), a dangling ``[ANSWER]`` with no closing tag, and a bare
    ``... == <literal>`` line. When neither an ``[ANSWER]`` tag nor ``==`` is
    present, the first non-empty line is taken as the answer (a bare literal).
    Returns the canonicalized literal, or ``None`` only when the completion is
    empty/whitespace (leaving the raw prediction for scoring).
    """
    if not isinstance(prediction, str):
        return None
    _text = prediction
    _match = _ANSWER_BLOCK.search(_text)
    if _match:
        _text = _match.group(1)
    else:
        _idx = _text.rfind("[ANSWER]")
        if _idx != -1:
            _text = _text[_idx + len("[ANSWER]"):]
    # Drop any leftover closing tag, then take the RHS of the last '=='.
    _text = _text.replace("[/ANSWER]", "")
    if "==" in _text:
        _text = _text.rsplit("==", 1)[1]
    _text = _text.strip()
    # Keep only the first non-empty line (the literal); ignore trailing prose.
    for _line in _text.splitlines():
        _line = _line.strip()
        if _line:
            _text = _line
            break
    if not _text:
        return None
    return _canon(_text)
