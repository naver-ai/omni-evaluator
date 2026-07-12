# Reference from https://github.com/MMMU-Benchmark/MMMU (Apache-2.0)

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

"""MMMU-style accuracy: ported from upstream MMMU eval_utils.

Source: https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e/eval/eval_utils.py
Mirror in this repo: omni_evaluator/evaluation/lmms_eval/resources/custom_tasks/mmmu_pro/utils.py

Two paths, one entry point (:func:`score_row`):

- **Multiple-choice**: ``parse_multi_choice_response`` finds the predicted
  option letter via three layered heuristics (bracketed, suffixed, content
  match), then ``eval_multi_choice`` compares to the gold letter.
- **Open-ended**: ``parse_open_response`` extracts candidate strings/numbers
  from the prediction (keyed off "therefore", "answer", "=", etc.) and
  ``eval_open`` substring-matches normalized gold against any candidate.

Both return a 0/1 correctness float, which the caller averages into accuracy.
"""
from __future__ import annotations

import random
import re
from typing import Any, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Number helpers (MMMU `eval_utils.extract_numbers` / `check_is_number`)
# ---------------------------------------------------------------------------

_NUM_COMMAS = r"-?\b\d{1,3}(?:,\d{3})+\b"
_NUM_SCIENTIFIC = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
_NUM_SIMPLE = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"


def _extract_numbers(text: str) -> List[str]:
    return (
        re.findall(_NUM_COMMAS, text)
        + re.findall(_NUM_SCIENTIFIC, text)
        + re.findall(_NUM_SIMPLE, text)
    )


def _check_is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(string: str) -> List[Any]:
    """Normalize to a list of candidate forms (float for numerics, lowercase string otherwise).

    Single-character strings get expanded with leading/trailing spaces to avoid
    trivial substring hits inside unrelated words.
    """
    string = string.strip()
    if _check_is_number(string):
        return [round(float(string.replace(",", "")), 2)]
    string = string.lower()
    if len(string) == 1:
        return [" " + string, string + " "]
    return [string]


# ---------------------------------------------------------------------------
# Multiple-choice parsing
# ---------------------------------------------------------------------------

def parse_multi_choice_response(
    response: str,
    all_choices: Sequence[str],
    index2ans: dict[str, str],
) -> str:
    """Extract a predicted choice letter from *response*.

    Falls back to a random choice when no signal is found (matches upstream).
    """
    if not isinstance(response, str):
        response = ""
    for ch in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(ch)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates: list[str] = []

    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if not candidates:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if not candidates:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # content match (only when the response is long enough to contain prose)
    if not candidates and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if isinstance(ans, str) and ans and ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if not candidates:
        return random.choice(list(all_choices)) if all_choices else ""

    if len(candidates) == 1:
        return candidates[0]

    # multiple candidates → take the last-occurring one (matches MMMU)
    start_indexes: list[int] = []
    if index_ans:
        if ans_with_brack:
            for can in candidates:
                start_indexes.append(response.rfind(f"({can})"))
        else:
            for can in candidates:
                start_indexes.append(response.rfind(f" {can} "))
    else:
        for can in candidates:
            start_indexes.append(response.lower().rfind(index2ans[can].lower()))
    return candidates[max(range(len(start_indexes)), key=start_indexes.__getitem__)]


def eval_multi_choice(gold: Any, pred: str) -> bool:
    """Exact-match against gold letter(s)."""
    if isinstance(gold, (list, tuple)):
        return any(g == pred for g in gold)
    return gold == pred


# ---------------------------------------------------------------------------
# Open-ended parsing
# ---------------------------------------------------------------------------

_OPEN_INDICATORS = (
    "could be ",
    "so ",
    "is ",
    "thus ",
    "therefore ",
    "final ",
    "answer ",
    "result ",
)


def _get_key_subresponses(response: str) -> List[str]:
    response = response.strip().strip(".").lower()
    sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)

    out: list[str] = []
    for idx, resp in enumerate(sub_responses):
        indicators = list(_OPEN_INDICATORS)
        if idx == len(sub_responses) - 1:
            indicators.append("=")
        shortest: Optional[str] = None
        for ind in indicators:
            if ind in resp:
                tail = resp.split(ind)[-1].strip()
                if shortest is None or len(tail) < len(shortest):
                    shortest = tail
        if shortest and shortest not in {":", ",", ".", "!", "?", ";", "'"}:
            out.append(shortest)
    return out if out else [response]


def parse_open_response(response: str) -> List[Any]:
    """Return candidate prediction strings/numbers extracted from *response*."""
    if not isinstance(response, str):
        return []
    keys = _get_key_subresponses(response)
    expanded: list[Any] = list(keys)
    for resp in keys:
        expanded.extend(_extract_numbers(resp))
    normalized: list[Any] = []
    for item in expanded:
        normalized.extend(normalize_str(item))
    # de-dup while preserving distinguishable values
    seen: list[Any] = []
    for n in normalized:
        if n not in seen:
            seen.append(n)
    return seen


def eval_open(gold: Any, pred_list: Iterable[Any]) -> bool:
    """Substring-match normalized gold against any pred candidate."""
    if isinstance(gold, (list, tuple)):
        norm_golds: list[Any] = []
        for g in gold:
            norm_golds.extend(normalize_str(g))
    else:
        norm_golds = normalize_str(gold)

    for pred in pred_list:
        if isinstance(pred, str):
            for g in norm_golds:
                if isinstance(g, str) and g in pred:
                    return True
        else:                                 # number
            if pred in norm_golds:
                return True
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_multi_choice_info(option_contents: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """A→option_contents[0], B→option_contents[1], …. Returns (index2ans, all_choices)."""
    index2ans: dict[str, str] = {}
    all_choices: list[str] = []
    for i, opt in enumerate(option_contents):
        idx = chr(ord("A") + i)
        index2ans[idx] = "" if opt is None else str(opt)
        all_choices.append(idx)
    return index2ans, all_choices


def score_row(
    prediction: str,
    labels: Sequence[Any],
    options: Optional[Sequence[str]] = None,
    option_contents: Optional[Sequence[str]] = None,
    question_type: Optional[str] = None,
) -> float:
    """Return 1.0 if the prediction is correct (MMMU semantics), else 0.0.

    Routing:
      - ``question_type`` starting with "open" → open-ended path
      - otherwise → multiple-choice path (requires options / option_contents)

    When question_type is missing/None, treat as multiple-choice if usable
    options data is present; otherwise fall back to open-ended.
    """
    if not isinstance(prediction, str):
        prediction = ""

    qt = (question_type or "").lower()
    is_open = qt.startswith("open")
    has_options = bool(options) or bool(option_contents)
    if not qt and not has_options:
        is_open = True

    if is_open:
        preds = parse_open_response(prediction)
        return 1.0 if eval_open(list(labels), preds) else 0.0

    # multiple-choice — need an index2ans / all_choices pair.
    if option_contents:
        index2ans, all_choices = get_multi_choice_info(option_contents)
    elif options:
        # options are letters; we have no contents → only bracket/suffix match works.
        all_choices = [str(o) for o in options]
        index2ans = {c: "" for c in all_choices}
    else:                                    # MC routing but no choices supplied
        return 0.0

    parsed = parse_multi_choice_response(
        response=prediction,
        all_choices=all_choices,
        index2ans=index2ans,
    )
    return 1.0 if eval_multi_choice(list(labels), parsed) else 0.0


__all__ = [
    "score_row",
    "parse_multi_choice_response",
    "parse_open_response",
    "eval_multi_choice",
    "eval_open",
    "normalize_str",
    "get_multi_choice_info",
]
