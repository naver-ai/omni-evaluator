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

import json
from typing import Dict, Any, List, Optional

from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)


INSTRUCTION = "Transcribe the speech."

# Lazy-initialized jiwer normalize pipeline (matches HiKE/src/utils.py normalize_text).
_normalize_pipeline = None


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
    query = INSTRUCTION
    label = sample["text_normalized"]

    _loanwords = sample.get("loanwords")
    if isinstance(_loanwords, str):
        try:
            _loanwords = json.loads(_loanwords)
        except json.JSONDecodeError:
            pass

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
        "sample_id": sample["sample_id"],
        "category": sample["category"],
        "cs_level": sample["cs_level"],
        "cs_levels_all": sample["cs_levels_all"],
        "text": sample["text"],
        "text_normalized": sample["text_normalized"],
        "text_pier_labeled": sample["text_pier_labeled"],
        "loanwords": _loanwords,
    }

    record = default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample["sample_id"],
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


def _get_normalize_pipeline():
    global _normalize_pipeline
    if _normalize_pipeline is not None:
        return _normalize_pipeline
    import jiwer
    _normalize_pipeline = jiwer.transforms.Compose([
        jiwer.transforms.ToLowerCase(),
        jiwer.transforms.ExpandCommonEnglishContractions(),
        jiwer.transforms.RemoveKaldiNonWords(),
        jiwer.transforms.SubstituteWords({"—": " "}),
        jiwer.transforms.RemovePunctuation(),
        jiwer.transforms.RemoveWhiteSpace(replace_by_space=True),
        jiwer.transforms.RemoveMultipleSpaces(),
        jiwer.transforms.Strip(),
        jiwer.transforms.ReduceToSingleSentence(),
        jiwer.transforms.ReduceToListOfListOfWords(),
    ])
    return _normalize_pipeline


def postprocess(
    prediction: str,
    loanwords: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> str:
    """HiKE-style normalization pipeline.

    Three stages applied in order:
      1. Replace Korean loanwords with their English equivalents.
      2. Apply the jiwer normalize pipeline (lowercase, multi-space collapse,
         contraction expansion, punctuation strip).
      3. Insert a space between trailing ASCII alphanumerics and the
         following Hangul characters, e.g. ``bug는`` → ``bug 는``.
    """
    if not isinstance(prediction, str):
        return prediction

    import regex

    # Step 1: loanword replacement (Korean → English).
    if loanwords:
        for _loanword in loanwords:
            prediction = prediction.replace(_loanword["Korean"], _loanword["English"])

    # Step 2: jiwer normalization. The pipeline ends with
    # `ReduceToListOfListOfWords`, so `[0]` yields the single sentence's tokens.
    _tokens = _get_normalize_pipeline()(prediction)[0]
    prediction = " ".join(_tokens)

    # Step 3: separate ASCII alnum from following Hangul characters.
    prediction = regex.sub(r'([A-Za-z0-9]+)(?=\p{Script=Hangul})', r'\1 ', prediction)

    return prediction