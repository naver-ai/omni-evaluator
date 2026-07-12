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

"""Prompts for the verifier judge.

Both variants share the same placeholders (``{reference}``, ``{prediction}``,
``{question}``, ``{choices}``) and the same parse contract — the response
ends with::

    Explanation: ...
    Rating: <0 or 1>

``{choices}`` is appended directly after ``{question}`` and is expected to
render as either the empty string (open-ended items) or a leading blank
line + ``[Options]`` block (MCQA items). The caller
(``Verifier._build_choices_block``) formats the record's ``options`` and
``option_contents`` — the prompt template stays agnostic to whether the
item is MCQA.

Consumers:

- ``Verifier`` (``omni_evaluator/evaluation/metrics/verifier.py``), dispatched by
  the ``enable_verifier`` flag in evaluate.py. It produces the ``verifier_score``
  metric and selects this prompt (CoT variant by ``verifier_reasoning``).
- Verifier training pipeline (``scripts/judge/verifier_train/``) — trains the
  Qwen2.5-Omni verifier to match this exact prompt + parse contract so the
  in-process inference path is a drop-in for the GPT call.

The COT variant is reasoning-heavy / longer; pick it for math / science / hard
samples. See ``scripts/judge/data_collection/judge_prompts.py`` for the
per-sample selection heuristic.
"""

VERIFIER_PROMPT = """\
[Reference Answer]
{reference}

[Model Answer]
{prediction}

[Question]
{question}{choices}

[Task]
Rate whether the model's FINAL answer matches the reference answer. Be strict and do NOT over-credit.
Count it as correct (Score 1) ONLY when it is equivalent to the reference under one of the following; otherwise Score 0:
- multiple acceptable answers: it matches any one of the reference's accepted answers;
- mathematical/numeric equivalence: the same value in different notation (e.g., 1/2 = 0.5 = 50%), an algebraically equal expression, or the same value with a correct unit;
- formatting only: the same answer with different wrapper/punctuation/case (e.g., \\boxed{{B}}, (B), "B") or a set/list given in a different order;
- multiple choice: the correct option's label (A / 1 / ①) or its exact option content.
Do NOT give Score 1 if the answer is only partially correct, omits a required part, adds incorrect or contradictory content, or is merely related/plausible but not the reference answer. A loose paraphrase only counts when it states the same answer with no change or loss of meaning.
Score 0: The answer refuses or says it cannot decide.
Score 0: The answer is wrong, incomplete, or irrelevant compared to the reference.
Score 1: The answer is correct under the equivalence criteria above.

[Task-wise instruction]
Apply the rule(s) that match the question's task type; otherwise use the general criteria above.
- Visual grounding (bounding box): if the model's answer contains no valid box (four normalized coordinates), Score 0; otherwise Score 1 only if its IoU with the reference box is >= 0.5.
- Temporal grounding (time interval): if there is no valid start-end time pair, Score 0; otherwise Score 1 only if the temporal IoU with the reference interval is >= 0.5.
- Numeric answers: Score 1 if the numeric value matches; omitting or adding a unit is acceptable, but a different or incorrect unit is Score 0.
- OCR / exact-string reading: Score 1 if the response contains all of the reference's characters/tokens (reading extra surrounding text is allowed); Score 0 if any required reference content is missing.
- Emotion / sentiment: treat adjacent affective states as equivalent (e.g., joy ~ excitement) -> Score 1.
- Translation / transcription (ASR): Score 1 if the meaning is equivalent; Score 0 if a requested target language is violated; ignore Chinese word-segmentation/spacing differences.
- Multiple reference answers: if the references are alternatives/surface variants, matching any one suffices (OR); if they are required components of a single answer, all must be present (AND) -- decide from the question.

Respond EXACTLY in this format (the last line must be 'Rating: 0' or 'Rating: 1', with nothing after the digit):
Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. "The reference answer is [XXX], while the model's answer is [YYY]. I think ...")
Rating: <0 or 1>"""


# CoT (rationale-first, longer) variant — for reasoning-heavy / math / science / hard
# samples. Same placeholders and parse contract as VERIFIER_PROMPT, so it is a
# drop-in where deeper reasoning helps.
VERIFIER_COT_PROMPT = """\
[Reference Answer]
{reference}

[Model Answer]
{prediction}

[Question]
{question}{choices}

[Task]
Decide whether the model's FINAL answer is CORRECT with respect to the reference answer. This may require multi-step reasoning, so reason step by step and check intermediate steps and units. Be strict and do NOT over-credit.
Score 0: The answer refuses / cannot decide, or is wrong, incomplete, or irrelevant compared to the reference.
Score 1: The model's final answer is equivalent to the reference under the equivalence rules below.

Work through it:
1. State the reference's final answer.
2. Extract the model's final answer from its response (ignore scratch work).
3. Judge equivalence — count as correct ONLY under one of these, otherwise incorrect:
   - multiple acceptable answers: matches any one accepted by the reference;
   - mathematical/numeric equivalence: same value in different notation (1/2 = 0.5 = 50%), an algebraically equal expression, or the same value with a correct unit;
   - formatting only: different wrapper/punctuation/case (e.g., \\boxed{{...}}, (X), "X") or a set/list in a different order;
   - multiple choice: the correct option's label (A / 1 / ①) or its exact option content.
   Do NOT credit partial answers, omissions, added incorrect/contradictory content, or merely related/plausible answers; a paraphrase counts only if it states the same answer with no change or loss of meaning.
4. Conclude.

[Task-wise instruction]
When the question matches a task type below, apply that rule in step 3; otherwise use the equivalence rules above.
- Visual grounding (bounding box): if the model's answer contains no valid box (four normalized coordinates), Score 0; otherwise Score 1 only if its IoU with the reference box is >= 0.5.
- Temporal grounding (time interval): if there is no valid start-end time pair, Score 0; otherwise Score 1 only if the temporal IoU with the reference interval is >= 0.5.
- Numeric answers: Score 1 if the numeric value matches; omitting or adding a unit is acceptable, but a different or incorrect unit is Score 0.
- OCR / exact-string reading: Score 1 if the response contains all of the reference's characters/tokens (reading extra surrounding text is allowed); Score 0 if any required reference content is missing.
- Emotion / sentiment: treat adjacent affective states as equivalent (e.g., joy ~ excitement) -> Score 1.
- Translation / transcription (ASR): Score 1 if the meaning is equivalent; Score 0 if a requested target language is violated; ignore Chinese word-segmentation/spacing differences.
- Multiple reference answers: if the references are alternatives/surface variants, matching any one suffices (OR); if they are required components of a single answer, all must be present (AND) -- decide from the question.

Respond EXACTLY in this format (the last line must be 'Rating: 0' or 'Rating: 1', with nothing after the digit):
Explanation: (your step-by-step reasoning ending in a clear conclusion)
Rating: <0 or 1>"""
