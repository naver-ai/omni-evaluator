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

"""Test case data for `omni_evaluator/postprocess/multichoice`."""

import pytest


# (prediction, expected) — used with options=["A","B","C","D"].
MC_LETTER_PREDICTIONS = [
    pytest.param("The answer is B.", "B", id="en_answer_B"),
    pytest.param("Final answer: C", "C", id="en_final_C"),
    pytest.param("정답은 A 입니다.", "A", id="ko_answer_A"),
    pytest.param("(B)", "B", id="parens_B"),
    pytest.param("**B**", "B", id="bold_B"),
    pytest.param("Therefore the correct answer is D.", "D", id="en_therefore_D"),
    pytest.param("\\boxed{C}", "C", id="boxed_C"),
]

# (prediction, expected) — used with options=[1,2,3,4].
MC_NUMBER_PREDICTIONS = [
    pytest.param("Answer: 2", "2", id="en_answer_2"),
    pytest.param("정답은 3 입니다.", "3", id="ko_answer_3"),
    pytest.param("(4)", "4", id="parens_4"),
]

# (prediction, expected) — used with options=["①","②","③","④"].
MC_CIRCLED_PREDICTIONS = [
    pytest.param("Answer: ②", "②", id="en_circled_2"),
    pytest.param("정답은 ③ 입니다.", "③", id="ko_circled_3"),
]

# Single-character prediction — returned as-is regardless of validity.
MC_PASSTHROUGH_SINGLE = [
    pytest.param("A", id="letter"),
    pytest.param("1", id="number"),
    pytest.param("①", id="circled"),
    pytest.param("Z", id="invalid_letter"),
]

# Operational pattern where the answer appears at the end of a long CoT. (prediction, expected) — options=["A","B","C","D"].
_MC_LONG_FILLER = (
    "Let's reason step by step. The question presents four options and we need to compare "
    "them carefully. Option A is about apples, option B is bananas, option C is cherries, "
    "and option D is dates. " * 5
)
MC_LETTER_COMPLEX = [
    # Explicit answer after long reasoning.
    pytest.param(_MC_LONG_FILLER + "Therefore the correct answer is C.", "C", id="long_cot_C"),
    # Even if a wrong candidate appears in the middle, the first match of the 'correct answer is' cue wins.
    pytest.param("First I thought A, but actually the correct answer is B.", "B", id="mid_wrong_then_B"),
    # When `\boxed{}` appears multiple times, LETTER_EXTRACT_PATTERNS[0] takes the last occurrence.
    pytest.param("Scratch work: \\boxed{A}. Final: \\boxed{D}.", "D", id="multiple_boxed_last_wins"),
    # Newline + tail-line fallback ("...B.\n\nB" — single token on the last line).
    pytest.param("After deliberation I'm sure.\n\nB", "B", id="tail_line_single_letter"),
    # Single character inside markdown bold.
    pytest.param("**Answer: C**", "C", id="markdown_bold_C"),
    # Korean reasoning + English answer character.
    pytest.param("두 번째 보기를 고민했지만 최종 정답은 D 입니다.", "D", id="ko_reasoning_D"),
]

# Complex cases for number scheme — options=[1,2,3,4].
MC_NUMBER_COMPLEX = [
    pytest.param("After comparing each option, the answer is 3.", "3", id="reasoning_then_3"),
    pytest.param("정답은 4 이다.", "4", id="ko_short_4"),
]

# Circled numbers — options=["①","②","③","④"].
MC_CIRCLED_COMPLEX = [
    pytest.param("선택지를 모두 검토한 결과, 정답은 ④ 입니다.", "④", id="ko_reasoning_circled_4"),
]
