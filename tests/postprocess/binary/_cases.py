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

"""Test case data for `omni_evaluator/postprocess/binary`."""

import pytest


BINARY_POSITIVE = [
    pytest.param("yes", id="en_yes"),
    pytest.param("true", id="en_true"),
    pytest.param("That is correct.", id="en_correct"),
    pytest.param("Right.", id="en_right"),
    pytest.param("affirmative", id="en_affirmative"),
    pytest.param("네", id="ko_yes"),
    pytest.param("맞다", id="ko_match"),
    pytest.param("참", id="ko_truth"),
    pytest.param("정답", id="ko_answer"),
    pytest.param("옳다", id="ko_correct"),
]

BINARY_NEGATIVE = [
    pytest.param("no", id="en_no"),
    pytest.param("false", id="en_false"),
    pytest.param("incorrect", id="en_incorrect"),
    pytest.param("wrong", id="en_wrong"),
    pytest.param("nope", id="en_nope"),
    pytest.param("아니다", id="ko_not"),
    pytest.param("틀리다", id="ko_wrong"),
    pytest.param("거짓", id="ko_false"),
    pytest.param("오답", id="ko_wrong_answer"),
    pytest.param("불일치", id="ko_mismatch"),
]

# Text with no polarity cue of either kind — is_binary False, extract None.
BINARY_NO_CUE = [
    pytest.param("maybe", id="en_maybe"),
    pytest.param("hello world", id="en_plain"),
    pytest.param("글쎄요", id="ko_unsure"),
    pytest.param("", id="empty"),
]

BINARY_NON_STRING = [
    pytest.param(None, id="none"),
    pytest.param(123, id="int"),
    pytest.param([], id="list"),
]

# Long CoT — needs sufficient length for `timeout(1)` guard to actually trigger backtracking.
_BINARY_LONG_FILLER = "Let me think about this step by step. " * 20  # ≈ 760 chars
_BINARY_LONG_FILLER_KO = "단계별로 차근차근 생각해 보겠습니다. " * 20  # ≈ 480 chars

BINARY_LONG_POSITIVE = [
    pytest.param(_BINARY_LONG_FILLER + "Final answer: yes.", id="en_long_reasoning_yes"),
    pytest.param(_BINARY_LONG_FILLER_KO + "결론적으로 답은 맞다.", id="ko_long_reasoning_match"),
]

BINARY_LONG_NEGATIVE = [
    pytest.param(_BINARY_LONG_FILLER + "Therefore the statement is false.", id="en_long_false"),
    pytest.param(_BINARY_LONG_FILLER_KO + "결론적으로 답은 틀리다.", id="ko_long_wrong"),
]

BINARY_LONG_NO_CUE = [
    pytest.param("Lorem ipsum dolor sit amet, " * 30, id="en_long_lipsum"),
    pytest.param("문장이 길지만 cue 가 없는 한국어 텍스트입니다. " * 15, id="ko_long_no_cue"),
]

# Mixed — EN/KO combined, or a case where a mid-sentence opposite cue is overridden by a final conclusion.
BINARY_MIXED_COMPLEX = [
    pytest.param("처음엔 맞다고 봤지만 다시 보니 거짓이다.", False, id="ko_match_mid_false_final"),
    pytest.param("The model said no, however the truth is yes.", False,
                 id="en_no_then_yes_negation_wins"),
    pytest.param("Yes 입니다.", True, id="en_ko_yes_combined"),
]
