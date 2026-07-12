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

"""Test case data for `omni_evaluator/postprocess/freeform`."""

import pytest


# (prediction, expected_answer).
FREEFORM_EN_CUE = [
    pytest.param("The answer is 14.", "14", id="answer_is_14"),
    pytest.param("Final answer: Paris", "Paris", id="final_answer_paris"),
    pytest.param('Answer: "Tokyo".', "Tokyo", id="quoted_tokyo"),
    pytest.param("The answer is 0.6 because of the ratio.", "0.6", id="float_with_because"),
]

FREEFORM_KO_CUE = [
    pytest.param("정답은 서울 입니다.", "서울", id="answer_seoul"),
    pytest.param("최종 정답은 42 이다.", "42", id="final_42"),
    pytest.param("답은 사과 입니다.", "사과", id="answer_apple"),
]

FREEFORM_NO_CUE = [
    pytest.param("Hello world.", id="en_plain"),
    pytest.param("그냥 평범한 문장입니다.", id="ko_plain"),
    pytest.param("", id="empty"),
]

# (raw_span, expected_cleaned) — for verifying _clean_span.
FREEFORM_CLEAN_SPAN = [
    pytest.param("  hello  ", "hello", id="outer_whitespace"),
    pytest.param('"hello"', "hello", id="double_quotes"),
    pytest.param("'hello'", "hello", id="single_quotes"),
    pytest.param("hello.", "hello", id="trailing_period"),
    pytest.param("hello;", "hello", id="trailing_semicolon"),
    pytest.param("hello,", "hello", id="trailing_comma"),
    pytest.param("hello", "hello", id="no_change"),
    pytest.param("“foo”", "foo", id="curly_double_quotes"),
    pytest.param("(foo)", "(foo)", id="parens_not_stripped"),  # The production code does not strip matched () [] pairs.
    pytest.param("hello。", "hello", id="cjk_period"),
]

# Long / various formats — shapes that frequently appear in production model responses.
_FREEFORM_LONG_FILLER = (
    "Let me reason through this problem carefully. " * 15
)
FREEFORM_EN_COMPLEX = [
    pytest.param(_FREEFORM_LONG_FILLER + "The answer is Paris.", "Paris", id="long_cot_paris"),
    # List alternative (`\[...\]`) captures *only the content inside brackets* — same as the
    # 'remove outer [...] when printing' rule in the production EXTRACT_PROMPT.
    pytest.param("The answer is [2007, 2008].", "2007, 2008", id="bracket_inner_only"),
    pytest.param("Answer: 1.45 because that is the total.", "1.45", id="float_with_because"),
    pytest.param("Step 1: think.\nStep 2: conclude.\nThe answer is 42.", "42", id="multiline_then_42"),
    pytest.param("Answer: '하늘색'.", "하늘색", id="ko_token_in_en_quotes"),
]

FREEFORM_KO_COMPLEX = [
    pytest.param("정답은 [1, 2, 3] 입니다.", "1, 2, 3", id="ko_bracket_inner_only"),
    pytest.param(
        "여러 단계로 추론한 결과, 최종 정답은 0.6 이며 그 근거는 비율이다.",
        "0.6",
        id="ko_long_then_decimal",
    ),
    pytest.param("Reasoning... 답은 사과 입니다.", "사과", id="ko_cue_after_en_prefix"),
]

# Regression safety net for when EN/KO cues appear together in one sentence — the loop in
# `_extract_freeform` matches both patterns and overwrites with the *last* (=KO) result.
# Common pattern where a production model starts incorrectly in English and corrects in Korean
# → KO taking priority is the desired behavior.
FREEFORM_EN_KO_MIXED_KO_WINS = [
    pytest.param(
        "The answer is 14. 그러나 정답은 42 이다.",
        "42",
        id="ko_overrides_en",
    ),
]

# API fallback empty response / 'N/A' normalization — any form must map to None.
FREEFORM_API_NULLISH_RETURNS = [
    pytest.param("N/A", id="upper"),
    pytest.param("n/a", id="lower"),
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace"),
]
