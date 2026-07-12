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

"""Test case data for `omni_evaluator/postprocess/asr` — Korean/English only."""

import pytest


# extract dispatcher — non-str / length-1 predictions pass through as-is before dispatch.
ASR_PASSTHROUGH = [
    pytest.param("A", id="single_char"),
    pytest.param(None, id="none"),
    pytest.param(123, id="int"),
]


# ---------------------------------------------------------------------------
# _extract_default (English) — extracts transcription via PATTERN__ASR / `:"` / `: "` matching.
# ---------------------------------------------------------------------------

ASR_DEFAULT_EXTRACT = [
    # source docstring example — single-quoted transcription.
    pytest.param(
        "The transcription of the speech is: 'Are you certain that this is the Mediterranean?'",
        "Are you certain that this is the Mediterranean?",
        id="docstring_example",
    ),
    # short marker + plain text.
    pytest.param("output: hello world", "hello world", id="output_marker_plain"),
    # double-quoted variant.
    pytest.param('transcription: "hello world"', "hello world", id="double_quoted"),
]

# If none of PATTERN__ASR / `:"` / `: "` / SPLIT_PATTERNS branches match, output=None.
ASR_DEFAULT_NO_MATCH = [
    pytest.param("Hello world", id="no_colon"),
    pytest.param("", id="empty_string"),
]


# ---------------------------------------------------------------------------
# _extract_default_korean — marker / quoted Korean extraction + punctuation/bracket cleanup.
# ---------------------------------------------------------------------------

ASR_KOREAN_EXTRACT = [
    pytest.param(
        "Final Transcription: '안녕하세요 반갑습니다'",
        "안녕하세요 반갑습니다",
        id="marker_quoted",
    ),
    pytest.param(
        "output: '서울에 가고 싶습니다'",
        "서울에 가고 싶습니다",
        id="output_quoted",
    ),
    pytest.param(
        'transcribed text is: "한국어 텍스트입니다"',
        "한국어 텍스트입니다",
        id="double_quoted",
    ),
]

# If no Hangul remains after extraction, the function returns an empty string.
ASR_KOREAN_NO_HANGUL = [
    pytest.param("Hello world", id="english_only"),
    pytest.param("12345", id="digits_only"),
    pytest.param("", id="empty_string"),
]


# ---------------------------------------------------------------------------
# normalize_korean — KsponSpeech marker / bracket / punctuation / whitespace normalization (pure Python).
# ---------------------------------------------------------------------------

ASR_NORMALIZE_KOREAN = [
    # KsponSpeech marker `X/` — removes single and multiple occurrences via `\b\S+/(?=\s|$)`.
    pytest.param("안녕 b/ 하세요", "안녕 하세요", id="kspon_marker_b"),
    pytest.param("o/ 안녕하세요 어/", "안녕하세요", id="kspon_markers_multi"),
    # bracket content — both () and [] are removed.
    pytest.param("안녕(영어) 하세요", "안녕 하세요", id="parens_removed"),
    pytest.param("안녕[배경음] 하세요", "안녕 하세요", id="brackets_removed"),
    # ASCII punctuation.
    pytest.param("안녕, 하세요.", "안녕 하세요", id="ascii_punct"),
    # Korean / fullwidth punctuation.
    pytest.param("안녕、 하세요。", "안녕 하세요", id="korean_punct"),
    # multiple whitespace collapse.
    pytest.param("안녕   하세요", "안녕 하세요", id="whitespace_collapse"),
]


# ---------------------------------------------------------------------------
# normalize_chinese — code-switch (Chinese + Latin) transcript normalizer:
# normalize_default chain + per-CJK char spacing (Latin runs preserved as-is).
# ---------------------------------------------------------------------------

ASR_NORMALIZE_CHINESE = [
    # Pure CJK — each hanzi gets a space, no other change.
    pytest.param("你好世界", "你 好 世 界", id="pure_cjk"),
    # Fullwidth Chinese punctuation — absorbed by jiwer RemovePunctuation (Unicode P).
    pytest.param("你好,世界。", "你 好 世 界", id="cjk_with_punct"),
    # Code-switch — Latin run preserved whole; only CJK runs are spaced.
    pytest.param("hello 你好 world", "hello 你 好 world", id="code_switch_latin"),
    # Whitespace collapsed by jiwer RemoveMultipleSpaces before CJK spacing.
    pytest.param("你好   世界", "你 好 世 界", id="whitespace_collapse"),
    # Lowercase + digit→word + CJK spacing (full normalize_default pipeline kicks in).
    pytest.param("HELLO 1 你好", "hello one 你 好", id="lower_digit_with_cjk"),
    # Bracketed/non-speech segments removed by normalize_default before CJK spacing.
    pytest.param("(bgm) 你好 [noise]", "你 好", id="brackets_and_nonspeech"),
    # English contraction expanded; CJK preserved + spaced.
    pytest.param("I'm 说 hello", "i am 说 hello", id="contraction_with_cjk"),
]
