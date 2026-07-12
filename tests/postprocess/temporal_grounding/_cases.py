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

"""Test case data for `omni_evaluator/postprocess/temporal_grounding`."""

import pytest


# ── Separator-pair (Stage 2 primary) ────────────────────────────────────────
# Same `[24.3, 30.4]` answer wrapped in various separators / prose / EN-KO.
TG_PAIR_PROSE = [
    pytest.param("The action occurs from 24.3 to 30.4 seconds.", id="en_to"),
    pytest.param("Answer: 24.3 - 30.4", id="en_hyphen"),
    pytest.param("24.3 – 30.4 seconds", id="en_en_dash"),
    pytest.param("24.3 — 30.4 seconds", id="en_em_dash"),
    pytest.param("24.3 ~ 30.4", id="en_tilde"),
    pytest.param("24.3 until 30.4", id="en_until"),
    pytest.param("[24.3, 30.4]", id="en_bracket_comma"),
    pytest.param("정답은 24.3부터 30.4까지입니다.", id="ko_to_hyphen"),
    pytest.param("결론: 24.3 - 30.4", id="ko_hyphen"),
]

# ── MM:SS / HH:MM:SS — `_to_seconds` token conversion ────────────────────────
TG_TIMESTAMP_MM_SS = [
    pytest.param("00:24.3 - 00:30.4", "[24.3, 30.4]", id="mm_ss_fractional"),
    pytest.param("01:24 to 02:30", "[84.0, 150.0]", id="mm_ss_integer"),
]

TG_TIMESTAMP_HH_MM_SS = [
    pytest.param("0:01:24 - 0:02:30", "[84.0, 150.0]", id="hh_mm_ss"),
    pytest.param("1:00:00 to 1:00:30", "[3600.0, 3630.0]", id="hh_mm_ss_hour"),
]

# ── Last-wins — model emits draft then final answer; final wins. ────────────
TG_LAST_WINS = [
    pytest.param(
        "First guess 12-15, but the right answer is 24.3 - 30.4 seconds.",
        "[24.3, 30.4]",
        id="en_draft_then_final",
    ),
]

# ── Fallback: first two bare tokens when no separator pair ──────────────────
TG_BARE_FALLBACK = [
    pytest.param("Start 24.3 End 30.4", "[24.3, 30.4]", id="en_no_sep_two_tokens"),
]

# ── No answer / non-string / empty ─────────────────────────────────────────
TG_NO_ANSWER = [
    pytest.param("I cannot determine the interval.", id="en_refusal"),
    pytest.param("모르겠습니다.", id="ko_refusal"),
    pytest.param("answer 24.3", id="single_token"),
]

TG_NON_STRING = [
    pytest.param(None, id="none"),
    pytest.param(123, id="int"),
    pytest.param([], id="list"),
]

TG_EMPTY = [
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace"),
]

# ── Invariant: 0 ≤ s < e ≤ 1e6; reverse / negative / out-of-range rejected ─
TG_INVARIANT_REJECTED = [
    pytest.param("30.4 - 24.3", id="reversed"),         # s > e
    pytest.param("-5 to 10", id="negative"),            # s < 0
    pytest.param("3.0 - 3.0", id="zero_duration"),      # s == e
]

# Reversed pair but a valid pair follows later — last-wins picks the valid one.
TG_INVARIANT_LATER_VALID = [
    pytest.param("first wrong: 30.4 - 24.3, then 24.3 - 30.4", "[24.3, 30.4]", id="reversed_then_valid"),
]

# ── NFKC normalize / `<think>` strip ────────────────────────────────────────
TG_NFKC_FULLWIDTH = [
    pytest.param("２４．３ － ３０．４", id="fullwidth_digits_dash"),
]

TG_THINK_STRIPPED = [
    pytest.param("<think>maybe 12-15</think>\n24.3 - 30.4", id="cot_trace_then_answer"),
]
