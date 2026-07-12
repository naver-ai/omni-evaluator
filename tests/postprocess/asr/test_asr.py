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

"""Unit tests for omni_evaluator/postprocess/asr."""

import os

import pytest

from omni_evaluator.postprocess.asr import _ENGLISH_SPELLING_MAPPING_PATH, AsrProcessor

from ._cases import (
    ASR_DEFAULT_EXTRACT,
    ASR_DEFAULT_NO_MATCH,
    ASR_KOREAN_EXTRACT,
    ASR_KOREAN_NO_HANGUL,
    ASR_NORMALIZE_CHINESE,
    ASR_NORMALIZE_KOREAN,
    ASR_PASSTHROUGH,
)


# `tests/postprocess/CLAUDE.md §8` — regex catastrophic-backtracking guard.
pytestmark = pytest.mark.timeout(1)

_ENGLISH_RESOURCE_AVAILABLE = os.path.exists(_ENGLISH_SPELLING_MAPPING_PATH)


# ============================================================================
# CASE STUDY — branch tests parametrized with case data from `_cases.py`.
# ============================================================================


@pytest.mark.parametrize("prediction", ASR_PASSTHROUGH)
def test_extract_passthrough(prediction):
    """non-str / length-1 prediction → returned as-is without entering dispatch."""
    assert AsrProcessor.extract(prediction=prediction) == prediction


@pytest.mark.parametrize("prediction,expected", ASR_DEFAULT_EXTRACT)
def test_extract_default(prediction, expected):
    """version_name=None default branch — extracts transcription after ':' using PATTERN__ASR."""
    assert AsrProcessor.extract(prediction=prediction) == expected


@pytest.mark.parametrize("prediction", ASR_DEFAULT_NO_MATCH)
def test_extract_default_no_match_returns_none(prediction):
    """Returns None when none of PATTERN__ASR / `:"` / `: "` / SPLIT_PATTERNS match."""
    assert AsrProcessor.extract(prediction=prediction) is None


@pytest.mark.parametrize("prediction,expected", ASR_KOREAN_EXTRACT)
def test_extract_korean(prediction, expected):
    """version_name='default_korean' — extracts quoted Korean text after marker."""
    assert AsrProcessor.extract(prediction=prediction, version_name="default_korean") == expected


@pytest.mark.parametrize("prediction", ASR_KOREAN_NO_HANGUL)
def test_extract_korean_no_hangul_returns_empty(prediction):
    """version_name='default_korean' — returns empty string when no Hangul remains after cleanup."""
    assert AsrProcessor.extract(prediction=prediction, version_name="default_korean") == ""


@pytest.mark.parametrize("text,expected", ASR_NORMALIZE_KOREAN)
def test_normalize_korean(text, expected):
    """Normalizes KsponSpeech markers / parentheses / ASCII and Korean punctuation / multiple spaces."""
    assert AsrProcessor.normalize_korean(text) == expected


# ============================================================================
# Standalone demonstrations — single inline input / raises / external dependency guards.
# ============================================================================


def test_extract_qwen_not_implemented():
    """version_name='qwen' branch raises NotImplementedError from `_extract_qwen` — not yet implemented in source."""
    with pytest.raises(NotImplementedError, match="_extract_qwen"):
        AsrProcessor.extract(prediction="some prediction", version_name="qwen")


def test_extract_api_path_not_implemented():
    """api_name specified raises NotImplementedError from `_extract_asr_api` — API fallback not yet implemented."""
    with pytest.raises(NotImplementedError, match="_extract_asr_api"):
        AsrProcessor.extract(prediction="some prediction", api_name="gpt-4o-mini")


@pytest.mark.timeout(30)  # `transformers` first import is heavy, overrides file-level timeout(1) — not a regex check.
@pytest.mark.requires_extra("transformers", "jiwer")
@pytest.mark.skipif(
    not _ENGLISH_RESOURCE_AVAILABLE,
    reason=f"english.json mapping missing at {_ENGLISH_SPELLING_MAPPING_PATH} — ASR resource not deployed in this environment",
)
def test_normalize_default_invariants():
    """Whisper-style English normalization — core invariants: lowercase / punctuation removal / whitespace collapse.

    Dependencies: `transformers` + `jiwer` packages + `asr/resources/english.json` resource file.
    Auto-skipped if any one is missing. Specific normalize steps are the external library's responsibility and are not verified here.
    """
    result = AsrProcessor.normalize_default("HELLO, world!")
    assert isinstance(result, str)
    assert result == result.lower()              # lowercasing applied
    assert "," not in result and "!" not in result  # punctuation removed
    assert "  " not in result                    # multiple spaces collapsed


@pytest.mark.parametrize("text,expected", ASR_NORMALIZE_CHINESE)
@pytest.mark.timeout(30)  # chains normalize_default — same heavy-import budget.
@pytest.mark.requires_extra("transformers", "jiwer")
@pytest.mark.skipif(
    not _ENGLISH_RESOURCE_AVAILABLE,
    reason=f"english.json mapping missing at {_ENGLISH_SPELLING_MAPPING_PATH} — ASR resource not deployed in this environment",
)
def test_normalize_chinese(text, expected):
    """normalize_default pipeline + per-CJK char spacing; Latin runs preserved whole."""
    assert AsrProcessor.normalize_chinese(text) == expected
