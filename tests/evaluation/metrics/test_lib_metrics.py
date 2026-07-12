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

"""Unit-validates library-dependent metrics (wer/cer/mer/cider/bleu/rouge/meteor) of TextEvaluator."""
from __future__ import annotations

import pytest

from omni_evaluator.evaluation.metrics.text_evaluator import TextEvaluator

pytestmark = pytest.mark.eval_engine("builtin")


# ─────────────────────────────────────────────────────────────────────────────
#  Offline deterministic: jiwer (wer / cer / mer)
# ─────────────────────────────────────────────────────────────────────────────

WER = [  # (labels: List[List[str]], predictions: List[str], expected_wer)
    pytest.param([["the cat sat"]], ["the cat sat"], 0.0, id="identical_0"),
    pytest.param([["the cat sat"]], ["the dog sat"], 1 / 3, id="one_word_sub_over_3"),
]

CER = [
    pytest.param([["abc"]], ["abc"], 0.0, id="identical_0"),
    pytest.param([["abc"]], ["abd"], 1 / 3, id="one_char_sub_over_3"),
]

MER = [  # (labels: List[str], prediction: str, expected)
    pytest.param(["the cat sat"], "the cat sat", 0.0, id="identical_0"),
    pytest.param(["the cat sat"], "the dog sat", 100 / 3, id="one_word_sub_scaled_x100"),
]


@pytest.mark.parametrize("labels, predictions, expected", WER)
def test_wer(labels, predictions, expected):
    """word error rate = edits/reference_length (jiwer, offline)."""
    metrics, _ = TextEvaluator.compute_wer(labels=labels, predictions=predictions)
    assert metrics["wer"] == pytest.approx(expected)


@pytest.mark.parametrize("labels, predictions, expected", CER)
def test_cer(labels, predictions, expected):
    """character error rate = char edits/reference_length (jiwer, offline)."""
    metrics, _ = TextEvaluator.compute_cer(labels=labels, predictions=predictions)
    assert metrics["cer"] == pytest.approx(expected)


@pytest.mark.parametrize("labels, prediction, expected", MER)
def test_mer(labels, prediction, expected):
    """mixed error rate = wer*100 (jiwer, offline). Korean normalization is disabled."""
    score, _ = TextEvaluator.compute_mer(labels=labels, prediction=prediction, do_normalize=False)
    assert score == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────────────────────
#  Offline deterministic: pycocoevalcap CIDEr
# ─────────────────────────────────────────────────────────────────────────────

def test_cider_single_doc_is_zero():
    """CIDEr is 0.0 for a single sample because idf=0."""
    scores, _ = TextEvaluator.compute_cider([["the cat sat"]], ["the cat sat"])
    assert scores["cider"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Network required: hf_evaluate.load (bleu / rouge / meteor) — slow gate
# ─────────────────────────────────────────────────────────────────────────────

def _compute_or_skip(fn):
    """Skips on hf_evaluate.load failure to prevent hard failures."""
    try:
        return fn()
    except Exception as ex:  # noqa: BLE001 — load failures have varied causes (network/ImportError, etc.)
        pytest.skip(f"hf_evaluate metric unavailable: {type(ex).__name__}: {str(ex)[:120]}")


@pytest.mark.slow
def test_bleu_identical():
    """Identical 3-word sentence: bleu-1/2/3=1.0, **bleu-4=0.0** (cannot form 4-grams)."""
    output, _ = _compute_or_skip(
        lambda: TextEvaluator.compute_bleu([["the cat sat"]], ["the cat sat"])
    )
    assert output["bleu-1"] == pytest.approx(1.0)
    assert output["bleu-2"] == pytest.approx(1.0)
    assert output["bleu-3"] == pytest.approx(1.0)
    assert output["bleu-4"] == pytest.approx(0.0)


@pytest.mark.slow
def test_rouge_identical():
    """Identical sentences yield rouge1/rougeL = 1.0 (requires rouge_score package → skip if absent)."""
    scores = _compute_or_skip(
        lambda: TextEvaluator.compute_rouge([["the cat sat"]], ["the cat sat"])
    )
    assert scores["rouge1"] == pytest.approx(1.0)
    assert scores["rougeL"] == pytest.approx(1.0)


@pytest.mark.slow
def test_meteor_identical_is_high():
    """METEOR for identical sentences is less than 1.0 due to fragmentation penalty but exceeds 0.9."""
    scores = _compute_or_skip(
        lambda: TextEvaluator.compute_meteor([["the cat sat"]], ["the cat sat"])
    )
    assert scores["meteor"] > 0.9
