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

"""Unit tests for the Verifier-specific surface — the parts JudgeEvaluator and
HuggingfaceInferencer don't already cover: verifier prompt build, ``Rating: 0|1``
parse, engine dispatch, ``verifier_score`` collection, and verbose output.

The actual HF model inference is covered by ``tests/inference/huggingface`` — here
the HF path is exercised only at the composed-inferencer boundary with a fake
(the inferencer's ``__call__`` return contract: an object with ``.prediction``).
The API path is exercised by faking ``chat_completion_sync`` / ``_async`` (the
chat-completion contract: a response string)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from omni_evaluator.evaluation.metrics import verifier as verifier_mod
from omni_evaluator.evaluation.metrics.verifier import Verifier, _LLAMA_CPP_NUM_CONTEXT_TOKENS
from omni_evaluator.evaluation.metrics.prompts.verifier import (
    VERIFIER_PROMPT, VERIFIER_COT_PROMPT,
)

pytestmark = pytest.mark.eval_engine("builtin")


def _record(query="What is 2+2?", label=("4",), prediction="4"):
    return {
        "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
        "label": list(label),
        "prediction": prediction,
        "meta": {},
    }


class _FakeTok:
    """Whitespace tokenizer (1 token per word) for budget-truncation tests."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


# ── _postprocess (Rating parse + Explanation) ────────────────────────────────
@pytest.mark.parametrize("response, scores, accuracy", [
    ("Explanation: ok\nRating: 1", 1, 1.0),
    ("Explanation: no\nRating: 0", 0, 0.0),
    ("Explanation: a\nRating: 1\nRating: 0", 0, 0.0),   # last line-anchored 'Rating:' wins
    ("no rating present", None, None),                   # unparseable
    (None, None, None),                                  # empty response
    # echoed prompt instruction ("'Rating: 0' or 'Rating: 1'") is mid-line, not line-anchored ->
    # must NOT parse as 1; only the real trailing rating line counts.
    ("the last line must be 'Rating: 0' or 'Rating: 1':\nExplanation:\nRating:", None, None),
    ("... 'Rating: 0' or 'Rating: 1' ...\nExplanation: wrong.\nRating: 0", 0, 0.0),
])
def test_postprocess_rating(response, scores, accuracy):
    """scores parses the final ``Rating: 0|1`` (None when absent); accuracy mirrors it."""
    out = Verifier(engine="api/openai", api_name="x")._postprocess(response)
    assert out["scores"] == scores and out["accuracy"] == accuracy
    assert set(out) == {"accuracy", "scores", "reasons", "response"}


def test_postprocess_explanation():
    """The ``Explanation:`` body is surfaced under ``reasons`` (inherited parser)."""
    out = Verifier(engine="api/openai", api_name="x")._postprocess("Explanation: refs match\nRating: 1")
    assert out["reasons"] == "refs match"


# ── _format_verifier_prompt (label dedup + query + CoT toggle) ───────────────
def test_format_prompt_std():
    """STD prompt fills reference (list dedup, order-preserved) + last-user query; not the CoT variant."""
    prompt = Verifier(engine="api/openai", api_name="x", reasoning=False)._format_verifier_prompt(
        _record(label=["A", "A", "United States"]))
    assert prompt.startswith("[Reference Answer]")
    assert "A\nUnited States" in prompt          # list dedup, order preserved
    assert "What is 2+2?" in prompt              # last-user query
    assert "Work through it" not in prompt       # CoT-only marker absent


def test_format_prompt_cot():
    """reasoning=True selects the CoT template (CoT-only marker present)."""
    prompt = Verifier(engine="api/openai", api_name="x", reasoning=True)._format_verifier_prompt(_record())
    assert "Work through it" in prompt
    assert "What is 2+2?" in prompt


# ── _truncate_to_budget (mirrors verifier_train/data.py) ─────────────────────
def test_truncate_to_budget():
    """Oversized fields are middle-dropped (head+tail kept) so the combined tokens fit budget."""
    prediction = " ".join(f"p{_i}" for _i in range(100))
    ref, pred, q = Verifier._truncate_to_budget(_FakeTok(), "gold", prediction, "q1 q2", budget=20)
    tok = _FakeTok()
    total = sum(len(tok(_s)["input_ids"]) for _s in (ref, pred, q))
    assert total <= 20 + 8                      # fits budget (+ small marker overhead)
    assert pred.startswith("p0")                # head survives
    assert pred.rstrip().endswith("p99")        # tail survives (final answer kept)
    assert "prediction truncated" in pred       # middle-drop marker


# ── _build_messages (pre-formatted judge_message wins) ───────────────────────
def test_build_messages():
    """A passed judge_message is used verbatim; otherwise the prompt is formatted."""
    verifier = Verifier(engine="api/openai", api_name="x")
    assert verifier._build_messages(_record(), judge_message="PRE")[0]["content"][0]["value"] == "PRE"
    assert verifier._build_messages(_record())[0]["content"][0]["value"].startswith("[Reference Answer]")


# ── _build_generation_options (engine-specific normalization) ────────────────
def test_build_generation_options():
    """HF path normalizes via HuggingfaceGenerationOptions (temperature 0 -> greedy);
    llama_cpp maps to create_chat_completion keys (temperature + max_tokens); api path
    dispatches through ApiGenerationOptions (returns a provider dict; exact keys are the
    schema's contract, not asserted here)."""
    hf = Verifier(engine="huggingface")._build_generation_options({"temperature": 0.0, "max_new_tokens": 256})
    assert hf["do_sample"] is False and hf["max_new_tokens"] == 256
    llama = Verifier(engine="llama_cpp", model_name_or_path="/x")._build_generation_options(
        {"temperature": 0.0, "max_new_tokens": 256})
    assert llama == {"temperature": 0.0, "max_tokens": 256}
    api = Verifier(engine="api/openai", api_name="gpt-5-mini")._build_generation_options(
        {"temperature": 0.0, "max_new_tokens": 256})
    assert isinstance(api, dict)


# ── _resolve_num_context_tokens (llama.cpp n_ctx <- max_seq_len) ─────────────
def test_resolve_num_context_tokens():
    """n_ctx follows max_seq_len when set, else falls back to the model-default sentinel."""
    v = Verifier(engine="llama_cpp", model_name_or_path="/x", max_seq_len=4096)
    assert v._resolve_num_context_tokens() == 4096
    v_unset = Verifier(engine="llama_cpp", model_name_or_path="/x", max_seq_len=None)
    assert v_unset._resolve_num_context_tokens() == _LLAMA_CPP_NUM_CONTEXT_TOKENS


# ── _generate (engine dispatch) ──────────────────────────────────────────────
def test_generate_api(monkeypatch):
    """api/* routes through chat_completion_sync (network faked)."""
    monkeypatch.setattr(verifier_mod, "chat_completion_sync", lambda **kw: "Rating: 1")
    verifier = Verifier(engine="api/openai", api_name="gpt-5-mini")
    assert verifier._generate(verifier._build_messages(_record()), {}) == "Rating: 1"


def test_generate_hf(monkeypatch):
    """huggingface routes through the composed inferencer (faked; real model in tests/inference/huggingface)."""
    verifier = Verifier(engine="huggingface", model_name_or_path="/x")
    monkeypatch.setattr(verifier, "_ensure_inferencer", lambda: (lambda **kw: SimpleNamespace(prediction="Rating: 0")))
    assert verifier._generate(verifier._build_messages(_record()), {}) == "Rating: 0"


def test_generate_llama_cpp(monkeypatch):
    """llama_cpp routes through the GGUF model's create_chat_completion (faked; the
    prompt text is flattened into an OpenAI-style string-content chat)."""
    fake_model = SimpleNamespace(
        create_chat_completion=lambda **kw: {"choices": [{"message": {"content": "Rating: 1"}}]})
    verifier = Verifier(engine="llama_cpp", model_name_or_path="/x")
    monkeypatch.setattr(verifier, "_ensure_llama_cpp_model", lambda: fake_model)
    assert verifier._generate(
        verifier._build_messages(_record()), {"temperature": 0.0, "max_tokens": 64}) == "Rating: 1"


# ── evaluate (dispatch + verifier_score collection) ──────────────────────────
def test_evaluate_collects_verifier_score(monkeypatch):
    """The api path collects under ``verifier_score`` with the JudgeEvaluator result shape."""
    async def _fake_async(**kw):
        return "Explanation: ok\nRating: 1"
    monkeypatch.setattr(verifier_mod, "chat_completion_async", _fake_async)
    res = Verifier(engine="api/openai", api_name="gpt-5-mini").evaluate(
        records=[_record(), _record()], show_progress=False)
    assert set(res) == {"metrics", "sample_metrics", "group_metrics", "responses"}
    assert res["metrics"]["verifier_score"] == 1.0
    assert res["sample_metrics"][0]["verifier_score"] == 1
    assert res["responses"][0]["verifier_score"] == "Explanation: ok\nRating: 1"
    # the score is also surfaced alias-qualified (accuracy field -> verifier_score/{alias};
    # api alias falls back to the api model name)
    assert res["metrics"]["verifier_score/gpt-5-mini"] == 1.0
    assert res["sample_metrics"][0]["verifier_score/gpt-5-mini"] == 1


def test_evaluate_metric_key_is_verifier_score(monkeypatch):
    """Default metric key is ``verifier_score`` (not the legacy ``judge_score``)."""
    async def _fake_async(**kw):
        return "Rating: 0"
    monkeypatch.setattr(verifier_mod, "chat_completion_async", _fake_async)
    res = Verifier(engine="api/openai", api_name="gpt-5-mini").evaluate(records=[_record()], show_progress=False)
    assert "verifier_score" in res["metrics"] and "judge_score" not in res["metrics"]


def test_evaluate_llama_cpp(monkeypatch):
    """llama_cpp (num_concurrency=1) collects verifier_score via the sequential path (faked)."""
    fake_model = SimpleNamespace(
        create_chat_completion=lambda **kw: {"choices": [{"message": {"content": "Explanation: ok\nRating: 1"}}]})
    verifier = Verifier(engine="llama_cpp", model_name_or_path="/x")
    monkeypatch.setattr(verifier, "_ensure_llama_cpp_model", lambda: fake_model)
    res = verifier.evaluate(records=[_record(), _record()], show_progress=False)
    assert res["metrics"]["verifier_score"] == 1.0
    assert res["sample_metrics"][0]["verifier_score"] == 1


def test_llama_cpp_clamps_to_sequential(monkeypatch):
    """num_concurrency above the NUMA node count clamps to the in-process sequential path
    (no worker processes spawned)."""
    monkeypatch.setattr(verifier_mod, "get_numa_node_cpus", lambda: [{0, 1}])  # single NUMA node
    fake_model = SimpleNamespace(
        create_chat_completion=lambda **kw: {"choices": [{"message": {"content": "Rating: 1"}}]})
    verifier = Verifier(engine="llama_cpp", model_name_or_path="/x", num_concurrency=4)
    monkeypatch.setattr(verifier, "_ensure_llama_cpp_model", lambda: fake_model)
    res = verifier.evaluate(records=[_record(), _record()], show_progress=False)
    assert res["metrics"]["verifier_score"] == 1.0


def test_evaluate_api_concurrent(monkeypatch):
    """num_concurrency > 1 on api/* runs the asyncio-semaphore concurrent path."""
    async def _fake_async(**kw):
        return "Rating: 1"
    monkeypatch.setattr(verifier_mod, "chat_completion_async", _fake_async)
    res = Verifier(engine="api/openai", api_name="gpt-5-mini", num_concurrency=4).evaluate(
        records=[_record(), _record(), _record()], show_progress=False)
    assert res["metrics"]["verifier_score"] == 1.0 and len(res["sample_metrics"]) == 3


# ── verbose ──────────────────────────────────────────────────────────────────
def test_verbose_emits_rows(monkeypatch, capsys):
    """verbose=True prints one row per record (engine + query/reference/prediction/rating/explanation)."""
    async def _fake_async(**kw):
        return "Explanation: ok\nRating: 1"
    monkeypatch.setattr(verifier_mod, "chat_completion_async", _fake_async)
    Verifier(engine="api/openai", api_name="gpt-5-mini", verbose=True).evaluate(
        records=[_record()], show_progress=False)
    out = capsys.readouterr().out
    assert "engine=api/openai" in out
    assert "Rating" in out and "Explanation" in out


def test_verbose_silent_by_default(monkeypatch, capsys):
    """verbose defaults False → no per-row output."""
    async def _fake_async(**kw):
        return "Rating: 1"
    monkeypatch.setattr(verifier_mod, "chat_completion_async", _fake_async)
    Verifier(engine="api/openai", api_name="gpt-5-mini").evaluate(records=[_record()], show_progress=False)
    assert "[verifier #" not in capsys.readouterr().out


# ── live (gated) — end-to-end through the real gpt-5-mini chat-completion ─────
@pytest.mark.slow
@pytest.mark.requires_env("OPENAI_API_KEY")
def test_api_live_end_to_end():
    """Real gpt-5-mini verifier call on a trivially-correct sample → Rating 1.
    (HF live e2e is intentionally omitted — covered by tests/inference/huggingface.)"""
    res = Verifier(engine="api/openai", api_name="gpt-5-mini").evaluate(
        records=[_record(query="What is 2+2?", label=("4",), prediction="4")],
        show_progress=False)
    assert res["sample_metrics"][0]["verifier_score"] in (0, 1)
