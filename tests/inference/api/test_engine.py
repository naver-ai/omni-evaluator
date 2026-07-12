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

"""Unit tests for inference/api/engine.py + per-provider live smoke."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from omegaconf import OmegaConf

from omni_evaluator.inference.api import engine
from tests.inference._smoke_inputs import (
    INFERENCE_CONFIG_DIR,
    LIVE_SMOKE_IMAGE_QUERY,
    LIVE_SMOKE_MAX_NEW_TOKENS,
    LIVE_SMOKE_TEXT_QUERY,
    LIVE_SMOKE_VIDEO_QUERY,
    LIVE_SMOKE_GOOGLE_MAX_NEW_TOKENS,
)
from tests.inference.test_engine_common import EngineMainCommonTests, _apply_output_to_record


# ── Per-provider endpoint config ─────────────────────────────────────────
# If the file location changes, update only this one place.
_API_CONFIG_OPENAI = INFERENCE_CONFIG_DIR / "api_openai.yaml"
_API_CONFIG_ANTHROPIC = INFERENCE_CONFIG_DIR / "api_anthropic.yaml"
_API_CONFIG_GOOGLE = INFERENCE_CONFIG_DIR / "api_google.yaml"


def _log_smoke_response(
    provider: str, api_name: str, modality: str, query: str, pred, latency,
) -> None:
    """Print the actual input/output of a live smoke test to stdout (check with `pytest -s`)."""
    latency_str = f"{latency:.3f}s" if isinstance(latency, (int, float)) else str(latency)
    print(
        f"\n  ── [{provider} live smoke / {modality}] ──"
        f"\n    api_name : {api_name}"
        f"\n    query    : {query!r}"
        f"\n    pred     : {pred!r}"
        f"\n    latency  : {latency_str}"
    )


@pytest.fixture(autouse=True)
def _cleanup_google_uploaded_files():
    """Delete files uploaded to the Google File API on a best-effort basis at test teardown."""
    try:
        from omni_evaluator.api.google.chat_completions import UPLOADED_FILES_CACHE
    except Exception:
        yield
        return

    before = set(UPLOADED_FILES_CACHE.values())
    yield
    after = set(UPLOADED_FILES_CACHE.values())
    new_names = after - before
    if not new_names:
        return

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return

    try:
        from google import genai
    except Exception:
        return

    client = genai.Client(api_key=api_key)
    for name in new_names:
        try:
            client.files.delete(name=name)
        except Exception:
            # best-effort — transient hiccup / already deleted / permission — ignore all
            pass


@pytest.mark.inference_engine("api")
@pytest.mark.timeout(60)
class TestApiEngineMain(EngineMainCommonTests):
    """Unit tests for API `engine.main()` + per-provider live smoke.

    Fixture source map:

        ── Defined in this class (base fixture overrides) ──
        engine_module           : engine module under test
        main_kwargs_factory     : baseline kwargs builder for engine.main()
        patch_boundary          : batch_chat_completion_sync monkeypatch helper

        ── Defined in EngineMainCommonTests (base) ──
        fake_inference_output   : response dict builder (uses base implementation as-is)

        ── Defined in tests/inference/conftest.py ──
        record_factory          : minimal Record list builder (text-only)
        image_record_url_factory    : image+text Record builder
        video_record_url_factory    : video+text Record builder
        task_config_factory     : minimal TaskConfig builder

        ── pytest builtin ──
        monkeypatch             : function patcher
    """

    # ── Common contract fixture overrides ────────────────────────────

    @pytest.fixture
    def engine_module(self):
        return engine

    @pytest.fixture
    def main_kwargs_factory(self):
        """Baseline kwargs for mock-based tests — api_name etc. are placeholders."""
        def _factory(records, task_config, **overrides):
            base = dict(
                rank=0, world_size=1, run_index=0, num_runs=1,
                inference_engine="api", evaluation_engine="builtin",
                benchmark="dummy", evaluation_method="generation",
                benchmark_idx=0, num_benchmarks=1,
                benchmark_dataset=records, task_config=task_config,
                default_generation_options={"temperature": 0.0},
                api_name="gpt-4o-mini",
                do_async=False, debug=True, verbose=False,
            )
            base.update(overrides)
            return base
        return _factory

    # ── Boundary abstraction (base fixture override) ──────────────────

    @pytest.fixture
    def patch_boundary(self, monkeypatch, fake_inference_output):
        """Monkeypatch `batch_chat_completion_sync` with a fake."""
        def _patch(output_fn=None):
            if output_fn is None:
                def output_fn(idx, messages):
                    return fake_inference_output(prediction=f"pred-{idx}")
            calls: List[Dict[str, Any]] = []

            def _fake_sync(*args, **kwargs):
                records = kwargs["records"]
                for i, record in enumerate(records):
                    calls.append({"messages": record.messages, "kwargs": kwargs})
                    try:
                        out = output_fn(i, record.messages)
                    except Exception:
                        out = None
                    if out is not None:
                        _apply_output_to_record(record, out)
                return records

            monkeypatch.setattr(engine, "batch_chat_completion_sync", _fake_sync)
            return calls
        return _patch

    # ── Batch boundary-specific contracts ────────────────────────────
    # Cannot occur in per-record engines, so omitted from base and verified in child.

    def test_empty_batch_returns_none(
        self, monkeypatch,
        engine_module, main_kwargs_factory,
        record_factory, task_config_factory,
    ):
        """`main()` returns `None` when `batch_chat_completion_sync` returns an empty list."""
        records = record_factory(n=2)
        monkeypatch.setattr(
            engine_module, "batch_chat_completion_sync",
            lambda *a, **kw: [],
        )
        out = engine_module.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=2),
        ))
        assert out is None

    def test_batch_exception_returns_none(
        self, monkeypatch,
        engine_module, main_kwargs_factory,
        record_factory, task_config_factory,
    ):
        """`main()` returns `None` (logging only) when `batch_chat_completion_sync` raises."""
        records = record_factory(n=2)

        def _raise(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(engine_module, "batch_chat_completion_sync", _raise)
        out = engine_module.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=2),
        ))
        assert out is None

    # ── API-specific tests ───────────────────────────────────────────

    def test_do_async(
        self, monkeypatch,
        engine_module, main_kwargs_factory, fake_inference_output,
        record_factory, task_config_factory,
    ):
        """`do_async=True` triggers the async batch and does not call sync.

        Verifications:
        - `batch_chat_completion_async` is called, `batch_chat_completion_sync` is not
        - `batch_size` is passed as the `semaphore_size` argument
        - prediction from the response is correctly applied to the record
        """
        records = record_factory(n=2)
        async_calls, sync_calls = [], []

        async def _fake_async(*a, **kw):
            records = kw["records"]
            async_calls.append((len(records), kw.get("semaphore_size")))
            for record in records:
                _apply_output_to_record(record, fake_inference_output(prediction="ap"))
            return records

        def _fake_sync(*a, **kw):
            sync_calls.append(1)
            return None

        monkeypatch.setattr(engine_module, "batch_chat_completion_async", _fake_async)
        monkeypatch.setattr(engine_module, "batch_chat_completion_sync", _fake_sync)

        out = engine_module.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=2),
            do_async=True,
            batch_size=4,
        ))

        assert async_calls == [(2, 4)]
        assert sync_calls == []
        assert len(out) == 2
        assert all(r["prediction"] == "ap" for r in out)

    # ── Provider × modality live smoke ───────────────────────────────
    # OpenAI/Anthropic: support text+image, not video. Google(Gemini): supports all modalities.
    # Unsupported combinations are expected to fail — the purpose is to verify endpoint capability.

    def _run_api_smoke(
        self,
        provider: str,
        config_path,
        modality: str,
        records,
        query: str,
        main_kwargs_factory,
        task_config_factory,
        max_new_tokens: int = LIVE_SMOKE_MAX_NEW_TOKENS,
    ) -> None:
        config = OmegaConf.load(config_path)
        out = engine.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=len(records)),
            api_name=str(config.api_name),
            default_generation_options={
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
        ))
        assert isinstance(out, list) and len(out) == 1
        pred = out[0]["prediction"]
        _log_smoke_response(
            provider=provider, api_name=str(config.api_name),
            modality=modality, query=query, pred=pred,
            latency=out[0]["latency"],
        )
        assert isinstance(pred, str) and len(pred) > 0

    # ── OpenAI ────────────────────────────────────────────────
    @pytest.mark.slow
    @pytest.mark.requires_env("OPENAI_API_KEY")
    def test_openai_smoke_text(
        self, main_kwargs_factory, record_factory, task_config_factory,
    ):
        """OpenAI endpoint returns a non-empty prediction for a text-only query."""
        self._run_api_smoke(
            provider="OpenAI", config_path=_API_CONFIG_OPENAI,
            modality="text",
            records=record_factory(n=1, query=LIVE_SMOKE_TEXT_QUERY),
            query=LIVE_SMOKE_TEXT_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )

    @pytest.mark.slow
    @pytest.mark.requires_env("OPENAI_API_KEY")
    def test_openai_smoke_image(
        self, main_kwargs_factory, image_record_url_factory, task_config_factory,
    ):
        """OpenAI endpoint returns a non-empty prediction for an image+text query."""
        self._run_api_smoke(
            provider="OpenAI", config_path=_API_CONFIG_OPENAI,
            modality="image",
            records=image_record_url_factory(n=1, query=LIVE_SMOKE_IMAGE_QUERY),
            query=LIVE_SMOKE_IMAGE_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )

    @pytest.mark.slow
    @pytest.mark.requires_env("OPENAI_API_KEY")
    def test_openai_smoke_video(
        self, main_kwargs_factory, video_record_url_factory, task_config_factory,
    ):
        """OpenAI endpoint returns a non-empty prediction for a video+text query."""
        self._run_api_smoke(
            provider="OpenAI", config_path=_API_CONFIG_OPENAI,
            modality="video",
            records=video_record_url_factory(n=1, query=LIVE_SMOKE_VIDEO_QUERY),
            query=LIVE_SMOKE_VIDEO_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )

    # ── Anthropic ─────────────────────────────────────────────

    @pytest.mark.slow
    @pytest.mark.requires_env("ANTHROPIC_API_KEY")
    def test_anthropic_smoke_text(
        self, main_kwargs_factory, record_factory, task_config_factory,
    ):
        """Anthropic endpoint text-only smoke — Anthropic version of `test_openai_smoke_text`."""
        self._run_api_smoke(
            provider="Anthropic", config_path=_API_CONFIG_ANTHROPIC,
            modality="text",
            records=record_factory(n=1, query=LIVE_SMOKE_TEXT_QUERY),
            query=LIVE_SMOKE_TEXT_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )

    @pytest.mark.slow
    @pytest.mark.requires_env("ANTHROPIC_API_KEY")
    def test_anthropic_smoke_image(
        self, main_kwargs_factory, image_record_url_factory, task_config_factory,
    ):
        """Anthropic endpoint image+text smoke — Anthropic version of `test_openai_smoke_image`."""
        self._run_api_smoke(
            provider="Anthropic", config_path=_API_CONFIG_ANTHROPIC,
            modality="image",
            records=image_record_url_factory(n=1, query=LIVE_SMOKE_IMAGE_QUERY),
            query=LIVE_SMOKE_IMAGE_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )

    @pytest.mark.slow
    @pytest.mark.requires_env("ANTHROPIC_API_KEY")
    def test_anthropic_smoke_video(
        self, main_kwargs_factory, video_record_url_factory, task_config_factory,
    ):
        """Anthropic endpoint returns a non-empty prediction for a video+text query."""
        self._run_api_smoke(
            provider="Anthropic", config_path=_API_CONFIG_ANTHROPIC,
            modality="video",
            records=video_record_url_factory(n=1, query=LIVE_SMOKE_VIDEO_QUERY),
            query=LIVE_SMOKE_VIDEO_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )

    # ── Google ────────────────────────────────────────────────

    @pytest.mark.slow
    @pytest.mark.requires_env("GOOGLE_API_KEY")
    def test_google_smoke_text(
        self, main_kwargs_factory, record_factory, task_config_factory,
    ):
        """Google (Gemini) endpoint text-only smoke — Google version of `test_openai_smoke_text`."""
        self._run_api_smoke(
            provider="Google", config_path=_API_CONFIG_GOOGLE,
            modality="text",
            records=record_factory(n=1, query=LIVE_SMOKE_TEXT_QUERY),
            query=LIVE_SMOKE_TEXT_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
            max_new_tokens=LIVE_SMOKE_GOOGLE_MAX_NEW_TOKENS,
        )

    @pytest.mark.slow
    @pytest.mark.requires_env("GOOGLE_API_KEY")
    def test_google_smoke_image(
        self, main_kwargs_factory, image_record_url_factory, task_config_factory,
    ):
        """Google (Gemini) endpoint image+text smoke — Google version of `test_openai_smoke_image`."""
        self._run_api_smoke(
            provider="Google", config_path=_API_CONFIG_GOOGLE,
            modality="image",
            records=image_record_url_factory(n=1, query=LIVE_SMOKE_IMAGE_QUERY),
            query=LIVE_SMOKE_IMAGE_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
            max_new_tokens=LIVE_SMOKE_GOOGLE_MAX_NEW_TOKENS,
        )

    @pytest.mark.slow
    @pytest.mark.requires_env("GOOGLE_API_KEY")
    @pytest.mark.timeout(600)
    def test_google_smoke_video(
        self, main_kwargs_factory, video_record_url_factory, task_config_factory,
    ):
        """Google (Gemini) endpoint returns a non-empty prediction for a video+text query."""
        self._run_api_smoke(
            provider="Google", config_path=_API_CONFIG_GOOGLE,
            modality="video",
            records=video_record_url_factory(n=1, query=LIVE_SMOKE_VIDEO_QUERY),
            query=LIVE_SMOKE_VIDEO_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )
