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

"""Unit tests for inference/vllm/engine.py + vLLM live smoke.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from omegaconf import OmegaConf

from omni_evaluator.inference.vllm import engine
from tests.conftest import skip_if_vllm_down
from tests.inference._smoke_inputs import (
    INFERENCE_CONFIG_DIR,
    LIVE_SMOKE_IMAGE_QUERY,
    LIVE_SMOKE_MAX_NEW_TOKENS,
    LIVE_SMOKE_TEXT_QUERY,
    LIVE_SMOKE_VIDEO_QUERY,
)
from tests.inference.test_engine_common import EngineMainCommonTests, _apply_output_to_record


_VLLM_CONFIG_V1 = INFERENCE_CONFIG_DIR / "vllm.yaml"

# ── placeholder URL for mock tests ───────────────────────────────────────────
# Must be a str so execution enters the healthcheck → batch path. The autouse fixture monkeypatches healthcheck to block real requests.
_MOCK_VLLM_URL = "http://placeholder:0000"


def _log_smoke_response(
    provider: str, url: str, modality: str, query: str, pred, latency,
) -> None:
    """Prints the actual input/output of a live smoke to stdout. Check with `pytest -s`."""
    latency_str = f"{latency:.3f}s" if isinstance(latency, (int, float)) else str(latency)
    print(
        f"\n  ── [{provider} live smoke / {modality}] ──"
        f"\n    url      : {url}"
        f"\n    query    : {query!r}"
        f"\n    pred     : {pred!r}"
        f"\n    latency  : {latency_str}"
    )


@pytest.mark.inference_engine("vllm")
@pytest.mark.timeout(60)
class TestVllmEngineMain(EngineMainCommonTests):
    """vLLM `engine.main()` unit tests + live smoke.

    Fixture source map:

        ── Defined in this class (base fixture override) ──
        engine_module           : vLLM engine module under test
        main_kwargs_factory     : baseline kwargs builder for engine.main()
        patch_boundary          : batch_chat_completion_sync monkeypatch helper
        fake_inference_output   : vLLM response schema (includes reasoning_content/perplexities)

        ── autouse in this class (mock tests only) ──
        _patch_healthcheck      : sets healthcheck=True for non-live-smoke tests

        ── tests/inference/conftest.py ──
        record_factory, image_record_factory, video_record_factory, task_config_factory

        ── pytest builtin ──
        monkeypatch
    """

    # ── common contract fixture overrides ────────────────────────────

    @pytest.fixture
    def engine_module(self):
        return engine

    @pytest.fixture
    def main_kwargs_factory(self):
        """Baseline kwargs for vLLM `engine.main()`."""
        def _factory(records, task_config, **overrides):
            base = dict(
                rank=0, world_size=1, run_index=0, num_runs=1,
                inference_engine="vllm", evaluation_engine="builtin",
                benchmark="dummy", evaluation_method="generation",
                benchmark_idx=0, num_benchmarks=1,
                benchmark_dataset=records, task_config=task_config,
                default_generation_options={},
                url=_MOCK_VLLM_URL,
                trust_remote_code=True,
                skip_chat_template=False,
                do_async=False, debug=True, verbose=False,
            )
            base.update(overrides)
            return base
        return _factory

    @pytest.fixture
    def fake_inference_output(self):
        """vLLM response dict builder."""
        def _factory(prediction: str = "x"):
            return {
                "prediction": prediction,
                "reasoning_content": None,
                "perplexities": None,
                "tool_calls": None,
                "latency": 0.0,
            }
        return _factory

    # ── autouse: healthcheck monkeypatch (mock tests only) ──

    @pytest.fixture(autouse=True)
    def _patch_healthcheck(self, monkeypatch, request):
        """Monkeypatches healthcheck to True for mock tests — live smokes use the real healthcheck."""
        if request.node.get_closest_marker("requires_env") is not None:
            return  # smoke — use real healthcheck
        monkeypatch.setattr(engine, "healthcheck", lambda **kw: True)

    # ── boundary abstraction (base fixture override) ──────────────────

    @pytest.fixture
    def patch_boundary(self, monkeypatch, fake_inference_output):
        """Monkeypatches `batch_chat_completion_sync` with a fake."""
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

    # ── batch boundary-specific contracts ────────────────────────────
    # Cannot occur in per-record engines, so excluded from base and verified in child.

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
        """`main()` returns `None` (logging only) when `batch_chat_completion_sync` raises an exception."""
        records = record_factory(n=2)

        def _raise(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(engine_module, "batch_chat_completion_sync", _raise)
        out = engine_module.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=2),
        ))
        assert out is None

    # ── vLLM-specific tests ───────────────────────────────────────────

    def test_do_async(
        self, monkeypatch,
        engine_module, main_kwargs_factory, fake_inference_output,
        record_factory, task_config_factory,
    ):
        """`do_async=True` + `skip_chat_template=False` calls only the async chat batch.

        Verifications:
        - `batch_chat_completion_async` is called, sync is not called
        - `batch_size` is passed as the `semaphore_size` argument
        - prediction in response is correctly written back to record
        """
        records = record_factory(n=2)
        async_calls, sync_calls = [], []

        async def _fake_async(*a, **kw):
            records = kw["records"]
            async_calls.append((kw["url"], len(records), kw.get("semaphore_size")))
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

        assert async_calls == [(_MOCK_VLLM_URL, 2, 4)]
        assert sync_calls == []
        assert len(out) == 2
        assert all(r["prediction"] == "ap" for r in out)

    def test_skip_chat_template(
        self, monkeypatch,
        engine_module, main_kwargs_factory, fake_inference_output,
        record_factory, task_config_factory,
    ):
        """`skip_chat_template=True` calls the completion path instead of the chat path.

        Verifications:
        - `batch_completion_sync` is called, `batch_chat_completion_sync` is not called
        - prompts are passed per record
        - prediction in response is correctly written back to record
        """
        records = record_factory(n=2)
        chat_calls, completion_calls = [], []

        def _fake_chat(*a, **kw):
            chat_calls.append(1)
            return None

        def _fake_completion(*a, **kw):
            records = kw["records"]
            completion_calls.append((kw["url"], len(records)))
            for record in records:
                _apply_output_to_record(record, fake_inference_output(prediction="cp"))
            return records

        monkeypatch.setattr(engine_module, "batch_chat_completion_sync", _fake_chat)
        monkeypatch.setattr(engine_module, "batch_completion_sync", _fake_completion)

        out = engine_module.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=2),
            skip_chat_template=True,
        ))

        assert chat_calls == []
        assert completion_calls == [(_MOCK_VLLM_URL, 2)]
        assert len(out) == 2
        assert all(r["prediction"] == "cp" for r in out)

    def test_skip_chat_template_async(
        self, monkeypatch,
        engine_module, main_kwargs_factory, fake_inference_output,
        record_factory, task_config_factory,
    ):
        """`skip_chat_template=True` + `do_async=True` calls only `batch_completion_async`."""
        records = record_factory(n=2)
        chat_async_calls, completion_async_calls = [], []

        async def _fake_chat_async(*a, **kw):
            chat_async_calls.append(1)
            return None

        async def _fake_completion_async(*a, **kw):
            records = kw["records"]
            completion_async_calls.append((kw["url"], len(records), kw.get("semaphore_size")))
            for record in records:
                _apply_output_to_record(record, fake_inference_output(prediction="ca"))
            return records

        monkeypatch.setattr(engine_module, "batch_chat_completion_async", _fake_chat_async)
        monkeypatch.setattr(engine_module, "batch_completion_async", _fake_completion_async)

        out = engine_module.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=2),
            skip_chat_template=True,
            do_async=True,
            batch_size=4,
        ))

        assert chat_async_calls == []
        assert completion_async_calls == [(_MOCK_VLLM_URL, 2, 4)]
        assert len(out) == 2
        assert all(r["prediction"] == "ca" for r in out)

    def test_healthcheck(
        self, monkeypatch,
        engine_module, main_kwargs_factory,
        record_factory, task_config_factory,
    ):
        """`RuntimeError("Failed to healthcheck: ...")` is propagated when `healthcheck()` returns False."""
        monkeypatch.setattr(engine_module, "healthcheck", lambda **kw: False)
        records = record_factory(n=1)

        with pytest.raises(RuntimeError, match="Failed to healthcheck"):
            engine_module.main(**main_kwargs_factory(
                records=records,
                task_config=task_config_factory(num_records=1),
            ))

    # ── live smoke ────────────────────────────────────────────────────
    # When adding a new vllm config, extend with `_VLLM_CONFIG_V<N>` + 3 tests `test_v<N>_smoke_{text,image,video}` following the same pattern.

    def _run_vllm_smoke(
        self,
        config_path,
        modality: str,
        records,
        query: str,
        main_kwargs_factory,
        task_config_factory,
    ) -> None:
        """Calls engine.main() with env(VLLM_URL/VLLM_API_VERSION) + yaml and asserts a non-empty prediction."""
        config = OmegaConf.load(config_path)
        url = os.environ["VLLM_URL"]
        api_version = os.environ["VLLM_API_VERSION"]
        skip_if_vllm_down(url)
        do_async = bool(config.do_async)
        out = engine.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=len(records)),
            url=url,
            api_version=api_version,
            trust_remote_code=bool(config.trust_remote_code),
            do_async=do_async,
            batch_size=int(config.semaphore_size if do_async else config.batch_size),
            timeout=int(config.timeout_async if do_async else config.timeout_sync),
            max_retry=int(config.max_retry),
            wait_between_retry=int(config.wait_between_retry),
            default_generation_options={
                "max_new_tokens": LIVE_SMOKE_MAX_NEW_TOKENS,
                "temperature": 0.0,
            },
        ))
        assert isinstance(out, list) and len(out) == 1
        pred = out[0]["prediction"]
        _log_smoke_response(
            provider=f"vLLM {api_version} ({config.exp_name})",
            url=url,
            modality=modality, query=query, pred=pred,
            latency=out[0]["latency"],
        )
        assert isinstance(pred, str) and len(pred) > 0

    @pytest.mark.requires_env("VLLM_API_KEY", "VLLM_URL", "VLLM_API_VERSION")
    def test_v1_smoke_text(
        self, main_kwargs_factory, record_factory, task_config_factory,
    ):
        """vLLM v1 endpoint text-only smoke — makes a real call to VLLM_URL from env."""
        self._run_vllm_smoke(
            config_path=_VLLM_CONFIG_V1, modality="text",
            records=record_factory(n=1, query=LIVE_SMOKE_TEXT_QUERY),
            query=LIVE_SMOKE_TEXT_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )

    @pytest.mark.requires_env("VLLM_API_KEY", "VLLM_URL", "VLLM_API_VERSION")
    def test_v1_smoke_image(
        self, main_kwargs_factory, image_record_url_factory, task_config_factory,
    ):
        """vLLM v1 endpoint image+text smoke — image version of `test_v1_smoke_text`."""
        self._run_vllm_smoke(
            config_path=_VLLM_CONFIG_V1, modality="image",
            records=image_record_url_factory(n=1, query=LIVE_SMOKE_IMAGE_QUERY),
            query=LIVE_SMOKE_IMAGE_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )

    @pytest.mark.requires_env("VLLM_API_KEY", "VLLM_URL", "VLLM_API_VERSION")
    def test_v1_smoke_video(
        self, main_kwargs_factory, video_record_url_factory, task_config_factory,
    ):
        """vLLM v1 endpoint video+text smoke — video version of `test_v1_smoke_text`."""
        self._run_vllm_smoke(
            config_path=_VLLM_CONFIG_V1, modality="video",
            records=video_record_url_factory(n=1, query=LIVE_SMOKE_VIDEO_QUERY),
            query=LIVE_SMOKE_VIDEO_QUERY,
            main_kwargs_factory=main_kwargs_factory,
            task_config_factory=task_config_factory,
        )
