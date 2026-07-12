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

"""Unit tests for api/google/chat_completions.py — chat response parsing, retry, generation_options normalization, UPLOADED_FILES_CACHE concurrency invariant."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("google.genai")

from omni_evaluator.api.google.chat_completions import (
    _UPLOADED_FILES_CACHE_MAX_SIZE,
    chat_completion_sync,
    file_upload_async,
    file_upload_sync,
)


# ── chat — response parsing / retry / gen_options ───────────────────────────


def test_chat_success_returns_parsed_dict(fake_google_client_factory):
    """Successful response → dict with `prediction` / `latency` / `finish_reason` populated and returned."""
    client, calls = fake_google_client_factory()
    out = chat_completion_sync(
        client=client, api_name="gemini-2.0-flash",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=1, wait_between_retry=0,
    )
    assert isinstance(out, dict)
    assert out["prediction"] == "hello"
    assert "latency" in out
    assert out["finish_reason"] == "STOP"
    assert len(calls) == 1


def test_chat_thinking_part_becomes_reasoning(
    fake_google_client_factory, fake_google_response_factory,
):
    """`content.parts[thought=True].text` → reasoning_content."""
    response = fake_google_response_factory(text="ok", thinking="why-step")
    client, _ = fake_google_client_factory(response=response)
    out = chat_completion_sync(
        client=client, api_name="gemini-2.5-pro",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=1, wait_between_retry=0,
    )
    assert out["reasoning_content"] == "why-step"


def test_chat_all_retries_fail_returns_none(fake_google_client_factory):
    """generate_content raises on every attempt → None."""
    client, calls = fake_google_client_factory(
        side_effect=[RuntimeError("boom")] * 3,
    )
    out = chat_completion_sync(
        client=client, api_name="gemini-2.0-flash",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=3, wait_between_retry=0,
    )
    assert out is None
    assert len(calls) == 3


def test_chat_partless_response_returns_none(
    fake_google_client_factory, fake_google_response_factory,
):
    """200-OK candidate with no `content.parts` (reasoning model burned the output budget on
    thinking → finish_reason=MAX_TOKENS) is handled gracefully: returns None, never raises.
    Regression: gemini-3.1-pro-preview on amber_test surfaced only as opaque 'Failed after N tries'.
    Fail-fast: a content-contract failure is not retried (identical request), so exactly 1 call."""
    resp = fake_google_response_factory(omit_parts=True, finish_reason="MAX_TOKENS")
    client, calls = fake_google_client_factory(response=resp)
    out = chat_completion_sync(
        client=client, api_name="gemini-3.1-pro-preview",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=5, wait_between_retry=0,
    )
    assert out is None
    assert len(calls) == 1  # not retried despite max_retry=5


def test_chat_empty_candidates_returns_none(
    fake_google_client_factory, fake_google_response_factory,
):
    """Empty `candidates` (safety block / refusal) → None, no IndexError, and not retried (1 call)."""
    resp = fake_google_response_factory(drop_candidates=True)
    client, calls = fake_google_client_factory(response=resp)
    out = chat_completion_sync(
        client=client, api_name="gemini-2.0-flash",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=5, wait_between_retry=0,
    )
    assert out is None
    assert len(calls) == 1


def test_chat_generation_options_camelcase_mapping(fake_google_client_factory):
    """`max_new_tokens` → `maxOutputTokens`, `stop_words` → `stopSequences` (Google camelCase mapping)."""
    client, calls = fake_google_client_factory()
    chat_completion_sync(
        client=client, api_name="gemini-2.0-flash",
        messages=[{"role": "user", "content": "hi"}],
        generation_options={"max_new_tokens": 77, "stop_words": ["END"]},
        max_retry=1, wait_between_retry=0,
    )
    # config is a GenerateContentConfig instance — validated via attributes
    config = calls[0].get("config")
    assert config is not None
    # GenerateContentConfig is pydantic-style; exposed as max_output_tokens / stop_sequences attributes
    assert getattr(config, "max_output_tokens", None) == 77
    assert getattr(config, "stop_sequences", None) == ["END"]


def test_chat_api_name_propagated(fake_google_client_factory):
    """api_name is passed to generate_content as `model=`."""
    client, calls = fake_google_client_factory()
    chat_completion_sync(
        client=client, api_name="gemini-2.5-pro",
        messages=[{"role": "user", "content": "hi"}],
        max_retry=1, wait_between_retry=0,
    )
    assert calls[0]["model"] == "gemini-2.5-pro"


# ── §5.6 — UPLOADED_FILES_CACHE concurrency invariant ────────────────────────


@pytest.mark.xfail(
    strict=False,
    reason="N uploads occur on concurrent miss due to missing lock between check-then-act. Adding in-flight guard changes xpass → fix signal.",
)
def test_cache_concurrent_miss_uploads_once_sync(
    isolated_upload_cache, upload_file, upload_sync_client,
):
    """When the same file_path has a concurrent miss, `client.files.upload` should be called exactly once."""
    client, counter = upload_sync_client
    N = 8

    with ThreadPoolExecutor(max_workers=N) as ex:
        results = list(ex.map(
            lambda _: file_upload_sync(client, upload_file),
            range(N),
        ))

    assert counter["upload"] == 1, (
        f"upload called {counter['upload']}x under {N} concurrent miss"
    )
    assert len({r.name for r in results}) == 1
    assert len(isolated_upload_cache) == 1


@pytest.mark.xfail(
    strict=False,
    reason="async version has the same race — all coroutines miss before N uploads occur due to event loop yield during await sleep.",
)
def test_cache_concurrent_miss_uploads_once_async(
    isolated_upload_cache, upload_file, upload_async_client,
):
    """With N coroutines entering concurrently via asyncio.gather, upload occurs once / cache has single entry."""
    client, counter = upload_async_client
    N = 8

    async def _run():
        return await asyncio.gather(*[
            file_upload_async(client, upload_file) for _ in range(N)
        ])

    results = asyncio.run(_run())

    assert counter["upload"] == 1
    assert len({r.name for r in results}) == 1
    assert len(isolated_upload_cache) == 1


def test_cache_eviction_invariant_under_concurrent_inserts(
    isolated_upload_cache, tmp_path, upload_sync_client,
):
    """Even when new keys are inserted concurrently into a full cache, size must not exceed MAX."""
    client, _ = upload_sync_client
    MAX = _UPLOADED_FILES_CACHE_MAX_SIZE

    for i in range(MAX - 1):
        isolated_upload_cache[f"prefilled-{i}/dummy"] = f"files/prefilled-{i}"

    paths = []
    for i in range(100):
        d = tmp_path / f"new-{i}"
        d.mkdir()
        p = d / "clip.mp4"
        p.write_bytes(b"\x00" * 32)
        paths.append(str(p))

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda fp: file_upload_sync(client, fp), paths))

    assert len(isolated_upload_cache) <= MAX, (
        f"cache size {len(isolated_upload_cache)} exceeds MAX={MAX}"
    )
    # FIFO: oldest prefilled key is evicted
    assert "prefilled-0/dummy" not in isolated_upload_cache
