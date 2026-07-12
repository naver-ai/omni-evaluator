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

"""Fixtures exclusive to tests/api/google/ — Google genai SDK responses and file upload cache."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from typing import Optional

import pytest


pytest.importorskip("google.genai", reason="Provider-level tests are meaningless without google.genai installed")


class _FakeGoogleResponse:
    """genai SDK response mock — exposes only `.json()`."""

    def __init__(self, data: dict):
        self._data = data

    def json(self) -> str:
        return json.dumps(self._data)


@pytest.fixture
def fake_google_response_factory():
    """genai generate_content response shape — `candidates[*].content.parts[*]`.

    Adversarial shapes (real 200-OK responses that carry no usable text):
      - ``omit_parts=True``     → candidate whose ``content`` has no ``parts`` key,
                                  as a reasoning model returns when it spends the whole
                                  output budget on thinking (``finish_reason='MAX_TOKENS'``).
      - ``drop_candidates=True`` → empty ``candidates`` list (safety block / refusal).
    Defaults are unchanged, so existing callers are unaffected.
    """
    def _factory(
        text: str = "hello",
        thinking: Optional[str] = None,
        finish_reason: str = "STOP",
        *,
        omit_parts: bool = False,
        drop_candidates: bool = False,
    ):
        if drop_candidates:
            return _FakeGoogleResponse({"candidates": []})
        content = {"role": "model"}
        if not omit_parts:
            parts = []
            if thinking is not None:
                parts.append({"text": thinking, "thought": True})
            parts.append({"text": text})
            content["parts"] = parts
        return _FakeGoogleResponse({
            "candidates": [{
                "content": content,
                "finish_reason": finish_reason,
            }],
        })
    return _factory


@pytest.fixture
def fake_google_client_factory(fake_google_response_factory):
    """genai client mock — records `.models.generate_content` calls and injects responses."""
    def _factory(*, response=None, side_effect=None):
        calls: list[dict] = []

        def _generate(**kwargs):
            calls.append(kwargs)
            if side_effect is not None:
                exc = side_effect.pop(0) if side_effect else None
                if isinstance(exc, BaseException):
                    raise exc
            return response or fake_google_response_factory()

        models = SimpleNamespace(generate_content=_generate)
        return SimpleNamespace(models=models), calls
    return _factory


# ── §5.6 file upload cache fixtures ──────────────────────────────────────────


_FILE_UPLOAD_RACE_WINDOW_SEC = 0.05


@pytest.fixture
def isolated_upload_cache(monkeypatch):
    """Replaces `UPLOADED_FILES_CACHE` with a fresh OrderedDict to prevent state leakage between tests."""
    from omni_evaluator.api.google import chat_completions as chat_mod
    fresh = OrderedDict()
    monkeypatch.setattr(chat_mod, "UPLOADED_FILES_CACHE", fresh)
    return fresh


@pytest.fixture
def upload_file(tmp_path):
    """Returns a dummy mp4 file with a stable cache key (parent_dir/file_name)."""
    p = tmp_path / "videos" / "clip.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * 32)
    return str(p)


def _active_file(name: str):
    return SimpleNamespace(name=name, state=SimpleNamespace(name="ACTIVE"))


@pytest.fixture
def upload_sync_client():
    """Sync client that counts `client.files.upload` calls and sleeps for the race window duration."""
    counter = {"upload": 0}
    lock = threading.Lock()

    def _upload(*, file, config):
        with lock:
            counter["upload"] += 1
            seq = counter["upload"]
        time.sleep(_FILE_UPLOAD_RACE_WINDOW_SEC)
        return _active_file(f"files/upload-{seq}")

    def _get(*, name):
        return _active_file(name)

    return SimpleNamespace(files=SimpleNamespace(upload=_upload, get=_get)), counter


@pytest.fixture
def upload_async_client():
    """Async version — async client with coroutine + asyncio.sleep race window."""
    counter = {"upload": 0}
    lock = threading.Lock()

    async def _upload(*, file, config):
        with lock:
            counter["upload"] += 1
            seq = counter["upload"]
        await asyncio.sleep(_FILE_UPLOAD_RACE_WINDOW_SEC)
        return _active_file(f"files/upload-{seq}")

    async def _get(*, name):
        return _active_file(name)

    return SimpleNamespace(files=SimpleNamespace(upload=_upload, get=_get)), counter
