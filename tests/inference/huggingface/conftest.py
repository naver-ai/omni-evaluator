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

"""Shared fixtures for tests/inference/huggingface/ — used by both HF engine and adapter tests."""
from __future__ import annotations

import io
import os
import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def audio_message_bytes() -> bytes:
    """0.5s 16kHz mono silence WAV bytes for use as generate input in the audio modality adapter."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(16000)
        fp.writeframes(np.zeros(8000, dtype=np.int16).tobytes())  # 0.5s
    return buffer.getvalue()


_VIDEO_FIXTURE_FPS = 8
_VIDEO_FIXTURE_DURATION_S = 100  # length sufficient to satisfy qwen-vl-utils sampler's nframes=128 requirement
_VIDEO_FIXTURE_FRAMES = _VIDEO_FIXTURE_FPS * _VIDEO_FIXTURE_DURATION_S


@pytest.fixture(scope="session")
def video_message_path(tmp_path_factory) -> str:
    """Path to an 8-fps × 100s 16×16 silent mpeg4 mp4 file for the video modality adapter (skipped if av is not installed)."""
    av = pytest.importorskip("av")

    path = tmp_path_factory.mktemp("video_fixture") / "tiny.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=_VIDEO_FIXTURE_FPS)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"

    for _ in range(_VIDEO_FIXTURE_FRAMES):
        frame = av.VideoFrame.from_ndarray(
            np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return str(path)


@pytest.fixture(scope="session")
def hf_cache_dir() -> Path:
    """HF Hub cache directory (prefers `HF_HUB_CACHE` env var, falls back to `~/.cache/huggingface/hub`)."""
    env_cache = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if env_cache:
        return Path(env_cache)
    return Path.home() / ".cache" / "huggingface" / "hub"
