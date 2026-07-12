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

"""tests/utils_ shared fixtures — byte fixtures and injection helpers that make codec / network / dynamic module load boundaries deterministic."""
import io
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def red_png_bytes() -> bytes:
    """1×1 red PNG bytes — image codec / format detection baseline."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    """0.01s silent 16kHz mono WAV bytes — audio format detection / MIME baseline."""
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(16000)
        fp.writeframes(np.zeros(160, dtype=np.int16).tobytes())
    return buffer.getvalue()


@pytest.fixture
def tiny_mp4_path(tmp_path) -> str:
    """3-frame 16×16 silent mpeg4 mp4 — input for count_video_frames / to_nparray_video."""
    av = pytest.importorskip("av")

    path = tmp_path / "tiny.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=1)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"

    for _ in range(3):
        frame = av.VideoFrame.from_ndarray(
            np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return str(path)


@pytest.fixture
def fake_module_factory(monkeypatch):
    """Factory that registers a fake module in `sys.modules` — teardown is handled automatically by monkeypatch."""

    def _factory(name: str, **attrs) -> types.ModuleType:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    return _factory


@pytest.fixture
def stub_requests_get(monkeypatch):
    """Fixture that replaces `requests.get` with a fake and returns a calls log."""

    def _stub(response_fn):
        calls = []

        def _fake(url, **kwargs):
            calls.append((url, kwargs))
            return response_fn(url, **kwargs)

        monkeypatch.setattr("requests.get", _fake)
        return calls

    return _stub
