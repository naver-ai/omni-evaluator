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

"""Unit tests for utils/multimodal.py — image/audio/video conversion, format detection, MIME, SSRF blocking."""
import pytest

pytest.importorskip("PIL")

import numpy as np  # noqa: E402

from omni_evaluator.utils.multimodal import (  # noqa: E402
    _validate_url_safe,
    audio_mime_type,
    count_video_frames,
    detect_audio_format,
    detect_image_format,
    detect_video_format,
    image_mime_type,
    media_mime_type,
    resize_image,
    safe_request_get,
    to_audio_bytes,
    to_audio_wav,
    to_image_bytes,
    to_nparray_audio,
    to_nparray_video,
    to_pil_image,
    to_video_bytes,
    video_mime_type,
)

# Containers whose format can be identified from the header alone — magic bytes only, no real encoding
_MP4_HEADER = b"\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00"
_WEBM_HEADER = b"\x1aE\xdf\xa3" + b"\x00" * 8
_AVI_HEADER = b"RIFF\x00\x00\x00\x00AVI "


# ─────────────────────────────────────────────────────────────
# _validate_url_safe — SSRF regression (loopback/private/link-local/non-http rejected)
# ─────────────────────────────────────────────────────────────

def test_validate_url_safe_blocks_internal():
    """Addresses that resolve to loopback / private / link-local are blocked (SSRF regression)."""
    for url in (
        "http://169.254.169.254/latest/meta-data/",
    ):
        with pytest.raises(ValueError):
            _validate_url_safe(url)


def test_validate_url_safe_rejects_non_http():
    """Non-http/https schemes (file://, ftp://) and missing hostname are rejected."""
    with pytest.raises(ValueError):
        _validate_url_safe("file:///etc/passwd")
    with pytest.raises(ValueError):
        _validate_url_safe("ftp://example.com/x")
    with pytest.raises(ValueError):
        _validate_url_safe("http://")


# ─────────────────────────────────────────────────────────────
# count_video_frames — frame count for video stream via av container
# ─────────────────────────────────────────────────────────────

@pytest.mark.requires_extra("av")
def test_count_video_frames(tiny_mp4_path):
    """Counts the number of frames in a video stream via av.open."""
    assert count_video_frames(tiny_mp4_path) == 3


# ─────────────────────────────────────────────────────────────
# detect_audio_format — WAV magic bytes detection
# ─────────────────────────────────────────────────────────────

@pytest.mark.requires_extra("soundfile")
def test_detect_audio_format(wav_bytes):
    """Returns 'wav' from WAV magic bytes, None for unknown."""
    assert detect_audio_format(wav_bytes) == "wav"
    assert detect_audio_format(b"\x00" * 12) is None


# ─────────────────────────────────────────────────────────────
# detect_image_format — PIL format name detection
# ─────────────────────────────────────────────────────────────

@pytest.mark.requires_extra("PIL")
def test_detect_image_format(red_png_bytes):
    """Detects PIL format name from PNG bytes; non-image returns None."""
    assert detect_image_format(red_png_bytes) == "PNG"
    assert detect_image_format(b"not an image") is None


# ─────────────────────────────────────────────────────────────
# detect_video_format — container detection via ftyp/EBML/RIFF magic bytes
# ─────────────────────────────────────────────────────────────

def test_detect_video_format():
    """Distinguishes mp4/webm/avi via ftyp/EBML/RIFF magic bytes."""
    assert detect_video_format(_MP4_HEADER) == "mp4"
    assert detect_video_format(_WEBM_HEADER) == "webm"
    assert detect_video_format(_AVI_HEADER) == "avi"
    assert detect_video_format(b"\x00" * 4) is None


# ─────────────────────────────────────────────────────────────
# image / audio / video / media _mime_type — MIME string assembly on top of detect_*
# ─────────────────────────────────────────────────────────────

@pytest.mark.requires_extra("PIL")
def test_mime_types(red_png_bytes, wav_bytes):
    """Builds MIME strings through each detector; media_mime_type auto-detects the media kind."""
    assert image_mime_type(red_png_bytes) == "image/png"
    assert audio_mime_type(wav_bytes) == "audio/wav"
    assert video_mime_type(_MP4_HEADER) == "video/mp4"
    assert media_mime_type(_MP4_HEADER) == "video/mp4"
    assert media_mime_type(red_png_bytes) == "image/png"


# ─────────────────────────────────────────────────────────────
# resize_image — PIL image resize to fit max_w/max_h ratio
# ─────────────────────────────────────────────────────────────

@pytest.mark.requires_extra("PIL")
def test_resize_image_passthrough():
    """Passes through without resizing when both dimensions are within max."""
    from PIL import Image

    img = Image.new("RGB", (60, 60))
    assert resize_image(img, max_height=100, max_width=100).size == (60, 60)


@pytest.mark.requires_extra("PIL")
def test_resize_image_width_dominant():
    """Scales down by width ratio when width_scale < height_scale."""
    from PIL import Image

    img = Image.new("RGB", (1000, 100))
    # width_scale=0.1, height_scale=1.0 → width_scale selected
    assert resize_image(img, max_height=100, max_width=100).size == (100, 10)


@pytest.mark.requires_extra("PIL")
def test_resize_image_height_dominant():
    """Scales down by height ratio when width_scale >= height_scale."""
    from PIL import Image

    img = Image.new("RGB", (100, 1000))
    assert resize_image(img, max_height=100, max_width=100).size == (10, 100)


@pytest.mark.requires_extra("PIL")
def test_resize_image_upscale_tiny():
    """Upscales to 50 first when min(w,h) < 50."""
    from PIL import Image

    img = Image.new("RGB", (10, 10))
    # 10x10 → upscaled to 50x50, then passthrough since w/h are within max
    assert resize_image(img, max_height=1000, max_width=1000).size == (50, 50)


# ─────────────────────────────────────────────────────────────
# safe_request_get — _validate_url_safe + mandatory timeout + raise_for_status wrapper
# ─────────────────────────────────────────────────────────────

def test_safe_request_get_blocks_internal():
    """URLs rejected by _validate_url_safe raise ValueError before the request is made."""
    with pytest.raises(ValueError):
        safe_request_get("http://169.254.169.254/latest/meta-data/")


def test_safe_request_get_raises_on_http_error(monkeypatch):
    """Non-2xx responses propagate HTTPError via raise_for_status."""
    import requests

    monkeypatch.setattr(
        "omni_evaluator.utils.multimodal._validate_url_safe", lambda url: None
    )

    class _Err:
        def raise_for_status(self):
            raise requests.HTTPError("500")

    monkeypatch.setattr("requests.get", lambda *a, **kw: _Err())
    with pytest.raises(requests.HTTPError):
        safe_request_get("http://example.com/x")


def test_safe_request_get_happy(monkeypatch):
    """Returns the Response and auto-injects the timeout kwarg on SSRF pass + 2xx."""
    monkeypatch.setattr(
        "omni_evaluator.utils.multimodal._validate_url_safe", lambda url: None
    )
    captured = {}

    class _Ok:
        status_code = 200

        def raise_for_status(self):
            pass

    def _fake(url, **kw):
        captured.update(kw)
        return _Ok()

    monkeypatch.setattr("requests.get", _fake)
    assert safe_request_get("http://example.com/x").status_code == 200
    assert "timeout" in captured


# ─────────────────────────────────────────────────────────────
# to_audio_bytes — normalization by input type + format match check + base64 option
# ─────────────────────────────────────────────────────────────

def test_to_audio_bytes_passthrough(wav_bytes):
    """Passes through unchanged when already WAV bytes (format match — no pydub conversion)."""
    assert to_audio_bytes(wav_bytes, extension="WAV") == wav_bytes


def test_to_audio_bytes_dict_recursion(wav_bytes):
    """Dict input is unwrapped in order of value/audio/input_audio/data keys."""
    assert to_audio_bytes({"value": wav_bytes}, extension="WAV") == wav_bytes
    assert to_audio_bytes({"audio": wav_bytes}, extension="WAV") == wav_bytes


def test_to_audio_bytes_base64(wav_bytes):
    """Returns bytes as a base64 utf-8 str when encode_base64=True."""
    import base64

    out = to_audio_bytes(wav_bytes, extension="WAV", encode_base64=True)
    assert isinstance(out, str)
    assert base64.b64decode(out) == wav_bytes


# ─────────────────────────────────────────────────────────────
# to_audio_wav — writes to_audio_bytes result to a path
# ─────────────────────────────────────────────────────────────

def test_to_audio_wav_writes_riff(tmp_path, wav_bytes):
    """Writes to path and returns path when the header is RIFF/WAVE."""
    out_path = str(tmp_path / "out.wav")
    assert to_audio_wav(wav_bytes, path=out_path) == out_path
    with open(out_path, "rb") as fp:
        data = fp.read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"


# ─────────────────────────────────────────────────────────────
# to_image_bytes — PIL/np/bytes → bytes round-trip
# ─────────────────────────────────────────────────────────────

@pytest.mark.requires_extra("PIL")
def test_to_image_bytes_roundtrip(red_png_bytes):
    """to_image_bytes → to_pil_image preserves image dimensions."""
    raw = to_image_bytes(red_png_bytes, encode_base64=False)
    assert isinstance(raw, bytes)
    assert to_pil_image(raw).size == (1, 1)


# ─────────────────────────────────────────────────────────────
# to_nparray_audio — np.ndarray passthrough / unknown type rejection
# ─────────────────────────────────────────────────────────────

def test_to_nparray_audio_passthrough_ndarray():
    """np.ndarray input passes through unchanged (sampling_rate also unchanged)."""
    arr = np.zeros(100, dtype=np.float32)
    out, sr, fmt = to_nparray_audio(arr, sampling_rate=16000)
    assert out is arr
    assert sr == 16000 and fmt is None


def test_to_nparray_audio_invalid_type():
    """Types other than str/bytes/np.ndarray/AudioDecoder raise TypeError."""
    with pytest.raises(TypeError):
        to_nparray_audio(12345)


# ─────────────────────────────────────────────────────────────
# to_nparray_video — np.ndarray passthrough / av container extraction
# ─────────────────────────────────────────────────────────────

def test_to_nparray_video_passthrough_ndarray():
    """np.ndarray input is returned immediately as (frames, None, None)."""
    arr = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    frames, wav, sr = to_nparray_video(arr)
    assert frames is arr
    assert wav is None and sr is None


# ─────────────────────────────────────────────────────────────
# to_pil_image — bytes / np.ndarray / PIL → RGB PIL.Image
# ─────────────────────────────────────────────────────────────

@pytest.mark.requires_extra("PIL")
def test_to_pil_image():
    """Converts bytes / np.ndarray / PIL input to an RGB PIL.Image."""
    from PIL import Image

    assert to_pil_image(np.zeros((2, 3, 3), dtype=np.uint8)).size == (3, 2)
    src = Image.new("RGB", (4, 4), (0, 255, 0))
    assert to_pil_image(src).mode == "RGB"


# ─────────────────────────────────────────────────────────────
# to_video_bytes — bytes/bytearray/BytesIO/str → bytes, base64 option, TypeError
# ─────────────────────────────────────────────────────────────

def test_to_video_bytes_passthrough():
    """bytes input passes through unchanged."""
    raw = b"\x00video\xff"
    assert to_video_bytes(raw) == raw


def test_to_video_bytes_bytearray_and_bytesio():
    """bytearray / BytesIO is converted to raw bytes."""
    import io as _io

    raw = b"vid"
    assert to_video_bytes(bytearray(raw)) == raw
    assert to_video_bytes(_io.BytesIO(raw)) == raw


def test_to_video_bytes_invalid_type():
    """Types other than str/bytes/BytesIO/bytearray raise TypeError."""
    with pytest.raises(TypeError):
        to_video_bytes(12345)


def test_to_video_bytes_base64():
    """Returns base64 utf-8 str when encode_base64=True."""
    import base64

    raw = b"vid"
    out = to_video_bytes(raw, encode_base64=True)
    assert isinstance(out, str)
    assert base64.b64decode(out) == raw
