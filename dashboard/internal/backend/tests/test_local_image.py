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

"""Tests for app/api/local_image.py helpers (called directly, no HTTP).

_decode_local_path(request) reads request.query_params.get("path", ""),
base64url-decodes it, realpath-canonicalizes it, and enforces the
LOCAL_IMAGE_ALLOWED_PREFIXES allowlist. It returns a 3-tuple:
    (real_path, mime, err_response)
where err_response is a fastapi Response (with .status_code) on failure and
None on success.

_sniff_media_class(real_path) reads the first 16 bytes off disk and maps a
known magic-byte signature to 'image' / 'video' / 'audio', else None.
"""

import base64
import os
import tempfile
import unittest

from app.api.local_image import _decode_local_path, _sniff_media_class
from app.config import LOCAL_IMAGE_ALLOWED_PREFIXES


def _b64url(path: str) -> str:
    """Encode a filesystem path the way the frontend does (urlsafe b64)."""
    return base64.urlsafe_b64encode(path.encode()).decode()


class _FakeQueryParams:
    def __init__(self, mapping):
        self._m = mapping

    def get(self, key, default=None):
        return self._m.get(key, default)


class _FakeRequest:
    """Minimal stand-in for fastapi.Request: only .query_params.get is used."""

    def __init__(self, **params):
        self.query_params = _FakeQueryParams(params)


class DecodeLocalPathTest(unittest.TestCase):
    def test_roundtrip_allowed_path(self):
        # A path under an allowed prefix that resolves to itself.
        target = "/mnt/some/dir/picture.png"
        req = _FakeRequest(path=_b64url(target))
        real_path, mime, err = _decode_local_path(req)
        self.assertIsNone(err)
        self.assertEqual(real_path, os.path.realpath(target))
        self.assertEqual(mime, "image/png")

    def test_missing_path_returns_400(self):
        req = _FakeRequest()  # no 'path' query param
        real_path, mime, err = _decode_local_path(req)
        self.assertIsNotNone(err)
        self.assertEqual(err.status_code, 400)
        self.assertEqual(real_path, "")
        self.assertIsNone(mime)

    def test_empty_path_returns_400(self):
        req = _FakeRequest(path="")
        _, _, err = _decode_local_path(req)
        self.assertIsNotNone(err)
        self.assertEqual(err.status_code, 400)

    def test_garbage_base64_returns_400(self):
        # A token whose length is invalid even after the "% 4" padding the code
        # applies (1 char -> still len 1 mod 4) makes b64decode raise -> 400.
        req = _FakeRequest(path="a")
        real_path, mime, err = _decode_local_path(req)
        self.assertIsNotNone(err)
        self.assertEqual(err.status_code, 400)
        self.assertEqual(real_path, "")
        self.assertIsNone(mime)

    def test_non_alphabet_chars_are_silently_dropped(self):
        # Characterization (NOT ideal-behavior): base64.urlsafe_b64decode is
        # lenient and discards non-alphabet bytes like '@', so '@@@@' decodes to
        # b'' -> the empty string -> realpath(cwd). This does NOT raise / 400.
        # Documented here so a future refactor that tightens decoding is a
        # deliberate, visible change rather than a silent one.
        req = _FakeRequest(path="@@@@")
        real_path, _, err = _decode_local_path(req)
        # '' realpaths to the process cwd; whether that yields err or not depends
        # on cwd being under an allowed prefix, so only assert the decode quirk:
        decoded = base64.urlsafe_b64decode(b"@@@@")
        self.assertEqual(decoded, b"")
        # Either way it must not be a 400 (decode did not raise).
        if err is not None:
            self.assertNotEqual(err.status_code, 400)

    def test_undecodable_utf8_returns_400(self):
        # Valid base64 of bytes that are not valid UTF-8 -> .decode() raises -> 400.
        bad = base64.urlsafe_b64encode(b"\xff\xfe\xfa").decode()
        _, _, err = _decode_local_path(_FakeRequest(path=bad))
        self.assertIsNotNone(err)
        self.assertEqual(err.status_code, 400)

    def test_etc_passwd_rejected_403(self):
        req = _FakeRequest(path=_b64url("/etc/passwd"))
        real_path, mime, err = _decode_local_path(req)
        self.assertIsNotNone(err)
        self.assertEqual(err.status_code, 403)
        self.assertEqual(real_path, "")
        self.assertIsNone(mime)

    def test_traversal_escaping_allowed_prefix_rejected_403(self):
        # realpath collapses '..' so '/mnt/../etc/passwd' -> '/etc/passwd'.
        req = _FakeRequest(path=_b64url("/mnt/../etc/passwd"))
        real_path, _, err = _decode_local_path(req)
        self.assertEqual(os.path.realpath("/mnt/../etc/passwd"), "/etc/passwd")
        self.assertIsNotNone(err)
        self.assertEqual(err.status_code, 403)

    def test_data_prefix_allowed(self):
        target = "/data/foo/bar.jpg"
        _, mime, err = _decode_local_path(_FakeRequest(path=_b64url(target)))
        self.assertIsNone(err)
        self.assertEqual(mime, "image/jpeg")

    def test_lookalike_prefix_not_allowed(self):
        # Trailing slash on the prefix prevents '/mnt_evil/...' from matching '/mnt'.
        # '/mnt_evil/x' starts with '/mnt' but not '/mnt/'.
        target = "/mnt_evil/x.png"
        # Guard: this path must actually resolve to itself (no symlink surprises).
        self.assertEqual(os.path.realpath(target), target)
        self.assertFalse(
            any(target.startswith(p) for p in LOCAL_IMAGE_ALLOWED_PREFIXES)
        )
        _, _, err = _decode_local_path(_FakeRequest(path=_b64url(target)))
        self.assertIsNotNone(err)
        self.assertEqual(err.status_code, 403)


class SniffMediaClassTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _write(self, data: bytes) -> str:
        fd, path = tempfile.mkstemp(dir=self._tmpdir.name)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return path

    def test_png_magic_is_image(self):
        path = self._write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        self.assertEqual(_sniff_media_class(path), "image")

    def test_jpeg_magic_is_image(self):
        path = self._write(b"\xff\xd8\xff\xe0" + b"\x00" * 12)
        self.assertEqual(_sniff_media_class(path), "image")

    def test_gif_magic_is_image(self):
        path = self._write(b"GIF89a" + b"\x00" * 10)
        self.assertEqual(_sniff_media_class(path), "image")

    def test_bmp_magic_is_image(self):
        path = self._write(b"BM" + b"\x00" * 14)
        self.assertEqual(_sniff_media_class(path), "image")

    def test_webp_magic_is_image(self):
        # RIFF....WEBP : bytes 0-3 'RIFF', 8-11 'WEBP'.
        path = self._write(b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 4)
        self.assertEqual(_sniff_media_class(path), "image")

    def test_mp4_ftyp_at_offset4_is_video(self):
        # 4-byte size box, then 'ftyp' at offset 4.
        path = self._write(b"\x00\x00\x00\x18" + b"ftyp" + b"isom" + b"\x00" * 4)
        self.assertEqual(_sniff_media_class(path), "video")

    def test_ogg_magic_is_audio(self):
        path = self._write(b"OggS" + b"\x00" * 12)
        self.assertEqual(_sniff_media_class(path), "audio")

    def test_mp3_id3_is_audio(self):
        path = self._write(b"ID3" + b"\x00" * 13)
        self.assertEqual(_sniff_media_class(path), "audio")

    def test_wav_magic_is_audio(self):
        path = self._write(b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 4)
        self.assertEqual(_sniff_media_class(path), "audio")

    def test_unknown_bytes_returns_none(self):
        path = self._write(b"not-a-real-magic-signature")
        self.assertIsNone(_sniff_media_class(path))

    def test_too_short_returns_none(self):
        # Fewer than 4 bytes -> None.
        path = self._write(b"AB")
        self.assertIsNone(_sniff_media_class(path))

    def test_missing_file_returns_none(self):
        missing = os.path.join(self._tmpdir.name, "does-not-exist.bin")
        self.assertIsNone(_sniff_media_class(missing))


if __name__ == "__main__":
    unittest.main()
