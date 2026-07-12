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

"""Regression tests for the GZip media-exclusion fix (app/main.py).

GZipMiddleware compresses any response over minimum_size by content-type, excluding only
text/event-stream by default. Compressing binary FileResponses (media at /api/local-*, zip
downloads) corrupts Range/206 semantics and wastes CPU. app/main.py extends Starlette's
DEFAULT_EXCLUDED_CONTENT_TYPES to also skip image/audio/video/zip. These tests pin that:
importing app.main applies the extension, and the middleware then leaves media untouched
while still compressing JSON. Driven at the ASGI layer (no httpx/TestClient needed).
"""

import asyncio
import gzip as gziplib
import unittest

# Importing app.main runs the module-level monkeypatch that extends the excluded types.
import app.main  # noqa: F401
import starlette.middleware.gzip as gz
from starlette.middleware.gzip import GZipMiddleware


def _drive(content_type: str, body: bytes, accept_encoding: str = "gzip"):
    """Run one request through GZipMiddleware and capture the response headers + body."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"accept-encoding", accept_encoding.encode())],
    }
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    async def downstream(scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", content_type.encode())],
        })
        await send({"type": "http.response.body", "body": body, "more_body": False})

    mw = GZipMiddleware(downstream, minimum_size=1024)
    asyncio.run(mw(scope, receive, send))

    start = next(m for m in sent if m["type"] == "http.response.start")
    headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    out = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return headers, out


class GzipExclusionTests(unittest.TestCase):
    BIG = b"A" * 4096  # over minimum_size so compression is eligible

    def test_app_main_extends_excluded_types(self):
        for t in ("image/", "audio/", "video/", "application/zip"):
            self.assertIn(t, gz.DEFAULT_EXCLUDED_CONTENT_TYPES)

    def test_image_passes_through_uncompressed(self):
        headers, body = _drive("image/png", self.BIG)
        self.assertNotIn("content-encoding", headers)
        self.assertEqual(body, self.BIG)  # bytes untouched (Range offsets stay valid)

    def test_video_passes_through_uncompressed(self):
        headers, _ = _drive("video/mp4", self.BIG)
        self.assertNotIn("content-encoding", headers)

    def test_audio_passes_through_uncompressed(self):
        headers, _ = _drive("audio/mpeg", self.BIG)
        self.assertNotIn("content-encoding", headers)

    def test_zip_download_passes_through_uncompressed(self):
        headers, _ = _drive("application/zip", self.BIG)
        self.assertNotIn("content-encoding", headers)

    def test_json_is_still_compressed(self):
        headers, body = _drive("application/json", self.BIG)
        self.assertEqual(headers.get("content-encoding"), "gzip")
        self.assertEqual(gziplib.decompress(body), self.BIG)

    def test_css_is_still_compressed(self):
        headers, _ = _drive("text/css", self.BIG)
        self.assertEqual(headers.get("content-encoding"), "gzip")


if __name__ == "__main__":
    unittest.main()
