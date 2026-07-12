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

"""Local image serving for Inference Viewer (filesystem images)."""

import logging
import mimetypes
import os
from urllib.parse import unquote

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse

from ..config import LOCAL_IMAGE_ALLOWED_PREFIXES, LOCAL_IMAGE_MAX_SIZE, LOCAL_MEDIA_PREFIX_MAP

logger = logging.getLogger(__name__)

router = APIRouter()


def _sniff_media_class(real_path: str) -> str | None:
    """Read the first bytes off disk and return the media class ('image',
    'video', 'audio') inferred from a known magic-byte signature, or None if
    unrecognized. Pure-Python byte checks only — no python-magic dependency.
    """
    try:
        with open(real_path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return None
    if len(head) < 4:
        return None

    # Images
    if head[:3] == b"\xff\xd8\xff":  # JPEG
        return "image"
    if head[:8] == b"\x89PNG\r\n\x1a\n":  # PNG
        return "image"
    if head[:4] == b"GIF8":  # GIF (GIF87a / GIF89a)
        return "image"
    if head[:2] == b"BM":  # BMP
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":  # WEBP
        return "image"
    # Video / audio (ISO base media: ....ftyp....)
    if head[4:8] == b"ftyp":  # MP4 / MOV / M4A family
        return "video"
    if head[:4] == b"OggS":  # Ogg (audio/video)
        return "audio"
    if head[:3] == b"ID3" or head[:2] == b"\xff\xfb":  # MP3
        return "audio"
    if head[:4] == b"fLaC":  # FLAC
        return "audio"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":  # WAV
        return "audio"
    return None


def _decode_local_path(request: Request) -> tuple[str, str | None, Response | None]:
    import base64

    path_b64 = unquote(request.query_params.get("path", ""))
    if not path_b64:
        return "", None, Response(status_code=400)

    try:
        padded = path_b64 + "=" * (-len(path_b64) % 4)
        file_path = base64.urlsafe_b64decode(padded.encode()).decode()
    except Exception as exc:
        # Truncate the offending token so we don't dump a long/secret-bearing
        # value into the logs while still aiding diagnosis.
        logger.debug("local path b64 decode failed (%s): %.32s", exc, path_b64)
        return "", None, Response(status_code=400)

    # Rewrite a configured path prefix to its local location (e.g. a dataset cache that lived
    # under /mnt on the inference host now mirrored under /data here). Done before realpath +
    # the allowlist check below, so the rewritten target is still fully validated and this
    # cannot widen access beyond LOCAL_IMAGE_ALLOWED_PREFIXES.
    for _old, _new in LOCAL_MEDIA_PREFIX_MAP:
        if file_path.startswith(_old):
            file_path = _new + file_path[len(_old):]
            break

    real_path = os.path.realpath(file_path)
    if not any(real_path.startswith(p) for p in LOCAL_IMAGE_ALLOWED_PREFIXES):
        return "", None, Response(status_code=403)

    mime, _ = mimetypes.guess_type(real_path)
    return real_path, mime, None


def _serve_local_file(real_path: str, mime: str) -> Response:
    try:
        fsize = os.path.getsize(real_path)
    except OSError:
        return Response(status_code=404)
    if fsize > LOCAL_IMAGE_MAX_SIZE:
        return Response(status_code=413)

    # Lightweight magic-byte sniff: the extension-derived MIME can lie. If the
    # sniffed media class (image/video/audio) conflicts with the extension's
    # class, refuse rather than serve a mislabeled file. Unknown signatures are
    # allowed through (sniff only catches a fixed set of common types).
    sniffed = _sniff_media_class(real_path)
    ext_class = mime.split("/", 1)[0] if mime else None
    if sniffed and ext_class and sniffed != ext_class:
        logger.debug(
            "media class mismatch: ext=%s sniffed=%s for %.64s",
            ext_class, sniffed, real_path,
        )
        return Response(status_code=403)

    # Stream from disk (FileResponse uses a threadpool + chunks) instead of
    # reading the whole file into memory — avoids a per-request RSS spike of up
    # to LOCAL_IMAGE_MAX_SIZE on small instances.
    return FileResponse(
        real_path,
        media_type=mime,
        headers={"Cache-Control": "public, max-age=300"},
    )


@router.get("/local-image")
def serve_local_image(request: Request) -> Response:
    """GET /api/local-image?path=<base64-encoded-path>

    Serve local filesystem images for Inference Viewer.
    Security: only paths under /mnt/ or /data/ allowed.
    """
    real_path, mime, err = _decode_local_path(request)
    if err:
        return err
    if not mime or not mime.startswith("image/"):
        return Response(status_code=403)
    return _serve_local_file(real_path, mime)


@router.get("/local-media")
def serve_local_media(request: Request) -> Response:
    """GET /api/local-media?path=<base64-encoded-path>

    Serve local filesystem audio/video for Inference Viewer.
    Security: only paths under /mnt/ or /data/ allowed.
    """
    real_path, mime, err = _decode_local_path(request)
    if err:
        return err
    if mime and (mime.startswith("audio/") or mime.startswith("video/")):
        return _serve_local_file(real_path, mime)
    return Response(status_code=403)
