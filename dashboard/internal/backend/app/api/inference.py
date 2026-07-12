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

"""Inference Viewer API: model options, sample-level results. Path-based (no DB)."""

import logging
import threading
import time
import zipfile
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse, parse_qs

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from ..config import (
    DIRECT_SUBMISSION_DIR,
    ENABLED_SOURCES,
    INFERENCE_SAMPLE_MAX_BYTES,
    INTERNAL_OUTPUTS_PATH,
    LOCAL_IMAGE_ALLOWED_PREFIXES,
    MEDIA_INLINE_MAX_BYTES,
    MEDIA_LEGACY_HOST,
    MEDIA_REBUCKET,
    MEDIA_S3_PREFIX_MAP,
    S3_BUCKET,
    S3_PREFIX,
    S3_PRESIGN_EXPIRE,
    ensure_upload_dir,
)
from ..services.json_io import read_json_head_tail, parse_head_tail_bytes
from ..services.sample_reader import (
    count_records,
    count_records_stream,
    stream_record_at,
    stream_record_at_stream,
    total_from_head,
)
from ..services.metric_extraction import extract_benchmark_from_filename
from ..services.inference_builder import message_text_from_messages
from ..services.inference_cache import get_bench_cache, get_s3_key_cache, set_bench_cache, set_s3_key_cache
from ..services.scan import (
    _base_benchmark,
    _normalize_engine_folder,
    find_inference_output,
    parse_model_id,
)
from ..services.scan_cache import get_models as sc_get_models, get_benchmark_modalities, get_model_benchmark_counts
from ..services.leaderboard_filters import classify_benchmark_modality
from ..services.s3_sync import _get_s3_client, _s3_list_with_retry

router = APIRouter()


# A file's engine is the path segment immediately preceding its output folder
# (evaluation_output / output / inference_output). Shared by the fs, zip and s3 sites
# so the "find the output folder, take its parent" rule lives in exactly one place.
_OUTPUT_FOLDERS = ("evaluation_output", "output", "inference_output")


def _engine_from_segments(parts: list[str], fallback: str = "built-in") -> str:
    """Derive the engine folder name from a list of path segments (most specific last).
    Returns the normalized segment before the first output folder, else ``fallback``."""
    for i, part in enumerate(parts):
        if part in _OUTPUT_FOLDERS and i > 0:
            return _normalize_engine_folder(parts[i - 1])
    return fallback


def _engine_from_path(path: Path | str) -> str:
    try:
        p = Path(path) if isinstance(path, str) else path
        # Preserve the fs fallback: a file not directly under an output folder maps to
        # its parent dir name (the segment-scan only matches the output-folder layout).
        return _engine_from_segments(list(p.parts), fallback=_normalize_engine_folder(p.parent.name))
    except Exception:
        return "built-in"


def _engine_from_zip_path(name: str) -> str:
    try:
        return _engine_from_segments(name.replace("\\", "/").split("/"))
    except Exception:
        return "built-in"


def _engine_from_key(key: str) -> str:
    try:
        return _engine_from_segments(key.replace("\\", "/").split("/"))
    except Exception:
        return "built-in"


def _is_allowed_local_path(p: Path, base_dir: Path | None) -> bool:
    """True iff p stays under base_dir or one of the configured allow-prefixes.
    Defense-in-depth against eval JSON echoing arbitrary absolute paths (e.g.
    '../../etc/passwd') in the API response; file serving itself stays gated by local_image.py."""
    try:
        sp = str(p)
    except Exception:
        return False
    if base_dir is not None:
        try:
            p.relative_to(base_dir.resolve())
            return True
        except Exception:
            pass
    return any(sp.startswith(pfx) for pfx in LOCAL_IMAGE_ALLOWED_PREFIXES)


def _resolve_media_path(val: str, base_dir: Path | None) -> str:
    if not val:
        return ""
    s = str(val)
    # Rewrite a configured local path prefix to an s3:// URL, so the image is served (presigned)
    # straight from object storage instead of this host's disk. Done before the local-path
    # branch; the result falls through to the s3:// handler below.
    for _old, _new in MEDIA_S3_PREFIX_MAP:
        if s.startswith(_old):
            s = _new + s[len(_old):]
            break
    if s.startswith(("http://", "https://")):
        return _maybe_refresh_storage_url(s)
    if s.startswith("s3://"):
        presigned = _presign_s3_url(s)
        return presigned or s
    if s.startswith("data:"):
        return s
    try:
        if s.startswith("/"):
            # Absolute path: gate by the configured local-image allowlist.
            resolved = Path(s).resolve()
            return str(resolved) if _is_allowed_local_path(resolved, None) else ""
        if base_dir:
            # Relative path: must stay UNDER base_dir (no '../' escape), regardless of
            # whether the escaped target happens to land in an allowed prefix.
            resolved = (base_dir / s).resolve()
            try:
                resolved.relative_to(base_dir.resolve())
            except ValueError:
                return ""
            return str(resolved)
        return ""
    except Exception:
        return ""


def _maybe_refresh_storage_url(url: str) -> str:
    """Refresh expiring presigned URLs that point at a configured legacy media host.

    No-op unless OMNI_MEDIA_LEGACY_HOST is set. Used when media was migrated between object
    stores: eval JSON may still carry old presigned URLs whose object key was copied verbatim
    into the new bucket, so re-presigning <new bucket>/<same key> serves it from the new home."""
    try:
        if not MEDIA_LEGACY_HOST:
            return url
        parsed = urlparse(url)
        host = (parsed.hostname or parsed.netloc or "").lower()
        if not (host == MEDIA_LEGACY_HOST or host.endswith("." + MEDIA_LEGACY_HOST)):
            return url
        parts = parsed.path.lstrip("/").split("/", 1)
        if len(parts) < 2:
            return url
        bucket, key = parts[0], parts[1]
        if not bucket or not key:
            return url
        # The same object key lives in MEDIA_REBUCKET on the active S3 client; re-presign there
        # (the embedded legacy signature/expiry is irrelevant — that host may be unreachable).
        if MEDIA_REBUCKET:
            return _presign_s3_object(MEDIA_REBUCKET, key) or url
        qs = parse_qs(parsed.query or "")
        date_str = (qs.get("X-Amz-Date") or [""])[0]
        exp_str = (qs.get("X-Amz-Expires") or [""])[0]
        if date_str and exp_str:
            try:
                dt = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                expires = int(exp_str)
                if datetime.now(timezone.utc) <= dt + timedelta(seconds=expires - 30):
                    return url
            except Exception:
                pass
        refreshed = _presign_s3_object(bucket, key)
        return refreshed or url
    except Exception:
        return url


def _presign_s3_url(s3_url: str) -> str:
    try:
        parts = s3_url.replace("s3://", "", 1).split("/", 1)
        if len(parts) < 2:
            return ""
        bucket, key = parts[0], parts[1]
        if not bucket or not key:
            return ""
        return _presign_s3_object(bucket, key)
    except Exception:
        return ""


# Bounded cache of presigned URLs: {(bucket, key): (url, expiry_epoch)}.
# Avoids re-presigning on every sample view; reused while >60s of validity remains.
_PRESIGN_CACHE_MAX = 512
_presign_cache: "OrderedDict[tuple[str, str], tuple[str, float]]" = OrderedDict()
_presign_lock = threading.Lock()


def _presign_s3_object(bucket: str, key: str) -> str:
    cache_key = (bucket, key)
    now = time.time()
    with _presign_lock:
        cached = _presign_cache.get(cache_key)
        if cached and cached[1] - now > 60:
            _presign_cache.move_to_end(cache_key)
            return cached[0]
    s3 = _get_s3_client()
    if not s3:
        return ""
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=S3_PRESIGN_EXPIRE,
        )
    except Exception:
        return ""
    if url:
        with _presign_lock:
            _presign_cache[cache_key] = (url, now + S3_PRESIGN_EXPIRE)
            _presign_cache.move_to_end(cache_key)
            while len(_presign_cache) > _PRESIGN_CACHE_MAX:
                _presign_cache.popitem(last=False)
    return url


# Cache of the resolved local output file for (source, model, base_bench) -> Path.
# get_sample_detail re-walks the tree (find_inference_output / path.rglob) on every
# request otherwise; this avoids re-walking for repeated views of the same comparison.
# Bounded LRU; stores the Path (or None for a confirmed miss) so misses aren't re-walked.
_TARGET_CACHE_MAX = 512
_target_cache: "OrderedDict[tuple[str, str, str], Path | None]" = OrderedDict()
_target_lock = threading.Lock()


def _get_cached_target(cache_key: tuple[str, str, str]):
    """Return (hit, value). value is a Path, None (confirmed miss), or unset on a cache miss."""
    with _target_lock:
        if cache_key in _target_cache:
            _target_cache.move_to_end(cache_key)
            return True, _target_cache[cache_key]
    return False, None


def _set_cached_target(cache_key: tuple[str, str, str], value) -> None:
    with _target_lock:
        _target_cache[cache_key] = value
        _target_cache.move_to_end(cache_key)
        while len(_target_cache) > _TARGET_CACHE_MAX:
            _target_cache.popitem(last=False)


# Magic prefixes of base64-encoded media (so we can serve embedded media inline,
# which is the only representation that is portable across hosts).
_B64_MAGIC = {
    "/9j/": "image/jpeg",
    "iVBORw0KGgo": "image/png",
    "R0lGOD": "image/gif",
    "Qk": "image/bmp",
    "SUQz": "audio/mpeg",
    "T2dnUw": "audio/ogg",
    "ZkxhQw": "audio/flac",
    "GkXf": "video/webm",
}
_B64_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-"
)


def _is_base64_blob(s: str) -> bool:
    """True if s looks like an embedded base64 blob rather than a path/URL.
    A filesystem path contains '.' (e.g. extensions) which is not a base64 char,
    so this also distinguishes base64 JPEG (starts with '/9j/') from real paths."""
    if len(s) < 64:
        return False
    return all(c in _B64_CHARS for c in s[:256])


def _b64_decoded_size(value: str) -> int:
    """Approximate decoded byte size of a base64 string without decoding it
    (3 bytes per 4 chars, minus '=' padding) — avoids materialising the bytes
    just to measure them on a 1GB box."""
    n = len(value)
    pad = value.count("=", max(0, n - 2))
    return (n // 4) * 3 - pad


def _is_embedded_media(value: str) -> bool:
    """True if value is embedded (inlineable) media — a data: URI or a bare base64 blob.
    Such values have no filesystem path, so when skipped they must NOT fall back to a path."""
    return isinstance(value, str) and (value.startswith("data:") or _is_base64_blob(value))


def _media_data_uri(value: str, kind: str) -> str | None:
    """Return a data: URI when value is embedded base64 media, else None.

    Embedded media whose decoded size exceeds MEDIA_INLINE_MAX_BYTES is dropped
    (returns None) to keep the sample-detail JSON from ballooning to many MB."""
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("data:"):
        b64 = value.split(",", 1)[1] if "," in value else ""
        if _b64_decoded_size(b64) > MEDIA_INLINE_MAX_BYTES:
            logging.warning("Skipping oversized inline %s media (%d B decoded)", kind, _b64_decoded_size(b64))
            return None
        return value
    if value.startswith(("http://", "https://", "s3://")):
        return None
    if not _is_base64_blob(value):
        return None
    if _b64_decoded_size(value) > MEDIA_INLINE_MAX_BYTES:
        logging.warning("Skipping oversized embedded %s media (%d B decoded)", kind, _b64_decoded_size(value))
        return None
    mime = None
    for pfx, m in _B64_MAGIC.items():
        if value.startswith(pfx):
            mime = m
            break
    if value.startswith("UklGR"):  # RIFF container: webp image or wav audio
        mime = "image/webp" if kind == "image" else "audio/wav"
    if not mime:
        mime = {"image": "image/jpeg", "audio": "audio/mpeg", "video": "video/mp4"}.get(
            kind, "application/octet-stream"
        )
    return f"data:{mime};base64,{value}"


def _dedup_prefer_embedded(items: list[str]) -> list[str]:
    """De-dup, and when an embedded (data:) copy exists drop bare filesystem-path
    references to the same media — those are non-portable and often absent."""
    seen: list[str] = []
    for x in items:
        if x and x not in seen:
            seen.append(x)
    if any(x.startswith("data:") for x in seen):
        seen = [x for x in seen if x.startswith(("data:", "http://", "https://"))]
    return seen


def _oversized_embedded_bytes(value: str) -> int | None:
    """If value is embedded base64 media (data: URI or bare blob) whose decoded
    size exceeds the inline cap, return that decoded byte count; else None.
    Used to emit the omni://oversized sentinel instead of silently dropping it."""
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("data:"):
        b64 = value.split(",", 1)[1] if "," in value else ""
        size = _b64_decoded_size(b64)
        return size if size > MEDIA_INLINE_MAX_BYTES else None
    if value.startswith(("http://", "https://", "s3://")):
        return None
    if not _is_base64_blob(value):
        return None
    size = _b64_decoded_size(value)
    return size if size > MEDIA_INLINE_MAX_BYTES else None


def _append_media(val, target: list[str], base_dir: Path | None, kind: str) -> None:
    def _one(v):
        if not isinstance(v, str) or not v:
            return
        uri = _media_data_uri(v, kind)
        if uri:
            target.append(uri)
            return
        # Embedded media (data:/base64) has no filesystem path. If it was dropped
        # because it exceeds the inline cap, append the shared oversized sentinel
        # so the frontend can render a clear warning instead of silently losing it.
        oversized = _oversized_embedded_bytes(v)
        if oversized is not None:
            target.append(f"omni://oversized?kind={kind}&bytes={oversized}")
            return
        if _is_embedded_media(v):
            return
        resolved = _resolve_media_path(v, base_dir)
        if resolved:
            target.append(resolved)
    if isinstance(val, str):
        _one(val)
    elif isinstance(val, list):
        for v in val:
            _one(v)


def _extract_media_from_messages(messages: list, base_dir: Path | None) -> tuple[list[str], list[str], list[str]]:
    images: list[str] = []
    videos: list[str] = []
    audios: list[str] = []
    if not isinstance(messages, list):
        return images, videos, audios
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type") or "").lower()
            if ptype in ("image", "image_url"):
                _append_media(part.get("value") or part.get("url") or part.get("image") or part.get("image_url"), images, base_dir, "image")
            elif ptype in ("video", "video_url"):
                _append_media(part.get("value") or part.get("url") or part.get("video") or part.get("video_url"), videos, base_dir, "video")
            elif ptype in ("audio", "audio_url"):
                _append_media(part.get("value") or part.get("url") or part.get("audio") or part.get("audio_url"), audios, base_dir, "audio")
    return images, videos, audios


def _extract_media_from_record(rec: dict, base_dir: Path | None) -> tuple[list[str], list[str], list[str]]:
    images: list[str] = []
    videos: list[str] = []
    audios: list[str] = []
    if not isinstance(rec, dict):
        return images, videos, audios
    for key in ("image", "image_path", "image_url", "image_paths", "images"):
        _append_media(rec.get(key), images, base_dir, "image")
    for key in ("video", "video_path", "video_url", "video_paths", "videos"):
        _append_media(rec.get(key), videos, base_dir, "video")
    for key in ("audio", "audio_path", "audio_url", "audio_paths", "audios"):
        _append_media(rec.get(key), audios, base_dir, "audio")
    meta = rec.get("meta")
    if isinstance(meta, dict):
        for key in ("image", "image_path", "image_url", "image_paths", "images"):
            _append_media(meta.get(key), images, base_dir, "image")
        for key in ("video", "video_path", "video_url", "video_paths", "videos"):
            _append_media(meta.get(key), videos, base_dir, "video")
        for key in ("audio", "audio_path", "audio_url", "audio_paths", "audios"):
            _append_media(meta.get(key), audios, base_dir, "audio")
    msg_images, msg_videos, msg_audios = _extract_media_from_messages(rec.get("messages") or [], base_dir)
    images.extend(msg_images)
    videos.extend(msg_videos)
    audios.extend(msg_audios)
    return _dedup_prefer_embedded(images), _dedup_prefer_embedded(videos), _dedup_prefer_embedded(audios)


def _resolve_local_path(source: str, model: str) -> Path | None:
    if source == "internal":
        d = INTERNAL_OUTPUTS_PATH / model
        if d.exists():
            return d
        z = INTERNAL_OUTPUTS_PATH / f"{model}.zip"
        return z if z.exists() else None
    if source == "direct":
        ensure_upload_dir()
        z = DIRECT_SUBMISSION_DIR / f"{model}.zip"
        return z if z.exists() else None
    return None


def _list_inference_benchmarks_from_dir(base: Path) -> tuple[list[str], dict[str, str]]:
    benches: set[str] = set()
    eng_map: dict[str, str] = {}
    inf_paths = list(base.rglob("inference_output/*.json"))
    if inf_paths:
        for p in inf_paths:
            bench = extract_benchmark_from_filename(p.name)
            benches.add(bench)
            eng_map.setdefault(bench, _engine_from_path(p))
        return sorted(benches), eng_map
    # No inference_output/ dirs: derive benchmarks from eval/output filenames (no file reads).
    for root_name in ("evaluation_output", "output"):
        for p in base.rglob(f"{root_name}/*.json"):
            bench = extract_benchmark_from_filename(p.name)
            benches.add(bench)
            eng_map.setdefault(bench, _engine_from_path(p))
    return sorted(benches), eng_map


def _list_inference_benchmarks_from_zip(zpath: Path) -> tuple[list[str], dict[str, str]]:
    benches: set[str] = set()
    eng_map: dict[str, str] = {}
    try:
        with zipfile.ZipFile(zpath, "r") as zf:
            names = zf.namelist()
            inf_names = [n for n in names if "inference_output/" in n and n.endswith(".json")]
            if inf_names:
                for n in inf_names:
                    bench = extract_benchmark_from_filename(Path(n).name)
                    benches.add(bench)
                    eng_map.setdefault(bench, _engine_from_zip_path(n))
                return sorted(benches), eng_map
            for n in names:
                if ("/evaluation_output/" not in n and "/output/" not in n) or not n.endswith(".json"):
                    continue
                bench = extract_benchmark_from_filename(Path(n).name)
                benches.add(bench)
                eng_map.setdefault(bench, _engine_from_zip_path(n))
    except Exception:
        pass
    return sorted(benches), eng_map


def _s3_prefix(model: str) -> str:
    prefix = (S3_PREFIX or "").rstrip("/")
    return f"{prefix}/{model}/" if prefix else f"{model}/"


def _list_s3_keys(model: str) -> dict[str, list[str]]:
    cached = get_s3_key_cache(model)
    if cached is not None:
        return cached
    keys = {"inference": [], "evaluation": []}
    s3 = _get_s3_client()
    if not s3 or not S3_BUCKET:
        set_s3_key_cache(model, keys)
        return keys
    prefix = _s3_prefix(model)
    token = None
    failed = False
    for i in range(20):
        params = {"Bucket": S3_BUCKET, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            params["ContinuationToken"] = token
        # _s3_list_with_retry already retries with backoff and returns {} (logging a
        # warning) on failure, so an S3 outage / bad creds degrades to an empty result
        # instead of raising a 500 through /benchmarks and /sample.
        resp = _s3_list_with_retry(s3, params)
        if not resp and i == 0:
            failed = True  # first page failed outright; don't cache the empty result
            break
        for obj in resp.get("Contents", []):
            key = (obj.get("Key") or "").strip()
            if not key or not key.endswith(".json"):
                continue
            if "/inference_output/" in key:
                keys["inference"].append(key)
            elif "/evaluation_output/" in key or "/output/" in key:
                keys["evaluation"].append(key)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break
    if not failed:
        set_s3_key_cache(model, keys)
    return keys


def _benchmarks_from_s3_model(model: str) -> tuple[list[str], dict[str, str]]:
    benches: set[str] = set()
    eng_map: dict[str, str] = {}
    keys = _list_s3_keys(model)
    if keys["inference"]:
        for k in keys["inference"]:
            bench = extract_benchmark_from_filename(Path(k).name)
            benches.add(bench)
            eng_map.setdefault(bench, _engine_from_key(k))
        return sorted(benches), eng_map
    # Derive from eval/output key names (no object downloads).
    for k in keys["evaluation"]:
        bench = extract_benchmark_from_filename(Path(k).name)
        benches.add(bench)
        eng_map.setdefault(bench, _engine_from_key(k))
    return sorted(benches), eng_map


def _get_benchmarks_for_model_with_engine(source: str, model: str) -> tuple[list[str], dict[str, str]]:
    # Gate disabled sources here too: model_ids arrive as a client-supplied query param, so a
    # stale/crafted request for an s3 model must not trigger heavy S3 listing when s3 is
    # disabled via OMNI_ENABLED_SOURCES (the box would hang). Mirrors scan_cache.get_models().
    if source not in ENABLED_SOURCES:
        return [], {}
    key = (source, model)
    cached = get_bench_cache(key)
    if cached is not None:
        return cached
    if source == "s3":
        benches, eng_map = _benchmarks_from_s3_model(model)
    else:
        path = _resolve_local_path(source, model)
        if not path:
            benches, eng_map = [], {}
        elif path.is_dir():
            benches, eng_map = _list_inference_benchmarks_from_dir(path)
        else:
            benches, eng_map = _list_inference_benchmarks_from_zip(path)
    set_bench_cache(key, benches, eng_map)
    return benches, eng_map


@router.get("/models")
def get_models(source: str | None = Query(None, description="internal|direct|s3")):
    """Get inference model options (DB when enabled, else paths)."""
    ensure_upload_dir()
    internal = sc_get_models("internal")
    direct = sc_get_models("direct")
    s3 = sc_get_models("s3")
    # Aggregate per-model benchmark counts once from warm caches (zero per-request I/O).
    counts = get_model_benchmark_counts()
    options: list[dict] = []

    def add_models(src: str, models: Iterable[str]) -> None:
        for m in models:
            if source and src != source:
                continue
            cached = get_bench_cache((src, m))
            cached_benchmarks = cached[0] if cached else []
            mc = counts.get((src, m)) or {}
            options.append({
                "id": f"{src}:{m}",
                "model": m,
                "source": src,
                "checkpoint": "checkpoint-none",
                "benchmarks": cached_benchmarks,
                "bench_count": mc.get("total", 0),
                "bench_counts": {
                    "text": mc.get("text", 0),
                    "image": mc.get("image", 0),
                    "video": mc.get("video", 0),
                    "audio": mc.get("audio", 0),
                },
            })

    add_models("internal", internal)
    add_models("direct", direct)
    add_models("s3", s3)
    return JSONResponse(content={"options": options}, headers={"Cache-Control": "no-cache"})


@router.get("/benchmarks")
def get_benchmarks(model_ids: str = Query("", description="Comma-separated model IDs")):
    """List benchmarks for the selected models from their inference/output FILES, not from
    evaluation score-keys. lm_eval_harness packs several sub-task scores into one file, so a
    score-key like 'click_cot' has no standalone file and can't be viewed — listing only
    file-backed benchmarks means selecting any listed benchmark always yields samples."""
    ids = [x.strip() for x in model_ids.split(",") if x.strip()]
    benches: set[str] = set()
    eng_sets: dict[str, set[str]] = {}
    for mid in ids:
        src, model = parse_model_id(mid)
        b_list, eng_map = _get_benchmarks_for_model_with_engine(src, model)
        benches.update(b_list)
        for b, eng in (eng_map or {}).items():
            if eng:
                eng_sets.setdefault(b, set()).add(eng)
    bench_engine_map: dict[str, str] = {}
    for b, engs in eng_sets.items():
        if not engs:
            continue
        if len(engs) == 1:
            bench_engine_map[b] = next(iter(engs))
        else:
            bench_engine_map[b] = " / ".join(sorted(engs))
    sorted_benches = sorted(benches)
    mod_meta = get_benchmark_modalities()
    return {
        "benchmarks": sorted_benches,
        "benchmark_engine_map": bench_engine_map,
        "benchmark_modality_map": {b: classify_benchmark_modality(b, mod_meta) for b in sorted_benches},
    }


@router.get("/sample/{sample_idx:int}")
def get_sample_detail(
    sample_idx: int,
    model_ids: str = Query(..., description="Comma-separated model IDs"),
    benchmark: str = Query(""),
):
    """Get detail for a single sample. Streams just the requested record from each model's
    output file (never loads the whole file). Path-based (internal/direct/s3)."""
    # benchmark is required: an empty value makes the file resolver's `base_bench in stem`
    # substring match succeed against the FIRST output file, silently returning an unrelated
    # benchmark's sample at 200. The UI always sends one; guard the raw endpoint too.
    if not benchmark.strip():
        raise HTTPException(status_code=422, detail="benchmark query parameter is required")
    ids = [x.strip() for x in model_ids.split(",") if x.strip()]
    result = {
        "question": "",
        "ground_truth": "",
        "choices": "",
        "predictions": [],
        "total_samples": 0,
    }
    for mid in ids:
        src, model = parse_model_id(mid)
        record, total, media_base = _get_one_sample(src, model, benchmark, sample_idx)
        if total:
            result["total_samples"] = max(result["total_samples"], total)
        if record:
            _merge_one_record(result, record, src, model, media_base)
    # Bounds check runs only once totals are known: an out-of-range index would otherwise
    # return 200 with an empty record and an unexplained total. Skip when total is unknown
    # (0) so a transient resolution failure doesn't masquerade as a bad index.
    total_samples = result["total_samples"]
    if total_samples and not (0 <= sample_idx < total_samples):
        raise HTTPException(
            status_code=404,
            detail=f"sample_idx {sample_idx} out of range (0..{total_samples - 1})",
        )
    # Media is a property of the sample (shared across models), but it is
    # collected once per model above — collapse identical entries so the same
    # image/audio isn't shown N times when N models are compared.
    media = result.get("media")
    if media:
        for key in ("images", "videos", "audios"):
            if media.get(key):
                media[key] = _dedup_prefer_embedded(media[key])
    return result


def _stringify_field(val) -> str:
    """Render a record field (ground truth / choices) as display text while
    preserving falsy-but-valid values such as 0 or False."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        return ", ".join(_stringify_field(v) for v in val)
    if isinstance(val, dict):
        return "; ".join(f"{k}: {_stringify_field(v)}" for k, v in val.items())
    return str(val)


def _record_ground_truth(rec: dict) -> str:
    """Ground truth may live under several keys (commonly `label`) and may be a
    falsy-but-valid value (e.g. label 0 / False), so test presence explicitly
    rather than truthiness — an `or` chain would silently drop those."""
    for key in ("ground_truth", "label", "answer", "target", "gold", "gt"):
        if key in rec and rec[key] is not None:
            return _stringify_field(rec[key])
    return ""


def _record_choices(rec: dict) -> str:
    for key in ("choices", "options", "option_contents"):
        if rec.get(key):
            return _stringify_field(rec[key])
    return ""


def _as_score(v):
    """Coerce a record value to a numeric score, or None if it isn't one.

    ijson decodes fractional JSON numbers as ``decimal.Decimal`` (neither int nor
    float), so a plain ``isinstance(v, (int, float))`` filter silently drops float
    scores read through the streaming path. Decimals are coerced to ``float`` so the
    JSON response serializes. ``bool`` is preserved as-is (an int subclass)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, Decimal):
        return float(v)
    return None


def _merge_one_record(result: dict, rec: dict, src: str, model: str, media_base: Path | None) -> None:
    """Fold a single streamed record into the viewer response."""
    pred = rec.get("prediction_postprocessed")
    if pred is None:
        pred = rec.get("prediction")
    if pred is None:
        pred = rec.get("answer")
    if pred is None:
        pred = "-"
    scores = {}
    for k, v in rec.items():
        sv = _as_score(v)
        if sv is not None:
            scores[k] = sv
    metrics = rec.get("metrics")
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            sv = _as_score(v)
            if sv is not None:
                scores[k] = sv
    result["predictions"].append({
        "model": model,
        "source": src,
        "prediction": pred,
        "scores": scores,
    })
    images, videos, audios = _extract_media_from_record(rec, media_base)
    if images or videos or audios:
        media = result.setdefault("media", {"images": [], "videos": [], "audios": []})
        media["images"].extend(images)
        media["videos"].extend(videos)
        media["audios"].extend(audios)
    if not result.get("question"):
        question = rec.get("question") or rec.get("prompt") or ""
        if not question:
            question = message_text_from_messages(rec.get("messages") or [])
        result["question"] = question
        result["ground_truth"] = _record_ground_truth(rec)
        result["choices"] = _record_choices(rec)


# Output files put num_records/num_samples in the head config/meta for built-in outputs,
# so reading only the HEAD (no tail) yields the total without a record-count pass. The
# zip/S3 sites slice the first 64KiB; that constant is the shared head-window size.
_HEAD_BYTES = 65536


def _too_large_record(size_bytes: int) -> dict:
    """A synthetic record whose prediction explains the sample's file is over the per-sample
    preview cap, so the viewer shows a clear note instead of the server buffering it."""
    mb = size_bytes / (1024 * 1024)
    cap_mb = INFERENCE_SAMPLE_MAX_BYTES / (1024 * 1024)
    return {
        "prediction": (
            f"[Preview unavailable] This output file is {mb:.0f} MB, over the "
            f"{cap_mb:.0f} MB per-sample preview limit (server memory protection). "
            f"Download the file to inspect this sample."
        ),
    }


def _read_sample_from_bytes(get_head, open_body, idx: int) -> tuple[dict | None, int]:
    """Shared read+count+stream for a byte-source (zip member / S3 object).

    ``get_head()`` returns the head slice bytes; ``total_from_head`` reads the count from it
    (empty tail — the count lives in the head). When absent, fall back to counting records by
    STREAMING the body. ``open_body`` is a factory returning a fresh readable stream each call
    (zip member / S3 Body), so the whole member is never buffered — memory stays bounded to
    ijson's parse buffers + one record. Head-read and body-read are independently guarded so a
    header miss never loses the body (and vice-versa)."""
    total = 0
    try:
        total = total_from_head(parse_head_tail_bytes(get_head(), b"")) or 0
    except Exception:
        pass
    rec = None
    try:
        if not total:
            total = count_records_stream(open_body)
        rec, _ = stream_record_at_stream(open_body, idx)
    except Exception:
        rec = None
    return rec, total


def _get_one_sample(source: str, model: str, benchmark: str, idx: int) -> tuple[dict | None, int, Path | None]:
    """Resolve the output file for (source, model, benchmark) and stream the idx-th record.
    Returns (record, total_samples, media_base). Never loads the whole file."""
    # Same gate as _get_benchmarks_for_model_with_engine: a client-supplied s3 model_id must
    # not reach _get_one_sample_s3 → _list_s3_keys when s3 is disabled (avoids the hang).
    if source not in ENABLED_SOURCES:
        return None, 0, None
    base_bench = _base_benchmark(benchmark)
    if source == "s3":
        return _get_one_sample_s3(model, base_bench, idx)
    path = _resolve_local_path(source, model)
    if not path:
        return None, 0, None
    if path.is_dir():
        cache_key = (source, model, base_bench)
        hit, target = _get_cached_target(cache_key)
        if not hit:
            target = find_inference_output(path, base_bench)
            if not target:
                for root_name in ("evaluation_output", "output"):
                    for p in path.rglob(f"{root_name}/*.json"):
                        if base_bench in p.stem:
                            target = p
                            break
                    if target:
                        break
            _set_cached_target(cache_key, target)
        if not target:
            return None, 0, None
        # Local files read head+tail and stream directly off disk (both mtime-cached), so the
        # whole file is never materialised — hence this path stays file-based rather than
        # routing through _read_sample_from_bytes. count_records only walks when the head
        # config/meta lacks the count.
        total = total_from_head(read_json_head_tail(target)) or count_records(target)
        return stream_record_at(target, idx), total, target.parent
    if path.suffix == ".zip" and path.exists():
        return _get_one_sample_zip(path, base_bench, idx)
    return None, 0, None


def _get_one_sample_zip(zpath: Path, base_bench: str, idx: int) -> tuple[dict | None, int, Path | None]:
    try:
        with zipfile.ZipFile(zpath, "r") as zf:
            names = zf.namelist()
            target = next(
                (n for n in names if "inference_output/" in n and n.endswith(".json") and base_bench in Path(n).stem),
                None,
            )
            if not target:
                target = next(
                    (n for n in names
                     if ("/evaluation_output/" in n or "/output/" in n) and n.endswith(".json") and base_bench in Path(n).stem),
                    None,
                )
            if not target:
                return None, 0, None
            # Guard: don't materialise/parse an oversized member to preview one record.
            # The count still comes from the head slice; show a "too large" note instead.
            try:
                member_size = zf.getinfo(target).file_size
            except Exception:
                member_size = 0
            if member_size > INFERENCE_SAMPLE_MAX_BYTES:
                total = 0
                try:
                    total = total_from_head(parse_head_tail_bytes(_zip_read(zf, target, _HEAD_BYTES), b"")) or 0
                except Exception:
                    pass
                return _too_large_record(member_size), total, None
            # zf.open() yields a fresh streaming member each call, so the reader streams the
            # record without buffering the whole member into RAM.
            rec, total = _read_sample_from_bytes(
                lambda: _zip_read(zf, target, _HEAD_BYTES),
                lambda: zf.open(target),
                idx,
            )
            return rec, total, None
    except Exception:
        return None, 0, None


def _zip_read(zf: zipfile.ZipFile, name: str, n: int | None = None) -> bytes:
    """Read a zip member: the first ``n`` bytes (head slice) or the whole member."""
    with zf.open(name) as s:
        return s.read(n) if n is not None else s.read()


def _get_one_sample_s3(model: str, base_bench: str, idx: int) -> tuple[dict | None, int, Path | None]:
    s3 = _get_s3_client()
    if not s3 or not S3_BUCKET:
        return None, 0, None
    keys = _list_s3_keys(model)
    key = next((k for k in keys["inference"] if base_bench in Path(k).stem), None)
    if not key:
        key = next((k for k in keys["evaluation"] if base_bench in Path(k).stem), None)
    if not key:
        return None, 0, None

    def _read_head() -> bytes:
        # HEAD Range read: the count lives in the head config/meta for built-in outputs.
        return s3.get_object(Bucket=S3_BUCKET, Key=key, Range=f"bytes=0-{_HEAD_BYTES - 1}")["Body"].read()

    # Guard: skip the body fetch entirely for oversized objects (head gives the count).
    try:
        clen = int(s3.head_object(Bucket=S3_BUCKET, Key=key).get("ContentLength", 0) or 0)
    except Exception:
        clen = 0
    if clen and clen > INFERENCE_SAMPLE_MAX_BYTES:
        total = 0
        try:
            total = total_from_head(parse_head_tail_bytes(_read_head(), b"")) or 0
        except Exception:
            pass
        return _too_large_record(clen), total, None

    def _open_body():
        # Factory: a fresh streaming Body per call (an S3 Body can't be rewound). The reader
        # streams records through ijson without buffering the whole object into RAM.
        return s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"]

    rec, total = _read_sample_from_bytes(_read_head, _open_body, idx)
    return rec, total, None
