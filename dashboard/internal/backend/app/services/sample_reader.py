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

"""Memory-bounded single-sample reader for the Inference Viewer.

The output files can be hundreds of MB (a single omni-modal sample embeds base64 media,
so one record alone can be ~1MB and a file holds thousands). Loading the whole file to
show one sample is what makes the viewer slow and blows past a 1GB box.

Instead we stream the records array with ijson (C backend) and stop at the requested
index — so memory stays bounded to ~one record regardless of file size.
"""

import io
import json
import threading
from collections import OrderedDict
from pathlib import Path

import ijson

# Record arrays live under one of these top-level paths, depending on the file format /
# evaluation engine. "inference" is often a list-of-lists (per-run batches), hence the
# ".item.item" variant that flattens it the same way build_inference_output_from_eval does.
_RECORD_PATHS = ("inference.item.item", "inference.item", "output.item", "outputs.item")

# NaN-tolerant fallback. Eval JSONs are written with Python's json.dump, which emits bare
# `NaN`/`Infinity` literals (e.g. `option_contents` padded with NaN). Those are invalid per the
# JSON spec, so ijson's strict streaming parser ABORTS at the first such token — every record
# after it silently becomes invisible in the viewer (one early NaN can blank a whole file).
# Python's json.loads accepts NaN/Infinity by default, so a bounded full parse recovers them.
#
# Memory note (1 GB box): a full json.loads materialises the whole object graph (~several× the
# raw bytes), so this is the one place that breaks the otherwise-streaming, ~one-record memory
# bound. Two guards keep it safe: (1) its OWN 64 MiB size cap (independent of, and smaller than,
# INFERENCE_SAMPLE_MAX_BYTES whose default is 256 MiB) — a file too large to full-parse is skipped
# rather than risk OOM; all NaN-affected files today are < 14 MB. (2) a semaphore that serialises
# concurrent full parses so N threadpool workers (THREADPOOL_TOKENS) can't each allocate a graph
# at once and blow the cgroup cap.
_FULL_PARSE_MAX_BYTES = 64 * 1024 * 1024  # independent 64 MiB bound (NOT tied to INFERENCE_SAMPLE_MAX_BYTES)
_full_parse_sem = threading.BoundedSemaphore(1)


def _flatten_records(obj):
    """Return the flat record list from a parsed file (handles inference: [[...]] nesting)."""
    if not isinstance(obj, dict):
        return None
    for key in ("inference", "output", "outputs"):
        cur = obj.get(key)
        if isinstance(cur, list):
            if cur and all(isinstance(s, list) for s in cur):
                flat = []
                for sub in cur:
                    flat.extend(sub)
                return flat
            return cur
    return None


def _full_parse_records(raw):
    """NaN/Infinity-tolerant parse → flat record list, or None. `raw` is bytes/str.

    Serialised by `_full_parse_sem` so concurrent fallbacks can't multiply peak RAM."""
    with _full_parse_sem:
        try:
            obj = json.loads(raw)  # Python json accepts NaN/Infinity by default
        except Exception:
            return None
        return _flatten_records(obj)

# total-count cache: {path_str: (mtime, count)}  — only used when the file has no num_records.
# Bounded LRU (most-recent at the end) so it can't grow without limit on a 1GB box.
_COUNT_CACHE_MAX = 512
_count_cache: "OrderedDict[str, tuple[float, int]]" = OrderedDict()
_count_lock = threading.Lock()


def total_from_head(head: dict | None) -> int | None:
    """Pull total sample count from already-parsed head fields (config/meta) — no scan."""
    if not isinstance(head, dict):
        return None
    cfg = head.get("config") if isinstance(head.get("config"), dict) else {}
    for m in (head, cfg, cfg.get("meta"), head.get("meta")):
        if isinstance(m, dict):
            for k in ("num_records", "num_samples"):
                v = m.get(k)
                if isinstance(v, int) and v > 0:
                    return v
    return None


def _iter_path(stream, prefix):
    return ijson.items(stream, prefix)


def _stream_provider(source):
    """Normalise a record source into (make_stream, owns).

    ``source`` may be:
      * a callable factory returning a FRESH readable file-like each call — preferred for
        large zip members / S3 bodies: nothing is buffered whole, so memory stays bounded
        to ijson's parse buffers + one record even for a 50MB+ file. owns=True (we close
        each stream we open).
      * raw bytes / a one-shot file-like — buffered ONCE into a seekable BytesIO and rewound
        per prefix attempt (single copy, not the previous bytes+BytesIO double copy).
        owns=False (the single buffer is reused across attempts, GC'd after).
    """
    if callable(source):
        return source, True
    raw = source if isinstance(source, (bytes, bytearray)) else source.read()
    buf = io.BytesIO(raw)

    def make():
        buf.seek(0)
        return buf

    return make, False


def _close(it) -> None:
    """Close an ijson iterator if it supports it. The pure-python backend returns a
    generator (worth closing deterministically on a 1GB box); the yajl2_c C backend
    returns an iterator with no .close(), so we guard with getattr — calling .close()
    blindly (e.g. via contextlib.closing) raises AttributeError on the C backend."""
    close = getattr(it, "close", None)
    if close is not None:
        try:
            close()
        except Exception:
            pass


def stream_record_at_stream(source, idx: int) -> tuple[dict | None, bool]:
    """Return (record_at_idx, matched_array). matched_array=True means the right array was
    found (even if idx was out of range). Tries record-path variants in priority order.

    ``source`` is a re-openable factory (preferred — streams without buffering the whole
    member) or raw bytes / a file-like (buffered once); see ``_stream_provider``. A parse
    error on one prefix falls through to the next prefix rather than aborting."""
    make, owns = _stream_provider(source)
    parse_error = False
    for prefix in _RECORD_PATHS:
        yielded = 0
        try:
            f = make()
        except Exception:
            continue
        try:
            items = _iter_path(f, prefix)
            try:
                for i, rec in enumerate(items):
                    yielded += 1
                    if i == idx:
                        return (rec if isinstance(rec, dict) else None), True
            finally:
                _close(items)
        except Exception:
            parse_error = True  # likely a bare NaN/Infinity aborting the strict stream
            yielded = 0
        finally:
            if owns:
                try:
                    f.close()
                except Exception:
                    pass
        if yielded:
            return None, True  # correct array, index out of range
    # Stream hit a parse error (NaN/Infinity) — recover via a bounded tolerant full parse.
    if parse_error:
        try:
            f = make()
            raw = f.read(_FULL_PARSE_MAX_BYTES + 1)
            if owns:
                try:
                    f.close()
                except Exception:
                    pass
            if raw is not None and len(raw) <= _FULL_PARSE_MAX_BYTES:
                recs = _full_parse_records(raw)
                if recs is not None:
                    if 0 <= idx < len(recs):
                        rec = recs[idx]
                        return (rec if isinstance(rec, dict) else None), True
                    return None, True
        except Exception:
            pass
    return None, False


def stream_record_at(path: Path, idx: int) -> dict | None:
    """Stream the idx-th record from a local JSON file without loading the whole file.

    Falls back to a bounded NaN/Infinity-tolerant full parse when the strict ijson stream
    can't yield the record (the file embeds bare NaN literals; see _full_parse_records)."""
    parse_error = False
    for prefix in _RECORD_PATHS:
        yielded = 0
        try:
            with path.open("rb") as f:
                items = ijson.items(f, prefix)
                try:
                    for i, rec in enumerate(items):
                        yielded += 1
                        if i == idx:
                            return rec if isinstance(rec, dict) else None
                finally:
                    _close(items)
        except Exception:
            parse_error = True  # likely a bare NaN/Infinity aborting the strict stream
            continue
        if yielded:
            return None  # this prefix matched and fully streamed — idx genuinely out of range
    # No prefix cleanly streamed the record. If the stream hit a parse error (NaN/Infinity),
    # recover the record via a bounded tolerant full parse.
    if parse_error:
        try:
            if path.stat().st_size <= _FULL_PARSE_MAX_BYTES:
                recs = _full_parse_records(path.read_bytes())
                if recs is not None and 0 <= idx < len(recs):
                    rec = recs[idx]
                    return rec if isinstance(rec, dict) else None
        except Exception:
            pass
    return None


def count_records_stream(source) -> int:
    """Count records from a re-openable factory or raw bytes / file-like (see
    ``_stream_provider``). Used only when a zip/S3 source's header lacks
    num_records/num_samples. Streams the source per prefix attempt without buffering the
    whole member when given a factory."""
    make, owns = _stream_provider(source)
    parse_error = False
    for prefix in _RECORD_PATHS:
        n = 0
        try:
            f = make()
        except Exception:
            continue
        try:
            items = _iter_path(f, prefix)
            try:
                for _ in items:
                    n += 1
            finally:
                _close(items)
        except Exception:
            parse_error = True  # NaN/Infinity aborts the strict stream short of the true count
            n = 0
        finally:
            if owns:
                try:
                    f.close()
                except Exception:
                    pass
        if n:
            return n
    if parse_error:
        try:
            f = make()
            raw = f.read(_FULL_PARSE_MAX_BYTES + 1)
            if owns:
                try:
                    f.close()
                except Exception:
                    pass
            if raw is not None and len(raw) <= _FULL_PARSE_MAX_BYTES:
                recs = _full_parse_records(raw)
                if recs is not None:
                    return len(recs)
        except Exception:
            pass
    return 0


def count_records(path: Path) -> int:
    """Count records via a single streaming pass (cached by mtime). Only needed when the
    file does not record num_records/num_samples in its header."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return 0
    key = str(path)
    with _count_lock:
        c = _count_cache.get(key)
        if c and c[0] == mtime:
            _count_cache.move_to_end(key)  # mark most-recently used
            return c[1]
    total = 0
    parse_error = False
    for prefix in _RECORD_PATHS:
        n = 0
        try:
            with path.open("rb") as f:
                items = ijson.items(f, prefix)
                try:
                    for _ in items:
                        n += 1
                finally:
                    _close(items)
        except Exception:
            parse_error = True  # NaN/Infinity aborts the strict stream short of the true count
            continue
        if n:
            total = n
            break
    if total == 0 and parse_error:
        try:
            if path.stat().st_size <= _FULL_PARSE_MAX_BYTES:
                recs = _full_parse_records(path.read_bytes())
                if recs is not None:
                    total = len(recs)
        except Exception:
            pass
    with _count_lock:
        _count_cache[key] = (mtime, total)
        _count_cache.move_to_end(key)
        while len(_count_cache) > _COUNT_CACHE_MAX:
            _count_cache.popitem(last=False)  # evict least-recently used
    return total
