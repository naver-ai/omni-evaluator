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

"""JSON I/O utilities: reading JSON from files (full, light, head+tail)."""

import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

import json  # stdlib: always available, and tolerates bare NaN/Infinity that orjson rejects

try:
    import orjson
    _USE_ORJSON = True
except ImportError:
    _USE_ORJSON = False


def _loads(data: bytes | str) -> dict | None:
    """Parse JSON. Prefers orjson for speed; on failure falls back to stdlib json.

    Eval outputs can embed bare NaN/Infinity literals (e.g. option_contents padded with NaN).
    orjson rejects those, so without the fallback a NaN-bearing file would silently parse to
    None and drop out of the scan. stdlib json.loads accepts them, recovering the record.
    """
    if _USE_ORJSON:
        try:
            return orjson.loads(data.encode("utf-8") if isinstance(data, str) else data)
        except Exception:
            pass  # orjson can't parse NaN/Infinity — retry with the tolerant stdlib parser
    try:
        s = data.decode("utf-8") if isinstance(data, bytes) else data
        return json.loads(s)
    except Exception:
        return None


def json_safe(obj, nan_repl=None):
    """Recursively replace non-finite floats (NaN/Inf) with `nan_repl`.

    Starlette's JSONResponse serializes with allow_nan=False, so any NaN/Inf reaching a
    response raises ValueError. Eval outputs can carry NaN — a literal that stdlib json.loads
    accepts, or computed (e.g. 0/0 in weight normalization / omni-index). Sanitize at the
    response boundary so a single bad metric can't 500 the whole leaderboard.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else nan_repl
    if isinstance(obj, dict):
        return {k: json_safe(v, nan_repl) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v, nan_repl) for v in obj]
    return obj


def read_json(path: Path, encoding: str = "utf-8") -> dict | None:
    """Read JSON from file. Returns None on error."""
    try:
        raw = path.read_bytes()
        return _loads(raw)
    except Exception:
        return None


def read_json_light(path: Path) -> dict | None:
    """Stream-parse JSON and keep only lightweight fields."""
    try:
        import ijson
        from ijson.common import ObjectBuilder
    except Exception:
        return read_json(path)
    allowed = {
        "meta",
        "config",
        "evaluation",
        "metrics",
        "metric_keys",
        "task_name",
        "evaluation_engine",
        "inference_engine",
    }
    data: dict = {}
    pending_key: str | None = None
    capture_key: str | None = None
    builder: ObjectBuilder | None = None
    depth = 0
    capture_next = False
    try:
        with path.open("rb") as f:
            for prefix, event, value in ijson.parse(f):
                if builder is None:
                    if prefix == "" and event == "map_key":
                        pending_key = value
                        capture_next = pending_key in allowed
                        continue
                    if capture_next:
                        builder = ObjectBuilder()
                        builder.event(event, value)
                        capture_key = pending_key
                        if event in ("start_map", "start_array"):
                            depth = 1
                        else:
                            data[capture_key] = builder.value
                            builder = None
                            capture_key = None
                        capture_next = False
                else:
                    builder.event(event, value)
                    if event in ("start_map", "start_array"):
                        depth += 1
                    elif event in ("end_map", "end_array"):
                        depth -= 1
                        if depth == 0:
                            data[capture_key] = builder.value
                            builder = None
                            capture_key = None
            return data or None
    except Exception:
        return read_json(path)


# Brace/string scanners below run a tiny state machine over a JSON slice.
# The key invariant: '{' and '}' (and a value-terminating '"') only count when
# we are NOT inside a quoted string. We track that with `in_string`, which
# flips on every quote that is *unescaped* -- i.e. preceded by an even number
# of consecutive backslashes (an odd count means the quote is part of the
# string, like \"). Without this, a brace or quote inside a string value
# corrupts the depth counter / truncates the value and json.loads fails.

def _find_string_end(text: str, start: int) -> int:
    """Given `start` pointing at an opening '"', return the index of the
    matching closing unescaped quote, or -1 if none is found in `text`."""
    i = start + 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2  # skip the escaped character (e.g. \" or \\)
            continue
        if ch == '"':
            return i
        i += 1
    return -1


def _find_object_end(text: str, start: int) -> int:
    """Given `start` pointing at an opening '{', return the index just past the
    matching closing '}' (string-aware), or -1 if the object never closes
    within `text`."""
    depth = 0
    in_string = False
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2  # skip the escaped character inside the string
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _parse_head_tail(head: str, tail: str) -> dict | None:
    """Extract lightweight fields from head (config/meta/scalars) and tail (evaluation/metrics)
    string slices of a large JSON. Shared by the file reader and the S3 byte-range reader."""
    import json as _json
    data: dict = {}

    def _extract_object(text: str, rest: str, key: str) -> None:
        """Parse the brace-balanced object at the start of `rest` into data[key].
        Logs (rather than silently dropping) when the object never closes."""
        end = _find_object_end(rest, 0)
        if end < 0:
            logger.warning("_parse_head_tail: unterminated %r object in head/tail slice", key)
            return
        try:
            data[key] = _json.loads(rest[:end])
        except Exception:
            logger.warning("_parse_head_tail: failed to json.loads %r object", key)

    # --- Head: config, meta, scalar fields ---
    for key in ("config", "meta", "task_name", "evaluation_engine", "inference_engine", "checkpoint"):
        marker = f'"{key}"'
        idx = head.find(marker)
        if idx < 0:
            continue
        colon = head.find(":", idx + len(marker))
        if colon < 0:
            continue
        rest = head[colon + 1:].lstrip()
        if rest.startswith('"'):
            end = _find_string_end(rest, 0)
            if end > 0:
                data[key] = rest[1:end]
        elif rest.startswith('{'):
            _extract_object(head, rest, key)

    # --- Tail: evaluation object ---
    ev_idx = tail.find('"evaluation"')
    if ev_idx >= 0:
        colon = tail.find(":", ev_idx + 12)
        if colon >= 0:
            rest = tail[colon + 1:].lstrip()
            if rest.startswith("{"):
                _extract_object(tail, rest, "evaluation")

    # --- Tail: top-level metrics (some formats put it at the end) ---
    if "metrics" not in data.get("evaluation", {}):
        m_idx = tail.rfind('"metrics"')
        if m_idx >= 0:
            colon = tail.find(":", m_idx + 9)
            if colon >= 0:
                rest = tail[colon + 1:].lstrip()
                if rest.startswith("{"):
                    _extract_object(tail, rest, "metrics")

    return data or None


def parse_head_tail_bytes(head: bytes, tail: bytes) -> dict | None:
    """Head+tail parse from raw byte slices (e.g. S3 Range responses)."""
    return _parse_head_tail(
        head.decode("utf-8", errors="ignore"),
        tail.decode("utf-8", errors="ignore"),
    )


def read_json_head_tail(path: Path, head_bytes: int = 64 * 1024, tail_bytes: int = 1024 * 1024) -> dict | None:
    """Read only head + tail of a large JSON file and extract lightweight fields.

    Much faster than full parse for files with big 'inference'/'output' arrays in the middle.
    Head gives us: config, meta. Tail gives us: evaluation, metrics.
    Falls back to read_json_light for small files.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= head_bytes + tail_bytes:
        return read_json_light(path)
    try:
        with path.open("rb") as f:
            head = f.read(head_bytes).decode("utf-8", errors="ignore")
            f.seek(max(0, size - tail_bytes))
            tail = f.read().decode("utf-8", errors="ignore")
    except Exception:
        return read_json_light(path)
    return _parse_head_tail(head, tail)
