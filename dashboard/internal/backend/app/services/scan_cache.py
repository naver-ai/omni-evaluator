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

"""Lazy mtime-based scan cache. The filesystem IS the database.

No explicit sync needed — files are scanned on demand, and only
re-parsed when their mtime changes.
"""

import logging
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import CACHE_DIR, DIRECT_SUBMISSION_DIR, INTERNAL_OUTPUTS_PATH
from ..config import SCAN_IJSON_THRESHOLD, SCAN_WORKERS, S3_SCAN_WORKERS, ENABLED_SOURCES
from .json_io import read_json, read_json_head_tail
from .leaderboard_filters import classify_benchmark_modality, STATIC_BENCH_MODALITY
from .metric_extraction import (
    _extract_meta_and_metrics_from_full,
    extract_benchmark_from_filename,
    extract_metrics_map,
)
from .scan import (
    _base_benchmark,
    _checkpoint_from_path,
    _fallback_modality_from_engine,
    normalize_eval_engine_from_path,
    scan_eval_zip,
)

_lock = threading.Lock()


def trim_memory() -> None:
    """Return freed heap back to the OS after a heavy scan.

    glibc malloc keeps freed arenas, so RSS stays high after a big transient spike
    (parsing many JSON files). On small instances (e2-micro) that retained RSS is the
    difference between fitting in 1GB and not. malloc_trim hands it back.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Per-file mtime cache: {abs_path_str: (mtime, extracted_data)}
# extracted_data = {
#   "benchmarks": {bench: score_str},
#   "metrics": {bench: {metric_key: val}},
#   "model": str, "inference": str, "checkpoint": str,
#   "eval_engine": str, "modality": {bench: str},
# }
# ---------------------------------------------------------------------------
_file_cache: dict[str, tuple[float, dict, int]] = {}

# Zip-level cache: {abs_path_str: (mtime, scan_result, size)}
_zip_cache: dict[str, tuple[float, tuple, int]] = {}

# Benchmark metadata (aggregated from all scans)
_bench_engines: dict[str, str] = {}
_bench_modalities: dict[str, str] = {}

# S3 cache: {model_name: (timestamp, rows)}
_s3_cache: dict[str, tuple[float, list[dict]]] = {}
_s3_models_cache: tuple[float, list[str]] | None = None
_S3_TTL = 300  # 5 minutes

# In-flight S3 fetches: {model_name: Event} — guarded by _lock. Ensures only one
# thread fetches a given model; others wait on the Event then read the cache.
_s3_inflight: dict[str, threading.Event] = {}


def _drop_file_cache_entry(path_str: str, entry: dict | None) -> None:
    """Remove a stale file-cache entry and its associated bench-map entries.

    Caller must hold _lock. `entry` is the cached extracted_data (or None).
    """
    _file_cache.pop(path_str, None)
    if not entry:
        return
    for bench in entry.get("benchmarks", {}):
        _bench_engines.pop(bench, None)
    for bench in entry.get("modality", {}):
        _bench_modalities.pop(bench, None)


def _scan_single_eval_json(
    json_path: Path, model_name: str
) -> dict | None:
    """Extract metrics from one evaluation JSON. Returns cached result keyed by mtime.

    Uses read_json_light for large files (>512KB) to avoid parsing full output arrays.
    Skips modality detection on first pass for speed (uses engine-based fallback only).
    """
    path_str = str(json_path)
    try:
        st = json_path.stat()
        mtime = st.st_mtime
        size = st.st_size
    except OSError:
        # File deleted/inaccessible — drop the stale entry (and its bench maps).
        with _lock:
            _drop_file_cache_entry(path_str, _file_cache.get(path_str, (0, None))[1])
        return None

    with _lock:
        cached = _file_cache.get(path_str)
        if cached and cached[0] == mtime and cached[2] == size:
            return cached[1]

    # Use head+tail reader for large files (skips huge inference/output arrays)
    data = read_json_head_tail(json_path) if size >= SCAN_IJSON_THRESHOLD else read_json(json_path)
    if not data:
        return None

    bench_hint = extract_benchmark_from_filename(json_path.name)
    results = _extract_meta_and_metrics_from_full(data, model_name, bench_hint)
    if not results:
        return None

    r = results[0]
    base_bench = _base_benchmark(r.get("benchmark") or bench_hint)
    metrics_map = extract_metrics_map(data)
    ev_eng = normalize_eval_engine_from_path(json_path)
    ckpt = _checkpoint_from_path(json_path)

    # Fast modality detection: engine-based fallback only (no extra file reads)
    modality = {}
    mod = _fallback_modality_from_engine(ev_eng)
    if mod:
        modality[base_bench] = mod

    entry = {
        "benchmarks": {base_bench: r["metric_value"]},
        "metrics": {base_bench: metrics_map} if metrics_map else {},
        "model": model_name,
        "inference": r["inference"],
        "checkpoint": ckpt,
        "eval_engine": ev_eng,
        "modality": modality,
    }

    with _lock:
        _file_cache[path_str] = (mtime, entry, size)
    return entry


def _scan_model_dir(model_dir: Path, source: str) -> list[dict]:
    """Scan a model directory tree for evaluation JSONs. Returns list of row dicts."""
    model_name = model_dir.name
    rows_map: dict[tuple[str, str, str], dict] = {}

    # Find all evaluation_output / output dirs recursively
    eval_dirs = []
    for root, dirs, files in os.walk(model_dir):
        rp = Path(root)
        if rp.name in ("evaluation_output", "output"):
            eval_dirs.append(rp)

    for eval_dir in eval_dirs:
        for json_path in eval_dir.glob("*.json"):
            entry = _scan_single_eval_json(json_path, model_name)
            if not entry:
                continue

            key = (entry["model"], entry["inference"], entry["checkpoint"])
            if key not in rows_map:
                rows_map[key] = {
                    "model": entry["model"],
                    "inference": entry["inference"],
                    "checkpoint": entry["checkpoint"],
                    "source": source,
                    "scores": {},
                    "metrics": {},
                }
            rows_map[key]["scores"].update(entry["benchmarks"])
            rows_map[key]["metrics"].update(entry.get("metrics", {}))

            # Update global metadata
            with _lock:
                for bench, eng in [(b, entry["eval_engine"]) for b in entry["benchmarks"]]:
                    if bench not in _bench_engines and eng != "-":
                        _bench_engines[bench] = eng
                for bench, mod in entry.get("modality", {}).items():
                    if bench not in _bench_modalities:
                        _bench_modalities[bench] = mod

    return list(rows_map.values())


def _scan_zip(zip_path: Path, source: str) -> list[dict]:
    """Scan a zip file, using mtime cache."""
    path_str = str(zip_path)
    try:
        st = zip_path.stat()
        mtime = st.st_mtime
        size = st.st_size
    except OSError:
        return []

    with _lock:
        cached = _zip_cache.get(path_str)
        if cached and cached[0] == mtime and cached[2] == size:
            return cached[1]

    model_name = zip_path.stem
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            benchmarks, rows_map, bench_eng, bench_mod = scan_eval_zip(zf, model_name, source)
    except Exception:
        logging.warning("Failed to scan zip: %s", zip_path)
        return []

    rows = []
    for key, r in rows_map.items():
        rows.append({
            "model": r["model"],
            "inference": r["inference"],
            "checkpoint": r["checkpoint"],
            "source": source,
            "scores": {b: r.get(b, "-") for b in benchmarks if not b.startswith("_")},
            "metrics": r.get("_metrics", {}),
        })

    with _lock:
        _zip_cache[path_str] = (mtime, rows, size)
        for bench, eng in bench_eng.items():
            if eng != "-" and bench not in _bench_engines:
                _bench_engines[bench] = eng
        for bench, mod in bench_mod.items():
            if mod and bench not in _bench_modalities:
                _bench_modalities[bench] = mod

    return rows


def _scan_s3_models() -> list[str]:
    """List S3 models with TTL cache."""
    global _s3_models_cache
    now = time.time()
    with _lock:
        if _s3_models_cache and (now - _s3_models_cache[0]) < _S3_TTL:
            return _s3_models_cache[1]
    try:
        from .s3_sync import _get_s3_client, _list_s3_models_sync
        s3 = _get_s3_client()
        if not s3:
            return []
        models = _list_s3_models_sync()
        with _lock:
            _s3_models_cache = (now, models)
        return models
    except Exception as e:
        logging.warning("S3 list failed: %s", e)
        return []


def _scan_s3_model(model_name: str) -> list[dict]:
    """Scan one S3 model with TTL cache."""
    now = time.time()
    with _lock:
        cached = _s3_cache.get(model_name)
        if cached and (now - cached[0]) < _S3_TTL:
            return cached[1]
        # In-flight guard: only one thread fetches a given model. Others wait.
        inflight = _s3_inflight.get(model_name)
        if inflight is None:
            inflight = _s3_inflight[model_name] = threading.Event()
            is_fetcher = True
        else:
            is_fetcher = False

    if not is_fetcher:
        # Another thread is fetching this model — wait, then read the cache.
        inflight.wait(_S3_TTL)
        with _lock:
            cached = _s3_cache.get(model_name)
        return cached[1] if cached else []

    try:
        try:
            from .s3_sync import _scan_one_s3_model_sync
            raw_rows, bench_eng = _scan_one_s3_model_sync(model_name)
        except Exception as e:
            logging.warning("S3 scan failed for %s: %s", model_name, e)
            return []

        rows = []
        for r in raw_rows:
            benchmarks = [k for k in r.keys() if k not in {"model", "inference", "checkpoint", "source", "_path", "_inference", "_metrics"} and not k.startswith("_")]
            rows.append({
                "model": r["model"],
                "inference": r.get("inference", "-"),
                "checkpoint": r.get("checkpoint", "checkpoint-none"),
                "source": "s3",
                "scores": {b: r[b] for b in benchmarks if r.get(b) is not None},
                "metrics": r.get("_metrics", {}),
            })

        with _lock:
            _s3_cache[model_name] = (now, rows)
            for bench, eng in bench_eng.items():
                if eng != "-" and bench not in _bench_engines:
                    _bench_engines[bench] = eng

        return rows
    finally:
        # Release waiters and clear the in-flight marker.
        with _lock:
            _s3_inflight.pop(model_name, None)
        inflight.set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_METRIC_SUFFIX_MAP: dict[str, str] = {
    "runtime_inference": "__runtime_inference",
    "runtime_total": "__runtime_total",
    "coverage_inference": "__coverage_inference",
    "coverage_evaluation": "__coverage_evaluation",
    "latency": "__latency",
    "throughput": "__throughput",
    "benchmark_time": "__benchmark_time",
}


def is_cache_warm() -> bool:
    """Check if any cache has entries (from disk or prior scan).

    File results live in _file_cache, zips in _zip_cache, S3 in _s3_cache.
    """
    with _lock:
        return bool(_file_cache or _zip_cache or _s3_cache)


def scan_all_sources(sources: set[str] | None = None, quick: bool = False) -> tuple[list[str], list[dict]]:
    """Scan internal + direct + s3. Returns (sorted_benchmarks, rows).

    quick=True: only return already-cached results (no new file I/O). Instant.
    quick=False: full scan with mtime cache. Slow on cold start.
    """
    # Restrict to whitelisted sources. A disabled source (e.g. s3 on a small box) is never
    # scanned/listed regardless of the request, so its heavy listing can't hang the server —
    # while S3 creds remain available for media URL re-signing (a separate, non-scan path).
    if sources is None:
        sources = set(ENABLED_SOURCES)
    else:
        sources = {s for s in sources if s in ENABLED_SOURCES}
    all_rows: list[dict] = []

    if quick:
        # Return only what's already in memory cache — no file I/O.
        # Take shallow snapshots under the lock, then build rows outside it.
        with _lock:
            file_items = list(_file_cache.items())
            zip_items = list(_zip_cache.items())
            s3_items = list(_s3_cache.items()) if "s3" in sources else []

        for path_str, (mtime, entry, size) in file_items:
            # Infer source from path (file-cache entries never carry a "_source" key).
            if str(DIRECT_SUBMISSION_DIR) in path_str:
                source = "direct"
            else:
                source = "internal"
            if source not in sources:
                continue
            all_rows.append({
                "model": entry["model"],
                "inference": entry.get("inference", "-"),
                "checkpoint": entry.get("checkpoint", "checkpoint-none"),
                "source": source,
                "scores": dict(entry.get("benchmarks", {})),
                "metrics": dict(entry.get("metrics", {})),
            })
        # Also include zip cache
        for path_str, (mtime, rows, size) in zip_items:
            for r in rows:
                if r.get("source") in sources:
                    all_rows.append(dict(r))
        # Also include S3 cache
        for model_name, (ts, rows) in s3_items:
            for r in rows:
                all_rows.append(dict(r))
    else:
        # Full scan
        if "internal" in sources and INTERNAL_OUTPUTS_PATH.exists():
            try:
                scan_tasks = []
                for entry in os.scandir(INTERNAL_OUTPUTS_PATH):
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            scan_tasks.append((Path(entry.path), True))
                        elif entry.name.endswith(".zip") and entry.is_file(follow_symlinks=False):
                            scan_tasks.append((Path(entry.path), False))
                    except OSError:
                        continue
                def _scan_internal_entry(args):
                    path, is_dir = args
                    try:
                        if is_dir:
                            return _scan_model_dir(path, "internal")
                        else:
                            return _scan_zip(path, "internal")
                    except Exception as e:
                        logging.warning("Scan failed: %s (%s)", path, e)
                        return []
                with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
                    for rows in pool.map(_scan_internal_entry, scan_tasks):
                        all_rows.extend(rows)
            except OSError:
                pass

        if "direct" in sources and DIRECT_SUBMISSION_DIR.exists():
            for zpath in DIRECT_SUBMISSION_DIR.glob("*.zip"):
                try:
                    all_rows.extend(_scan_zip(zpath, "direct"))
                except Exception as e:
                    logging.warning("Direct scan failed: %s (%s)", zpath, e)

        if "s3" in sources:
            try:
                s3_models = _scan_s3_models()
                with ThreadPoolExecutor(max_workers=S3_SCAN_WORKERS) as pool:
                    results = pool.map(_scan_s3_model, s3_models)
                    for rows in results:
                        all_rows.extend(rows)
            except Exception as e:
                logging.warning("S3 scan failed: %s", e)

    # Expand metric suffixes into score columns
    benchmarks_set: set[str] = set()
    for r in all_rows:
        scores = r.get("scores", {})
        metrics = r.get("metrics", {})
        for bench, m in metrics.items():
            if not isinstance(m, dict):
                continue
            for key, suffix in _METRIC_SUFFIX_MAP.items():
                val = m.get(key)
                if val is not None and val != "":
                    scores[f"{bench}{suffix}"] = val
        benchmarks_set.update(scores.keys())
        r["scores"] = scores

    benchmarks = sorted(benchmarks_set)
    # Fill missing with "-"
    for r in all_rows:
        r["scores"] = {b: r["scores"].get(b, "-") for b in benchmarks}

    return benchmarks, all_rows


def get_models(source: str) -> list[str]:
    """List model names for a source by scanning the filesystem."""
    # A source disabled via OMNI_ENABLED_SOURCES is never listed (keeps the heavy S3 listing
    # off small boxes even when creds are present for media re-signing).
    if source not in ENABLED_SOURCES:
        return []
    if source == "internal":
        if not INTERNAL_OUTPUTS_PATH.exists():
            return []
        names = []
        try:
            for entry in os.scandir(INTERNAL_OUTPUTS_PATH):
                try:
                    if entry.is_dir(follow_symlinks=False):
                        names.append(entry.name)
                    elif entry.name.endswith(".zip") and entry.is_file(follow_symlinks=False):
                        names.append(entry.name[:-4])
                except OSError:
                    continue
        except OSError:
            pass
        return sorted(set(names))
    elif source == "direct":
        if not DIRECT_SUBMISSION_DIR.exists():
            return []
        return sorted(p.stem for p in DIRECT_SUBMISSION_DIR.glob("*.zip"))
    elif source == "s3":
        return _scan_s3_models()
    return []


def get_benchmark_engines() -> dict[str, str]:
    with _lock:
        return dict(_bench_engines)


def get_benchmark_modalities() -> dict[str, str]:
    with _lock:
        m = dict(_bench_modalities)
    # Overlay the data-derived map: a concrete media type from real content (image/video/audio)
    # outranks the scan's engine-only guess, so the leaderboard modality filter and the shipped
    # benchmark_modality_map match the inference viewer's classify_benchmark_modality.
    for b, mod in STATIC_BENCH_MODALITY.items():
        if mod in ("audio", "video", "image"):
            m[b] = mod
    return m


_SYNTHETIC_SUFFIXES = tuple(_METRIC_SUFFIX_MAP.values())


def _is_real_benchmark(bench: str) -> bool:
    """Return False for synthetic leaderboard columns (runtime, coverage, latency, etc.)."""
    return not bench.endswith(_SYNTHETIC_SUFFIXES)


def get_model_benchmark_counts() -> dict[tuple[str, str], dict[str, int]]:
    """Count distinct real benchmarks per (source, model), split by modality.

    Aggregates PURELY from the in-memory caches (_file_cache, _zip_cache,
    _s3_cache) under _lock — no filesystem I/O and no S3 fetch. The returned
    keys match sc_get_models("internal"/"direct"/"s3"):
      * internal  -> first path component under INTERNAL_OUTPUTS_PATH
      * direct    -> zip stem under DIRECT_SUBMISSION_DIR
      * s3        -> the _s3_cache model_name
    Counts use the same filtering as get_benchmarks_for_models
    (_is_real_benchmark and value != "-").

    Each value is a breakdown dict with keys "total", "text", "image",
    "video", "audio"; the four modality buckets sum to "total". Modality is
    derived via classify_benchmark_modality using the warm _bench_modalities
    map (name-heuristic fallback when a benchmark isn't in the map).
    """
    benches: dict[tuple[str, str], set[str]] = {}
    internal_root = str(INTERNAL_OUTPUTS_PATH)
    direct_root = str(DIRECT_SUBMISSION_DIR)

    with _lock:
        modality_map = dict(_bench_modalities)
        # Per-file cache: derive (source, model) from the path. internal model is
        # the first path component under INTERNAL_OUTPUTS_PATH (matches sc_get_models).
        for path_str, (_mtime, entry, _size) in _file_cache.items():
            if path_str.startswith(internal_root):
                try:
                    model = Path(path_str).relative_to(INTERNAL_OUTPUTS_PATH).parts[0]
                except (ValueError, IndexError):
                    continue
                src = "internal"
            elif path_str.startswith(direct_root):
                # Direct submissions are zips, not loose files; skip loose files here.
                continue
            else:
                continue
            bset = benches.setdefault((src, model), set())
            for b, v in (entry.get("benchmarks") or {}).items():
                if v != "-" and _is_real_benchmark(b):
                    bset.add(b)

        # Zip cache: distinguish internal-zip vs direct by path prefix; the model is
        # the zip stem (matches sc_get_models for both internal zips and direct).
        for path_str, (_mtime, rows, _size) in _zip_cache.items():
            if path_str.startswith(internal_root):
                src = "internal"
            elif path_str.startswith(direct_root):
                src = "direct"
            else:
                continue
            model = Path(path_str).stem
            bset = benches.setdefault((src, model), set())
            for r in rows:
                for b, v in (r.get("scores") or {}).items():
                    if v != "-" and _is_real_benchmark(b):
                        bset.add(b)

        # S3 cache: keyed by model_name (matches sc_get_models("s3")).
        for model_name, (_ts, rows) in _s3_cache.items():
            bset = benches.setdefault(("s3", model_name), set())
            for r in rows:
                for b, v in (r.get("scores") or {}).items():
                    if v != "-" and _is_real_benchmark(b):
                        bset.add(b)

    result: dict[tuple[str, str], dict[str, int]] = {}
    for key, bset in benches.items():
        breakdown = {"total": 0, "text": 0, "image": 0, "video": 0, "audio": 0}
        for b in bset:
            mod = classify_benchmark_modality(b, modality_map)
            if mod not in breakdown:
                mod = "text"
            breakdown[mod] += 1
            breakdown["total"] += 1
        result[key] = breakdown
    return result


def get_benchmarks_for_models(ids: list[tuple[str, str]]) -> list[str]:
    """Get benchmarks available for given (source, model) pairs by scanning."""
    benches: set[str] = set()
    for src, model in ids:
        if src == "internal" and INTERNAL_OUTPUTS_PATH.exists():
            model_dir = INTERNAL_OUTPUTS_PATH / model
            if model_dir.is_dir():
                for row in _scan_model_dir(model_dir, "internal"):
                    benches.update(k for k, v in row.get("scores", {}).items() if v != "-" and _is_real_benchmark(k))
            zip_path = INTERNAL_OUTPUTS_PATH / f"{model}.zip"
            if zip_path.exists():
                for row in _scan_zip(zip_path, "internal"):
                    benches.update(k for k, v in row.get("scores", {}).items() if v != "-" and _is_real_benchmark(k))
        elif src == "direct" and DIRECT_SUBMISSION_DIR.exists():
            zip_path = DIRECT_SUBMISSION_DIR / f"{model}.zip"
            if zip_path.exists():
                for row in _scan_zip(zip_path, "direct"):
                    benches.update(k for k, v in row.get("scores", {}).items() if v != "-" and _is_real_benchmark(k))
        elif src == "s3":
            for row in _scan_s3_model(model):
                benches.update(k for k, v in row.get("scores", {}).items() if v != "-" and _is_real_benchmark(k))
    return sorted(benches)


_DISK_CACHE_PATH = CACHE_DIR / "scan_cache.json"


def save_cache_to_disk() -> None:
    """Persist file + zip caches to disk for fast startup.

    Persisting _zip_cache avoids re-parsing every zip member (zf.read) on each
    restart — the cold-start CPU/memory spike this cache exists to prevent.
    rows are already JSON-serializable (same shape returned in API responses).
    S3 results are intentionally NOT persisted: the 5-min TTL would invalidate
    them almost immediately, so persistence is pointless.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        data = {
            "file_cache": {k: [v[0], v[1], v[2]] for k, v in _file_cache.items()},
            "zip_cache": {k: [v[0], v[1], v[2]] for k, v in _zip_cache.items()},
            "bench_engines": _bench_engines,
            "bench_modalities": _bench_modalities,
        }
    try:
        import json
        _DISK_CACHE_PATH.write_text(json.dumps(data, default=str))
    except Exception:
        logging.exception("Failed to save scan cache to disk")


def load_cache_from_disk() -> bool:
    """Load persisted file + zip caches. Returns True if loaded.

    A missing 'zip_cache' key (older on-disk schema) loads as empty — those
    zips simply get re-scanned on first access. Stale entries (path gone or
    mtime/size mismatch) are re-scanned by _scan_zip's validation, same as
    _file_cache. S3 results are not persisted (see save_cache_to_disk).
    """
    if not _DISK_CACHE_PATH.exists():
        return False
    try:
        import json
        data = json.loads(_DISK_CACHE_PATH.read_text())
        with _lock:
            for k, v in data.get("file_cache", {}).items():
                # New format: [mtime, entry, size]. Old format: [mtime, entry].
                # For old entries, default size to -1 so the (mtime, size) check
                # forces a one-time re-scan (old cache had no size identity).
                size = v[2] if len(v) > 2 else -1
                _file_cache[k] = (v[0], v[1], size)
            for k, v in data.get("zip_cache", {}).items():
                # New format: [mtime, rows, size]. Tolerate any legacy 2-element
                # form by defaulting size to -1 (forces a one-time re-scan).
                size = v[2] if len(v) > 2 else -1
                _zip_cache[k] = (v[0], v[1], size)
            _bench_engines.update(data.get("bench_engines", {}))
            _bench_modalities.update(data.get("bench_modalities", {}))
        logging.info(
            "Loaded scan cache from disk: %d file, %d zip entries",
            len(_file_cache), len(_zip_cache),
        )
        return True
    except Exception:
        logging.exception("Failed to load scan cache from disk")
        return False


def evict_zip(zip_path: Path) -> None:
    """Remove a single zip from cache (after deletion)."""
    path_str = str(zip_path)
    with _lock:
        _zip_cache.pop(path_str, None)


def clear_cache() -> None:
    """Clear all caches (forces full re-scan on next request)."""
    global _s3_models_cache
    with _lock:
        _file_cache.clear()
        _zip_cache.clear()
        _s3_cache.clear()
        _s3_models_cache = None
        _bench_engines.clear()
        _bench_modalities.clear()
    try:
        _DISK_CACHE_PATH.unlink(missing_ok=True)
    except Exception:
        pass
