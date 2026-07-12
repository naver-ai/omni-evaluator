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

"""Leaderboard API: aggregated results from Internal/Direct/S3."""

import json
import logging
import re
import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from ..auth import require_api_key
from ..config import (
    CHART_ROWS_MAX,
    LEADERBOARD_WEIGHT_FILE,
)
from ..services.scan_cache import scan_all_sources, get_benchmark_engines, get_benchmark_modalities, is_cache_warm, load_cache_from_disk, save_cache_to_disk, trim_memory
from ..services.json_io import json_safe
from ..services.leaderboard_filters import (
    apply_engine_modality_fallback,
    apply_metric_override,
    build_modality_matcher,
    collect_metric_keys,
    compute_averages,
    compute_minmax,
    filter_benchmarks_by_eval_engine,
    filter_benchmarks_by_value_type,
    filter_by_inference_engine,
    filter_by_model_mode,
    merge_inference_rows,
    parse_float,
    sort_rows,
    strip_numeric_suffix_benchmarks,
)

router = APIRouter()

# Guards a single background full rescan triggered by ?refresh=true so only one
# runs at a time (e2-micro has 1 GB RAM — concurrent scans would OOM).
_REFRESH_LOCK = threading.Lock()
_refresh_running = False

# Safe charset for benchmark names used as response keys. Benchmark names are
# derived from ZIP/file names (extract_benchmark_from_filename), which are
# attacker-influenced on the S3/upload path.
_SAFE_BENCHMARK_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _trigger_background_refresh() -> None:
    """Kick off a guarded background full rescan; return immediately.

    Mirrors main.py's _background_refresh. The existing `scanning` flag covers
    the UI while this runs, so the request is never blocked on the rescan.
    """
    global _refresh_running
    with _REFRESH_LOCK:
        if _refresh_running:
            return
        _refresh_running = True

    def _run():
        global _refresh_running
        try:
            scan_all_sources(quick=False)
            save_cache_to_disk()
            trim_memory()  # hand freed heap back to the OS so RSS fits small instances
            logging.info("Leaderboard refresh scan complete, cache saved")
        except Exception:
            logging.exception("Leaderboard refresh scan failed")
        finally:
            with _REFRESH_LOCK:
                _refresh_running = False

    threading.Thread(target=_run, daemon=True).start()


def _log_suspicious_benchmarks(benchmarks) -> None:
    """Log benchmark names containing chars outside the safe charset.

    Conservative: we only log, not rewrite. Normalizing here would risk breaking
    key joins against the weight file (LEADERBOARD_WEIGHT_FILE) and the per-source
    scan caches, which all key on the raw benchmark name.
    """
    for b in benchmarks:
        if not _SAFE_BENCHMARK_RE.match(str(b)):
            logging.warning("Suspicious benchmark name (unsafe charset): %r", b)


# --- Response memoization -----------------------------------------------------------------
# The handler recomputes the full result set from the scan cache on every request (CPU-bound
# filtering/sort/min-max). On the 1-vCPU box, concurrent identical requests (the 60s auto-
# refresh from several tabs, or a refresh storm) stack behind the GIL, spiking latency and
# starving /health. Memoize the built response under a short TTL keyed by the query params, and
# build it under one lock so a burst of identical requests does the work once. The frontend
# auto-refreshes every 60s, so an 8s TTL is invisible to users while collapsing CPU under load.
_LB_CACHE: dict = {}            # cache_key -> (monotonic_ts, content)
_LB_LOCK = threading.Lock()
_LB_CACHE_TTL = 8.0
_LB_CACHE_MAX = 32
_VALID_VALUE_TYPES = {"score", "time", "coverage"}


@router.get("")
def get_leaderboard_endpoint(
    sources: str = Query("internal,direct,s3", description="Comma-separated: internal,direct,s3"),
    page: int = Query(0, ge=0),
    page_size: int = Query(50, ge=1, le=100000),
    refresh: bool = Query(False, description="Bypass cache"),
    sort: str = Query("", description="model|inference|checkpoint|avg|<benchmark>"),
    sort_dir: str = Query("asc", description="asc|desc"),
    hidden_benchmarks: str = Query("", description="Comma-separated benchmarks to hide"),
    show_avg: bool = Query(False, description="Show avg column"),
    modality: str = Query("", description="Filter by modality: comma-separated text,vision,audio (OR)"),
    inference_engine: str = Query("", description="Filter by inference engine"),
    evaluation_engine: str = Query("", description="Filter by evaluation engine"),
    value_type: str = Query("score", description="Value to display: score|time|coverage"),
    value_subtype: str = Query("", description="For time/coverage subtypes"),
    merge_inference: bool = Query(True, description="Merge rows with same model+checkpoint"),
    metric_map: str = Query("", description="JSON map of benchmark->metric key"),
    model_mode: str = Query("normal", description="Model mode: normal|debug|all"),
):
    """Get leaderboard data with sort, column visibility, avg (memoized, TTL=8s)."""
    if refresh:
        # Trigger a guarded background full rescan; don't block this request.
        # The `scanning` flag below keeps the UI informed while it runs.
        _trigger_background_refresh()

    cache_key = (
        sources, page, page_size, sort, sort_dir, hidden_benchmarks, show_avg, modality,
        inference_engine, evaluation_engine, value_type, value_subtype, merge_inference,
        metric_map, model_mode,
    )
    with _LB_LOCK:
        hit = _LB_CACHE.get(cache_key)
        if hit is not None and not refresh and (time.monotonic() - hit[0]) < _LB_CACHE_TTL:
            content = hit[1]
        else:
            content = _build_leaderboard(
                sources, page, page_size, sort, sort_dir, hidden_benchmarks, show_avg,
                modality, inference_engine, evaluation_engine, value_type, value_subtype,
                merge_inference, metric_map, model_mode,
            )
            _LB_CACHE[cache_key] = (time.monotonic(), content)
            if len(_LB_CACHE) > _LB_CACHE_MAX:
                _LB_CACHE.pop(min(_LB_CACHE, key=lambda k: _LB_CACHE[k][0]), None)
    # NaN/Inf (data literal or computed in weighting/omni-index) would 500 the strict-JSON
    # response; map non-finite scores to "-" (the leaderboard's missing-value marker).
    return JSONResponse(content=json_safe(content, "-"), headers={"Cache-Control": "no-cache"})


def _build_leaderboard(
    sources, page, page_size, sort, sort_dir, hidden_benchmarks, show_avg, modality,
    inference_engine, evaluation_engine, value_type, value_subtype, merge_inference,
    metric_map, model_mode,
):
    """Compute the leaderboard response payload (pure compute; caller memoizes)."""
    source_set = {s.strip() for s in sources.split(",") if s.strip()}

    # --- Data loading ---
    # Always use quick mode (cache only) to never block the server.
    # Background process handles full scanning and disk cache updates.
    if not is_cache_warm():
        load_cache_from_disk()  # Try to pick up background scan results
    benchmarks, rows = scan_all_sources(source_set, quick=True)
    _log_suspicious_benchmarks(benchmarks)
    scanning = not is_cache_warm()
    benchmark_modality_map = get_benchmark_modalities()
    benchmark_engine_map = get_benchmark_engines()

    # --- Row-level filters ---
    if source_set:
        rows = [r for r in rows if r["source"] in source_set]
    rows = filter_by_model_mode(rows, model_mode)
    rows = filter_by_inference_engine(rows, inference_engine.split(","))

    # --- Metadata ---
    benchmark_modality_map = apply_engine_modality_fallback(benchmark_engine_map, benchmark_modality_map)
    metric_keys_map = collect_metric_keys(rows)

    # --- Value type ---
    value_subs = [s.strip() for s in (value_subtype or "").strip().lower().split(",") if s.strip()]
    vt = (value_type or "").strip().lower() or "score"
    # Whitelist the value type so an unknown value isn't echoed back verbatim while silently
    # serving score data; clamp to the safe default instead.
    if vt not in _VALID_VALUE_TYPES:
        vt = "score"
    if vt == "time":
        value_subs = [s for s in value_subs if s != "evaluation"]
    value_sub_str = ",".join(value_subs)

    # --- Metric override ---
    if vt == "score":
        apply_metric_override(rows, metric_map)

    # --- Merge ---
    if merge_inference:
        rows = merge_inference_rows(rows)

    # --- Column filters ---
    benchmarks = strip_numeric_suffix_benchmarks(benchmarks)
    available_cols = list(benchmarks)

    modality_matcher = build_modality_matcher(modality.split(","), benchmark_modality_map)
    if modality_matcher:
        available_cols = [b for b in available_cols if modality_matcher(b)]

    available_cols = filter_benchmarks_by_eval_engine(available_cols, evaluation_engine.split(","), benchmark_engine_map)
    available_cols = filter_benchmarks_by_value_type(available_cols, vt, value_subs)

    hidden = {b.strip() for b in hidden_benchmarks.split(",") if b.strip()}
    visible_cols = [b for b in available_cols if b not in hidden]

    # Drop benchmark columns with no actual value in ANY row of the current view. A column that
    # is "-" for every visible model carries no information; without this a Direct upload that
    # was evaluated on 1 of 238 benchmarks would still render 238 mostly-empty columns.
    nonempty = set()
    for _r in rows:
        for _b, _v in (_r.get("scores") or {}).items():
            if parse_float(_v) is not None:
                nonempty.add(_b)
    available_cols = [b for b in available_cols if b in nonempty]
    visible_cols = [b for b in visible_cols if b in nonempty]

    # --- Weights + averages ---
    weights: dict[str, float] = {}
    if vt == "score" and LEADERBOARD_WEIGHT_FILE.exists():
        try:
            weights = json.loads(LEADERBOARD_WEIGHT_FILE.read_text())
        except Exception:
            pass
    show_sum = compute_averages(rows, visible_cols, weights, vt, value_subs)

    # --- Sort ---
    rows = sort_rows(rows, sort, sort_dir, visible_cols)
    total = len(rows)

    # --- Min/max (needs full scores) ---
    minmax = compute_minmax(rows, visible_cols, show_sum)

    # --- Cleanup ---
    for r in rows:
        r.pop("metrics", None)

    # Ship only the benchmark scores that can actually be displayed for this view. The client
    # reads row.scores[c] solely for the visible `columns` (⊆ available_cols); the raw dict
    # carries every metric variant across all value_types (~8x the visible set), and the client
    # re-fetches when value_type/filters change. Project into shallow copies so the shared scan
    # cache is never mutated.
    _avail = set(available_cols)

    def _project(r):
        sc = r.get("scores")
        if not isinstance(sc, dict):
            return r
        nr = dict(r)
        nr["scores"] = {k: v for k, v in sc.items() if k in _avail}
        return nr

    page_rows = [_project(r) for r in rows[page * page_size : (page + 1) * page_size]]
    # The chart plots only a handful of models (default top 7) and its picker lists this set;
    # cap it so the full result set isn't shipped (and re-fetched every 60s) on a large corpus.
    chart_rows = [_project(r) for r in rows[:CHART_ROWS_MAX]]

    return {
        "columns": visible_cols,
        "all_columns": available_cols,
        "rows": page_rows,
        "chart_rows": chart_rows,
        "total": total,
        "page": page,
        "page_size": page_size,
        "minmax": minmax,
        "show_avg": show_avg,
        "show_sum": show_sum,
        "value_type": vt,
        "value_subtype": value_sub_str,
        "benchmark_engine_map": benchmark_engine_map,
        "benchmark_modality_map": benchmark_modality_map,
        "metric_keys_map": metric_keys_map,
        "scanning": scanning,
    }


@router.get("/weights")
def get_weights():
    """Get leaderboard benchmark weights."""
    if LEADERBOARD_WEIGHT_FILE.exists():
        try:
            return json.loads(LEADERBOARD_WEIGHT_FILE.read_text())
        except Exception:
            pass
    return {}


@router.put("/weights", dependencies=[Depends(require_api_key)])
def put_weights(weights: dict):
    """Save leaderboard benchmark weights."""
    # Validate: every value must be a real number (reject bools, strings, null,
    # nested objects). bool is a subclass of int, so exclude it explicitly.
    clean: dict[str, float] = {}
    for k, v in weights.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise HTTPException(status_code=422, detail=f"non-numeric weight for {k!r}")
        clean[str(k)] = float(v)
    LEADERBOARD_WEIGHT_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_WEIGHT_FILE.write_text(json.dumps(clean, indent=2))
    return {"saved": True}
