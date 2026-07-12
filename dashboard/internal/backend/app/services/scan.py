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

"""Scanning utilities for evaluation JSON files and zip archives."""

import logging
import os
import time
import zipfile
from pathlib import Path
from threading import Lock

from ..config import INFERENCE_SAMPLE_MAX_BYTES
from .json_io import _loads as _load_json_bytes
from .metric_extraction import (
    _extract_meta_and_metrics_from_full,
    extract_benchmark_from_filename,
    extract_metrics_map,
)
from .inference_builder import build_inference_output_from_eval


def _read_member_if_small(zf: zipfile.ZipFile, name: str):
    """Read a zip member only if its uncompressed size is under the per-sample cap, else
    None. The pre-warmed inference payload is transient (never cached) and re-read on demand
    by the streaming viewer, so reading a huge (media-laden) member here is pure scan-peak
    RAM with no benefit — skip it and let the eval-derived fallback cover the viewer."""
    try:
        if zf.getinfo(name).file_size > INFERENCE_SAMPLE_MAX_BYTES:
            return None
    except Exception:
        return None
    try:
        return zf.read(name)
    except Exception:
        return None


def parse_model_id(model_id: str) -> tuple[str, str]:
    """Parse 'source:model' into (source, model)."""
    parts = model_id.split(":", 2)
    if len(parts) > 1:
        return parts[0], parts[1]
    return "internal", model_id


EVAL_ENGINE_KNOWN = frozenset({"lmms_eval", "vlm_eval_kit", "audio_bench", "lm_eval_harness"})


def _normalize_engine_folder(name: str) -> str:
    raw = (name or "").strip().lower().replace("-", "_")
    if raw == "audeio_bench":
        raw = "audio_bench"
    if raw in EVAL_ENGINE_KNOWN:
        return raw
    return "built-in"


def _fallback_modality_from_engine(engine: str) -> str | None:
    eng = (engine or "").strip().lower().replace("-", "_")
    if eng == "audio_bench":
        return "audio"
    return None


def normalize_eval_engine_from_path(path: Path | str) -> str:
    """Extract evaluation engine from path: .../checkpoint/ENGINE/evaluation_output/..."""
    try:
        p = Path(path) if isinstance(path, str) else path
        parent = p.parent
        if parent.name in ("evaluation_output", "output"):
            engine_folder = parent.parent.name
        else:
            engine_folder = parent.name
        return _normalize_engine_folder(engine_folder)
    except Exception:
        return "built-in"


def normalize_eval_engine_from_zip_path(name: str) -> str:
    """Extract evaluation engine from zip member path."""
    try:
        parts = name.replace("\\", "/").split("/")
        for i, part in enumerate(parts):
            if part in ("evaluation_output", "output") and i > 0:
                engine_folder = parts[i - 1]
                return _normalize_engine_folder(engine_folder)
        return "built-in"
    except Exception:
        return "built-in"


def _normalize_checkpoint(raw: str) -> str:
    if not raw or raw == "-":
        return "checkpoint-none"
    return raw.strip()


def _checkpoint_from_path(path: Path) -> str:
    """Extract checkpoint from path: .../MODEL/CHECKPOINT/ENGINE/evaluation_output/file.json."""
    try:
        p = path.parent
        for _ in range(2):
            p = p.parent
        raw = p.name if p and p.name else ""
        return _normalize_checkpoint(raw)
    except Exception:
        return "checkpoint-none"


def _checkpoint_from_zip_path(name: str) -> str:
    """Extract checkpoint from zip member: MODEL/CHECKPOINT/ENGINE/evaluation_output/file.json."""
    try:
        parts = name.replace("\\", "/").split("/")
        for i, part in enumerate(parts):
            if part in ("evaluation_output", "output") and i >= 2:
                return _normalize_checkpoint(parts[i - 2])
        return "checkpoint-none"
    except Exception:
        return "checkpoint-none"


def _base_benchmark(bench: str) -> str:
    """Extract base benchmark name (before __suffix) for inference file lookup."""
    return bench.split("__")[0] if "__" in bench else bench


# Inference output index with TTL cache
_INF_INDEX_CACHE: dict[str, tuple[float, dict[str, Path]]] = {}
_INF_INDEX_LOCK = Lock()
_INF_INDEX_TTL = int(os.environ.get("OMNI_INF_INDEX_TTL", "30"))
# Keys are scan base paths (bounded by the FS layout, not user input), but cap anyway so a
# long-lived process can't accumulate stale entries without bound. Entries rebuild cheaply.
_INF_INDEX_MAX_ENTRIES = 256


def _build_inference_index(base: Path) -> dict[str, Path]:
    # Recursive walk: inference_output/ dirs can sit at any depth (e.g. the 2-level
    # <ckpt>/<engine>/inference_output layout), so rglob matches them regardless of position.
    # Key by _base_benchmark() so it matches find_inference_output()'s lookup. The rglob cost is
    # mitigated by the 30s TTL cache in _get_inference_index(); if ever pointed at very large
    # trees it should be scoped (narrower base) and profiled.
    index: dict[str, Path] = {}
    try:
        for p in base.rglob("inference_output/*.json"):
            bench = _base_benchmark(extract_benchmark_from_filename(p.name))
            if bench and bench not in index:
                index[bench] = p
    except OSError:
        pass
    return index


def _get_inference_index(base: Path) -> dict[str, Path]:
    key = str(base)
    now = time.time()
    with _INF_INDEX_LOCK:
        cached = _INF_INDEX_CACHE.get(key)
        if cached and _INF_INDEX_TTL > 0 and (now - cached[0]) < _INF_INDEX_TTL:
            return cached[1]
    idx = _build_inference_index(base)
    with _INF_INDEX_LOCK:
        if key not in _INF_INDEX_CACHE and len(_INF_INDEX_CACHE) >= _INF_INDEX_MAX_ENTRIES:
            _INF_INDEX_CACHE.clear()  # bounded; TTL means entries are short-lived anyway
        _INF_INDEX_CACHE[key] = (now, idx)
    return idx


def find_inference_output(base: Path, benchmark: str) -> Path | None:
    """Find inference_output JSON for benchmark under base dir."""
    base_bench = _base_benchmark(benchmark)
    index = _get_inference_index(base)
    return index.get(base_bench)


def scan_eval_zip(zf: zipfile.ZipFile, model_name: str, source: str, namelist: list[str] | None = None) -> tuple[set[str], dict, dict, dict]:
    """Scan evaluation_output in zip. Returns (benchmarks_set, rows_map, benchmark_engine_map, benchmark_modality_map)."""
    benchmarks: set[str] = set()
    rows_map: dict[tuple[str, str, str], dict] = {}
    benchmark_engine_map: dict[str, str] = {}
    benchmark_modality_map: dict[str, str] = {}
    names = namelist if namelist is not None else zf.namelist()
    names_set = set(names)

    for name in names:
        if ("/evaluation_output/" not in name and "/output/" not in name) or not name.endswith(".json"):
            continue
        if "/inference_output/" in name:
            continue
        ev_eng = normalize_eval_engine_from_zip_path(name)
        # Cap the eval-member read like inference members do: a pathological multi-hundred-MB
        # score file would otherwise spike scan-peak RAM on the small box. Over the cap → skip.
        raw = _read_member_if_small(zf, name)
        if raw is None:
            logging.warning("Skipping oversized/unreadable eval member: %s", name)
            continue
        bench_hint = extract_benchmark_from_filename(name)
        data = _load_json_bytes(raw)
        if not data:
            continue
        results = _extract_meta_and_metrics_from_full(data, model_name, bench_hint)
        if not results:
            continue
        metrics_map = extract_metrics_map(data)
        inf_from_eval = build_inference_output_from_eval(data)
        ckpt = _checkpoint_from_zip_path(name)
        base_bench = _base_benchmark(bench_hint)
        inf_name = name.replace("evaluation_output/", "inference_output/", 1)
        if inf_name == name:
            inf_name = name.replace("output/", "inference_output/", 1)
        inf_data = None
        if inf_name != name and inf_name in names_set:
            raw_inf = _read_member_if_small(zf, inf_name)
            if raw_inf is not None:
                inf_data = _load_json_bytes(raw_inf)
        if inf_data is None:
            for n in names_set:
                if "inference_output/" in n and base_bench in Path(n).stem and n.endswith(".json"):
                    raw_inf = _read_member_if_small(zf, n)
                    if raw_inf is not None:
                        inf_data = _load_json_bytes(raw_inf)
                        break
        for r in results:
            effective_model = model_name
            bench = r.get("benchmark") or bench_hint
            base_bench = _base_benchmark(bench)
            key = (effective_model, r["inference"], ckpt)
            if key not in rows_map:
                rows_map[key] = {
                    "model": effective_model,
                    "inference": r["inference"],
                    "checkpoint": ckpt,
                    "source": source,
                    "_inference": {},
                    "_metrics": {},
                }
            rows_map[key][base_bench] = r["metric_value"]
            if inf_data and isinstance(inf_data, dict):
                rows_map[key]["_inference"][base_bench] = inf_data
            elif inf_from_eval and base_bench not in rows_map[key]["_inference"]:
                rows_map[key]["_inference"][base_bench] = inf_from_eval
            if metrics_map:
                rows_map[key]["_metrics"][base_bench] = metrics_map
            benchmarks.add(base_bench)
            if base_bench not in benchmark_engine_map:
                benchmark_engine_map[base_bench] = ev_eng
            if base_bench not in benchmark_modality_map:
                mod = _fallback_modality_from_engine(ev_eng)
                if mod:
                    benchmark_modality_map[base_bench] = mod

    return benchmarks, rows_map, benchmark_engine_map, benchmark_modality_map
