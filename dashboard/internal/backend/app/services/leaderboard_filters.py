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

"""Leaderboard filtering, sorting, merging, and aggregation logic."""

import json
import logging
import re
from pathlib import Path

# Data-derived benchmark -> modality map, built by public_tools/build_modality_map.py from the
# ACTUAL media-part types in each benchmark's generation.json (image/video/audio/text). A
# detected non-text media type is authoritative — it fixes the many domain benchmark names the
# keyword heuristic below can't (refcoco/pope/amber/gqa land in image; charades/mlvu/tomato in
# video). Benches absent here, or mapped "text" (no displayable media found, e.g. lmms_eval runs
# that don't store media paths), fall through to the keyword heuristic so nothing regresses.
# Regenerate when the benchmark set changes.
_STATIC_MODALITY_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark_modality.json"
try:
    _loaded = json.loads(_STATIC_MODALITY_PATH.read_text())
    STATIC_BENCH_MODALITY: dict[str, str] = _loaded if isinstance(_loaded, dict) else {}
except Exception:
    STATIC_BENCH_MODALITY = {}

# Guards for client-supplied metric_map override (cheap DoS protection).
_METRIC_MAP_MAX_BYTES = 64 * 1024
_METRIC_MAP_MAX_ENTRIES = 200


def parse_float(v) -> float | None:
    """Parse score to float for sorting/avg. Returns None if not numeric."""
    if v is None or v == "-" or v == "":
        return None
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def is_debug_model(name: str) -> bool:
    if not name:
        return False
    return str(name).lower().startswith("debug__")


def base_model_name(name: str) -> str:
    if not name:
        return ""
    s = str(name)
    if s.lower().startswith("debug__"):
        rest = s[len("debug__"):]
        base = rest.split("__")[0] if "__" in rest else rest
        return "debug__" + base if base else "debug__"
    return s.split("__")[0] if "__" in s else s


def apply_engine_modality_fallback(
    benchmark_engine_map: dict[str, str], benchmark_modality_map: dict[str, str]
) -> dict[str, str]:
    """Fill modality map from engine when missing (e.g., audio_bench -> audio)."""
    if not benchmark_engine_map:
        return benchmark_modality_map
    mod_map = benchmark_modality_map or {}
    for bench, eng in benchmark_engine_map.items():
        if bench in mod_map:
            continue
        norm = (eng or "").strip().lower().replace("-", "_")
        if norm == "audio_bench":
            mod_map[bench] = "audio"
    return mod_map


# Modality classification term lists (lowercase-substring match on bench name).
# Used by classify_benchmark_modality for the Inference Viewer "Select Benchmark"
# grouping. Order of checks: audio -> video -> image -> text (catch-all).
_AUDIO_TERMS = [
    "speech", "asr", "librispeech", "fleurs", "covost", "clotho", "gtzan",
    "meld", "mmau", "cochlscene", "audiobench", "audio_bench", "audio",
    "voice", "vocal", "music", "chomusic", "mmsu", "spoonspeech", "hike",
    "omni_bench", "omnibench", "avqa", "sound",
]
_VIDEO_TERMS = [
    "video", "videomme", "video_mmmu", "mvbench", "lvbench",
    "longvideobench", "perceptiontest", "tempcompass",
]
_IMAGE_TERMS = [
    "vision", "vlm", "mmvet", "mmstar", "mathvista", "mathvision", "mathverse",
    "ocrbench", "ocr", "hallusion", "ai2d", "chartqa", "docvqa", "infovqa",
    "textvqa", "mmmu", "seedbench", "seedbench_img", "realworldqa", "mmbench",
    "llavabench", "llava", "konet", "click", "mme", "image",
]


def classify_benchmark_modality(bench: str, modality_map: dict | None = None) -> str:
    """Classify a benchmark into one of "audio"|"video"|"image"|"text".

    Pure function. Precedence:
      (1) an EXPLICIT 'video'/'audio' modality_map value for ``bench`` (or its
          base name ``bench.split("__")[0]``) wins: these are specific and
          authoritative;
      (2) else a name-keyword heuristic checked IN ORDER: audio-terms ->
          video-terms -> image-terms. A clear NAME-based video signal
          (e.g. 'longvideobench') thus overrides a COARSE 'vision'/'image'
          map value, which cannot distinguish image from video;
      (3) else a 'vision'/'image'/'text' modality_map value: 'vision'/'image'
          -> "image", 'text' -> "text";
      (4) else "text".

    Video is checked BEFORE image so 'video_mmmu' classifies as video, not image.

    (0) A data-derived STATIC_BENCH_MODALITY entry (real media content) outranks everything when
        it names a concrete media type (image/video/audio); a "text"/missing entry falls through.
    """
    s = STATIC_BENCH_MODALITY.get(bench)
    if s is None and "__" in (bench or ""):
        s = STATIC_BENCH_MODALITY.get(bench.split("__")[0])
    if s in ("audio", "video", "image"):
        return s

    norm = ""
    if modality_map:
        meta = modality_map.get(bench)
        if meta is None:
            meta = modality_map.get(bench.split("__")[0] if "__" in bench else bench)
        norm = (meta or "").strip().lower()
        # (1) Explicit, specific map values are authoritative.
        if norm == "audio":
            return "audio"
        if norm == "video":
            return "video"

    # (2) Name heuristic: a clear video name beats a coarse 'vision'/'image' map.
    b = (bench or "").lower()
    if any(t in b for t in _AUDIO_TERMS):
        return "audio"
    if any(t in b for t in _VIDEO_TERMS):
        return "video"
    if any(t in b for t in _IMAGE_TERMS):
        return "image"

    # (3) Fall back to a coarse map value when the name was inconclusive.
    if norm in ("vision", "image"):
        return "image"
    if norm == "text":
        return "text"

    # (4) Default.
    return "text"


def filter_by_model_mode(rows: list[dict], mode: str) -> list[dict]:
    mode = (mode or "normal").strip().lower()
    if mode not in {"normal", "debug", "all"}:
        mode = "normal"
    if mode == "normal":
        return [r for r in rows if not is_debug_model(r.get("model", ""))]
    if mode == "debug":
        return [r for r in rows if is_debug_model(r.get("model", ""))]
    return rows


def strip_numeric_suffix_benchmarks(benchmarks: list[str]) -> list[str]:
    """Remove benchmarks with __N numeric suffixes."""
    return [b for b in benchmarks if not re.search(r"__\d+$", b)]


def build_modality_matcher(
    modalities: list[str],
    benchmark_modality_map: dict[str, str],
):
    """Return a function bench->bool for modality column filtering, or None if no filter."""
    mod_list = [m.strip().lower() for m in modalities if m.strip()]
    if not mod_list or set(mod_list) == {"text", "vision", "audio"}:
        return None

    vision_terms = ["vision", "vlm", "mmvet", "mmstar", "mme", "mathvista", "mathvision", "mvbench", "ocrbench", "ocr", "hallusion", "videomme", "image"]
    audio_terms = ["audio", "asr"]

    def bench_matches_one(bench: str, mod: str) -> bool:
        meta_mod = (
            (benchmark_modality_map.get(bench) or benchmark_modality_map.get(bench.split("__")[0] if "__" in bench else bench) or "")
            .lower()
            if benchmark_modality_map
            else ""
        )
        if meta_mod:
            if mod == "vision":
                return meta_mod in ("vision", "image", "video")
            if mod == "audio":
                return meta_mod == "audio"
            if mod == "text":
                return meta_mod == "text"
        b = bench.lower()
        if mod == "vision":
            return any(x in b for x in vision_terms)
        if mod == "audio":
            return any(x in b for x in audio_terms)
        if mod == "text":
            return not (any(x in b for x in vision_terms) or any(x in b for x in audio_terms))
        return False

    def bench_matches_any(bench: str) -> bool:
        return any(bench_matches_one(bench, mod) for mod in mod_list)

    return bench_matches_any


def filter_by_inference_engine(rows: list[dict], engines: list[str]) -> list[dict]:
    inf_list = [x.strip().lower() for x in engines if x.strip()]
    ALL_OPTS = {"api", "huggingface", "vllm", "sglang"}
    if not inf_list or set(inf_list) == ALL_OPTS:
        return rows

    def matches(r: dict) -> bool:
        inf_str = (r.get("inference") or "-").strip()
        if not inf_str or inf_str == "-":
            return False
        parts = [p.strip().lower() for p in inf_str.split("/") if p.strip()]
        return any(any(f in p or p == f for f in inf_list) for p in parts)

    return [r for r in rows if matches(r)]


def filter_benchmarks_by_eval_engine(
    benchmarks: list[str], engines: list[str], benchmark_engine_map: dict[str, str]
) -> list[str]:
    ev_list = [x.strip().lower().replace("-", "_") for x in engines if x.strip()]
    ALL_OPTS = {"built_in", "lmms_eval", "vlm_eval_kit", "audio_bench"}
    if not ev_list or set(ev_list) == ALL_OPTS or not benchmark_engine_map:
        return benchmarks

    def matches(bench: str) -> bool:
        eng = (benchmark_engine_map.get(bench) or "-").lower().replace("-", "_")
        if eng == "-":
            return False
        return any(eng == ev or ev in eng for ev in ev_list)

    return [b for b in benchmarks if matches(b)]


def filter_benchmarks_by_value_type(
    benchmarks: list[str], value_type: str, subtypes: list[str]
) -> list[str]:
    vt = (value_type or "").strip().lower() or "score"
    if vt == "time":
        time_cols = [
            b for b in benchmarks
            if (b.endswith("__runtime_inference") or b.endswith("__runtime_total")
                or b.endswith("__latency") or b.endswith("__throughput")
                or b.endswith("__benchmark_time"))
        ]
        if not subtypes:
            return time_cols

        def _time_match(col: str) -> bool:
            return (
                ("inference" in subtypes and col.endswith("__runtime_inference"))
                or ("total" in subtypes and col.endswith("__runtime_total"))
                or ("latency" in subtypes and col.endswith("__latency"))
                or ("throughput" in subtypes and col.endswith("__throughput"))
                or ("benchmarks" in subtypes and col.endswith("__benchmark_time"))
            )
        return [b for b in time_cols if _time_match(b)]
    elif vt == "coverage":
        cov_cols = [b for b in benchmarks if b.endswith("__coverage_inference") or b.endswith("__coverage_evaluation")]
        if not subtypes:
            return cov_cols

        def _cov_match(col: str) -> bool:
            return (
                ("inference" in subtypes and col.endswith("__coverage_inference"))
                or ("evaluation" in subtypes and col.endswith("__coverage_evaluation"))
            )
        return [b for b in cov_cols if _cov_match(b)]
    else:
        return [b for b in benchmarks if not (
            b.endswith("__runtime_inference") or b.endswith("__runtime_total")
            or b.endswith("__coverage_inference") or b.endswith("__coverage_evaluation")
            or b.endswith("__latency") or b.endswith("__throughput") or b.endswith("__benchmark_time")
        )]


def apply_metric_override(rows: list[dict], metric_map_json: str) -> None:
    """Apply metric key overrides to row scores in-place."""
    if not metric_map_json:
        return
    if len(metric_map_json) > _METRIC_MAP_MAX_BYTES:
        logging.warning("Ignoring metric_map override: too large (%d bytes)", len(metric_map_json))
        return
    try:
        parsed = json.loads(metric_map_json)
        if not isinstance(parsed, dict):
            return
        if len(parsed) > _METRIC_MAP_MAX_ENTRIES:
            logging.warning("Ignoring metric_map override: too many entries (%d)", len(parsed))
            return
        metric_override = {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        return
    for r in rows:
        metrics = r.get("metrics") or {}
        for bench, key in metric_override.items():
            bench_metrics = metrics.get(bench) or {}
            val = bench_metrics.get(key) if isinstance(bench_metrics, dict) else None
            if "scores" in r:
                r["scores"][bench] = val if val is not None and val != "" else "-"


def collect_metric_keys(rows: list[dict]) -> dict[str, list[str]]:
    """Collect available metric keys per benchmark."""
    metric_keys_map: dict[str, list[str]] = {}
    for r in rows:
        metrics = r.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        for bench, m in metrics.items():
            if not isinstance(m, dict):
                continue
            keys = metric_keys_map.setdefault(bench, [])
            for k in m.keys():
                if k not in keys:
                    keys.append(k)
    return metric_keys_map


def merge_inference_rows(rows: list[dict]) -> list[dict]:
    """Merge rows with same (model, checkpoint, source) keeping best scores."""
    if not rows:
        return rows
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        model_key = base_model_name(r.get("model") or "")
        ckpt = r.get("checkpoint") or "checkpoint-none"
        key = (model_key, ckpt, r.get("source") or "")
        groups.setdefault(key, []).append(r)

    merged = []
    for key, group in groups.items():
        model, checkpoint, source = key
        inferences = sorted(set(g.get("inference") or "-" for g in group if g.get("inference")))
        inf_str = " / ".join(inferences) if inferences else "-"
        scores: dict[str, str] = {}
        all_benchmarks = set()
        for g in group:
            all_benchmarks.update(g.get("scores") or {})
        for b in all_benchmarks:
            best_val = None
            best_str = "-"
            lower_is_better = (
                b.endswith("__runtime_inference")
                or b.endswith("__runtime_total") or b.endswith("__latency")
                or b.endswith("__benchmark_time")
            )
            for g in group:
                v = (g.get("scores") or {}).get(b, "-")
                nv = parse_float(v)
                if nv is not None:
                    if best_val is None or (nv < best_val if lower_is_better else nv > best_val):
                        best_val = nv
                        best_str = v
            scores[b] = best_str
        merged.append({
            "model": model,
            "inference": inf_str,
            "checkpoint": checkpoint,
            "source": source,
            "scores": scores,
        })
    return merged


def compute_averages(
    rows: list[dict], visible_cols: list[str],
    weights: dict[str, float], value_type: str, value_subs: list[str],
) -> bool:
    """Compute _avg and _sum for each row in-place. Returns show_sum flag."""
    show_sum = value_type == "time" and not (set(value_subs) & {"latency", "throughput"})
    for r in rows:
        vals = []
        wsum = 0.0
        for b in visible_cols:
            v = parse_float(r.get("scores", {}).get(b))
            if v is not None:
                w = weights.get(b, 1.0) if value_type == "score" else 1.0
                vals.append((v, w))
                wsum += w
        r["_avg"] = (sum(v * w for v, w in vals) / wsum) if vals and wsum > 0 else None
        if show_sum:
            time_vals = [parse_float(r.get("scores", {}).get(b)) for b in visible_cols]
            time_vals = [v for v in time_vals if v is not None]
            r["_sum"] = sum(time_vals) if time_vals else None
        else:
            r["_sum"] = None
    return show_sum


def _checkpoint_sort_num(ckpt: str) -> int:
    """Extract numeric part from 'checkpoint-1234' for sorting. Returns -1 for 'checkpoint-none'."""
    import re
    m = re.search(r"(\d+)", ckpt or "")
    return int(m.group(1)) if m else -1


def sort_rows(rows: list[dict], sort_col: str, sort_dir: str, visible_cols: list[str]) -> list[dict]:
    if not sort_col:
        # Default: model asc → inference asc → checkpoint numeric asc
        return sorted(rows, key=lambda r: (
            (r.get("model") or "").lower(),
            (r.get("inference") or "").lower(),
            _checkpoint_sort_num(r.get("checkpoint") or ""),
        ))

    def sort_key(r):
        if sort_col == "model":
            return (r.get("model") or "").lower()
        if sort_col == "inference":
            return (r.get("inference") or "").lower()
        if sort_col == "checkpoint":
            return _checkpoint_sort_num(r.get("checkpoint") or "")
        if sort_col == "avg":
            v = r.get("_avg")
            return (v is not None, v if v is not None else -1e9)
        if sort_col == "_sum":
            v = r.get("_sum")
            return (v is not None, v if v is not None else -1e9)
        if sort_col in visible_cols:
            v = parse_float(r.get("scores", {}).get(sort_col))
            return (v is not None, v if v is not None else -1e9)
        return ()

    return sorted(rows, key=sort_key, reverse=sort_dir.lower() == "desc")


def compute_minmax(rows: list[dict], visible_cols: list[str], show_sum: bool) -> dict:
    def col_min_max(col: str):
        nums = []
        for r in rows:
            if col == "_avg":
                v = r.get("_avg")
            elif col == "_sum":
                v = r.get("_sum")
            else:
                v = parse_float(r.get("scores", {}).get(col))
            if v is not None:
                nums.append(v)
        if not nums:
            return None
        mn, mx = min(nums), max(nums)
        if mn == mx or abs(mn - mx) < 1e-9:
            return None
        return mn, mx

    minmax: dict = {}
    for b in visible_cols:
        mm = col_min_max(b)
        if mm is not None:
            minmax[b] = mm
    mm_avg = col_min_max("_avg")
    if mm_avg is not None:
        minmax["_avg"] = mm_avg
    if show_sum:
        mm_sum = col_min_max("_sum")
        if mm_sum is not None:
            minmax["_sum"] = mm_sum
    return minmax
