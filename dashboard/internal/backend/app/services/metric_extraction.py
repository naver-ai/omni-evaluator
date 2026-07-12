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

"""Metric extraction from evaluation/inference JSON outputs."""

from pathlib import Path


# Top-level keys that are NOT benchmark metrics
_NON_METRIC_KEYS = frozenset({
    "meta", "output", "config", "inference", "results",
    "num_runs", "run_index", "num_samples", "num_empty_predictions",
    "latency", "throughput", "coverage_inference", "coverage_evaluation",
    "runtime_inference", "runtime_evaluation", "run_outputs",
    "evaluation_engine", "task_name", "evaluation_method", "metric_keys",
    "group_metrics", "sample_metrics", "outputs",
})


def extract_benchmark_from_filename(filename: str) -> str:
    """Extract benchmark name from filename like 'bench__method.json'."""
    stem = Path(filename).stem
    return stem.split("__")[0] if "__" in stem else stem


def _resolve_latency_and_samples(ev: dict) -> tuple:
    """Resolve (latency, num_samples), falling back to run_outputs[0] when absent at top level."""
    run_outputs = ev.get("run_outputs") or []
    r0 = run_outputs[0] if run_outputs and isinstance(run_outputs[0], dict) else None
    lat = ev.get("latency") if ev.get("latency") is not None else (r0.get("latency") if r0 else None)
    n = ev.get("num_samples") if ev.get("num_samples") is not None else (r0.get("num_samples") if r0 else None)
    return lat, n


def _extract_runtime_and_coverage(data: dict) -> dict:
    """Extract runtime_inference, runtime_evaluation, runtime_total, coverage_inference, coverage_evaluation."""
    ev = data.get("evaluation") or {}
    if not isinstance(ev, dict):
        ev = {}
    rt_inf = ev.get("runtime_inference") if ev.get("runtime_inference") is not None else data.get("runtime_inference")
    rt_eval = ev.get("runtime_evaluation") if ev.get("runtime_evaluation") is not None else data.get("runtime_evaluation")
    cov_inf = ev.get("coverage_inference") if ev.get("coverage_inference") is not None else data.get("coverage_inference")
    cov_eval = ev.get("coverage_evaluation") if ev.get("coverage_evaluation") is not None else data.get("coverage_evaluation")
    run_outputs = ev.get("run_outputs") or []
    r0 = run_outputs[0] if run_outputs and isinstance(run_outputs[0], dict) else None
    if r0:
        if rt_inf is None and isinstance(r0.get("runtime_inference"), (int, float)):
            rt_inf = r0["runtime_inference"]
        if rt_eval is None and isinstance(r0.get("runtime_evaluation"), (int, float)):
            rt_eval = r0["runtime_evaluation"]
    if rt_inf is None:
        lat, n = _resolve_latency_and_samples(ev)
        if isinstance(lat, (int, float)) and isinstance(n, (int, float)) and n > 0:
            rt_inf = lat * n
    rt_total = None
    if isinstance(rt_inf, (int, float)) and isinstance(rt_eval, (int, float)):
        rt_total = rt_inf + rt_eval
    elif isinstance(rt_inf, (int, float)):
        rt_total = rt_inf
    elif isinstance(rt_eval, (int, float)):
        rt_total = rt_eval

    def _clamp_coverage(v):
        if not isinstance(v, (int, float)):
            return v
        return max(0.0, min(1.0, float(v)))

    cov_inf = _clamp_coverage(cov_inf) if cov_inf is not None else cov_inf
    cov_eval = _clamp_coverage(cov_eval) if cov_eval is not None else cov_eval
    return {
        "runtime_inference": rt_inf,
        "runtime_evaluation": rt_eval,
        "runtime_total": rt_total,
        "coverage_inference": cov_inf,
        "coverage_evaluation": cov_eval,
    }


def extract_metrics_all(data: dict) -> list[tuple[str, str]]:
    """Extract ALL numeric metrics from evaluation output. Returns [(metric_key, value_str), ...]."""
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        metrics = None
    ev = data.get("evaluation")
    if isinstance(ev, dict):
        ev_metrics = ev.get("metrics")
        if isinstance(ev_metrics, dict) and metrics is None:
            metrics = ev_metrics
    if not metrics:
        return []
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    metric_keys = data.get("metric_keys") or (ev.get("metric_keys") if isinstance(ev, dict) else None)
    if isinstance(metric_keys, (list, tuple)):
        for k in metric_keys:
            if k in metrics and k not in seen:
                v = metrics[k]
                if isinstance(v, (int, float)):
                    result.append((k, f"{v:.2f}" if isinstance(v, float) else str(v)))
                    seen.add(k)
    for k, v in metrics.items():
        if k not in seen and isinstance(v, (int, float)):
            result.append((k, f"{v:.2f}" if isinstance(v, float) else str(v)))
            seen.add(k)
    return result


def extract_numeric_metric(data: dict) -> tuple[str, str] | None:
    """Extract primary numeric metric. Returns (metric_key, value_str) or None."""
    all_metrics = extract_metrics_all(data)
    if all_metrics:
        return all_metrics[0]
    out = data.get("output") or data.get("results")
    if isinstance(out, dict):
        for k, v in out.items():
            if isinstance(v, (int, float)):
                return (k, f"{v:.2f}" if isinstance(v, float) else str(v))
    for k, v in data.items():
        if k in _NON_METRIC_KEYS or not isinstance(v, (int, float)):
            continue
        return (k, f"{v:.2f}" if isinstance(v, float) else str(v))
    return None


def extract_metrics_map(data: dict) -> dict[str, str]:
    """Extract metrics as key-value mapping (includes runtime/coverage/latency)."""
    metrics: dict[str, str] = {k: v for k, v in extract_metrics_all(data)}
    rc = _extract_runtime_and_coverage(data)
    if isinstance(rc.get("runtime_inference"), (int, float)):
        metrics["runtime_inference"] = f"{rc['runtime_inference']:.2f}"
    if isinstance(rc.get("runtime_evaluation"), (int, float)):
        metrics["runtime_evaluation"] = f"{rc['runtime_evaluation']:.2f}"
    if isinstance(rc.get("runtime_total"), (int, float)):
        metrics["runtime_total"] = f"{rc['runtime_total']:.2f}"
    ev_data = data.get("evaluation") if isinstance(data.get("evaluation"), dict) else {}
    lat, n_samples = _resolve_latency_and_samples(ev_data)
    thr = ev_data.get("throughput")
    if isinstance(lat, (int, float)):
        metrics["latency"] = f"{lat:.4f}"
    if isinstance(thr, (int, float)):
        metrics["throughput"] = f"{thr:.4f}"
    if isinstance(lat, (int, float)) and isinstance(n_samples, (int, float)) and n_samples > 0:
        bench_time = lat * n_samples
        metrics["benchmark_time"] = f"{bench_time:.2f}"
    if isinstance(rc.get("coverage_inference"), (int, float)):
        metrics["coverage_inference"] = f"{rc['coverage_inference']:.4f}"
    if isinstance(rc.get("coverage_evaluation"), (int, float)):
        metrics["coverage_evaluation"] = f"{rc['coverage_evaluation']:.4f}"
    return metrics


def _extract_meta_and_metrics_from_full(data: dict, model_hint: str, benchmark_hint: str = "") -> list[dict]:
    """Extract model, inference, checkpoint, primary metric.
    Returns list of {model, inference, checkpoint, benchmark, metric_value, evaluation_engine}."""
    meta = data.get("meta") or (data.get("config") or {}).get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    ev_data = data.get("evaluation") or {}
    if not isinstance(ev_data, dict):
        ev_data = {}
    cfg = data.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    model = meta.get("model_name") or meta.get("model") or model_hint
    inf_eng = (
        meta.get("inference_engine")
        or meta.get("inference_engine_name")
        or ev_data.get("inference_engine")
        or cfg.get("inference_engine")
        or (cfg.get("arguments") or {}).get("inference_engine")
        or data.get("inference_engine")
        or "-"
    )
    ckpt = (
        data.get("checkpoint")
        or ev_data.get("checkpoint")
        or meta.get("checkpoint")
        or meta.get("checkpoint_name")
        or cfg.get("checkpoint")
        or "-"
    )
    if not ckpt or ckpt == "-":
        ckpt = "checkpoint-none"
    bench = data.get("task_name") or ev_data.get("task_name") or benchmark_hint
    ev_engine = (
        data.get("evaluation_engine")
        or ev_data.get("evaluation_engine")
        or cfg.get("evaluation_engine")
        or "-"
    )
    all_metrics = extract_metrics_all(data)
    if not all_metrics:
        primary = extract_numeric_metric(data)
        if not primary:
            return []
        all_metrics = [primary]
    _, value = all_metrics[0]
    return [{
        "model": model,
        "inference": inf_eng,
        "checkpoint": ckpt,
        "metric_value": value,
        "benchmark": bench,
        "evaluation_engine": ev_engine,
    }]
