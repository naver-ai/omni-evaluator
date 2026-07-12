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

"""S3 sync: list models and scan evaluation_output from S3."""

import json
import logging
import os
import time
from threading import Lock
from pathlib import Path

from ..config import (
    HEAD_BYTES,
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT,
    S3_MAX_MODELS,
    S3_PREFIX,
    S3_REGION,
    S3_SECRET_KEY,
    TAIL_BYTES,
)
from .json_io import parse_head_tail_bytes
from .metric_extraction import (
    extract_benchmark_from_filename,
    extract_metrics_map,
    extract_numeric_metric,
)
from .scan import normalize_eval_engine_from_zip_path

_S3_LIST_CACHE: dict[str, tuple[float, list[str]]] = {}
_S3_LIST_LOCK = Lock()

_S3_CLIENT = None
_S3_CLIENT_LOCK = Lock()


def _s3_list_cache_key() -> str:
    return f"{S3_BUCKET}|{S3_PREFIX}|{S3_MAX_MODELS}"


def _s3_list_ttl() -> int:
    try:
        return max(0, int(os.environ.get("OMNI_S3_LIST_TTL", "30")))
    except Exception:
        return 30


def _s3_list_with_retry(s3, params: dict, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            return s3.list_objects_v2(**params)
        except Exception as e:
            last = e
            time.sleep(0.5 * (2 ** attempt))
    logging.warning("S3 list_objects_v2 failed: %s", last)
    return {}

def _get_s3_client():
    """Return a cached boto3 S3 client (credentials from env).

    Built once and reused: boto3 clients are thread-safe for concurrent use, and
    this is called from scan threads and request handlers, so creating a fresh
    Session/connection-pool/SSL context per call was pure overhead. Thread-safe
    lazy init via double-checked locking. Returns None when boto3 is missing or
    credentials are absent (unchanged behavior)."""
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    try:
        import boto3
        from botocore.config import Config as BotocoreConfig
    except ImportError:
        return None
    if not S3_ACCESS_KEY or not S3_SECRET_KEY:
        return None
    with _S3_CLIENT_LOCK:
        if _S3_CLIENT is not None:
            return _S3_CLIENT
        session_kwargs = {"region_name": S3_REGION} if S3_REGION else {}
        if S3_ENDPOINT:
            session_kwargs["endpoint_url"] = S3_ENDPOINT
        session_kwargs["aws_access_key_id"] = S3_ACCESS_KEY
        session_kwargs["aws_secret_access_key"] = S3_SECRET_KEY
        # Bounded timeouts + no retry storm. If the host cannot reach the S3 endpoint (e.g. the
        # object-storage API is not routable from this network), every S3 call must FAIL FAST
        # and return empty rather than hang — otherwise one unreachable scan freezes the whole
        # Submission/Inference Viewer (which await all sources together). Presigning is local
        # crypto (no network), so these timeouts never affect media URL signing.
        config_kwargs = {
            "connect_timeout": 3,
            "read_timeout": 5,
            "retries": {"max_attempts": 1},
        }
        if S3_ENDPOINT:
            config_kwargs["signature_version"] = "s3v4"
            config_kwargs["s3"] = {"addressing_style": "path"}
        session_kwargs["config"] = BotocoreConfig(**config_kwargs)
        _S3_CLIENT = boto3.client("s3", **session_kwargs)
    return _S3_CLIENT


def _checkpoint_sort_key(name: str) -> tuple[int, object]:
    """Sort key that orders 'checkpoint-<step>' names by numeric step.

    Lexical sort misorders steps ('checkpoint-1000' < 'checkpoint-500'); pull the
    trailing integer out so 500 < 1000. Non-numeric names fall back to lexical
    order and sort after numeric ones."""
    suffix = name.rsplit("-", 1)[-1]
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, name)


def _is_eval_output_key(key: str) -> bool:
    if not key or not key.endswith(".json"):
        return False
    if "/inference_output/" in key:
        return False
    return "/evaluation_output/" in key or "/output/" in key


def _list_s3_models_sync() -> list[str]:
    """List S3 model names only (fast, no evaluation scan)."""
    s3 = _get_s3_client()
    if not s3 or not S3_BUCKET:
        return []
    ttl = _s3_list_ttl()
    cache_key = _s3_list_cache_key()
    if ttl > 0:
        with _S3_LIST_LOCK:
            cached = _S3_LIST_CACHE.get(cache_key)
        if cached and (time.time() - cached[0]) < ttl:
            return cached[1]
    prefix = (S3_PREFIX or "").rstrip("/")
    if prefix:
        prefix = prefix + "/"
    model_names = []
    seen = set()
    token = None
    truncated = False
    for _ in range(20):
        params = {"Bucket": S3_BUCKET, "Prefix": prefix, "Delimiter": "/", "MaxKeys": 1000}
        if token:
            params["ContinuationToken"] = token
        resp = _s3_list_with_retry(s3, params)
        for common in resp.get("CommonPrefixes", []):
            cp = common.get("Prefix", "")
            rel = cp[len(prefix) :] if prefix and cp.startswith(prefix) else cp
            top = rel.strip("/").split("/", 1)[0].strip()
            if not top or top in seen:
                continue
            seen.add(top)
            model_names.append(top)
            if len(model_names) >= S3_MAX_MODELS:
                return model_names
        if not model_names:
            for obj in resp.get("Contents", []):
                key = (obj.get("Key") or "").strip()
                if not key:
                    continue
                rel = key[len(prefix) :] if prefix and key.startswith(prefix) else key
                top = rel.strip("/").split("/", 1)[0].strip()
                if not top or top in seen:
                    continue
                seen.add(top)
                model_names.append(top)
                if len(model_names) >= S3_MAX_MODELS:
                    return model_names
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break
    else:
        # for-loop exhausted its page cap while S3 still had more to list
        truncated = True
    if truncated:
        logging.warning(
            "S3 model listing hit page cap for prefix %r; listing truncated at %s models",
            prefix,
            len(model_names),
        )
    return model_names


def _fetch_eval_head_tail(s3, key: str, size: int) -> dict | None:
    """Read just enough of an eval JSON to extract metrics: full body for small files,
    head+tail byte ranges for large ones (avoids downloading hundreds of MB)."""
    try:
        if size and size <= HEAD_BYTES + TAIL_BYTES:
            body = s3.get_object(Bucket=S3_BUCKET, Key=key)
            return json.loads(body["Body"].read())
        head = s3.get_object(Bucket=S3_BUCKET, Key=key, Range=f"bytes=0-{HEAD_BYTES - 1}")["Body"].read()
        tail = s3.get_object(Bucket=S3_BUCKET, Key=key, Range=f"bytes=-{TAIL_BYTES}")["Body"].read()
        return parse_head_tail_bytes(head, tail)
    except Exception:
        return None


def _scan_one_s3_model_sync(model_name: str) -> tuple[list[dict], dict[str, str]]:
    """Scan one S3 model for leaderboard rows (scores + metrics only).

    Memory-bounded: large eval JSONs are read via head+tail byte ranges, and inference
    output is NOT fetched here (the Inference Viewer loads it on demand). The leaderboard
    only needs scores/metrics, so downloading full inference bodies during scan was pure waste.
    """
    s3 = _get_s3_client()
    if not s3 or not S3_BUCKET:
        return [], {}
    prefix = (S3_PREFIX or "").rstrip("/")
    if prefix:
        prefix = prefix + "/"
    benchmarks: set[str] = set()
    rows_map: dict[tuple[str, str, str], dict] = {}
    benchmark_engine_map: dict[str, str] = {}
    err_count = 0

    model_prefix = f"{prefix}{model_name}/"
    checkpoints: list[str] = []
    cp_token = None
    for _ in range(10):
        cp_params = {"Bucket": S3_BUCKET, "Prefix": model_prefix, "Delimiter": "/", "MaxKeys": 500}
        if cp_token:
            cp_params["ContinuationToken"] = cp_token
        cp_resp = _s3_list_with_retry(s3, cp_params)
        for common in cp_resp.get("CommonPrefixes", []):
            cp_prefix = common.get("Prefix", "")
            rel = cp_prefix[len(model_prefix) :] if cp_prefix.startswith(model_prefix) else cp_prefix
            cp_name = rel.strip("/").split("/", 1)[0].strip()
            if cp_name:
                checkpoints.append(cp_name)
        if not checkpoints:
            for obj in cp_resp.get("Contents", []):
                if _is_eval_output_key((obj.get("Key") or "").strip()):
                    checkpoints.append("checkpoint-none")
                    break
        if not cp_resp.get("IsTruncated"):
            break
        cp_token = cp_resp.get("NextContinuationToken")
        if not cp_token:
            break

    if not checkpoints:
        checkpoints = ["checkpoint-none"]

    ordered_ckpts = sorted(set(checkpoints), key=_checkpoint_sort_key)
    if len(ordered_ckpts) > 20:
        logging.warning(
            "S3 checkpoint cap hit for %s: scanning 20 of %s checkpoints, %s dropped",
            model_name,
            len(ordered_ckpts),
            len(ordered_ckpts) - 20,
        )
    for ckpt in ordered_ckpts[:20]:
        base_prefix = f"{model_prefix}{ckpt}/"
        list_params = {"Bucket": S3_BUCKET, "Prefix": base_prefix, "MaxKeys": 1000}
        list_token = None
        for _ in range(15):
            if list_token:
                list_params["ContinuationToken"] = list_token
            list_resp = _s3_list_with_retry(s3, list_params)
            for obj in list_resp.get("Contents", []):
                key = (obj.get("Key") or "").strip()
                if not _is_eval_output_key(key):
                    continue
                data = _fetch_eval_head_tail(s3, key, int(obj.get("Size") or 0))
                if not data:
                    err_count += 1
                    continue
                bench = extract_benchmark_from_filename(Path(key).name)
                benchmarks.add(bench)
                if bench not in benchmark_engine_map:
                    benchmark_engine_map[bench] = normalize_eval_engine_from_zip_path(key)
                meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
                inf_eng = meta.get("inference_engine") or data.get("inference_engine") or "-"
                row_key = (model_name, inf_eng, ckpt)
                if row_key not in rows_map:
                    rows_map[row_key] = {
                        "model": model_name,
                        "inference": inf_eng,
                        "checkpoint": ckpt,
                        "source": "s3",
                        "_path": f"s3://{S3_BUCKET}/{base_prefix}",
                        "_metrics": {},
                    }
                metric = extract_numeric_metric(data)
                if metric:
                    rows_map[row_key][bench] = metric[1]
                metrics_map = extract_metrics_map(data)
                if metrics_map:
                    rows_map[row_key]["_metrics"][bench] = metrics_map
            if not list_resp.get("IsTruncated"):
                break
            list_token = list_resp.get("NextContinuationToken")
            if not list_token:
                break
        else:
            # page cap exhausted while S3 still had more objects under this prefix
            logging.warning(
                "S3 object listing hit page cap for %s; some eval files under this checkpoint were not scanned",
                base_prefix,
            )

    if err_count:
        logging.warning("S3 eval read failures for %s: %s", model_name, err_count)
    return list(rows_map.values()), benchmark_engine_map

