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

"""Internal dashboard backend configuration."""

import os
from pathlib import Path

# Cache / upload paths
CACHE_DIR = Path(
    os.environ.get("OMNI_INTERNAL_CACHE_DIR", "/tmp/omni-internal-dashboard")
)
UPLOAD_DIR = Path(
    os.environ.get("OMNI_INTERNAL_UPLOAD_DIR", "/tmp/omni-internal-dashboard-uploads")
)
DIRECT_SUBMISSION_DIR = UPLOAD_DIR / "direct"
INTERNAL_OUTPUTS_PATH = Path(
    os.environ.get("OMNI_INTERNAL_OUTPUTS_PATH", "./eval_outputs")
)

# CORS
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000"
).split(",")

# Data-source whitelist. Models are auto-detected from internal/direct/s3. On a small instance
# the S3 *source* scan (listing every model + scanning every object) is too heavy and can hang
# the box — yet S3 credentials may still be wanted purely to RE-SIGN expired presigned media
# URLs referenced by internal-source samples (that path is a single presign, not a scan).
# This gates which sources are scanned/listed, INDEPENDENTLY of whether creds exist.
#   default (unset) = "internal,direct,s3" (unchanged behavior)
#   e.g. OMNI_ENABLED_SOURCES=internal,direct → S3 creds stay active for media re-signing,
#        but S3 is never scanned as a data source. Empty/garbage falls back to all sources.
_ALL_SOURCES = ("internal", "direct", "s3")
ENABLED_SOURCES = tuple(
    s.strip()
    for s in os.environ.get("OMNI_ENABLED_SOURCES", ",".join(_ALL_SOURCES)).split(",")
    if s.strip() in _ALL_SOURCES
) or _ALL_SOURCES

# Leaderboard weights (persisted user preference)
LEADERBOARD_WEIGHT_FILE = CACHE_DIR / "leaderboard_weights.json"

# Scan performance
SCAN_IJSON_THRESHOLD = 512 * 1024  # Use ijson for files > 512KB
# Bounded worker pools for the background scan. Kept low on purpose: memory peak and
# CPU contention scale with concurrency, which is what hurts on small instances (e2-micro).
SCAN_WORKERS = max(1, int(os.environ.get("OMNI_SCAN_WORKERS", "2")))
S3_SCAN_WORKERS = max(1, int(os.environ.get("OMNI_S3_SCAN_WORKERS", "2")))
# Head/tail byte windows for large-file metric extraction (local + S3).
HEAD_BYTES = int(os.environ.get("OMNI_HEAD_BYTES", str(64 * 1024)))
TAIL_BYTES = int(os.environ.get("OMNI_TAIL_BYTES", str(1024 * 1024)))

# Local image API (Inference Viewer)
# 50 MB serving cap (memory/bandwidth management). Env-configurable if a deployment
# wants larger; over the cap the viewer shows a "too large" warning rather than serving.
LOCAL_IMAGE_MAX_SIZE = int(os.environ.get("OMNI_LOCAL_MEDIA_MAX_BYTES", str(50 * 1024 * 1024)))  # 50 MiB
# Trailing slash is REQUIRED on each prefix: prevents '/mnt' matching '/mnt_evil/...'.
LOCAL_IMAGE_ALLOWED_PREFIXES = ("/mnt/", "/data/")

# Optional media path rewriting. Eval JSON sometimes references images by an absolute path
# that only exists on the host that ran inference (e.g. a dataset cache under /mnt/...). On a
# different deployment those files live elsewhere, so map the old path prefix to the local one.
# Format: "old=new" pairs, ';'-separated. Applied to the decoded path BEFORE the realpath +
# allowlist check, so it cannot widen access — the rewritten path must still resolve under
# LOCAL_IMAGE_ALLOWED_PREFIXES. Empty (default) = no rewriting.
#   e.g. OMNI_LOCAL_MEDIA_PREFIX_MAP=/path/to/source_images/=/data/local_images/


def _parse_prefix_map(raw: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        old, new = item.split("=", 1)
        old, new = old.strip(), new.strip()
        if not old or not new:
            continue
        # Force a trailing slash on both ends. The match is a startswith() on the decoded path,
        # so a prefix without the slash ("/mnt/ds") would also match a sibling ("/mnt/ds_evil/")
        # and rewrite it — the same '/mnt' vs '/mnt_evil' pitfall the allowlist guards against.
        # Normalising to a directory prefix makes the boundary unambiguous.
        if not old.endswith("/"):
            old += "/"
        if not new.endswith("/"):
            new += "/"
        pairs.append((old, new))
    return tuple(pairs)


LOCAL_MEDIA_PREFIX_MAP = _parse_prefix_map(os.environ.get("OMNI_LOCAL_MEDIA_PREFIX_MAP", ""))

# Optional rewrite of a local media path prefix to an s3:// URL prefix. Like
# LOCAL_MEDIA_PREFIX_MAP, but the rewritten value is an s3:// URL that the inference viewer
# PRESIGNS — the browser then loads the image directly from object storage, so the media never
# has to live on this host's disk. Applied during sample media resolution (api/inference.py),
# before the local-path branch; requires S3 creds (the same ones used for media re-signing).
# Format: "old=s3://bucket/prefix/" pairs, ';'-separated. Default empty = no rewriting.
#   e.g. OMNI_MEDIA_S3_PREFIX_MAP=/path/to/source_images/=s3://your-bucket/media/images/
MEDIA_S3_PREFIX_MAP = _parse_prefix_map(os.environ.get("OMNI_MEDIA_S3_PREFIX_MAP", ""))

# When set, http(s) media URLs whose host matches OMNI_MEDIA_LEGACY_HOST are re-presigned
# against THIS bucket (MEDIA_REBUCKET) on the active S3 client, instead of the bucket embedded
# in the URL. Useful after migrating media between object stores: eval JSON may still carry the
# old presigned URLs, but the object key was copied verbatim into the new bucket, so re-presigning
# <new bucket>/<same key> serves it from the new home. Both empty (default) = feature off.
MEDIA_LEGACY_HOST = os.environ.get("OMNI_MEDIA_LEGACY_HOST", "").strip()
MEDIA_REBUCKET = os.environ.get("OMNI_MEDIA_REBUCKET", "").strip()

# S3 (optional — credentials + bucket config). All default to empty: set them via env to enable
# an S3 / S3-compatible data source or media presigning. With no creds + empty bucket, S3 is off
# and the dashboard serves from the local filesystem source only.
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", os.environ.get("AWS_ACCESS_KEY_ID", "")).strip()
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", "")).strip()
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", os.environ.get("OMNI_S3_BUCKET", "")).strip()
S3_PREFIX = os.environ.get("S3_PREFIX", os.environ.get("OMNI_S3_PREFIX", "")).strip()
S3_REGION = os.environ.get("S3_REGION", os.environ.get("OMNI_S3_REGION", "")).strip()
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", os.environ.get("OMNI_S3_ENDPOINT", "")).strip()
S3_MAX_MODELS = int(os.environ.get("OMNI_S3_MAX_MODELS", "200"))
S3_PRESIGN_EXPIRE = int(os.environ.get("OMNI_S3_PRESIGN_EXPIRE", str(3600)))  # seconds
# True iff both S3 credentials are set (used by /health s3_configured). Keyed off credentials,
# not bucket, so an unconfigured deployment reports s3_configured=false.
S3_HAS_CREDS = bool(S3_ACCESS_KEY and S3_SECRET_KEY)

# Upload / archive limits
DASHBOARD_API_KEY = os.environ.get("DASHBOARD_API_KEY", "").strip()  # '' => auth disabled
UPLOAD_MAX_BYTES = int(os.environ.get("OMNI_UPLOAD_MAX_BYTES", str(2 * 1024 * 1024 * 1024)))  # 2 GiB compressed upload cap
# Total direct-upload disk quota. The per-file cap above doesn't bound how many files pile up;
# on the small box (30 GiB disk) repeated uploads could fill the disk, so cap the dir total too.
UPLOAD_TOTAL_QUOTA_BYTES = int(os.environ.get("OMNI_UPLOAD_TOTAL_QUOTA_BYTES", str(8 * 1024 * 1024 * 1024)))  # 8 GiB
ZIP_MEMBER_MAX_BYTES = int(os.environ.get("OMNI_ZIP_MEMBER_MAX_BYTES", str(1024 * 1024 * 1024)))  # 1 GiB per-member uncompressed cap
ZIP_TOTAL_MAX_BYTES = int(os.environ.get("OMNI_ZIP_TOTAL_MAX_BYTES", str(4 * 1024 * 1024 * 1024)))  # 4 GiB total uncompressed cap
ZIP_MAX_RATIO = int(os.environ.get("OMNI_ZIP_MAX_RATIO", str(200)))  # max uncompressed/compressed ratio per member
MEDIA_INLINE_MAX_BYTES = int(os.environ.get("OMNI_MEDIA_INLINE_MAX_BYTES", str(4 * 1024 * 1024)))  # 4 MiB embedded-media inline cap (memory mgmt); over this the viewer shows a "too large" warning

# Inference Viewer per-sample read cap. A zip member / S3 object above this shows a "too large
# to preview" note instead of being read. Under the cap the record is STREAMED via ijson (never
# fully buffered), so peak RAM is ~one record regardless of file size — raising this is memory-
# safe. The genuine RAM risk is the NaN/Infinity-tolerant FULL parse in sample_reader.py, which
# stays capped INDEPENDENTLY at its own 64 MiB (_FULL_PARSE_MAX_BYTES) + a 1-wide semaphore; a
# file over that cap simply isn't full-parsed (a rare NaN sample renders blank) rather than OOM.
# Set to 256 MiB so large API outputs (e.g. 113 MB) preview by streaming. Cost of a large preview
# is CPU/network (ijson parses up to the index; S3 streams the body), not memory. Env-tunable.
INFERENCE_SAMPLE_MAX_BYTES = int(os.environ.get("OMNI_INFERENCE_SAMPLE_MAX_BYTES", str(256 * 1024 * 1024)))  # 256 MiB
# Leaderboard chart_rows cap: the chart only plots a handful, so bound the full set shipped
# to the client (the 60s auto-refresh re-fetches it). Generous vs realistic corpora.
CHART_ROWS_MAX = int(os.environ.get("OMNI_CHART_ROWS_MAX", "100"))
# anyio worker-thread cap: every blocking (sync def) endpoint runs in this pool. Default
# anyio gives 40 tokens; on 1 vCPU that lets too many sync handlers run at once. Sized to
# the box so concurrent requests can't multiply transient RAM past 1GB.
THREADPOOL_TOKENS = int(os.environ.get("OMNI_THREADPOOL_TOKENS", "12"))


def ensure_upload_dir() -> None:
    DIRECT_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
