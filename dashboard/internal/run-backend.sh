#!/usr/bin/env bash
# Run FastAPI backend for Internal Dashboard (port 11592)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DASHBOARD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$DASHBOARD_ROOT/../../.." && pwd)"
PORT_INTERNAL=11592

cd "$WORKSPACE_ROOT"
[ -d .venv ] && source .venv/bin/activate
cd "$SCRIPT_DIR/backend"
pip install -r requirements.txt -q

# --- e2-micro (1GB / 1 vCPU) tuning ---
# Scan parsing is GIL-bound, so 1 worker is nearly as fast as 2 and roughly halves the
# transient scan-peak buffer footprint. Override via env if running on a larger box.
export OMNI_SCAN_WORKERS="${OMNI_SCAN_WORKERS:-1}"
export OMNI_S3_SCAN_WORKERS="${OMNI_S3_SCAN_WORKERS:-1}"
# Admission control: bound in-flight requests so a burst returns 503 instead of OOM-killing
# the single worker. The anyio threadpool cap (config.THREADPOOL_TOKENS) bounds parallelism.
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT_INTERNAL" \
  --limit-concurrency 32 --timeout-keep-alive 5
