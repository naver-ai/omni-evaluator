# Internal Dashboard

Internal dashboard is a FastAPI app that serves a static UI (Vanilla JS + Tailwind CSS)
and exposes API endpoints for submission sync, leaderboard, and inference viewer.

## Tech Stack

| Area | Tech |
|------|------|
| Backend | FastAPI (Python 3.10+). Blocking handlers are sync `def` → run in a threadpool so the event loop never stalls on file/S3 I/O |
| Frontend | Vanilla JS + Tailwind CSS (CDN) + Chart.js — light theme, 3 tabs (Submission / Leaderboard / Inference Viewer). No build step |
| Storage | Local filesystem (no database) and/or S3-compatible. Large JSON is streamed (head+tail + `ijson`), never fully loaded |

## Requirements

- Python 3.10+
- Optional: S3 credentials if you use the S3 source

## Installation

Python environment (creates `.venv` at repo root so `run-backend.sh` auto-activates):

```bash
cd /path/to/repo
python -m venv .venv
source .venv/bin/activate
pip install -r HyperCLOVA-VLM-Evaluator/dashboard/internal/backend/requirements.txt
```

## Environment

Create `.env` in either `dashboard/internal/backend/.env` or `dashboard/internal/.env`.
Both locations are loaded on startup.

Optional:
- `OMNI_INTERNAL_OUTPUTS_PATH` (default: `./eval_outputs`) — eval-outputs source root
- `OMNI_INTERNAL_UPLOAD_DIR` (default: `/tmp/omni-internal-dashboard-uploads`) — Direct-upload zips
- `OMNI_INTERNAL_CACHE_DIR` (default: `/tmp/omni-internal-dashboard`) — scan cache on disk
- `OMNI_SCAN_WORKERS` / `OMNI_S3_SCAN_WORKERS` (default: `2` each) — scan parallelism; keep small (≤2) on tiny instances
- `OMNI_HEAD_BYTES` / `OMNI_TAIL_BYTES` (default: `64KB` / `1MB`) — head+tail window for large-JSON parsing
- `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET_NAME`, `S3_PREFIX`, `S3_REGION`, `S3_ENDPOINT_URL`, `OMNI_S3_MAX_MODELS`

See `backend/.env.example` for a full template. **Never commit a real `.env`** — it is gitignored.

## Run

```bash
cd dashboard/internal
./run-backend.sh
```

- URL: http://localhost:11592/
- Health: http://localhost:11592/health

## Usage (Workflow)

- Submission
  - Local path: scans outputs under `OMNI_INTERNAL_OUTPUTS_PATH`
  - Direct upload: uploads to `OMNI_INTERNAL_UPLOAD_DIR`
  - S3: syncs from bucket/prefix configured by `S3_*` variables
- Leaderboard: browse and compare aggregated metrics
- Inference Viewer: inspect per-sample inputs/outputs and metrics

## Resource Footprint

Designed to run on a **tiny instance (~1 GB RAM, 1–2 vCPU)**. Large output
files are streamed (head+tail + `ijson` for single records), never loaded whole, so memory stays
flat regardless of file size.

Verified under a hard 1 GB cgroup cap (swap off, 2 vCPU):

| Scenario | Result |
|----------|--------|
| Idle RSS | ~60 MB |
| Cold leaderboard scan (50 models) | ~0.4 s, peak ~110 MB |
| Inference Viewer on a 383 MB file | ~0.1 s, peak ~115 MB |
| 6 concurrent viewer requests | peak ~120 MB, `/health` stays ~1 ms |
| cgroup OOM events | 0 |

- **CPU note:** the above used 2 full cores. Real micro instances are *burstable/shared*, so
  latencies are higher and sustained heavy concurrency can throttle (it won't OOM/crash). Fine for
  typical dashboard traffic.
- **Disk:** scan cache is small (filename + metadata, not file bodies). No 1.5× raw-output reserve needed.
- **Network:** required only for the S3 source and presigned-URL refresh.

## Deployment (small instance)

1. **Data location** — point `OMNI_INTERNAL_OUTPUTS_PATH` at your evaluator's output directory
   (default `./eval_outputs`). Or **use the S3 source only** (set the `S3_*` vars and leave the
   internal path unset).
2. **Python env** — create a fresh venv and `pip install -r backend/requirements.txt`
   (`ijson` should pull its fast `yajl2_c` backend; a pure-Python fallback also works, just slower).
3. **Process** — run a single uvicorn worker (don't add `--workers` on 1 GB), `--reload` **off**.
   Prefer a systemd service for auto-restart + reboot survival over `nohup`.
4. **Secrets** — put real values in `.env` (gitignored), `chmod 600`. Do not commit them.

## Notes

- No frontend build step (static HTML served by FastAPI).
- No database needed — models are auto-scanned from the filesystem.
- Default folder structure (internal/direct/S3 compatible):
  - `<model>/<checkpoint>/<engine>/output/*.json`
  - `output` is treated like `evaluation_output` (legacy)

## Structure

```
internal/
├── backend/     # FastAPI
│   ├── app/main.py, config.py
│   └── requirements.txt
├── assets/
└── run-backend.sh
```
