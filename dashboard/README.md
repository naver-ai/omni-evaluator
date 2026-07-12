# OmniEvaluator Dashboard

Two dashboards live here:

- **Internal** (`internal/`) — operational tool (FastAPI + vanilla JS). 3 tabs:
  Submission / Leaderboard / Inference Viewer. Auto-scans models from the filesystem
  and/or an S3-compatible source. Runs comfortably on a tiny instance (~1 GB RAM);
  see [internal/README.md](internal/README.md) for the verified footprint and run steps.
- **Public** (`public/`) — static leaderboard site, no backend. Renders a generated
  `data.json` snapshot: a leaderboard on an absolute 0–100 scale (with per-model benchmark
  coverage), illustrative per-modality examples, and the pipeline figure. Open
  `public/index.html` directly or serve the folder with any static file server.

## Quick Start

Internal dashboard:

```bash
cd HyperCLOVA-VLM-Evaluator/dashboard/internal/backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 11592
```

No database needed. Models are auto-scanned from the filesystem.

Public dashboard (static — pick any server):

```bash
cd HyperCLOVA-VLM-Evaluator/dashboard/public
python -m http.server 10017
```

## Environment (.env)

Place `.env` at `dashboard/internal/.env`:

```bash
# Eval outputs root (required) — point this at your evaluator's output directory
OMNI_INTERNAL_OUTPUTS_PATH=/path/to/evaluator/outputs

# S3 (optional) — set these to read from an S3 / S3-compatible source
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
S3_BUCKET_NAME=your-bucket
S3_PREFIX=your-prefix/
S3_ENDPOINT_URL=https://your-s3-endpoint
```

## Architecture

- **Backend:** FastAPI, no database — filesystem is the source of truth
- **Scanning:** mtime-based cache with head+tail parser (reads only first 64KB + last 1MB of large JSON files)
- **Frontend:** Static HTML + Vanilla JS, Tailwind CSS (CDN), Chart.js
- **Background:** Server starts → loads disk cache → background thread scans for updates

## Ports

- Internal dashboard: **11592** (`http://localhost:11592/`)
- Public dashboard (static): any port you serve it on (e.g. `10017`)
