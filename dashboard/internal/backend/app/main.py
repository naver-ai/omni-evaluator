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

"""Internal dashboard FastAPI application."""

from contextlib import asynccontextmanager
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from backend/ and from its parent internal/. Either location works; both are
# optional, and load_dotenv is a no-op when the file is absent.
_env_dir = Path(__file__).resolve().parent.parent  # backend/
load_dotenv(_env_dir / ".env")          # backend/.env
load_dotenv(_env_dir.parent / ".env")   # internal/.env

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .api import admin, inference, leaderboard, local_image, submission
from .config import CORS_ORIGINS, INTERNAL_OUTPUTS_PATH, S3_BUCKET, S3_ENDPOINT, S3_PREFIX


# Scan freshness, surfaced via /health so monitoring can alert on stale data.
_last_scan_ok_ts = None  # epoch seconds of last successful background scan
_last_scan_error = None  # str of last background scan failure, or None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cap the anyio worker-thread pool that runs every sync def endpoint. The default is 40
    # tokens; on a 1 vCPU / 1GB box that lets too many blocking handlers run at once and
    # multiply transient RAM. Sizing it to the box gives admission control at the thread layer.
    try:
        import anyio
        anyio.to_thread.current_default_thread_limiter().total_tokens = config.THREADPOOL_TOKENS
        logging.info("anyio threadpool capped at %d tokens", config.THREADPOOL_TOKENS)
    except Exception:
        logging.exception("Could not cap anyio threadpool")
    if not INTERNAL_OUTPUTS_PATH.exists():
        logging.warning("Internal outputs path not found: %s", INTERNAL_OUTPUTS_PATH)
    # Load disk cache first (instant), then refresh in background thread
    import threading
    from .services.scan_cache import load_cache_from_disk, save_cache_to_disk, scan_all_sources, trim_memory
    loaded = load_cache_from_disk()
    if loaded:
        logging.info("Disk cache loaded — serving immediately")
    # Background thread is fine now — head+tail parser is fast (~0.03s per file)
    def _background_refresh():
        global _last_scan_ok_ts, _last_scan_error
        try:
            scan_all_sources(quick=False)
            save_cache_to_disk()
            trim_memory()  # hand freed heap back to the OS so RSS fits small instances
            _last_scan_ok_ts = time.time()
            _last_scan_error = None
            logging.info("Background scan complete, cache saved")
        except Exception as exc:
            _last_scan_error = repr(exc)
            logging.exception("Background scan failed")
    threading.Thread(target=_background_refresh, daemon=True).start()
    logging.info("Background scan thread started")
    yield


class ImmutableStaticFiles(StaticFiles):
    """Static assets are versioned via ?v=<token> cache-busting (bumped on every change),
    so they can be cached hard. Long-lived immutable caching avoids re-downloading JS/CSS on
    every page load (the previous no-store also defeated 304s). HTML and /api JSON keep
    no-store via their own response headers, so only fingerprinted assets are cached."""
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


app = FastAPI(
    title="OmniEvaluator Internal Dashboard",
    version="0.1.0",
    lifespan=lifespan,
    # The interactive docs + schema would hand an anonymous visitor a machine-readable map of
    # every admin route and the auth header names. The dashboard has no login, so disable them.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.middleware("http")
async def _security_headers(request, call_next):
    """Add baseline security response headers (clickjacking, referrer, feature policy).
    nginx sets HSTS / X-Content-Type-Options / X-Robots-Tag; these complement without overlap.
    Only a `frame-ancestors` CSP is set (not script-src) so it cannot break the inline bootstrap
    or the vendored Tailwind runtime."""
    response = await call_next(request)
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Content-Security-Policy", "frame-ancestors 'none'")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response

# Compress responses: the leaderboard JSON (repetitive benchmark-key strings) and the
# served JS/CSS shrink ~10x on the wire, cutting socket-write time on the single vCPU.
# GZipMiddleware decides by content-type, excluding only text/event-stream by default.
# Compressing binary FileResponses (media at /api/local-*, zip downloads) would corrupt
# Range/206 semantics (gzipped length vs uncompressed byte offsets, breaking <video>/<audio>
# seeking) and waste CPU recompressing already-compressed bytes — so extend the exclusion
# list it checks. JSON/JS/CSS (the intended wins) still compress; media/archives stream
# through untouched. Done before the app serves traffic; the responder reads it per response.
try:
    import starlette.middleware.gzip as _sgz
    _sgz.DEFAULT_EXCLUDED_CONTENT_TYPES = (
        "text/event-stream", "image/", "audio/", "video/",
        "application/zip", "application/octet-stream",
    )
except Exception:
    logging.exception("Could not extend GZip excluded content types")
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

app.include_router(submission.router, prefix="/api/submission", tags=["submission"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(leaderboard.router, prefix="/api/leaderboard", tags=["leaderboard"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(local_image.router, prefix="/api", tags=["local-image"])

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", ImmutableStaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/health")
def health():
    s3_configured = config.S3_HAS_CREDS
    s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}" if S3_BUCKET else ""
    return {
        "status": "ok",
        "internal_path": str(INTERNAL_OUTPUTS_PATH),
        "internal_path_exists": INTERNAL_OUTPUTS_PATH.exists(),
        "s3_path": s3_path,
        "s3_endpoint": S3_ENDPOINT if S3_BUCKET else "",
        "s3_configured": s3_configured,
        "scan_ok": _last_scan_error is None and _last_scan_ok_ts is not None,
        "last_scan_ok_ts": _last_scan_ok_ts,
        "last_scan_error": _last_scan_error,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve dashboard page."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return HTMLResponse(
            template_path.read_text(encoding="utf-8"),
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )
    return HTMLResponse("<h1>Dashboard</h1><p>Template not found.</p>")
