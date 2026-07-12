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

"""Submission API: Internal models, Direct upload."""

import logging
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from ..auth import require_api_key
from ..config import (
    DIRECT_SUBMISSION_DIR,
    UPLOAD_MAX_BYTES,
    UPLOAD_TOTAL_QUOTA_BYTES,
    ZIP_MAX_RATIO,
    ZIP_MEMBER_MAX_BYTES,
    ZIP_TOTAL_MAX_BYTES,
    ensure_upload_dir,
)
from ..services.scan_cache import get_models, evict_zip, _scan_zip

# Direct-Upload example bundle. Served from app/static (tracked + shipped on deploy); the
# legacy internal/assets path is kept as a fallback for local runs that still have it.
_STATIC_MYMODEL = Path(__file__).resolve().parent.parent / "static" / "MyModel.zip"
_ASSETS_MYMODEL = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "MyModel.zip"
_MYMODEL_ZIP = _STATIC_MYMODEL if _STATIC_MYMODEL.exists() else _ASSETS_MYMODEL

router = APIRouter()


def _dir_total_bytes(d: Path, exclude: Path | None = None) -> int:
    """Sum of *.zip sizes in a directory (optionally excluding one path)."""
    total = 0
    try:
        for p in d.glob("*.zip"):
            if exclude is not None and p == exclude:
                continue
            try:
                total += p.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


@router.get("/internal/models")
def internal_models():
    """List models from filesystem (internal source)."""
    models = get_models("internal")
    return JSONResponse(
        content={"models": models},
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/direct/models")
def direct_models():
    """List models from filesystem (direct source)."""
    models = get_models("direct")
    return JSONResponse(
        content={"models": models},
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/s3/models")
def s3_models():
    """List models from S3 (auto-cached with TTL)."""
    models = get_models("s3")
    return JSONResponse(
        content={"models": models},
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/direct/upload", dependencies=[Depends(require_api_key)])
async def direct_upload(file: UploadFile = File(...)):
    """Upload ZIP file for direct submission."""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files allowed")
    ensure_upload_dir()
    dest = DIRECT_SUBMISSION_DIR / file.filename
    # Existing direct-upload disk usage (excluding the destination we're about to overwrite),
    # so a new upload can't push the directory past the total quota and fill the small disk.
    existing = _dir_total_bytes(DIRECT_SUBMISSION_DIR, exclude=dest)
    # Stream the upload to disk in chunks (don't buffer the whole body in RAM)
    # and enforce the compressed-size cap as we go.
    try:
        written = 0
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                written += len(chunk)
                if written > UPLOAD_MAX_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="Upload too large")
                if existing + written > UPLOAD_TOTAL_QUOTA_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(status_code=507, detail="Upload storage quota exceeded")
                out.write(chunk)
    except HTTPException:
        raise
    except Exception:
        logging.exception("Direct upload failed for %s", file.filename)
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail="File upload failed")
    # Guard against zip bombs before scanning: cap per-member size, total
    # uncompressed size, and per-member compression ratio.
    try:
        total = 0
        with zipfile.ZipFile(dest, "r") as zf:
            for info in zf.infolist():
                if info.file_size > ZIP_MEMBER_MAX_BYTES:
                    raise HTTPException(status_code=400, detail="Invalid zip file")
                total += info.file_size
                if total > ZIP_TOTAL_MAX_BYTES:
                    raise HTTPException(status_code=400, detail="Invalid zip file")
                if info.compress_size and info.file_size / info.compress_size > ZIP_MAX_RATIO:
                    raise HTTPException(status_code=400, detail="Invalid zip file")
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception:
        logging.exception("Failed to validate uploaded zip %s", file.filename)
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid zip file")
    # Scan the uploaded zip immediately so it appears in leaderboard.
    # Runs in the threadpool so the (blocking) zip parse doesn't stall the loop.
    await run_in_threadpool(_scan_zip, dest, "direct")
    return {"filename": file.filename, "model": dest.stem}


@router.delete("/direct/{model}", dependencies=[Depends(require_api_key)])
def direct_delete(model: str):
    """Delete direct-upload model (zip file)."""
    safe = "".join(c for c in model if c.isalnum() or c in "-_")
    if safe != model:
        raise HTTPException(status_code=400, detail="Invalid model name")
    ensure_upload_dir()
    path = DIRECT_SUBMISSION_DIR / f"{model}.zip"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        path.unlink()
    except OSError:
        raise HTTPException(status_code=500, detail="Failed to delete file")
    evict_zip(path)
    return {"deleted": model}


@router.delete("/model/{source}/{model}", dependencies=[Depends(require_api_key)])
def delete_model_endpoint(source: str, model: str):
    """Delete model by source (internal/direct)."""
    src = (source or "").strip().lower()
    if src not in {"internal", "direct"}:
        raise HTTPException(status_code=400, detail="Invalid source")
    safe = "".join(c for c in model if c.isalnum() or c in "-_")
    if safe != model:
        raise HTTPException(status_code=400, detail="Invalid model name")
    # Internal-source models are read-only scanned outputs, not deletable artifacts. Reject
    # rather than returning a misleading 200 that claims a deletion that never happened.
    if src == "internal":
        raise HTTPException(status_code=405, detail="Internal models are read-only and cannot be deleted")
    if src == "direct":
        ensure_upload_dir()
        path = DIRECT_SUBMISSION_DIR / f"{model}.zip"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        try:
            path.unlink()
        except OSError:
            raise HTTPException(status_code=500, detail="Failed to delete file")
        evict_zip(path)
    return {"deleted": model, "source": src}


@router.get("/example/download")
def download_example():
    """Download MyModel.zip template for Direct Upload."""
    if not _MYMODEL_ZIP.exists():
        raise HTTPException(status_code=404, detail="MyModel.zip not found")
    return FileResponse(
        path=_MYMODEL_ZIP,
        media_type="application/zip",
        filename="MyModel.zip",
    )
