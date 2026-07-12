#!/usr/bin/env python3
"""Preprocess a directory of raw video files for omni_evaluator ingestion.

Two-stage pipeline on ``--video_dirpath``:

  1. Container unification — stream-copy remux every non-``.mp4`` video into
     ``<parent>/<stem>.mp4`` with ``+faststart``. No re-encode: codec / GOP /
     fps / resolution / pixel format are all preserved verbatim. On success,
     the original (non-``.mp4``) source is removed. Existing ``.mp4`` files are
     left untouched.

  2. Integrity validation — for every resulting ``.mp4``:
       - ``ffprobe`` extracts codec / width / height / fps / duration /
         nb_frames.
       - ``ffmpeg -err_detect explode -f null -`` walks the whole stream to
         catch truncation or corrupted frames.
     Successful entries are appended to ``<video_dirpath>/.metadata.jsonl``
     (one row per video); failures land in ``<video_dirpath>/.errors.jsonl``
     rather than blocking the run.

Idempotent: re-runs skip files already ``.mp4`` and passing validation.

Usage:
    python scripts/create_task/preprocess_video.py \
        --video_dirpath /path/to/videos \
        --workers 8
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".mpg", ".mpeg", ".m4v", ".ts", ".flv", ".wmv"}


@dataclasses.dataclass
class Job:
    src_path: str
    dst_path: str
    action: str                                    # "validate" or "remux"


@dataclasses.dataclass
class JobResult:
    src: str
    dst: str
    ok: bool
    action: str                                    # "validated" / "remuxed" / "skipped" / "failed"
    meta: Optional[dict[str, Any]] = None
    error: Optional[str] = None


def _parse_rational(r: str) -> float:
    try:
        num, den = r.split("/")
        return float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _ffprobe_streams(path: Path) -> dict[str, Any]:
    """Return codec / width / height / fps / duration / nb_frames. May raise."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
        "-of", "json",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    data = json.loads(out.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream: {path}")
    s = streams[0]
    fps = _parse_rational(s.get("avg_frame_rate") or "0/1")
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    nb_frames = s.get("nb_frames")
    return {
        "codec": s.get("codec_name"),
        "width": int(s.get("width") or 0),
        "height": int(s.get("height") or 0),
        "fps": round(fps, 3) if fps else None,
        "duration": round(duration, 3),
        "nb_frames": int(nb_frames) if nb_frames and nb_frames.isdigit() else None,
    }


def _full_decode_probe(path: Path) -> None:
    """Walk the entire stream to catch truncation / corrupted frames. May raise."""
    cmd = [
        "ffmpeg", "-v", "error", "-err_detect", "explode",
        "-i", str(path),
        "-f", "null", "-",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1800)


def _remux_to_mp4(src: Path, dst: Path) -> None:
    """Stream-copy remux *src* → *dst*.mp4 with +faststart. No re-encode.

    Uses ``mkstemp`` in *dst*'s parent so a partial output never appears
    at the final path even if two runs race on the same file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.stem + ".", suffix=".remux.mp4", dir=str(dst.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(src),
        "-c", "copy",
        "-movflags", "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1800)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _process_one(job_dict: dict[str, Any]) -> dict[str, Any]:
    job = Job(**job_dict)
    src = Path(job.src_path)
    dst = Path(job.dst_path)
    try:
        if job.action == "remux":
            if dst.exists() and dst.stat().st_size > 0:
                # target already there from a prior run — validate only, leave src alone
                meta = _ffprobe_streams(dst)
                _full_decode_probe(dst)
                return dataclasses.asdict(JobResult(
                    src=str(src), dst=str(dst), ok=True, action="skipped", meta=meta,
                ))
            _remux_to_mp4(src, dst)
            meta = _ffprobe_streams(dst)
            _full_decode_probe(dst)
            if src.resolve() != dst.resolve():
                try:
                    src.unlink()
                except OSError:
                    pass
            return dataclasses.asdict(JobResult(
                src=str(src), dst=str(dst), ok=True, action="remuxed", meta=meta,
            ))
        # validate-only path: dst == src (already .mp4)
        meta = _ffprobe_streams(dst)
        _full_decode_probe(dst)
        return dataclasses.asdict(JobResult(
            src=str(src), dst=str(dst), ok=True, action="validated", meta=meta,
        ))
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "")[-500:]
        return dataclasses.asdict(JobResult(
            src=str(src), dst=str(dst), ok=False, action="failed",
            error=f"{type(e).__name__}: {err}",
        ))
    except Exception as e:
        return dataclasses.asdict(JobResult(
            src=str(src), dst=str(dst), ok=False, action="failed",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
        ))


def _enumerate_jobs(video_dirpath: Path) -> list[Job]:
    jobs: list[Job] = []
    for path in sorted(video_dirpath.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        # skip our own transient outputs
        if path.name.endswith((".remux.mp4", ".enc.mp4", ".tmp.mp4")):
            continue
        if path.suffix.lower() == ".mp4":
            jobs.append(Job(src_path=str(path), dst_path=str(path), action="validate"))
        else:
            dst = path.with_suffix(".mp4")
            jobs.append(Job(src_path=str(path), dst_path=str(dst), action="remux"))
    return jobs


def run(video_dirpath: Path, workers: int, progress_every: int = 100) -> int:
    if not video_dirpath.is_dir():
        print(f"[error] not a directory: {video_dirpath}", flush=True)
        return 2

    jobs = _enumerate_jobs(video_dirpath)
    if not jobs:
        print(f"[skip] {video_dirpath}: no video files found", flush=True)
        return 0
    print(f"[start] {video_dirpath}: {len(jobs)} files, {workers} workers", flush=True)

    counts = {"validated": 0, "remuxed": 0, "skipped": 0, "failed": 0}
    meta_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_one, dataclasses.asdict(j)) for j in jobs]
        for done_i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            counts[r["action"]] = counts.get(r["action"], 0) + 1
            if r["ok"]:
                meta_rows.append({
                    "path": str(Path(r["dst"]).relative_to(video_dirpath)),
                    "action": r["action"],
                    "meta": r.get("meta"),
                })
            else:
                failures.append({
                    "src": r["src"], "dst": r["dst"], "error": r["error"],
                })
            if done_i % progress_every == 0 or done_i == len(jobs):
                elapsed = time.time() - t0
                rate = done_i / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - done_i) / rate if rate > 0 else 0
                print(f"  [{done_i}/{len(jobs)}] rate={rate:.1f}/s elapsed={elapsed:.0f}s "
                      f"eta={eta:.0f}s | val={counts['validated']} remux={counts['remuxed']} "
                      f"skip={counts['skipped']} fail={counts['failed']}", flush=True)

    print(f"[done] {video_dirpath}: val={counts['validated']} remux={counts['remuxed']} "
          f"skip={counts['skipped']} fail={counts['failed']}", flush=True)

    meta_path = video_dirpath / ".metadata.jsonl"
    with meta_path.open("w") as fm:
        for row in sorted(meta_rows, key=lambda r: r["path"]):
            fm.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[meta] {len(meta_rows)} → {meta_path}", flush=True)

    if failures:
        err_path = video_dirpath / ".errors.jsonl"
        with err_path.open("w") as fe:
            for f in failures:
                fe.write(json.dumps(f, ensure_ascii=False) + "\n")
        print(f"[errors] {len(failures)} → {err_path}", flush=True)
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--video_dirpath", required=True, type=Path,
                   help="directory containing raw video files (rglob'd)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    args = p.parse_args()
    return run(args.video_dirpath, workers=args.workers)


if __name__ == "__main__":
    sys.exit(main())
