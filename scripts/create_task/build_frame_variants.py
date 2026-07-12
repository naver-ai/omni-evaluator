#!/usr/bin/env python3
"""Build N-frame compressed variants for a directory of preprocessed videos.

Layout transformation (in-place on ``--video_dirpath``):

  <video_dirpath>/*.mp4
    →
  <video_dirpath>/base/*.mp4               (moved verbatim; no re-encode)
  <video_dirpath>/8_frames/*.mp4           (built)
  <video_dirpath>/64_frames/*.mp4          (built)
  <video_dirpath>/128_frames/*.mp4         (built)

The N-frame variants use ffmpeg's fps filter to resample the video stream to
``N / duration`` fps while preserving the original duration. Audio is stream-
copied (``-c:a copy``) so audio sync stays intact for audio-aware models.
``-g 1`` makes every output frame a keyframe, eliminating inter-frame
dependencies (cheap seek; favorable for vllm's sequential ``cap.grab()`` loop).

Edge case: source has ≤ target N total frames (extremely short clips). Output
fps would be very high; ffmpeg's fps filter would then duplicate frames, which
wastes space without helping decoding. We instead hardlink the base file into
the variant dir for those cases — same content, no extra disk.

Idempotency: each step skips work if its sentinel output is already present
(``.base_moved`` marker, or the variant file existing with positive size).
Safe to re-run.

Assumes ``preprocess_video.py`` has already unified containers to ``.mp4``
and validated integrity; only ``*.mp4`` under ``<video_dirpath>`` are picked up.

Usage:
    python scripts/create_task/build_frame_variants.py \
        --video_dirpath /path/to/videos \
        --frames 8 64 128 \
        --workers 16
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

CRF = 23
PRESET = "fast"          # ~2× faster encode vs medium; file size +5–10% (negligible at <5 MB)
FFMPEG_THREADS = 2       # fixed per-worker thread budget; with workers=N total threads ≈ 2N


@dataclasses.dataclass
class Job:
    src_path: str
    dst_path: str
    target_frames: int


@dataclasses.dataclass
class JobResult:
    src: str
    dst: str
    target_frames: int
    ok: bool
    action: str                                    # "encoded" / "hardlinked" / "skipped" / "failed"
    error: Optional[str] = None


def _ffprobe_duration_and_frames(path: Path) -> tuple[float, int]:
    """Return (duration_seconds, total_video_frames). Either may be 0 on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets:format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        data = json.loads(out.stdout)
        duration = float((data.get("format") or {}).get("duration") or 0.0)
        streams = data.get("streams") or [{}]
        nb_packets = streams[0].get("nb_read_packets", "0")
        total_frames = int(nb_packets) if isinstance(nb_packets, str) and nb_packets.isdigit() else 0
        return duration, total_frames
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError):
        return 0.0, 0


def _process_one(job_dict: dict) -> dict:
    """Subprocess entry: probe, decide encode vs hardlink, run ffmpeg if needed."""
    job = Job(**job_dict)
    src = Path(job.src_path)
    dst = Path(job.dst_path)
    try:
        if dst.exists() and dst.stat().st_size > 0:
            return dataclasses.asdict(JobResult(
                src=str(src), dst=str(dst), target_frames=job.target_frames,
                ok=True, action="skipped",
            ))
        dst.parent.mkdir(parents=True, exist_ok=True)

        duration, total_frames = _ffprobe_duration_and_frames(src)
        if duration <= 0:
            raise RuntimeError(f"ffprobe failed or duration=0: {src}")

        # Short-source edge case: source has at most as many frames as target →
        # hardlink the original (no re-encode, no extra disk).
        if total_frames > 0 and total_frames <= job.target_frames:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            return dataclasses.asdict(JobResult(
                src=str(src), dst=str(dst), target_frames=job.target_frames,
                ok=True, action="hardlinked",
            ))

        # Normal path: fps filter resamples video to target_frames/duration.
        # ``scale=trunc(iw/2)*2:trunc(ih/2)*2`` forces even dimensions (libx264
        # with yuv420p rejects odd width/height).
        target_fps = max(job.target_frames / duration, 1e-3)
        fd, tmp_name = tempfile.mkstemp(prefix=dst.stem + ".", suffix=".enc.mp4", dir=str(dst.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-threads", str(FFMPEG_THREADS),
            "-i", str(src),
            "-vf", f"fps={target_fps:.6f},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
            "-pix_fmt", "yuv420p",
            "-g", "1",
            "-c:a", "copy",
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
        return dataclasses.asdict(JobResult(
            src=str(src), dst=str(dst), target_frames=job.target_frames,
            ok=True, action="encoded",
        ))
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "")[-400:]
        return dataclasses.asdict(JobResult(
            src=str(src), dst=str(dst), target_frames=job.target_frames,
            ok=False, action="failed", error=f"ffmpeg: {err}",
        ))
    except Exception as e:
        return dataclasses.asdict(JobResult(
            src=str(src), dst=str(dst), target_frames=job.target_frames,
            ok=False, action="failed",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
        ))


def _move_to_base(video_dirpath: Path) -> int:
    """Move video_dirpath/*.mp4 → video_dirpath/base/ (idempotent, single FS rename).

    Only top-level ``*.mp4`` are moved; anything already under ``base/`` or
    ``{N}_frames/`` is left alone."""
    base_dir = video_dirpath / "base"
    marker = video_dirpath / ".base_moved"
    if marker.exists() and base_dir.is_dir():
        return sum(1 for _ in base_dir.glob("*.mp4"))
    base_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for mp4 in list(video_dirpath.glob("*.mp4")):
        target = base_dir / mp4.name
        if target.exists():
            mp4.unlink()
            continue
        mp4.rename(target)
        moved += 1
    marker.write_text("ok")
    return moved


def _enumerate_jobs(base_dir: Path, variant_dirs: dict[int, Path]) -> list[Job]:
    jobs: list[Job] = []
    # rglob so nested layouts (e.g. base/cat1/<id>.mp4) are picked up
    # alongside flat layouts (base/<id>.mp4). dst preserves subdir.
    for src in sorted(base_dir.rglob("*.mp4")):
        # skip leftover tmp/work files
        if src.name.endswith((".tmp.mp4", ".enc.mp4", ".remux.mp4")):
            continue
        rel = src.relative_to(base_dir)
        for n_frames, vdir in variant_dirs.items():
            dst = vdir / rel
            jobs.append(Job(
                src_path=str(src),
                dst_path=str(dst),
                target_frames=n_frames,
            ))
    return jobs


def run(
    video_dirpath: Path,
    frames: list[int],
    workers: int,
    limit: Optional[int] = None,
    progress_every: int = 100,
) -> int:
    if not video_dirpath.is_dir():
        print(f"[error] not a directory: {video_dirpath}", flush=True)
        return 2

    print(f"\n========== {video_dirpath} ==========", flush=True)
    moved = _move_to_base(video_dirpath)
    base_dir = video_dirpath / "base"
    if moved:
        print(f"  [move] {moved} mp4s → {base_dir}", flush=True)
    else:
        print(f"  [move] (already in base/)", flush=True)
    total_base = sum(1 for _ in base_dir.rglob("*.mp4"))
    print(f"  [base] {total_base} mp4s ready", flush=True)

    variant_dirs = {n: video_dirpath / f"{n}_frames" for n in frames}
    for n, vdir in variant_dirs.items():
        vdir.mkdir(parents=True, exist_ok=True)

    jobs = _enumerate_jobs(base_dir, variant_dirs)
    if limit:
        jobs = jobs[:limit]
        print(f"  [build] truncated to limit={limit}", flush=True)
    print(f"  [build] {len(jobs)} encode jobs across {len(variant_dirs)} variants "
          f"({sorted(frames)}), {workers} workers", flush=True)

    counts = {"encoded": 0, "hardlinked": 0, "skipped": 0, "failed": 0}
    failures: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_one, dataclasses.asdict(j)) for j in jobs]
        for done_i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            counts[r["action"]] = counts.get(r["action"], 0) + 1
            if not r["ok"]:
                failures.append(r)
            if done_i % progress_every == 0 or done_i == len(jobs):
                elapsed = time.time() - t0
                rate = done_i / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - done_i) / rate if rate > 0 else 0
                print(f"  [{done_i}/{len(jobs)}] rate={rate:.1f}/s elapsed={elapsed:.0f}s "
                      f"eta={eta:.0f}s | enc={counts['encoded']} hl={counts['hardlinked']} "
                      f"skip={counts['skipped']} fail={counts['failed']}", flush=True)

    print(f"  [done] enc={counts['encoded']} hl={counts['hardlinked']} "
          f"skip={counts['skipped']} fail={counts['failed']}", flush=True)
    if failures:
        err_path = video_dirpath / ".variant_errors.jsonl"
        with err_path.open("w") as fe:
            for f in failures:
                fe.write(json.dumps(f, ensure_ascii=False) + "\n")
        print(f"  [errors] {len(failures)} → {err_path}", flush=True)
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--video_dirpath", required=True, type=Path,
                   help="directory of preprocessed *.mp4 videos (rearranged in place)")
    p.add_argument("--frames", type=int, nargs="+", default=[8, 64, 128],
                   help="target frame counts per variant (default: 8 64 128)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    p.add_argument("--limit", type=int, default=None,
                   help="dev only: cap total encode jobs for a quick pilot")
    args = p.parse_args()
    return run(args.video_dirpath, frames=args.frames, workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
