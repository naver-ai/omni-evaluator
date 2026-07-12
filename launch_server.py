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

import argparse
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from fastapi import FastAPI, HTTPException
import logging
import os
from pathlib import Path
from pydantic import BaseModel
import shlex
from typing import Union
import uuid
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_CONCURRENT = 1
BASE = "python evaluate.py"
LOG_DIR = "./logs"
MAX_JOB_HISTORY = 1000

class JobState(str, Enum):
    PENDING = "pending"
    INPROGRESS = "inprogress"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"

class AddJobRequest(BaseModel):
    arguments: Union[str, dict]

class RemoveJobRequest(BaseModel):
    pid: str

class GetStateRequest(BaseModel):
    pid: str


@dataclass
class Job:
    pid: str
    arguments: Union[str, dict]
    state: JobState = JobState.PENDING
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    log_file: str = ""


app = FastAPI(title="Remote Job Server")

# In-memory state
jobs: dict[str, Job] = {}
pending_queue: list[str] = []  # pids in FIFO order
running_count: int = 0
_lock = None  # initialized on startup

def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock

def _build_command(arguments: Union[str, dict]) -> list[str]:
    """Build command as a list of arguments (safe from shell injection)."""
    base = shlex.split(BASE)
    if isinstance(arguments, str):
        return base + shlex.split(arguments) if arguments else base
    if isinstance(arguments, dict):
        parts = list(base)
        for k, v in arguments.items():
            if v is None:
                continue
            elif isinstance(v, bool):
                if v:
                    parts.append(f"--{k}")
            else:
                parts.append(f"--{k}={v}")
        return parts
    return base

async def _run_job(job: Job) -> None:
    """Execute a single job as a subprocess, stream output to a log file."""
    global running_count

    log_path = os.path.join(LOG_DIR, f"{job.pid}.log")
    os.makedirs(Path(log_path).parent, exist_ok=True)
    job.log_file = log_path
    job.state = JobState.INPROGRESS

    cmd = _build_command(job.arguments)
    logger.info(f"[{job.pid}] Starting: {' '.join(cmd)}")

    try:
        with open(log_path, "w") as log_fh:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_fh,
                stderr=asyncio.subprocess.STDOUT,
            )
            job.process = proc
            returncode = await proc.wait()

        # If already terminated by remove_job, don't overwrite state
        if job.state == JobState.TERMINATED:
            return

        if returncode == 0:
            job.state = JobState.COMPLETED
            logger.info(f"[{job.pid}] Completed (rc=0)")
        else:
            job.state = JobState.FAILED
            logger.warning(f"[{job.pid}] Failed (rc={returncode})")

    except Exception as e:
        if job.state != JobState.TERMINATED:
            job.state = JobState.FAILED
            logger.error(f"[{job.pid}] Exception: {e}")

    finally:
        job.process = None
        # Only decrement if not already handled by remove_job
        if job.state != JobState.TERMINATED:
            async with _get_lock():
                running_count -= 1
        await _try_dispatch()


async def _try_dispatch() -> None:
    """Dispatch pending jobs if capacity is available."""
    global running_count

    async with _get_lock():
        while running_count < MAX_CONCURRENT and pending_queue:
            pid = pending_queue.pop(0)
            job = jobs.get(pid)
            if job is None or job.state != JobState.PENDING:
                continue
            running_count += 1
            asyncio.create_task(_run_job(job))


def _cleanup_finished_jobs() -> None:
    """Remove oldest terminal jobs when history exceeds MAX_JOB_HISTORY."""
    terminal = [
        pid for pid, j in jobs.items()
        if j.state in (JobState.COMPLETED, JobState.FAILED, JobState.TERMINATED)
    ]
    if len(terminal) > MAX_JOB_HISTORY:
        for pid in terminal[:len(terminal) - MAX_JOB_HISTORY]:
            del jobs[pid]


@app.post("/add_job")
async def add_job(req: AddJobRequest):
    pid = uuid.uuid4().hex[:12]
    job = Job(pid=pid, arguments=req.arguments)
    async with _get_lock():
        jobs[pid] = job
        pending_queue.append(pid)
    logger.info(f"[{pid}] Enqueued (arguments={req.arguments})")

    _cleanup_finished_jobs()
    await _try_dispatch()

    return {"pid": pid}


@app.post("/get_state")
async def get_state(req: GetStateRequest):
    job = jobs.get(req.pid)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {req.pid} not found")

    state = job.state.value
    if job.state == JobState.COMPLETED:
        state = f"completed({job.log_file})"

    return {"state": state}


@app.post("/remove_job")
async def remove_job(req: RemoveJobRequest):
    global running_count

    job = jobs.get(req.pid)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {req.pid} not found")

    if job.state == JobState.INPROGRESS:
        # Mark as terminated first to prevent _run_job from decrementing running_count
        job.state = JobState.TERMINATED
        if job.process and job.process.returncode is None:
            try:
                job.process.kill()
                await job.process.wait()
                logger.info(f"[{req.pid}] Killed running process")
            except ProcessLookupError:
                pass
        job.process = None
        async with _get_lock():
            running_count -= 1
        await _try_dispatch()

    elif job.state == JobState.PENDING:
        async with _get_lock():
            if req.pid in pending_queue:
                pending_queue.remove(req.pid)
        job.state = JobState.TERMINATED
        logger.info(f"[{req.pid}] Removed from pending queue")

    else:
        # Already completed or failed — just acknowledge
        logger.info(f"[{req.pid}] Already in terminal state: {job.state.value}")

    return {"result": True}


@app.get("/jobs")
async def list_jobs():
    """Utility endpoint: list all jobs and their states."""
    return {
        pid: {
            "state": j.state.value,
            "arguments": j.arguments,
            "log_file": j.log_file,
        }
        for pid, j in jobs.items()
    }


@app.on_event("startup")
async def startup_event():
    """Ensure log directory exists regardless of how the server is started."""
    os.makedirs(LOG_DIR, exist_ok=True)


if __name__ == "__main__":
    """
    uvicorn launch_server:app --host 0.0.0.0 --port 11169
    python launch_server.py --host 0.0.0.0 --port 11169 --log_dir="./logs_temp"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--base", type=str, default="python evaluate.py")
    args = parser.parse_args()

    LOG_DIR = args.log_dir
    os.makedirs(LOG_DIR, exist_ok=True)
    MAX_CONCURRENT = args.max_concurrent
    if args.base:
        BASE = args.base

    uvicorn.run(app, host=args.host, port=args.port)
