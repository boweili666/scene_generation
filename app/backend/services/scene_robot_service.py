"""scene_robot pipeline service.

Mirrors `pipeline_service.py` in shape but launches scene_robot scripts
(which live in the `env_isaaclab` conda env, not the web backend's env).
The integration is loose: subprocess + filesystem artifacts. Each job
streams its child stdout/stderr into a log file the UI polls via
`/scene_robot/log`.

Skeleton only: just the `collect` stage today. `train` / `eval` slot in
later as additional `start_*_job` entry points reusing `_run_subprocess`.
"""

from __future__ import annotations

import os
import selectors
import subprocess
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import (
    ENV_ISAACLAB_PYTHON,
    PROJECT_ROOT,
    SCENE_ROBOT_AUTO_GRASP_COLLECT_SCRIPT,
    SCENE_ROBOT_LOG_PATH,
)


_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_log_path(log_path: str | os.PathLike[str] | None = None) -> Path:
    return Path(str(log_path or SCENE_ROBOT_LOG_PATH)).resolve()


def _append_log_entry(entry: str, *, log_path: str | os.PathLike[str] | None = None) -> None:
    target = _resolve_log_path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as logf:
        logf.write(entry)


def log_scene_robot_event(
    message: str,
    job_id: str | None = None,
    level: str = "INFO",
    *,
    log_path: str | os.PathLike[str] | None = None,
) -> None:
    prefix = f"[{datetime.now().isoformat()}] [{level}]"
    if job_id:
        prefix += f" [job={job_id}]"
    lines = message.splitlines() or [""]
    entries: list[str] = []
    for line in lines:
        entry = f"{prefix} {line}\n"
        print(entry, end="")
        entries.append(entry)
    _append_log_entry("".join(entries), log_path=log_path)


def get_scene_robot_log_size(*, log_path: str | os.PathLike[str] | None = None) -> int:
    path = _resolve_log_path(log_path)
    return path.stat().st_size if path.exists() else 0


def read_scene_robot_log(
    offset: int = 0,
    limit: int = 65536,
    *,
    log_path: str | os.PathLike[str] | None = None,
) -> dict:
    path = _resolve_log_path(log_path)
    if not path.exists():
        return {"content": "", "next_offset": 0, "size": 0, "truncated": False}

    file_size = path.stat().st_size
    safe_offset = max(0, min(int(offset), file_size))
    safe_limit = max(1024, min(int(limit), 262144))

    with path.open("rb") as f:
        f.seek(safe_offset)
        chunk = f.read(safe_limit)
        next_offset = f.tell()

    return {
        "content": chunk.decode("utf-8", errors="ignore"),
        "next_offset": next_offset,
        "size": file_size,
        "truncated": next_offset < file_size,
    }


def _run_subprocess(
    cmd: list[str],
    *,
    label: str,
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    job_id: str | None = None,
    log_path: str | os.PathLike[str] | None = None,
) -> int:
    """Stream a subprocess' stdout/stderr line-by-line into the log file.

    Returns the child's exit code. Raises subprocess.TimeoutExpired if
    `timeout` (seconds) is exceeded. Does NOT raise on nonzero exit —
    callers decide how to react.
    """
    ts = datetime.now().isoformat()
    header = (
        f"[{ts}]"
        + (f" [job={job_id}]" if job_id else "")
        + f" === {label} ===\n"
        f"CMD: {' '.join(cmd)}\n"
        f"CWD: {cwd or os.getcwd()}\n"
        f"--- stream start ---\n"
    )
    print(header, end="")
    _append_log_entry(header, log_path=log_path)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or os.environ.copy(),
        cwd=str(cwd) if cwd else None,
        bufsize=1,
    )

    sel = selectors.DefaultSelector()
    if process.stdout is not None:
        sel.register(process.stdout, selectors.EVENT_READ, data="STDOUT")
    if process.stderr is not None:
        sel.register(process.stderr, selectors.EVENT_READ, data="STDERR")

    start_time = time.monotonic()
    last_activity = start_time
    heartbeat_interval = 30.0

    def _write_stream_line(stream_name: str, line: str) -> None:
        ts_line = datetime.now().isoformat()
        entry = f"[{ts_line}] [{stream_name}]"
        if job_id:
            entry += f" [job={job_id}]"
        entry += f" {line}"
        print(entry, end="")
        _append_log_entry(entry, log_path=log_path)

    while True:
        now = time.monotonic()
        if timeout is not None and now - start_time > timeout:
            process.kill()
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        events = sel.select(timeout=0.5)
        if not events:
            if process.poll() is not None:
                break
            if now - last_activity >= heartbeat_interval:
                heartbeat = (
                    f"[{datetime.now().isoformat()}] [HEARTBEAT]"
                    + (f" [job={job_id}]" if job_id else "")
                    + f" {label} still running ({int(now - start_time)}s)\n"
                )
                print(heartbeat, end="")
                _append_log_entry(heartbeat, log_path=log_path)
                last_activity = now
            continue

        for key, _ in events:
            stream_name = key.data
            stream = key.fileobj
            line = stream.readline()
            if line:
                last_activity = time.monotonic()
                _write_stream_line(stream_name, line)
            else:
                try:
                    sel.unregister(stream)
                except Exception:
                    pass
                stream.close()

        if process.poll() is not None and not sel.get_map():
            break

    return_code = process.wait()
    footer = (
        f"[{datetime.now().isoformat()}]"
        + (f" [job={job_id}]" if job_id else "")
        + f" --- stream end --- exit={return_code} label={label}\n"
    )
    print(footer, end="")
    _append_log_entry(footer, log_path=log_path)
    return return_code


def _build_collect_cmd(payload: dict[str, Any]) -> list[str]:
    python_bin = str(payload.get("python_bin") or ENV_ISAACLAB_PYTHON)
    script = str(payload.get("script") or SCENE_ROBOT_AUTO_GRASP_COLLECT_SCRIPT)

    cmd: list[str] = [python_bin, "-u", script]

    session_id = payload.get("session_id")
    run_id = payload.get("run_id")
    if session_id:
        cmd += ["--session", str(session_id)]
    if run_id:
        cmd += ["--run", str(run_id)]

    if payload.get("robot"):
        cmd += ["--robot", str(payload["robot"])]
    if payload.get("target"):
        cmd += ["--target", str(payload["target"])]
    if payload.get("support"):
        cmd += ["--support", str(payload["support"])]
    if payload.get("num_episodes") is not None:
        cmd += ["--num-episodes", str(int(payload["num_episodes"]))]

    if payload.get("headless", True):
        cmd += ["--headless"]
    if payload.get("wait_for_run_request") is False:
        cmd += ["--no-wait-for-run-request"]

    extra_args = payload.get("extra_args") or []
    if isinstance(extra_args, (list, tuple)):
        cmd.extend(str(part) for part in extra_args)
    return cmd


def run_scene_robot_collect(payload: dict[str, Any], job_id: str | None = None) -> dict:
    log_path = str(payload.get("log_path") or SCENE_ROBOT_LOG_PATH)
    cmd = _build_collect_cmd(payload)
    cwd = str(payload.get("cwd") or PROJECT_ROOT)

    log_scene_robot_event(
        f"Starting scene_robot collect: session={payload.get('session_id')} "
        f"run={payload.get('run_id')} robot={payload.get('robot')} "
        f"target={payload.get('target')} num_episodes={payload.get('num_episodes')}",
        job_id=job_id,
        log_path=log_path,
    )

    timeout = payload.get("timeout")
    timeout_value = float(timeout) if timeout is not None else None

    return_code = _run_subprocess(
        cmd,
        label="scene_robot_auto_grasp_collect",
        cwd=cwd,
        timeout=timeout_value,
        job_id=job_id,
        log_path=log_path,
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(returncode=return_code, cmd=cmd)

    return {
        "stage": "collect",
        "cmd": cmd,
        "cwd": cwd,
        "log_path": log_path,
    }


def start_scene_robot_collect_job(payload: dict[str, Any]) -> dict[str, object]:
    job_id = uuid.uuid4().hex
    log_path = str(payload.get("log_path") or SCENE_ROBOT_LOG_PATH)
    log_start_offset = get_scene_robot_log_size(log_path=log_path)

    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "stage": "collect",
            "status": "queued",
            "created_at": _utcnow_iso(),
            "updated_at": _utcnow_iso(),
            "error": None,
            "traceback": None,
            "artifacts": {},
            "log_path": log_path,
            "log_start_offset": log_start_offset,
            "payload": dict(payload or {}),
        }

    def _runner() -> None:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["updated_at"] = _utcnow_iso()

        log_scene_robot_event("Job queued -> running", job_id=job_id, log_path=log_path)
        try:
            artifacts = run_scene_robot_collect(payload, job_id=job_id)
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
                if not job:
                    return
                job["status"] = "succeeded"
                job["updated_at"] = _utcnow_iso()
                job["artifacts"] = artifacts
            log_scene_robot_event("Job completed successfully", job_id=job_id, log_path=log_path)
        except Exception as exc:  # noqa: BLE001 - skeleton: surface raw failure
            tb = traceback.format_exc()
            log_scene_robot_event(f"Job failed: {exc}", job_id=job_id, level="ERROR", log_path=log_path)
            log_scene_robot_event(tb.rstrip(), job_id=job_id, level="TRACE", log_path=log_path)
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
                if not job:
                    return
                job["status"] = "failed"
                job["updated_at"] = _utcnow_iso()
                job["error"] = str(exc)
                job["traceback"] = tb

    threading.Thread(target=_runner, daemon=True).start()
    return {
        "job_id": job_id,
        "log_start_offset": log_start_offset,
        "log_path": log_path,
    }


def get_scene_robot_job_status(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        base = _JOBS.get(job_id)
        if not base:
            return None
        job = dict(base)

    payload = job.get("payload") or {}
    if isinstance(payload.get("session_id"), str) and payload.get("session_id"):
        job["session_id"] = payload["session_id"]
    if isinstance(payload.get("run_id"), str) and payload.get("run_id"):
        job["run_id"] = payload["run_id"]
    job.pop("payload", None)
    return job
