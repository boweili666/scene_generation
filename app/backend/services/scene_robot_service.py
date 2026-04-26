"""scene_robot pipeline service.

Mirrors `pipeline_service.py` in shape but launches scene_robot scripts
(which live in the `env_isaaclab` / `lerobot` conda envs, not the web
backend's env). Integration is loose: subprocess + filesystem artifacts.

Stages:
  - collect: scene_auto_grasp_collect.py (env_isaaclab)
  - convert: tools/convert_hdf5_to_lerobot.py (lerobot env, fast)
  - train:   lerobot-train binary (lerobot env, hours)
  - eval:    scene_robot/scripts/collect/scene_eval_policy.py (env_isaaclab)

All four share `_run_subprocess` + `_JOBS` so `/scene_robot/status/<id>`
works regardless of stage. Each stage has its own per-run log file so
streaming doesn't interleave.
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
    CONVERT_HDF5_SCRIPT,
    ENV_ISAACLAB_PYTHON,
    LEROBOT_DATASETS_DIR,
    LEROBOT_PYTHON,
    LEROBOT_TRAIN_BIN,
    OUTPUTS_EVAL_DIR,
    OUTPUTS_TRAIN_DIR,
    PROJECT_ROOT,
    SCENE_ROBOT_AUTO_GRASP_COLLECT_SCRIPT,
    SCENE_ROBOT_CONVERT_LOG_PATH,
    SCENE_ROBOT_EVAL_LOG_PATH,
    SCENE_ROBOT_EVAL_SCRIPT,
    SCENE_ROBOT_LOG_PATH,
    SCENE_ROBOT_TRAIN_LOG_PATH,
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


def _start_stage_job(
    *,
    stage: str,
    payload: dict[str, Any],
    runner: Any,
    log_path: str,
    intro_message: str,
) -> dict[str, object]:
    """Generic skeleton: dict-tracked job + bg thread + log streaming.

    `runner` is a callable taking (payload, job_id) -> dict (artifacts).
    """
    job_id = uuid.uuid4().hex
    log_start_offset = get_scene_robot_log_size(log_path=log_path)

    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "stage": stage,
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

    def _bg_runner() -> None:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["updated_at"] = _utcnow_iso()

        log_scene_robot_event(intro_message, job_id=job_id, log_path=log_path)
        try:
            artifacts = runner(payload, job_id=job_id)
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

    threading.Thread(target=_bg_runner, daemon=True).start()
    return {
        "job_id": job_id,
        "stage": stage,
        "log_start_offset": log_start_offset,
        "log_path": log_path,
    }


def start_scene_robot_collect_job(payload: dict[str, Any]) -> dict[str, object]:
    log_path = str(payload.get("log_path") or SCENE_ROBOT_LOG_PATH)
    intro = (
        f"Starting scene_robot collect: session={payload.get('session_id')} "
        f"run={payload.get('run_id')} robot={payload.get('robot')} "
        f"target={payload.get('target')} num_episodes={payload.get('num_episodes')}"
    )
    return _start_stage_job(
        stage="collect",
        payload=payload,
        runner=run_scene_robot_collect,
        log_path=log_path,
        intro_message=intro,
    )


# --------- Convert (HDF5 -> LeRobotDataset) ---------


def _build_convert_cmd(payload: dict[str, Any]) -> list[str]:
    python_bin = str(payload.get("python_bin") or LEROBOT_PYTHON)
    script = str(payload.get("script") or CONVERT_HDF5_SCRIPT)
    cmd: list[str] = [python_bin, "-u", script]
    cmd += ["--hdf5", str(payload["hdf5"])]
    cmd += ["--repo-id", str(payload["repo_id"])]
    cmd += ["--output-root", str(payload["output_root"])]
    if payload.get("task"):
        cmd += ["--task", str(payload["task"])]
    if payload.get("fps") is not None:
        cmd += ["--fps", str(payload["fps"])]
    if payload.get("max_episodes") is not None:
        cmd += ["--max-episodes", str(int(payload["max_episodes"]))]
    if payload.get("include_failed"):
        cmd += ["--include-failed"]
    if payload.get("no_videos"):
        cmd += ["--no-videos"]
    if payload.get("overwrite"):
        cmd += ["--overwrite"]
    extra_args = payload.get("extra_args") or []
    if isinstance(extra_args, (list, tuple)):
        cmd.extend(str(part) for part in extra_args)
    return cmd


def run_scene_robot_convert(payload: dict[str, Any], job_id: str | None = None) -> dict:
    log_path = str(payload.get("log_path") or SCENE_ROBOT_CONVERT_LOG_PATH)
    cmd = _build_convert_cmd(payload)
    cwd = str(payload.get("cwd") or PROJECT_ROOT)
    timeout = payload.get("timeout")
    timeout_value = float(timeout) if timeout is not None else None
    return_code = _run_subprocess(
        cmd,
        label="scene_robot_convert_hdf5",
        cwd=cwd,
        timeout=timeout_value,
        job_id=job_id,
        log_path=log_path,
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(returncode=return_code, cmd=cmd)
    return {
        "stage": "convert",
        "cmd": cmd,
        "cwd": cwd,
        "log_path": log_path,
        "repo_id": payload.get("repo_id"),
        "output_root": str(payload.get("output_root", "")),
    }


def start_scene_robot_convert_job(payload: dict[str, Any]) -> dict[str, object]:
    log_path = str(payload.get("log_path") or SCENE_ROBOT_CONVERT_LOG_PATH)
    intro = (
        f"Starting scene_robot convert: hdf5={payload.get('hdf5')} "
        f"repo_id={payload.get('repo_id')} output_root={payload.get('output_root')}"
    )
    return _start_stage_job(
        stage="convert",
        payload=payload,
        runner=run_scene_robot_convert,
        log_path=log_path,
        intro_message=intro,
    )


# --------- Train (lerobot-train) ---------


def _build_train_cmd(payload: dict[str, Any]) -> list[str]:
    train_bin = str(payload.get("train_bin") or LEROBOT_TRAIN_BIN)
    cmd: list[str] = [train_bin]
    cmd += [f"--dataset.repo_id={payload['repo_id']}"]
    cmd += [f"--dataset.root={payload['dataset_root']}"]
    cmd += [f"--policy.type={payload.get('policy_type', 'diffusion')}"]
    cmd += [f"--policy.device={payload.get('device', 'cuda')}"]
    cmd += ["--policy.push_to_hub=false"]
    cmd += [f"--output_dir={payload['output_dir']}"]
    if payload.get("steps") is not None:
        cmd += [f"--steps={int(payload['steps'])}"]
    if payload.get("batch_size") is not None:
        cmd += [f"--batch_size={int(payload['batch_size'])}"]
    if payload.get("save_freq") is not None:
        cmd += [f"--save_freq={int(payload['save_freq'])}"]
    extra_args = payload.get("extra_args") or []
    if isinstance(extra_args, (list, tuple)):
        cmd.extend(str(part) for part in extra_args)
    return cmd


def run_scene_robot_train(payload: dict[str, Any], job_id: str | None = None) -> dict:
    log_path = str(payload.get("log_path") or SCENE_ROBOT_TRAIN_LOG_PATH)
    cmd = _build_train_cmd(payload)
    cwd = str(payload.get("cwd") or PROJECT_ROOT)
    timeout = payload.get("timeout")
    timeout_value = float(timeout) if timeout is not None else None
    return_code = _run_subprocess(
        cmd,
        label="scene_robot_lerobot_train",
        cwd=cwd,
        timeout=timeout_value,
        job_id=job_id,
        log_path=log_path,
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(returncode=return_code, cmd=cmd)
    output_dir = str(payload.get("output_dir", ""))
    return {
        "stage": "train",
        "cmd": cmd,
        "cwd": cwd,
        "log_path": log_path,
        "repo_id": payload.get("repo_id"),
        "output_dir": output_dir,
        "checkpoint_dir": (
            str(Path(output_dir) / "checkpoints" / "last" / "pretrained_model") if output_dir else None
        ),
    }


def start_scene_robot_train_job(payload: dict[str, Any]) -> dict[str, object]:
    log_path = str(payload.get("log_path") or SCENE_ROBOT_TRAIN_LOG_PATH)
    intro = (
        f"Starting scene_robot train: repo_id={payload.get('repo_id')} "
        f"policy={payload.get('policy_type', 'diffusion')} "
        f"steps={payload.get('steps')} output_dir={payload.get('output_dir')}"
    )
    return _start_stage_job(
        stage="train",
        payload=payload,
        runner=run_scene_robot_train,
        log_path=log_path,
        intro_message=intro,
    )


# --------- Eval (closed-loop sim rollout) ---------


def _build_eval_cmd(payload: dict[str, Any]) -> list[str]:
    python_bin = str(payload.get("python_bin") or ENV_ISAACLAB_PYTHON)
    script = str(payload.get("script") or SCENE_ROBOT_EVAL_SCRIPT)
    cmd: list[str] = [python_bin, "-u", script]
    if payload.get("session_id"):
        cmd += ["--session", str(payload["session_id"])]
    if payload.get("run_id"):
        cmd += ["--run", str(payload["run_id"])]
    if payload.get("robot"):
        cmd += ["--robot", str(payload["robot"])]
    cmd += ["--target", str(payload["target"])]
    cmd += ["--checkpoint", str(payload["checkpoint"])]
    cmd += ["--dataset-root", str(payload["dataset_root"])]
    if payload.get("num_episodes") is not None:
        cmd += ["--num-episodes", str(int(payload["num_episodes"]))]
    if payload.get("max_steps_per_episode") is not None:
        cmd += ["--max-steps-per-episode", str(int(payload["max_steps_per_episode"]))]
    if payload.get("record_dir"):
        cmd += ["--record-dir", str(payload["record_dir"])]
    if payload.get("headless", True):
        cmd += ["--headless"]
    extra_args = payload.get("extra_args") or []
    if isinstance(extra_args, (list, tuple)):
        cmd.extend(str(part) for part in extra_args)
    return cmd


def run_scene_robot_eval(payload: dict[str, Any], job_id: str | None = None) -> dict:
    log_path = str(payload.get("log_path") or SCENE_ROBOT_EVAL_LOG_PATH)
    cmd = _build_eval_cmd(payload)
    cwd = str(payload.get("cwd") or PROJECT_ROOT)
    timeout = payload.get("timeout")
    timeout_value = float(timeout) if timeout is not None else None
    return_code = _run_subprocess(
        cmd,
        label="scene_robot_eval_policy",
        cwd=cwd,
        timeout=timeout_value,
        job_id=job_id,
        log_path=log_path,
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(returncode=return_code, cmd=cmd)
    return {
        "stage": "eval",
        "cmd": cmd,
        "cwd": cwd,
        "log_path": log_path,
        "checkpoint": str(payload.get("checkpoint", "")),
        "record_dir": str(payload.get("record_dir", "")) if payload.get("record_dir") else None,
    }


def start_scene_robot_eval_job(payload: dict[str, Any]) -> dict[str, object]:
    log_path = str(payload.get("log_path") or SCENE_ROBOT_EVAL_LOG_PATH)
    intro = (
        f"Starting scene_robot eval: target={payload.get('target')} "
        f"checkpoint={payload.get('checkpoint')} "
        f"num_episodes={payload.get('num_episodes')}"
    )
    return _start_stage_job(
        stage="eval",
        payload=payload,
        runner=run_scene_robot_eval,
        log_path=log_path,
        intro_message=intro,
    )


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
