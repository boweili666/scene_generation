"""Lifecycle helper for the Isaac-backed scene service.

The scene service (`app.backend.services.scene_service`) and the
scene_robot collect script both spawn their own SimulationApp and each
needs ~5-8 GB of GPU memory. On a single-GPU dev machine they cannot
coexist, so the agent flips the service ON for `generate_scene` and
OFF for `run_scene_robot_collect`.

This module is the only place that knows how to start, stop, and probe
the scene service. It writes a PID file under `runtime/scene_service.pid`
so multiple agent turns (and the web process across restarts) can find
the running instance.

Process model:
- start() spawns the service in a new session group via Popen
  (start_new_session=True) so we can SIGTERM the whole tree on stop().
- start() blocks until /health responds (default 120s for Isaac cold-start)
  or the child exits early; on timeout it kills the child and surfaces a
  log tail via RuntimeError.
- stop() reads the PID, SIGTERM the process group, waits 10s, SIGKILLs
  if still alive, then sleeps a couple of seconds for the GPU to release.
- is_alive() does a short HTTP /health probe — that's the source of truth.
  PID files can lie (PID reuse, stale files); the health probe cannot.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import NamedTuple, Optional

import requests

from ..config import LOGS_DIR, PROJECT_ROOT, RUNTIME_DIR, SCENE_SERVICE_URL


_PID_FILE = Path(RUNTIME_DIR) / "scene_service.pid"
_LOG_FILE = Path(LOGS_DIR) / "scene_service.log"
_HEALTH_TIMEOUT_SECONDS = 0.7
DEFAULT_START_TIMEOUT_SECONDS = float(os.environ.get("SCENE_HEALTH_TIMEOUT_SECONDS", "120"))
_STOP_GRACE_SECONDS = 10.0
_GPU_RELEASE_GRACE_SECONDS = 2.0
_LOG_TAIL_LINES = 25
_LOG_TAIL_BYTES = 16384


class ServiceStatus(NamedTuple):
    alive: bool
    pid: Optional[int]
    url: str


def _pid_file() -> Path:
    return _PID_FILE


def _log_file() -> Path:
    return _LOG_FILE


def _health_url() -> str:
    return f"{SCENE_SERVICE_URL.rstrip('/')}/health"


def _read_pid() -> Optional[int]:
    try:
        text = _pid_file().read_text(encoding="utf-8").strip()
    except (OSError, FileNotFoundError):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _write_pid(pid: int) -> None:
    pf = _pid_file()
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(str(pid), encoding="utf-8")


def _delete_pid() -> None:
    try:
        _pid_file().unlink()
    except FileNotFoundError:
        pass


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


def _check_health() -> bool:
    try:
        r = requests.get(_health_url(), timeout=_HEALTH_TIMEOUT_SECONDS)
        return r.status_code == 200
    except Exception:
        return False


def _tail_log(n: int = _LOG_TAIL_LINES) -> str:
    log = _log_file()
    if not log.exists():
        return ""
    try:
        size = log.stat().st_size
        with log.open("rb") as f:
            f.seek(max(0, size - _LOG_TAIL_BYTES))
            blob = f.read().decode("utf-8", errors="replace")
        lines = blob.splitlines()[-n:]
        return "\n".join(lines)
    except Exception:
        return ""


def _kill_pgid(pid: int, sig: int) -> None:
    """SIGTERM/SIGKILL the whole process group (start_new_session=True
    means the child is its own pgid leader). Falls back to single-pid
    kill if pgid lookup fails."""
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def status() -> ServiceStatus:
    """Snapshot for diagnostics / inspect_state. Cleans up stale PID file."""
    pid = _read_pid()
    if pid is not None and not _process_alive(pid):
        _delete_pid()
        pid = None
    return ServiceStatus(alive=_check_health(), pid=pid, url=SCENE_SERVICE_URL)


def is_alive() -> bool:
    """Single source of truth: does /health respond?"""
    return _check_health()


def start(timeout: float = DEFAULT_START_TIMEOUT_SECONDS) -> ServiceStatus:
    """Idempotent start.

    Returns immediately if /health already responds. Otherwise spawns
    the scene service in a new session group, polls /health until
    healthy, and returns. Raises RuntimeError on early child exit or
    health-check timeout, with the last log lines included in the
    message.
    """
    if _check_health():
        return status()

    parsed = urllib.parse.urlparse(SCENE_SERVICE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 8001)

    # SCENE_PYTHON env var matches scripts/start_with_scene_service.sh.
    # Falls back to whatever python is currently running the web backend
    # (which only works if isaacsim is installed in that env).
    python_bin = os.environ.get("SCENE_PYTHON") or sys.executable

    cmd = [
        python_bin,
        "-m",
        "app.backend.services.scene_service",
        "--host", host,
        "--port", port,
        "--headless",
    ]

    log_path = _log_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("a", buffering=1, encoding="utf-8")
    log_fh.write(
        f"\n=== Scene service start (agent-managed) at "
        f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} via {python_bin} ===\n"
    )

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(PROJECT_ROOT),
        )
    except (OSError, FileNotFoundError) as exc:
        log_fh.close()
        raise RuntimeError(
            f"Failed to spawn scene service via '{python_bin}': {exc}"
        ) from exc

    _write_pid(proc.pid)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = _tail_log()
            _delete_pid()
            try:
                log_fh.close()
            except Exception:
                pass
            raise RuntimeError(
                f"scene_service exited before becoming healthy "
                f"(exit={proc.returncode}). Last log lines:\n{tail}"
            )
        if _check_health():
            return status()
        time.sleep(0.5)

    # Timed out — kill the half-started service so we don't leak.
    _kill_pgid(proc.pid, signal.SIGTERM)
    time.sleep(0.5)
    if proc.poll() is None:
        _kill_pgid(proc.pid, signal.SIGKILL)
    _delete_pid()
    try:
        log_fh.close()
    except Exception:
        pass
    tail = _tail_log()
    raise RuntimeError(
        f"scene_service did not pass health check within {timeout:.0f}s. "
        f"Last log lines:\n{tail}"
    )


def stop(
    timeout: float = _STOP_GRACE_SECONDS,
    gpu_release_grace: float = _GPU_RELEASE_GRACE_SECONDS,
) -> ServiceStatus:
    """Idempotent stop.

    No-op if the service is not running. Otherwise SIGTERM the process
    group, wait up to `timeout` seconds, SIGKILL if still alive, then
    sleep `gpu_release_grace` seconds so the GPU can release before the
    caller (typically scene_robot collect) starts its own SimulationApp.
    """
    pid = _read_pid()
    # Even without a tracked PID, if /health says alive someone else
    # started it manually — we still try to be polite, but only kill
    # what we tracked. If health is alive but we have no PID, leave it.
    if pid is None:
        _delete_pid()
        if gpu_release_grace > 0 and _check_health():
            # Orphan we don't own; nothing to kill, just report status.
            pass
        return status()

    if not _process_alive(pid):
        _delete_pid()
        return status()

    _kill_pgid(pid, signal.SIGTERM)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            break
        time.sleep(0.2)
    else:
        _kill_pgid(pid, signal.SIGKILL)
        # Brief pause for kernel to reap.
        for _ in range(15):
            if not _process_alive(pid):
                break
            time.sleep(0.2)

    _delete_pid()

    if gpu_release_grace > 0:
        time.sleep(gpu_release_grace)

    return status()
