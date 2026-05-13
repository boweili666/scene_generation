"""Background heartbeat for long-running pipeline jobs.

When real2sim or any scene_robot stage starts (via the existing
`_run_subprocess`-backed wrappers), this module spawns a daemon thread
that periodically writes a status line into the session's agent
history. The user sees these inline in the chat as `🔄 …` updates,
even when they walk away and come back later.

Design notes:

- One heartbeat row per (session, run, job_id). We UPSERT into
  history rather than appending: every tick replaces the previous
  heartbeat for the same job, so a one-hour train job doesn't bury
  real conversation under 20 status lines.
- Heartbeats stop firing as soon as the live job dict reports a
  non-running status (succeeded / failed / cancelled). The success
  / failure handlers in `agent_service` write their own dedicated
  history line, so heartbeats don't need to mirror the terminal state.
- Best-effort writes. If the runtime context is gone (web restart) or
  agent_state.json is mid-write, we just skip this tick — the next one
  will catch up. There IS a small race with the main agent flow (both
  read-modify-write the same state file), but at a 3-minute interval
  it's vanishingly small in practice; if it ever causes lost messages,
  add a per-session lock around `_load → mutate → _save`.
"""
from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


HEARTBEAT_INTERVAL_SECONDS = int(os.environ.get("AGENT_HEARTBEAT_INTERVAL_SECONDS", "180"))
HEARTBEAT_INITIAL_DELAY_SECONDS = int(os.environ.get("AGENT_HEARTBEAT_INITIAL_DELAY_SECONDS", "60"))


# Mirror of the small log-extraction helpers in agent_loop. Kept local so
# pipeline_service / scene_robot_service can import this module without a
# transitive dependency on the agent loop (which depends on them).
_TQDM_NOISE = re.compile(r"\b(it/s|loading weights|materializing param|\d+\s*%\|)", re.IGNORECASE)
_PROGRESS_R2S_ATTEMPT = re.compile(r"\battempt=(\d+)\b")
_PROGRESS_R2S_MASKS = re.compile(r"selected=(\d+)\s+total=(\d+)")
_PROGRESS_EPISODE = re.compile(r"[Ee]pisode\s+(\d+)\s*/\s*(\d+)")
_PROGRESS_STEP = re.compile(r"\bstep[:= ]\s*(\d+)\s*/\s*(\d+)\b")


def _read_log_tail(log_path: Any, max_lines: int = 40, tail_kb: int = 64) -> str:
    if not log_path:
        return ""
    try:
        path = Path(str(log_path))
        if not path.exists() or not path.is_file():
            return ""
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - tail_kb * 1024))
            blob = f.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""
    lines = blob.splitlines()
    filtered = [ln for ln in lines if ln.strip() and not _TQDM_NOISE.search(ln)]
    return "\n".join(filtered[-max_lines:])


def _extract_progress(log_tail: str, kind: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not log_tail:
        return out
    if kind == "real2sim":
        attempts = _PROGRESS_R2S_ATTEMPT.findall(log_tail)
        if attempts:
            out["last_attempt"] = int(attempts[-1])
        masks = _PROGRESS_R2S_MASKS.findall(log_tail)
        if masks:
            sel, tot = masks[-1]
            out["mask_selection"] = {"selected": int(sel), "total": int(tot)}
    elif kind in {"scene_robot_collect", "scene_robot_eval"}:
        ep = _PROGRESS_EPISODE.findall(log_tail)
        if ep:
            cur, tot = ep[-1]
            out["episode"] = {"current": int(cur), "total": int(tot)}
    elif kind == "scene_robot_train":
        step = _PROGRESS_STEP.findall(log_tail)
        if step:
            cur, tot = step[-1]
            out["step"] = {"current": int(cur), "total": int(tot)}
    return out


def _format_elapsed(created_at: Any) -> str:
    if not created_at:
        return ""
    try:
        started = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        elapsed = int((datetime.now(timezone.utc) - started).total_seconds())
    except Exception:  # noqa: BLE001
        return ""
    if elapsed < 60:
        return f"{elapsed}s"
    mins, secs = divmod(elapsed, 60)
    if mins < 60:
        return f"{mins}m{secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h{mins:02d}m"


def _kind_label(kind: str) -> str:
    return {
        "real2sim": "Real2Sim",
        "scene_robot_collect": "scene_robot collect",
        "scene_robot_train": "scene_robot train",
        "scene_robot_eval": "scene_robot eval",
    }.get(kind, kind)


def format_heartbeat(kind: str, job_id: str, job: dict[str, Any], log_path: Any) -> str:
    """Compose the one-line chat message for a single heartbeat tick."""
    label = _kind_label(kind)
    bits: list[str] = [f"🔄 {label} heartbeat"]

    elapsed = _format_elapsed(job.get("created_at"))
    if elapsed:
        bits.append(f"elapsed {elapsed}")

    log_tail = _read_log_tail(log_path)
    progress = _extract_progress(log_tail, kind)

    if "episode" in progress:
        ep = progress["episode"]
        bits.append(f"episode {ep['current']}/{ep['total']}")
    if "step" in progress:
        st = progress["step"]
        bits.append(f"step {st['current']}/{st['total']}")
    if "mask_selection" in progress:
        ms = progress["mask_selection"]
        bits.append(f"masks {ms['selected']}/{ms['total']}")
    if "last_attempt" in progress:
        bits.append(f"attempt {progress['last_attempt']}")

    if log_tail and "Traceback (most recent call last):" in log_tail:
        bits.append("⚠ traceback detected in log")

    short_id = (job_id or "")[:8]
    if short_id:
        bits.append(f"job {short_id}")

    return " — ".join(bits) + "."


def upsert_heartbeat_in_history(
    state: dict[str, Any],
    *,
    kind: str,
    job_id: str,
    content: str,
) -> None:
    """Insert or replace the heartbeat-tagged entry for a specific job_id.

    Heartbeats are tagged with `kind="heartbeat"` and `job_id=...` in the
    history entry so we can find and replace them in-place. This keeps the
    history compact regardless of how long the job runs."""
    history = state.setdefault("history", [])
    if not isinstance(history, list):
        history = []
        state["history"] = history
    entry = {
        "role": "assistant",
        "kind": "heartbeat",
        "stage": kind,
        "job_id": job_id,
        "content": content,
        "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    for i in range(len(history) - 1, -1, -1):
        existing = history[i]
        if (
            isinstance(existing, dict)
            and existing.get("kind") == "heartbeat"
            and existing.get("job_id") == job_id
        ):
            history[i] = entry
            return
    history.append(entry)
    if len(history) > 20:
        del history[: len(history) - 20]


def spawn(
    *,
    kind: str,
    job_id: str,
    session_id: Optional[str],
    run_id: Optional[str],
    log_path: str,
    get_status: Callable[[str], Optional[dict[str, Any]]],
    interval_seconds: int = HEARTBEAT_INTERVAL_SECONDS,
    initial_delay_seconds: int = HEARTBEAT_INITIAL_DELAY_SECONDS,
) -> Optional[threading.Thread]:
    """Start the heartbeat daemon thread for a single job.

    Skips spawning when session_id/run_id are missing (no session to
    write to) or when interval_seconds <= 0 (heartbeats disabled).
    """
    if not session_id or not run_id:
        return None
    if interval_seconds <= 0:
        return None

    def _runner() -> None:
        # Avoid noisy heartbeats while Isaac is still booting.
        time.sleep(max(0, initial_delay_seconds))
        # Local imports to avoid a circular dependency at module load.
        from .agent_service import _load_agent_state, _save_agent_state
        from .runtime_context import resolve_runtime_context

        while True:
            try:
                job = get_status(job_id)
            except Exception:  # noqa: BLE001
                job = None
            status = ""
            if isinstance(job, dict):
                status = str(job.get("status") or "")
            # Stop firing once the job is no longer pending. The
            # success/failure paths write their own terminal message.
            if status not in ("queued", "running"):
                return

            try:
                ctx = resolve_runtime_context(session_id=session_id, run_id=run_id)
                if ctx is not None:
                    state = _load_agent_state(ctx)
                    content = format_heartbeat(kind, job_id, job or {}, log_path)
                    upsert_heartbeat_in_history(
                        state, kind=kind, job_id=job_id, content=content
                    )
                    _save_agent_state(ctx, state)
            except Exception:  # noqa: BLE001 - heartbeats are best-effort
                pass

            time.sleep(max(1, interval_seconds))

    t = threading.Thread(
        target=_runner,
        daemon=True,
        name=f"heartbeat-{kind}-{(job_id or '')[:8]}",
    )
    t.start()
    return t
