"""Per-session cooperative interrupt flag.

`/agent/interrupt` sets a flag here; the loop / plan executors poll it
between turns/steps and bail out cleanly so the operator can interact
again without waiting for (or being blocked by) the agent. Long pipeline
work runs as background subprocess jobs, so the interrupt route ALSO
kills those jobs (see pipeline_service.cancel_real2sim_jobs_for_session /
scene_robot_service.cancel_scene_robot_jobs_for_session); this module
only handles the in-process "stop the agent loop" half.

Thread-safe: `/agent/interrupt` runs on a different request thread than
the in-flight agent loop.
"""

from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_INTERRUPTS: dict[str, float] = {}


def _key(session_id: str | None) -> str:
    return session_id or "-"


def request_interrupt(session_id: str | None) -> None:
    """Mark this session as interrupt-requested (set by /agent/interrupt)."""
    with _LOCK:
        _INTERRUPTS[_key(session_id)] = time.time()


def interrupt_pending(session_id: str | None) -> bool:
    """Non-destructive check (use between cheap steps)."""
    with _LOCK:
        return _key(session_id) in _INTERRUPTS


def consume_interrupt(session_id: str | None) -> bool:
    """Check AND clear in one atomic op (use at a turn boundary so a
    single interrupt fires exactly once)."""
    with _LOCK:
        return _INTERRUPTS.pop(_key(session_id), None) is not None


def clear_interrupt(session_id: str | None) -> None:
    """Drop a stale flag (e.g. at the very start of a fresh message so a
    never-consumed interrupt from a prior turn can't kill the new one)."""
    with _LOCK:
        _INTERRUPTS.pop(_key(session_id), None)
