#!/usr/bin/env python3
"""One-shot migration for runtime/sessions/* after the run/session merge.

Before:
    runtime/sessions/<sid>/
        session.json
        current_run.txt
        agent_state.json   (state["runs"][rid]["real2sim"] = {...})
        runs/<rid>/
            uploads/, scene_graph/, real2sim/, scene_service/, logs/, ...

After:
    runtime/sessions/<sid>/
        session.json
        agent_state.json   (state["real2sim"] = {...} flat)
        uploads/, scene_graph/, real2sim/, scene_service/, logs/, ...

For sessions that have multiple runs (none exist in the current snapshot,
but defensively): we pick the run pointed to by current_run.txt and discard
the others, printing a warning. If current_run.txt is absent, we pick the
most-recently-modified run.

Idempotent: if a session has no `runs/` subdir, it's left alone.

Usage:
    python tools/migrate_sessions_flatten_runs.py [--dry-run] [<sessions_dir>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _pick_keep_run(session_dir: Path) -> Path | None:
    runs_dir = session_dir / "runs"
    if not runs_dir.is_dir():
        return None
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        return None
    cur_file = session_dir / "current_run.txt"
    keep_name: str | None = None
    if cur_file.exists():
        try:
            keep_name = cur_file.read_text(encoding="utf-8").strip()
        except Exception:
            keep_name = None
    keep = None
    if keep_name:
        keep = next((r for r in run_dirs if r.name == keep_name), None)
    if keep is None:
        run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        keep = run_dirs[0]
    return keep


def _flatten_agent_state(session_dir: Path, keep_run_name: str, dry_run: bool) -> None:
    state_path = session_dir / "agent_state.json"
    if not state_path.exists():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  ! could not read {state_path}: {exc}", file=sys.stderr)
        return
    if not isinstance(state, dict):
        return

    runs = state.get("runs")
    target = None
    if isinstance(runs, dict):
        target = runs.get(keep_run_name) or next(iter(runs.values()), None)
    if isinstance(target, dict):
        for key in ("real2sim", "scene_robot", "scene_generation"):
            if key in target and key not in state:
                state[key] = target[key]

    state.pop("runs", None)
    state.pop("current_run_id", None)
    state.pop("latest_real2sim_run_id", None)
    state.pop("latest_scene_robot_run_id", None)
    state.pop("latest_scene_generation_run_id", None)

    if dry_run:
        print(f"  [dry-run] would rewrite {state_path}")
        return
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _move_run_contents_up(run_dir: Path, session_dir: Path, dry_run: bool) -> None:
    """Move every immediate child of run_dir into session_dir.

    Skips the `run.json` metadata file (no longer meaningful at session
    granularity). Collisions with already-present session files (e.g. a
    session-level placeholder) bail out loudly so we don't silently lose
    data.
    """
    for entry in sorted(run_dir.iterdir()):
        if entry.name == "run.json":
            continue
        dest = session_dir / entry.name
        if dest.exists():
            print(
                f"  ! collision: {dest} already exists; skipping {entry}",
                file=sys.stderr,
            )
            continue
        if dry_run:
            print(f"  [dry-run] would move {entry} -> {dest}")
            continue
        shutil.move(str(entry), str(dest))


def migrate_session(session_dir: Path, dry_run: bool) -> None:
    print(f"session: {session_dir.name}")
    keep = _pick_keep_run(session_dir)
    if keep is None:
        print("  no runs/ subdir — already flat or empty")
        return
    runs_dir = session_dir / "runs"
    other_runs = [r for r in runs_dir.iterdir() if r.is_dir() and r.name != keep.name]
    if other_runs:
        print(
            f"  ! multiple runs found; keeping {keep.name} and DISCARDING: "
            + ", ".join(r.name for r in other_runs),
            file=sys.stderr,
        )

    _flatten_agent_state(session_dir, keep.name, dry_run)
    _move_run_contents_up(keep, session_dir, dry_run)

    if dry_run:
        print(f"  [dry-run] would rm -rf {runs_dir}")
        print(f"  [dry-run] would rm {session_dir / 'current_run.txt'} (if present)")
        return
    shutil.rmtree(runs_dir, ignore_errors=False)
    cur_file = session_dir / "current_run.txt"
    if cur_file.exists():
        cur_file.unlink()
    print(f"  ok (kept {keep.name})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sessions_dir",
        nargs="?",
        default="runtime/sessions",
        help="Path to runtime/sessions (default: runtime/sessions)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.sessions_dir).resolve()
    if not root.is_dir():
        print(f"sessions dir not found: {root}", file=sys.stderr)
        return 1

    for ses in sorted(root.iterdir()):
        if not ses.is_dir():
            continue
        migrate_session(ses, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
