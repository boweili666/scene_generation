"""Read-only dashboard data for the right-panel Robot tab.

Aggregates everything the UI needs into one JSON payload so the panel
can poll a single endpoint:

  - collect: latest scene_robot collect job + HDF5 episode breakdown
  - train:   latest train output + checkpoints + parsed step/loss
  - eval:    latest eval record dir + per-episode mp4 URLs

All file reads are best-effort: a missing dataset / output dir just
yields empty fields rather than erroring out, so the UI can render an
"idle" card cleanly. h5py reads use locking=False so a still-running
collect job that holds the HDF5 doesn't block our read.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import (
    DATASETS_DIR,
    LEROBOT_DATASETS_DIR,
    OUTPUTS_EVAL_DIR,
    OUTPUTS_TRAIN_DIR,
    RUNTIME_DIR,
)
from .runtime_context import RuntimeContext
from .scene_robot_service import get_scene_robot_job_status


# --------- HDF5 collect summary ---------


def _open_hdf5_readonly(path: Path):
    """Open an HDF5 file even if another process is currently writing to it.
    Returns the open File or None on error."""
    try:
        import h5py  # type: ignore
    except ImportError:
        return None
    try:
        # locking=False bypasses HDF5's POSIX locking so we can read while
        # the collect process still holds the file open. swmr=True would be
        # cleaner but only works if the writer was opened in SWMR mode,
        # which our collect script isn't.
        return h5py.File(str(path), "r", locking=False)
    except Exception:
        # Fall back to standard open in case the file isn't actually being
        # held — older h5py versions also don't accept locking kw.
        try:
            return h5py.File(str(path), "r")
        except Exception:
            return None


def summarize_hdf5(path: Path) -> dict[str, Any]:
    """Return episode-level summary of a collect HDF5 file.

    Output:
      {
        "exists": bool,
        "path": "<abs path>",
        "size_bytes": int,
        "mtime_iso": "...",
        "num_demos": int,
        "episodes": [{"name": "demo_0", "steps": 53, "success": True}, ...],
        "error": str | None,   # populated on read failure
      }
    """
    out: dict[str, Any] = {"exists": False, "path": str(path)}
    if not path.exists():
        return out
    try:
        st = path.stat()
        out.update(
            {
                "exists": True,
                "size_bytes": int(st.st_size),
                "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    except OSError as exc:
        out["error"] = f"stat failed: {exc}"
        return out

    f = _open_hdf5_readonly(path)
    if f is None:
        out["error"] = "could not open hdf5 (file locked or h5py unavailable)"
        return out

    try:
        episodes: list[dict[str, Any]] = []
        if "data" in f:
            data_group = f["data"]
            # Sort demo_N by N when possible; fall back to lexicographic.
            def _sort_key(name: str) -> Any:
                m = re.match(r"demo_(\d+)", name)
                return int(m.group(1)) if m else (10**9, name)
            for name in sorted(list(data_group.keys()), key=_sort_key):
                ep = data_group[name]
                attrs = dict(ep.attrs) if hasattr(ep, "attrs") else {}
                steps: Optional[int] = None
                # Common: actions[N, A] is the per-step action sequence.
                try:
                    if "actions" in ep:
                        steps = int(ep["actions"].shape[0])
                    elif "obs" in ep:
                        # Take the first leaf in obs and use its first dim.
                        obs = ep["obs"]
                        for k in obs.keys():
                            arr = obs[k]
                            if hasattr(arr, "shape") and arr.shape:
                                steps = int(arr.shape[0])
                                break
                except Exception:
                    pass
                if steps is None and "num_samples" in attrs:
                    try:
                        steps = int(attrs["num_samples"])
                    except (TypeError, ValueError):
                        steps = None
                success = attrs.get("success")
                if success is not None:
                    success = bool(success)
                episodes.append({"name": str(name), "steps": steps, "success": success})
        out["num_demos"] = len(episodes)
        out["episodes"] = episodes
    except Exception as exc:
        out["error"] = f"read failed: {exc}"
    finally:
        try:
            f.close()
        except Exception:
            pass
    return out


def derive_collect_hdf5_path(
    context: RuntimeContext,
    robot: Optional[str],
    target: Optional[str],
) -> Optional[Path]:
    """Match the naming convention used by scene_auto_grasp_collect.py:
        <DATASETS_DIR>/<session>_<run>_<robot>_<target_slug>.hdf5
    Falls back to globbing `<session>_<run>_*.hdf5` so we still find a
    file when robot/target weren't tracked in session state yet.
    """
    target_slug: str = "target"
    if isinstance(target, str) and target.strip():
        cleaned = target.strip().lstrip("/").replace("/", "_")
        if cleaned:
            target_slug = cleaned
    robot_slug = (robot or "").strip().lower() or "agibot"
    candidate = DATASETS_DIR / f"{context.session_id}_{context.run_id}_{robot_slug}_{target_slug}.hdf5"
    if candidate.exists():
        return candidate
    pattern = f"{context.session_id}_{context.run_id}_*.hdf5"
    matches = sorted(DATASETS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


# --------- Train summary ---------


_TRAIN_STEP_RE = re.compile(r"\bstep[:= ]\s*(\d+)\s*/?\s*(\d+)?\b", re.IGNORECASE)
_TRAIN_LOSS_RE = re.compile(r"\bloss[:= ]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)")


def _list_checkpoints(output_dir: Path) -> list[dict[str, Any]]:
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for entry in sorted(ckpt_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        step: Optional[int] = None
        try:
            step = int(name)
        except ValueError:
            step = None
        items.append(
            {
                "name": name,
                "step": step,
                "path": str(entry),
                "mtime": int(entry.stat().st_mtime) if entry.exists() else 0,
            }
        )
    items.sort(key=lambda x: (x.get("step") is None, x.get("step") or 0, x.get("mtime") or 0))
    return items


def _parse_train_log_progress(log_path: Path, *, tail_kb: int = 64) -> dict[str, Any]:
    """Pull the latest step / loss out of a train log tail."""
    out: dict[str, Any] = {}
    if not log_path.exists():
        return out
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            f.seek(max(0, size - tail_kb * 1024))
            blob = f.read().decode("utf-8", errors="replace")
    except Exception:
        return out
    # Last step = N(/M)
    last_step = None
    last_total = None
    for m in _TRAIN_STEP_RE.finditer(blob):
        try:
            last_step = int(m.group(1))
            last_total = int(m.group(2)) if m.group(2) else last_total
        except (TypeError, ValueError):
            pass
    if last_step is not None:
        out["step"] = last_step
        if last_total is not None:
            out["total_steps"] = last_total
    # Last loss
    last_loss = None
    for m in _TRAIN_LOSS_RE.finditer(blob):
        try:
            last_loss = float(m.group(1))
        except (TypeError, ValueError):
            pass
    if last_loss is not None:
        out["loss"] = last_loss
    return out


def find_train_output(repo_id: Optional[str]) -> Optional[Path]:
    """Best-effort: try `outputs/train/<repo_basename>` first, else newest."""
    if not OUTPUTS_TRAIN_DIR.exists():
        return None
    if repo_id:
        basename = repo_id.split("/", 1)[-1]
        cand = OUTPUTS_TRAIN_DIR / basename
        if cand.exists() and cand.is_dir():
            return cand
    candidates = [p for p in OUTPUTS_TRAIN_DIR.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def summarize_train(
    context: RuntimeContext,
    *,
    repo_id: Optional[str],
    train_log_path: Path,
) -> dict[str, Any]:
    """Return train output summary for the dashboard."""
    out: dict[str, Any] = {"exists": False}
    output_dir = find_train_output(repo_id)
    if output_dir is None:
        return out
    out["exists"] = True
    out["output_dir"] = str(output_dir)
    out["repo_id"] = repo_id or output_dir.name

    ckpts = _list_checkpoints(output_dir)
    out["checkpoint_count"] = len(ckpts)
    if ckpts:
        # Latest checkpoint = highest step (or last by mtime when step missing).
        latest = ckpts[-1]
        out["latest_checkpoint"] = latest
        out["checkpoints"] = [{"name": c["name"], "step": c["step"]} for c in ckpts]

    progress = _parse_train_log_progress(train_log_path)
    if progress:
        out["progress"] = progress
    return out


# --------- Eval summary ---------


_EVAL_VIDEO_RE = re.compile(r"^episode_(\d+)_([a-zA-Z_]+)\.mp4$")


def _runtime_file_url(abs_path: Path) -> Optional[str]:
    """Convert an absolute path into a /runtime_file/<rel> URL when the
    path is under RUNTIME_DIR. The frontend already serves /runtime_file/
    as a static-passthrough."""
    try:
        runtime_root = Path(RUNTIME_DIR).resolve()
        target = abs_path.resolve()
    except Exception:
        return None
    if target == runtime_root or runtime_root in target.parents:
        rel = target.relative_to(runtime_root).as_posix()
        ts = int(target.stat().st_mtime_ns) if target.exists() else 0
        return f"/runtime_file/{rel}?ts={ts}"
    return None


def _project_file_url(abs_path: Path) -> Optional[str]:
    """For files outside RUNTIME_DIR (eval videos under outputs/), we expose
    them via a sibling endpoint. The route /robot_file/<...> implements this;
    see routes.py."""
    try:
        target = abs_path.resolve()
    except Exception:
        return None
    project_root = Path(RUNTIME_DIR).resolve().parent  # PROJECT_ROOT
    if target == project_root or project_root in target.parents:
        rel = target.relative_to(project_root).as_posix()
        ts = int(target.stat().st_mtime_ns) if target.exists() else 0
        return f"/robot_file/{rel}?ts={ts}"
    return None


def _gather_eval_videos(record_dir: Path) -> list[dict[str, Any]]:
    """Group <episode_NN_<view>.mp4> files by episode. Each episode entry
    surfaces one URL per camera view (head / left_hand / right_hand)."""
    if not record_dir.exists() or not record_dir.is_dir():
        return []
    by_episode: dict[int, dict[str, Any]] = {}
    for entry in record_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".mp4":
            continue
        m = _EVAL_VIDEO_RE.match(entry.name)
        if not m:
            continue
        try:
            ep_idx = int(m.group(1))
        except ValueError:
            continue
        view = m.group(2)
        url = _project_file_url(entry)
        if url is None:
            continue
        ep = by_episode.setdefault(
            ep_idx, {"index": ep_idx, "videos": {}, "size_bytes": 0}
        )
        ep["videos"][view] = url
        try:
            ep["size_bytes"] += int(entry.stat().st_size)
        except OSError:
            pass
    return [by_episode[i] for i in sorted(by_episode.keys())]


# --------- Grasp / placement plan summary ---------


def _read_json_silent(path: Path) -> Optional[dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize_grasp_plan(context: RuntimeContext) -> dict[str, Any]:
    """Pull the latest robot-placement plan + selected grasp proposal.

    `plan_robot_base_pose` writes its outputs to a shared dir (the script's
    `--plan_output_dir` default is `<PROJECT_ROOT>/runtime/robot_placement`)
    rather than under the per-run dir, so we read from the shared location.
    Each new collect run overwrites the four files in-place, which means
    what we read here is the plan that drove the LATEST collect call —
    the right thing for the UI to show.
    """
    out: dict[str, Any] = {"exists": False}
    plan_root = Path(RUNTIME_DIR) / "robot_placement"
    base_plan_path = plan_root / "robot_base_plan.json"
    grasp_path = plan_root / "selected_grasp_proposal.json"

    base = _read_json_silent(base_plan_path)
    grasp = _read_json_silent(grasp_path)
    if base is None and grasp is None:
        return out

    out["exists"] = True
    out["plan_dir"] = str(plan_root)

    if isinstance(base, dict):
        try:
            mtime = base_plan_path.stat().st_mtime
            out["mtime_iso"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except OSError:
            pass
        # Compact base pose: [x, y, z, yaw_deg]
        bp = base.get("base_pose")
        if isinstance(bp, list) and len(bp) >= 4:
            out["base_pose"] = {
                "x": float(bp[0]),
                "y": float(bp[1]),
                "z": float(bp[2]),
                "yaw_deg": float(bp[3]),
            }
        out["robot"] = base.get("robot")
        out["target_prim"] = base.get("target_prim")
        out["support_prim"] = base.get("support_prim")
        out["chosen_side"] = base.get("chosen_side")
        if isinstance(base.get("candidates"), list):
            out["candidate_count"] = len(base["candidates"])
        # Top candidate score (already chosen side is rendered first)
        if isinstance(base.get("candidates"), list) and base["candidates"]:
            top = base["candidates"][0]
            tie = top.get("tie_break_score")
            if tie is not None:
                out["base_score"] = float(tie)

    if isinstance(grasp, dict):
        sel = grasp.get("selected_grasp_proposal") or {}
        g = sel.get("grasp") or {}
        pos = g.get("position_world")
        if isinstance(pos, list) and len(pos) >= 3:
            out["grasp_position"] = {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            }
        approach = g.get("approach_axis_world")
        if isinstance(approach, list) and len(approach) >= 3:
            out["approach_axis"] = [float(approach[0]), float(approach[1]), float(approach[2])]
        if g.get("score") is not None:
            try:
                out["grasp_score"] = float(g.get("score"))
            except (TypeError, ValueError):
                pass
        if grasp.get("selected_grasp_candidate_id"):
            out["selected_candidate_id"] = str(grasp.get("selected_grasp_candidate_id"))
        if grasp.get("agibot_ee_frame_remap"):
            out["ee_frame"] = str(grasp.get("agibot_ee_frame_remap"))
        # Total grasp candidates considered.
        shortlist_path = plan_root / "grasp_candidate_shortlist.json"
        shortlist = _read_json_silent(shortlist_path)
        if isinstance(shortlist, dict):
            cands = shortlist.get("candidates") or shortlist.get("shortlist") or []
            if isinstance(cands, list):
                out["grasp_candidate_count"] = len(cands)

    return out


# --------- LeRobot dataset summary ---------


def summarize_lerobot_dataset(repo_id: Optional[str], context: RuntimeContext) -> dict[str, Any]:
    """Find the LeRobotDataset for this session/run and return preview info.

    Looks for `<LEROBOT_DATASETS_DIR>/<basename>` where basename is derived
    from repo_id; falls back to globbing `<session>_<run>_*` so a slightly
    different naming still lights up.
    """
    out: dict[str, Any] = {"exists": False}
    if not LEROBOT_DATASETS_DIR.exists():
        return out

    candidates: list[Path] = []
    if repo_id:
        basename = repo_id.split("/", 1)[-1]
        cand = LEROBOT_DATASETS_DIR / basename
        if cand.is_dir():
            candidates.append(cand)
    if not candidates:
        # Glob session/run-prefixed dirs (covers `..._partial4` etc.)
        for entry in LEROBOT_DATASETS_DIR.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            if context.session_id[:8] in name or context.run_id[:8] in name:
                candidates.append(entry)
    if not candidates:
        return out

    # Pick most recently modified.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ds_dir = candidates[0]

    info = _read_json_silent(ds_dir / "meta" / "info.json")
    out["exists"] = True
    out["dataset_dir"] = str(ds_dir)
    out["repo_id"] = ds_dir.name

    if isinstance(info, dict):
        for k in ("total_episodes", "total_frames", "total_tasks", "fps"):
            if k in info:
                out[k] = info[k]
        feats = info.get("features") or {}
        # List the camera/video features.
        cameras: list[str] = []
        for key, spec in feats.items():
            if not key.startswith("observation.images."):
                continue
            view = key.split(".", 2)[-1]
            cameras.append(view)
        out["cameras"] = cameras
        # Action / state shapes for a quick eyeball.
        if isinstance(feats.get("action"), dict):
            out["action_shape"] = feats["action"].get("shape")
        if isinstance(feats.get("observation.state"), dict):
            out["state_shape"] = feats["observation.state"].get("shape")

    # Locate one mp4 per camera and expose a /robot_file URL. The standard
    # LeRobot v3 layout puts videos at:
    #   videos/<key>/chunk-000/file-000.mp4
    videos_dir = ds_dir / "videos"
    video_urls: list[dict[str, Any]] = []
    if videos_dir.is_dir():
        for cam_dir in sorted(videos_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            view = cam_dir.name.replace("observation.images.", "")
            mp4s = sorted(cam_dir.rglob("*.mp4"))
            if not mp4s:
                continue
            mp4 = mp4s[0]
            url = _project_file_url(mp4)
            if url:
                try:
                    sz = int(mp4.stat().st_size)
                except OSError:
                    sz = 0
                video_urls.append({"view": view, "url": url, "size_bytes": sz})
    out["videos"] = video_urls
    return out


def find_latest_eval_dir(context: RuntimeContext) -> Optional[Path]:
    """Pick the most-recent <session>_<run>_*_runs/ dir under outputs/eval/.
    Falls back to the newest directory overall when the session-tagged
    convention isn't followed."""
    if not OUTPUTS_EVAL_DIR.exists():
        return None
    pattern = f"{context.session_id}_{context.run_id}_*"
    cands = sorted(OUTPUTS_EVAL_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    # Fallback: newest dir overall.
    fallback = sorted(
        [p for p in OUTPUTS_EVAL_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return fallback[0] if fallback else None


def summarize_eval(context: RuntimeContext) -> dict[str, Any]:
    out: dict[str, Any] = {"exists": False}
    record_dir = find_latest_eval_dir(context)
    if record_dir is None:
        return out
    out["exists"] = True
    out["record_dir"] = str(record_dir)
    episodes = _gather_eval_videos(record_dir)
    out["episodes"] = episodes
    out["episode_count"] = len(episodes)
    # Comparison_all.mp4 if the eval script wrote a side-by-side video.
    comparison = record_dir / "comparison_all.mp4"
    if comparison.exists():
        url = _project_file_url(comparison)
        if url:
            out["comparison_all_url"] = url
    return out


# --------- Top-level dashboard ---------


def _format_elapsed(created_at: Any) -> Optional[int]:
    if not created_at:
        return None
    try:
        started = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return int((datetime.now(timezone.utc) - started).total_seconds())
    except Exception:
        return None


def build_dashboard(context: RuntimeContext, agent_state: dict[str, Any]) -> dict[str, Any]:
    """Assemble the per-tab payload."""
    run_state = (agent_state.get("runs") or {}).get(context.run_id) or {}
    sr_state = run_state.get("scene_robot") if isinstance(run_state.get("scene_robot"), dict) else {}

    # --- Collect ---
    collect: dict[str, Any] = {
        "status": str((sr_state or {}).get("status") or "idle"),
        "robot": (sr_state or {}).get("robot"),
        "target": (sr_state or {}).get("target"),
        "num_episodes": (sr_state or {}).get("num_episodes"),
        "job_id": (sr_state or {}).get("job_id"),
    }
    job_id = collect.get("job_id")
    if job_id:
        live = get_scene_robot_job_status(str(job_id))
        if isinstance(live, dict):
            # The live job dict tracks all 4 stages; only treat it as the
            # collect job if its stage matches.
            stage = str(live.get("stage") or "")
            if stage == "collect":
                collect["status"] = str(live.get("status") or collect["status"])
                collect["created_at"] = live.get("created_at")
                collect["updated_at"] = live.get("updated_at")
                elapsed = _format_elapsed(live.get("created_at"))
                if elapsed is not None:
                    collect["elapsed_seconds"] = elapsed
                if live.get("error"):
                    collect["error"] = str(live.get("error"))[:600]

    hdf5_path = derive_collect_hdf5_path(context, collect.get("robot"), collect.get("target"))
    collect["hdf5"] = summarize_hdf5(hdf5_path) if hdf5_path else {"exists": False}

    # --- Train ---
    repo_id: Optional[str] = None
    if collect.get("robot") and collect.get("target"):
        target_slug = (collect["target"] or "").strip().lstrip("/").replace("/", "_") or "obj"
        repo_id = f"local/{context.session_id}_{context.run_id}_{collect['robot']}_{target_slug}"
    train = summarize_train(context, repo_id=repo_id, train_log_path=Path(context.scene_robot_train_log_path))
    train["status"] = "idle"
    if train.get("exists"):
        train["status"] = "trained"
    # If the live job dict has a train stage, override with its status.
    if job_id:
        live = get_scene_robot_job_status(str(job_id))
        if isinstance(live, dict) and str(live.get("stage") or "") == "train":
            train["status"] = str(live.get("status") or train["status"])
            elapsed = _format_elapsed(live.get("created_at"))
            if elapsed is not None:
                train["elapsed_seconds"] = elapsed
            if live.get("error"):
                train["error"] = str(live.get("error"))[:600]

    # --- Eval ---
    evald = summarize_eval(context)
    evald["status"] = "idle"
    if evald.get("exists"):
        evald["status"] = "evaluated"
    if job_id:
        live = get_scene_robot_job_status(str(job_id))
        if isinstance(live, dict) and str(live.get("stage") or "") == "eval":
            evald["status"] = str(live.get("status") or evald["status"])
            elapsed = _format_elapsed(live.get("created_at"))
            if elapsed is not None:
                evald["elapsed_seconds"] = elapsed
            if live.get("error"):
                evald["error"] = str(live.get("error"))[:600]

    grasp = summarize_grasp_plan(context)
    dataset = summarize_lerobot_dataset(repo_id, context)

    return {
        "session_id": context.session_id,
        "run_id": context.run_id,
        "collect": collect,
        "grasp": grasp,
        "dataset": dataset,
        "train": train,
        "eval": evald,
    }
