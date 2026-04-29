"""Tool-use multi-turn agent loop.

Opt-in alternative to `agent_service.handle_agent_message`. Wraps the
existing pipeline helpers as OpenAI Responses-API tools and runs a loop
that lets the model chain `inspect_state` -> `create_scene_graph` ->
`run_real2sim` -> `generate_scene` -> `run_scene_robot_collect` (or any
subset) within a single user message.

Long-running tools (`run_real2sim`, `run_scene_robot_collect`) start
their job in a background thread and return immediately with a job id;
the loop does NOT poll. The frontend's existing `monitorReal2SimJob` /
`monitorSceneRobotJob` keep streaming logs after the response returns.

Skeleton scope: 5 tools, no edit_scene_graph, no per-message message
history persistence (each user message starts a fresh tool-use loop and
sees current state via `inspect_state`).
"""

from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import SCENE_SERVICE_URL
from .agent_service import (
    DEFAULT_SCENE_ROBOT_NUM_EPISODES,
    DEFAULT_SCENE_ROBOT_ROBOT,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_RUN_REAL2SIM,
    STATE_RUN_SCENE_ROBOT_COLLECT,
    _agent_paths_payload,
    _append_agent_history,
    _build_agent_response,
    _collect_real2sim_artifacts_for_context,
    _ensure_run_state,
    _load_agent_state,
    _normalize_scene_result_for_agent,
    _pick_default_scene_robot_target,
    _record_real2sim_state,
    _record_scene_generation_state,
    _record_scene_robot_state,
    _run_scene_service_with_repair,
    _save_agent_state,
    _scene_graph_iter_objects,
    _session_state_snapshot,
    _set_current_state,
    sync_real2sim_job_to_session,
    sync_scene_robot_job_to_session,
)
from .instruction_service import create_scene_graph_from_input
from .openai_service import _get_openai_client
from .pipeline_service import get_real2sim_job_status, start_real2sim_job
from .runtime_context import (
    RuntimeContext,
    create_session,
    resolve_runtime_context,
)
from .scene_robot_service import (
    get_scene_robot_job_status,
    start_scene_robot_collect_job,
    start_scene_robot_convert_job,
    start_scene_robot_eval_job,
    start_scene_robot_train_job,
)
from ..config import (
    DATASETS_DIR,
    LEROBOT_DATASETS_DIR,
    OUTPUTS_EVAL_DIR,
    OUTPUTS_TRAIN_DIR,
)


AGENT_LOOP_MODEL = os.getenv("AGENT_LOOP_MODEL", os.getenv("AGENT_ROUTER_MODEL", "gpt-5.4-mini"))
MAX_TOOL_TURNS = int(os.getenv("AGENT_LOOP_MAX_TURNS", "8"))


SYSTEM_PROMPT = """
You are the orchestrator agent for a 3D scene + robot pipeline. You decide,
within one user message, which tools to call and in what order.

Available tools (in normal pipeline order):
- inspect_state: read what already exists in the current run (scene graph,
  input image, scene USD, real2sim status, scene_robot status).
- create_scene_graph: build a new scene graph from text and/or the
  current uploaded image. Replaces the existing scene graph for this run.
- run_real2sim: launch the Real2Sim observed-object reconstruction job
  for the current scene graph + image. Returns immediately with a job id;
  it runs in the background and the UI streams its log.
- generate_scene: call the Isaac scene service to produce the scene USD
  preview using the current scene graph (pick resample_mode joint or
  lock_real2sim).
- run_scene_robot_collect: launch the scene_robot auto-grasp data
  collection job. Defaults: robot=agibot, num_episodes=5, target=first
  real2sim object in the scene graph. Returns immediately with a job id.
- run_scene_robot_convert: convert the latest collect HDF5 into a
  LeRobotDataset directory. Fast (seconds-to-minutes); fire-and-return.
- run_scene_robot_train: launch `lerobot-train` on the dataset. LONG
  (potentially hours). Returns immediately with a job id; checkpoint
  lands in the train output dir under `checkpoints/last/pretrained_model`.
- run_scene_robot_eval: closed-loop sim rollout of a trained checkpoint.
  Long job (minutes-to-tens of minutes). Returns immediately with a
  job id and a record directory for per-episode videos.

Operating rules:
1. Always call `inspect_state` first if you do not already know the
   current run's state from earlier tool results in this turn.
2. `run_real2sim`, `run_scene_robot_collect`, `run_scene_robot_train`,
   and `run_scene_robot_eval` are LONG jobs. Do NOT poll them inside
   this turn. Return to the user as soon as you start them; the UI
   streams their progress and the user can come back in a follow-up
   message.
3. If the user asks for an end-to-end pipeline ("set up everything to
   train a pickup policy"), you may call create_scene_graph ->
   run_real2sim in one turn, then stop and tell the user you'll continue
   once Real2Sim finishes.
4. Do NOT call generate_scene or run_scene_robot_collect while Real2Sim
   is still running or the scene graph contains real2sim objects without
   completed artifacts.
5. If the user's request is ambiguous, ask a clarifying question instead
   of guessing. Tools you do not call are free; nothing happens.
6. Final assistant text should be concise: state what you did and what
   the user should do or wait for next.
""".strip()


# --------- Tool registry ---------


def _tool_inspect_state(_args: dict[str, Any], lctx: "LoopContext") -> dict[str, Any]:
    context = lctx.runtime_context
    state = lctx.state
    run_state = _ensure_run_state(state, context)
    real2sim_state = run_state.get("real2sim") if isinstance(run_state.get("real2sim"), dict) else {}
    scene_robot_state = run_state.get("scene_robot") if isinstance(run_state.get("scene_robot"), dict) else {}

    scene_graph: dict[str, Any] | None = None
    if context.scene_graph_path.exists():
        try:
            scene_graph = json.loads(context.scene_graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            scene_graph = None

    objects: list[dict[str, str]] = []
    if isinstance(scene_graph, dict):
        for row in _scene_graph_iter_objects(scene_graph):
            path = row.get("path") or row.get("id")
            if not isinstance(path, str) or not path:
                continue
            objects.append(
                {
                    "path": path,
                    "class": str(row.get("class_name") or row.get("class") or ""),
                    "source": str(row.get("source") or ""),
                }
            )

    real2sim_block: dict[str, Any] = {
        "status": str(real2sim_state.get("status") or "idle"),
        "job_id": str(real2sim_state.get("job_id") or "") or None,
    }
    r2s_error_info = real2sim_state.get("error_info") if isinstance(real2sim_state.get("error_info"), dict) else None
    if r2s_error_info:
        real2sim_block["error_info"] = {
            "code": r2s_error_info.get("code"),
            "step": r2s_error_info.get("step"),
            "retryable": r2s_error_info.get("retryable"),
            "user_message": r2s_error_info.get("user_message"),
            "technical_detail_tail": _tail_text(r2s_error_info.get("technical_detail")),
        }
    if real2sim_state.get("error"):
        real2sim_block["last_error"] = str(real2sim_state.get("error"))
    if str(real2sim_state.get("status") or "") == "failed":
        digest = _extract_log_failure_digest(real2sim_state.get("log_path") or context.real2sim_log_path)
        if digest:
            real2sim_block["log_digest"] = digest

    scene_robot_block: dict[str, Any] = {
        "status": str(scene_robot_state.get("status") or "idle"),
        "job_id": str(scene_robot_state.get("job_id") or "") or None,
        "robot": scene_robot_state.get("robot"),
        "target": scene_robot_state.get("target"),
    }
    if scene_robot_state.get("error"):
        scene_robot_block["last_error"] = str(scene_robot_state.get("error"))
    if str(scene_robot_state.get("status") or "") == "failed":
        digest = _extract_log_failure_digest(scene_robot_state.get("log_path") or context.scene_robot_log_path)
        if digest:
            scene_robot_block["log_digest"] = digest

    return {
        "session_id": context.session_id,
        "run_id": context.run_id,
        "has_scene_graph": isinstance(scene_graph, dict),
        "scene_graph_objects": objects[:40],
        "has_uploaded_image_this_turn": lctx.has_uploaded_image,
        "has_saved_image": context.latest_input_image.exists(),
        "has_scene_usd": context.scene_service_usd_path.exists(),
        "real2sim": real2sim_block,
        "scene_robot": scene_robot_block,
    }


def _tool_create_scene_graph(args: dict[str, Any], lctx: "LoopContext") -> dict[str, Any]:
    instruction = str(args.get("instruction") or "").strip()
    use_image = bool(args.get("use_uploaded_image", lctx.has_uploaded_image))
    if not instruction and not use_image and not lctx.runtime_context.latest_input_image.exists():
        return {"ok": False, "error": "create_scene_graph requires either an instruction or an image."}

    result = create_scene_graph_from_input(
        instruction,
        image_bytes=None,
        scene_graph_path=lctx.runtime_context.scene_graph_path,
        placements_path=lctx.runtime_context.default_placements_path,
        latest_input_image_path=lctx.runtime_context.latest_input_image,
        reuse_saved_image=use_image,
    )
    scene_graph = result.get("scene_graph") if isinstance(result.get("scene_graph"), dict) else None
    object_count = 0
    if isinstance(scene_graph, dict):
        if isinstance(scene_graph.get("obj"), dict):
            object_count = len(scene_graph["obj"])
        elif isinstance(scene_graph.get("objects"), list):
            object_count = len(scene_graph["objects"])
    lctx.scene_graph_changed = True
    return {
        "ok": True,
        "object_count": object_count,
        "warnings": result.get("warnings") or [],
    }


def _tool_run_real2sim(_args: dict[str, Any], lctx: "LoopContext") -> dict[str, Any]:
    context = lctx.runtime_context
    if not context.scene_graph_path.exists():
        return {"ok": False, "error": "No scene graph in this run; create_scene_graph first."}
    if not context.latest_input_image.exists():
        return {"ok": False, "error": "No input image in this run; upload one first."}

    payload = {
        "session_id": context.session_id,
        "run_id": context.run_id,
        "image_path": str(context.latest_input_image),
        "scene_graph_path": str(context.scene_graph_path),
        "log_path": str(context.real2sim_log_path),
        "real2sim_root_dir": str(context.run_root),
        "mask_output": str(context.real2sim_masks_dir),
        "mesh_output_dir": str(context.real2sim_meshes_dir),
        "reuse_mesh_dir": str(context.real2sim_meshes_dir),
        "scene_results_dir": str(context.real2sim_scene_results_dir),
    }
    job = start_real2sim_job(payload)
    _record_real2sim_state(
        lctx.state,
        context,
        status="running",
        job_id=str(job.get("job_id") or ""),
        artifacts=_collect_real2sim_artifacts_for_context(context),
        log_path=str(job.get("log_path") or context.real2sim_log_path),
        log_start_offset=int(job.get("log_start_offset") or 0),
    )
    lctx.real2sim_job = job
    lctx.last_started_state = STATE_RUN_REAL2SIM
    return {
        "ok": True,
        "job_id": job.get("job_id"),
        "status": "running",
        "note": "Real2Sim runs in background. Do not poll within this turn.",
    }


def _tool_generate_scene(args: dict[str, Any], lctx: "LoopContext") -> dict[str, Any]:
    context = lctx.runtime_context
    if not context.scene_graph_path.exists():
        return {"ok": False, "error": "No scene graph in this run; create_scene_graph first."}

    resample_mode = str(args.get("resample_mode") or "joint").strip().lower()
    if resample_mode not in {"joint", "lock_real2sim"}:
        resample_mode = "joint"
    scene_endpoint = str(args.get("scene_endpoint") or "scene_new").strip().lower()
    if scene_endpoint not in {"scene", "scene_new"}:
        scene_endpoint = "scene_new"

    try:
        scene_graph = json.loads(context.scene_graph_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        scene_graph = None

    try:
        result, repair_warnings = _run_scene_service_with_repair(
            context,
            scene_endpoint=scene_endpoint,
            resample_mode=resample_mode,
            scene_service_url=lctx.scene_service_url,
            scene_graph=scene_graph,
        )
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}

    normalized = _record_scene_generation_state(
        lctx.state,
        context,
        scene_result=result,
        scene_endpoint=scene_endpoint,
        resample_mode=resample_mode,
    )
    lctx.scene_result = normalized
    lctx.warnings.extend(repair_warnings)
    return {
        "ok": True,
        "resample_mode": normalized.get("resample_mode") or resample_mode,
        "warnings": repair_warnings,
        "scene_usd_present": context.scene_service_usd_path.exists(),
    }


def _tool_run_scene_robot_collect(args: dict[str, Any], lctx: "LoopContext") -> dict[str, Any]:
    context = lctx.runtime_context
    if not context.scene_graph_path.exists():
        return {"ok": False, "error": "No scene graph in this run; create_scene_graph first."}

    try:
        scene_graph = json.loads(context.scene_graph_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        scene_graph = None

    target = (args.get("target") or "").strip() if isinstance(args.get("target"), str) else None
    if not target:
        target = _pick_default_scene_robot_target(scene_graph)
    if not target:
        return {"ok": False, "error": "No target available; the scene graph has no objects."}

    robot = (args.get("robot") or DEFAULT_SCENE_ROBOT_ROBOT).strip().lower()
    if robot not in {"agibot", "kinova", "r1lite"}:
        robot = DEFAULT_SCENE_ROBOT_ROBOT

    num_episodes_raw = args.get("num_episodes")
    try:
        num_episodes = int(num_episodes_raw) if num_episodes_raw is not None else DEFAULT_SCENE_ROBOT_NUM_EPISODES
    except (TypeError, ValueError):
        num_episodes = DEFAULT_SCENE_ROBOT_NUM_EPISODES
    num_episodes = max(1, min(num_episodes, 200))

    payload = {
        "session_id": context.session_id,
        "run_id": context.run_id,
        "log_path": str(context.scene_robot_log_path),
        "robot": robot,
        "target": target,
        "num_episodes": num_episodes,
        "headless": True,
        "wait_for_run_request": False,
    }
    job = start_scene_robot_collect_job(payload)
    _record_scene_robot_state(
        lctx.state,
        context,
        status="running",
        job_id=str(job.get("job_id") or ""),
        log_path=str(job.get("log_path") or context.scene_robot_log_path),
        log_start_offset=int(job.get("log_start_offset") or 0),
        robot=robot,
        target=target,
        num_episodes=num_episodes,
    )
    lctx.scene_robot_job = job
    lctx.last_started_state = STATE_RUN_SCENE_ROBOT_COLLECT
    return {
        "ok": True,
        "job_id": job.get("job_id"),
        "status": "running",
        "robot": robot,
        "target": target,
        "num_episodes": num_episodes,
        "note": "scene_robot collect runs in background. Do not poll within this turn.",
    }


def _default_repo_id(context: RuntimeContext, robot: str | None, target: str | None) -> str:
    """Derive a sensible default LeRobot repo_id from the current run."""
    target_slug = "obj"
    if isinstance(target, str) and target:
        cleaned = target.strip().lstrip("/").replace("/", "_")
        if cleaned:
            target_slug = cleaned[:40]
    robot_slug = (robot or "agibot").strip().lower() or "agibot"
    return f"local/{context.session_id}_{context.run_id}_{robot_slug}_{target_slug}"


def _default_dataset_root(repo_id: str) -> str:
    return str(LEROBOT_DATASETS_DIR / repo_id.split("/", 1)[-1])


def _latest_collect_hdf5(context: RuntimeContext) -> Path | None:
    """Find the most recent HDF5 dataset produced by collect for this run.

    scene_auto_grasp_collect.py writes to
    `datasets/<session>_<run>_<robot>_<target>.hdf5` by default, so we glob
    project-level `datasets/` for files matching this run.
    """
    pattern = f"{context.session_id}_{context.run_id}_*.hdf5"
    candidates = sorted(DATASETS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _latest_train_output(repo_id: str | None = None) -> Path | None:
    if not OUTPUTS_TRAIN_DIR.exists():
        return None
    candidates = [p for p in OUTPUTS_TRAIN_DIR.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _tool_run_scene_robot_convert(args: dict[str, Any], lctx: "LoopContext") -> dict[str, Any]:
    context = lctx.runtime_context

    hdf5_arg = args.get("hdf5")
    if isinstance(hdf5_arg, str) and hdf5_arg.strip():
        hdf5_path = Path(hdf5_arg.strip())
        if not hdf5_path.is_absolute():
            hdf5_path = (context.run_root / hdf5_path).resolve()
    else:
        candidate = _latest_collect_hdf5(context)
        if candidate is None:
            return {"ok": False, "error": "No HDF5 dataset found for this run; run scene_robot_collect first or pass hdf5 explicitly."}
        hdf5_path = candidate
    if not hdf5_path.exists():
        return {"ok": False, "error": f"HDF5 file not found: {hdf5_path}"}

    repo_id = (args.get("repo_id") or "").strip() if isinstance(args.get("repo_id"), str) else ""
    if not repo_id:
        repo_id = _default_repo_id(context, robot=None, target=None)

    output_root = (args.get("output_root") or "").strip() if isinstance(args.get("output_root"), str) else ""
    if not output_root:
        output_root = _default_dataset_root(repo_id)

    payload = {
        "session_id": context.session_id,
        "run_id": context.run_id,
        "log_path": str(context.scene_robot_convert_log_path),
        "hdf5": str(hdf5_path),
        "repo_id": repo_id,
        "output_root": output_root,
        "task": str(args.get("task") or "pick up the target object"),
        "overwrite": bool(args.get("overwrite", True)),
    }
    job = start_scene_robot_convert_job(payload)
    lctx.scene_robot_job = job
    lctx.last_started_state = STATE_RUN_SCENE_ROBOT_COLLECT
    return {
        "ok": True,
        "job_id": job.get("job_id"),
        "status": "running",
        "stage": "convert",
        "hdf5": str(hdf5_path),
        "repo_id": repo_id,
        "output_root": output_root,
        "note": "convert runs in background. Do not poll within this turn.",
    }


def _tool_run_scene_robot_train(args: dict[str, Any], lctx: "LoopContext") -> dict[str, Any]:
    context = lctx.runtime_context

    repo_id = (args.get("repo_id") or "").strip() if isinstance(args.get("repo_id"), str) else ""
    if not repo_id:
        repo_id = _default_repo_id(context, robot=None, target=None)

    dataset_root = (args.get("dataset_root") or "").strip() if isinstance(args.get("dataset_root"), str) else ""
    if not dataset_root:
        dataset_root = _default_dataset_root(repo_id)

    output_dir = (args.get("output_dir") or "").strip() if isinstance(args.get("output_dir"), str) else ""
    if not output_dir:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = str(OUTPUTS_TRAIN_DIR / f"{context.session_id}_{context.run_id}_{timestamp}")

    payload = {
        "session_id": context.session_id,
        "run_id": context.run_id,
        "log_path": str(context.scene_robot_train_log_path),
        "repo_id": repo_id,
        "dataset_root": dataset_root,
        "output_dir": output_dir,
        "policy_type": str(args.get("policy_type") or "diffusion"),
        "device": str(args.get("device") or "cuda"),
    }
    if args.get("steps") is not None:
        payload["steps"] = int(args["steps"])
    if args.get("batch_size") is not None:
        payload["batch_size"] = int(args["batch_size"])
    if args.get("save_freq") is not None:
        payload["save_freq"] = int(args["save_freq"])

    job = start_scene_robot_train_job(payload)
    lctx.scene_robot_job = job
    lctx.last_started_state = STATE_RUN_SCENE_ROBOT_COLLECT
    return {
        "ok": True,
        "job_id": job.get("job_id"),
        "status": "running",
        "stage": "train",
        "repo_id": repo_id,
        "dataset_root": dataset_root,
        "output_dir": output_dir,
        "expected_checkpoint_dir": str(Path(output_dir) / "checkpoints" / "last" / "pretrained_model"),
        "note": "train runs in background and may take hours. Do not poll within this turn.",
    }


def _tool_run_scene_robot_eval(args: dict[str, Any], lctx: "LoopContext") -> dict[str, Any]:
    context = lctx.runtime_context

    target = (args.get("target") or "").strip() if isinstance(args.get("target"), str) else ""
    if not target:
        try:
            scene_graph = json.loads(context.scene_graph_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, FileNotFoundError):
            scene_graph = None
        target = _pick_default_scene_robot_target(scene_graph) or ""
    if not target:
        return {"ok": False, "error": "No target available; pass target explicitly or ensure scene graph has objects."}

    checkpoint = (args.get("checkpoint") or "").strip() if isinstance(args.get("checkpoint"), str) else ""
    if not checkpoint:
        latest = _latest_train_output()
        if latest is None:
            return {"ok": False, "error": "No checkpoint provided and no train outputs found; run train first or pass checkpoint."}
        checkpoint = str(latest / "checkpoints" / "last" / "pretrained_model")

    repo_id = (args.get("repo_id") or "").strip() if isinstance(args.get("repo_id"), str) else ""
    if not repo_id:
        repo_id = _default_repo_id(context, robot=None, target=target)

    dataset_root = (args.get("dataset_root") or "").strip() if isinstance(args.get("dataset_root"), str) else ""
    if not dataset_root:
        dataset_root = _default_dataset_root(repo_id)

    num_episodes_raw = args.get("num_episodes")
    try:
        num_episodes = int(num_episodes_raw) if num_episodes_raw is not None else 20
    except (TypeError, ValueError):
        num_episodes = 20
    num_episodes = max(1, min(num_episodes, 500))

    record_dir = (args.get("record_dir") or "").strip() if isinstance(args.get("record_dir"), str) else ""
    if not record_dir:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        record_dir = str(OUTPUTS_EVAL_DIR / f"{context.session_id}_{context.run_id}_{timestamp}_runs")

    payload = {
        "session_id": context.session_id,
        "run_id": context.run_id,
        "log_path": str(context.scene_robot_eval_log_path),
        "target": target,
        "checkpoint": checkpoint,
        "dataset_root": dataset_root,
        "num_episodes": num_episodes,
        "record_dir": record_dir,
        "headless": True,
    }
    if args.get("robot"):
        payload["robot"] = str(args["robot"])

    job = start_scene_robot_eval_job(payload)
    lctx.scene_robot_job = job
    lctx.last_started_state = STATE_RUN_SCENE_ROBOT_COLLECT
    return {
        "ok": True,
        "job_id": job.get("job_id"),
        "status": "running",
        "stage": "eval",
        "target": target,
        "checkpoint": checkpoint,
        "num_episodes": num_episodes,
        "record_dir": record_dir,
        "note": "eval runs in background. Do not poll within this turn.",
    }


TOOL_HANDLERS: dict[str, Callable[[dict[str, Any], "LoopContext"], dict[str, Any]]] = {
    "inspect_state": _tool_inspect_state,
    "create_scene_graph": _tool_create_scene_graph,
    "run_real2sim": _tool_run_real2sim,
    "generate_scene": _tool_generate_scene,
    "run_scene_robot_collect": _tool_run_scene_robot_collect,
    "run_scene_robot_convert": _tool_run_scene_robot_convert,
    "run_scene_robot_train": _tool_run_scene_robot_train,
    "run_scene_robot_eval": _tool_run_scene_robot_eval,
}


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "inspect_state",
        "description": "Read the current run's state: scene graph object summary, whether an input image exists, whether a scene USD has been generated, and the latest real2sim / scene_robot job status.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "create_scene_graph",
        "description": "Build a new scene graph for the current run. Replaces any existing graph. Use the saved/uploaded input image when use_uploaded_image is true.",
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Free-text scene description; may be empty when relying on the image.",
                },
                "use_uploaded_image": {
                    "type": "boolean",
                    "description": "Reuse the saved/uploaded image for vision-grounded graph creation.",
                },
            },
            "required": ["instruction"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_real2sim",
        "description": "Launch the Real2Sim reconstruction job for the current run (requires scene_graph + input image). Returns immediately with a job_id; do not poll within this turn.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "generate_scene",
        "description": "Call the Isaac scene service to produce a scene USD preview using the current scene graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "resample_mode": {
                    "type": "string",
                    "enum": ["joint", "lock_real2sim"],
                    "description": "Layout strategy. lock_real2sim keeps observed real2sim support chains rigid.",
                },
                "scene_endpoint": {
                    "type": "string",
                    "enum": ["scene", "scene_new"],
                    "description": "scene_new resamples; scene preserves the existing layout.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_scene_robot_collect",
        "description": "Launch the scene_robot auto-grasp data collection job for the current scene USD. Returns immediately with a job_id; do not poll within this turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot": {"type": "string", "enum": ["agibot", "kinova", "r1lite"]},
                "target": {
                    "type": "string",
                    "description": "USD prim path of the target object, e.g. /World/bolt_2. If omitted, the first real2sim object in the scene graph is used.",
                },
                "num_episodes": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_scene_robot_convert",
        "description": "Convert the latest collect HDF5 dataset for this run into a LeRobotDataset directory. Defaults pick the most recently modified HDF5 matching this session/run plus a derived repo_id and output_root. Returns immediately with a job_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "hdf5": {
                    "type": "string",
                    "description": "Optional path to a specific HDF5 file. Defaults to the most recent collect output for this run.",
                },
                "repo_id": {
                    "type": "string",
                    "description": "LeRobot repo id. Default: local/<session>_<run>_<robot>_<target>.",
                },
                "output_root": {
                    "type": "string",
                    "description": "Directory to write the dataset into. Default: datasets/lerobot/<repo_id>.",
                },
                "task": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_scene_robot_train",
        "description": "Launch lerobot-train on the LeRobotDataset. Long job (potentially hours). Returns immediately with a job_id; the train output dir contains the eventual checkpoint.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_id": {"type": "string"},
                "dataset_root": {"type": "string"},
                "output_dir": {
                    "type": "string",
                    "description": "Training output directory. Default: outputs/train/<session>_<run>_<timestamp>.",
                },
                "policy_type": {"type": "string", "enum": ["diffusion", "act", "tdmpc", "vqbet"]},
                "device": {"type": "string"},
                "steps": {"type": "integer", "minimum": 100, "maximum": 1000000},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 256},
                "save_freq": {"type": "integer", "minimum": 100, "maximum": 100000},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_scene_robot_eval",
        "description": "Launch closed-loop sim eval (scene_eval_policy.py) on a trained checkpoint. Returns immediately with a job_id. Defaults to the most recent train output's `pretrained_model` checkpoint when checkpoint is omitted.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "checkpoint": {
                    "type": "string",
                    "description": "Path to the LeRobot policy `pretrained_model` directory. Defaults to the latest train output.",
                },
                "repo_id": {"type": "string"},
                "dataset_root": {"type": "string"},
                "robot": {"type": "string", "enum": ["agibot", "kinova", "r1lite"]},
                "num_episodes": {"type": "integer", "minimum": 1, "maximum": 500},
                "record_dir": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
]


# --------- Loop context ---------


class LoopContext:
    def __init__(
        self,
        runtime_context: RuntimeContext,
        state: dict[str, Any],
        *,
        has_uploaded_image: bool,
        scene_service_url: str,
    ) -> None:
        self.runtime_context = runtime_context
        self.state = state
        self.has_uploaded_image = has_uploaded_image
        self.scene_service_url = scene_service_url

        self.real2sim_job: dict[str, Any] | None = None
        self.scene_robot_job: dict[str, Any] | None = None
        self.scene_result: dict[str, Any] | None = None
        self.scene_graph_changed = False
        self.last_started_state: str | None = None
        self.warnings: list[str] = []
        self.tool_steps: list[dict[str, Any]] = []


# --------- OpenAI Responses API plumbing ---------


def _save_input_image(context: RuntimeContext, image_bytes: bytes | None) -> bool:
    if not image_bytes:
        return False
    target = context.latest_input_image
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(image_bytes)
    return True


def _output_item_to_input(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    if isinstance(item, dict):
        return dict(item)
    raise TypeError(f"Unsupported response item type: {type(item)!r}")


def _extract_text_from_message_item(item: Any) -> str:
    content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if block_type in {"output_text", "text"}:
            text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _summarize_tool_result(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return str(result)
    if result.get("error"):
        return f"error: {result['error']}"
    if result.get("job_id"):
        return f"started job {str(result['job_id'])[:8]}"
    return "ok"


def _execute_tool(name: str, args_json: str, lctx: LoopContext) -> dict[str, Any]:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        args = json.loads(args_json or "{}") if args_json else {}
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    try:
        return handler(args, lctx)
    except Exception as exc:  # noqa: BLE001 - surface tool-side errors back to the model
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# --------- Public entry point ---------


def _sync_inflight_jobs(state: dict[str, Any], context: RuntimeContext) -> list[dict[str, Any]]:
    """Refresh real2sim/scene_robot job state from the in-process job tables.

    Returns a list of "fresh failure" notes (one per job that just transitioned
    to `failed`) so the caller can surface them to the LLM. Mutates and
    re-saves `state` indirectly via the sync_*_to_session helpers.
    """
    notes: list[dict[str, Any]] = []
    run_state = _ensure_run_state(state, context)

    real2sim_state = run_state.get("real2sim") if isinstance(run_state.get("real2sim"), dict) else {}
    real2sim_status = str(real2sim_state.get("status") or "")
    real2sim_job_id = str(real2sim_state.get("job_id") or "")
    if real2sim_job_id and real2sim_status in {"queued", "running"}:
        try:
            raw_job = get_real2sim_job_status(real2sim_job_id)
        except Exception:  # noqa: BLE001 - best-effort sync; never break the loop
            raw_job = None
        if isinstance(raw_job, dict):
            new_status = str(raw_job.get("status") or "")
            sync_real2sim_job_to_session(raw_job)
            if new_status == "failed" and real2sim_status != "failed":
                err_info = raw_job.get("error_info") if isinstance(raw_job.get("error_info"), dict) else {}
                log_path = raw_job.get("log_path") or real2sim_state.get("log_path") or context.real2sim_log_path
                notes.append({
                    "kind": "real2sim",
                    "job_id": real2sim_job_id,
                    "code": err_info.get("code"),
                    "step": err_info.get("step"),
                    "retryable": err_info.get("retryable"),
                    "user_message": err_info.get("user_message") or raw_job.get("error"),
                    "technical_detail_tail": _tail_text(err_info.get("technical_detail")),
                    "log_digest": _extract_log_failure_digest(log_path),
                })

    scene_robot_state = run_state.get("scene_robot") if isinstance(run_state.get("scene_robot"), dict) else {}
    scene_robot_status = str(scene_robot_state.get("status") or "")
    scene_robot_job_id = str(scene_robot_state.get("job_id") or "")
    if scene_robot_job_id and scene_robot_status in {"queued", "running"}:
        try:
            raw_job = get_scene_robot_job_status(scene_robot_job_id)
        except Exception:  # noqa: BLE001
            raw_job = None
        if isinstance(raw_job, dict):
            new_status = str(raw_job.get("status") or "")
            sync_scene_robot_job_to_session(raw_job)
            if new_status == "failed" and scene_robot_status != "failed":
                log_path = raw_job.get("log_path") or scene_robot_state.get("log_path") or context.scene_robot_log_path
                notes.append({
                    "kind": "scene_robot",
                    "job_id": scene_robot_job_id,
                    "user_message": raw_job.get("error") or "scene_robot job failed.",
                    "log_digest": _extract_log_failure_digest(log_path),
                })

    return notes


def _tail_text(text: Any, max_chars: int = 600) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    if len(text) <= max_chars:
        return text
    return "...\n" + text[-max_chars:]


_TQDM_NOISE = re.compile(r"\b(it/s|loading weights|materializing param|\d+\s*%\|)", re.IGNORECASE)
_EXCEPTION_RE = re.compile(
    r"((?:[A-Za-z_][\w.]*\.)*[A-Z][A-Za-z0-9_]*(?:Error|Exception|Timeout|Refused))\s*:\s*(.+)"
)
_EXIT_CODE_RE = re.compile(r"returned non-zero exit status\s+(\d+)")
_CLASSIFIED_RE = re.compile(r"Classified failure:\s*code=(\S+)\s+retryable=(\S+)\s+step=(\S+)")
_URL_RE = re.compile(r"https?://[\w.\-]+(?::\d+)?(?:/[^\s'\")]*)?")
_HOSTPORT_RE = re.compile(r"host=['\"]([^'\"]+)['\"][^)]*?port=(\d+)")


def _extract_log_failure_digest(log_path: Any, *, tail_kb: int = 256, max_lines: int = 400) -> dict[str, Any] | None:
    """Read the tail of a job log and pull out the most likely failure signal.

    Best-effort. Returns None if log is missing/unreadable. Skips tqdm-style
    progress noise so we keep the meaningful ERROR/WARN/Traceback lines.
    """
    if not log_path:
        return None
    try:
        path = Path(str(log_path))
        if not path.exists() or not path.is_file():
            return None
        with path.open("rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            chunk = min(file_size, tail_kb * 1024)
            f.seek(max(0, file_size - chunk))
            tail_bytes = f.read()
        tail_text = tail_bytes.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None

    raw_lines = tail_text.splitlines()
    filtered: list[str] = [ln for ln in raw_lines if ln.strip() and not _TQDM_NOISE.search(ln)]
    last_lines = filtered[-max_lines:]

    digest: dict[str, Any] = {"log_path": str(log_path)}

    # Final exception line (works for stacked "X.Y.Z: msg" forms).
    for line in reversed(last_lines):
        m = _EXCEPTION_RE.search(line)
        if m:
            cls = m.group(1)
            msg = m.group(2).strip()
            digest["last_exception"] = f"{cls}: {msg}"[:600]
            break

    # subprocess exit code.
    for line in reversed(last_lines):
        m = _EXIT_CODE_RE.search(line)
        if m:
            digest["exit_code"] = int(m.group(1))
            break

    # Classifier verdict (already-classified by pipeline_service).
    for line in reversed(last_lines):
        m = _CLASSIFIED_RE.search(line)
        if m:
            digest["classified"] = {
                "code": m.group(1),
                "retryable": m.group(2),
                "step": m.group(3),
            }
            break

    # URLs / host:port in errors.
    urls: list[str] = []
    for u in _URL_RE.findall(tail_text):
        if u not in urls:
            urls.append(u)
        if len(urls) >= 5:
            break
    if urls:
        digest["urls"] = urls

    hostports: list[str] = []
    for h, p in _HOSTPORT_RE.findall(tail_text):
        hp = f"{h}:{p}"
        if hp not in hostports:
            hostports.append(hp)
        if len(hostports) >= 5:
            break
    if hostports:
        digest["hostports"] = hostports

    # Last 12 ERROR/WARN/Traceback lines for narrative context.
    important: list[str] = [
        ln for ln in last_lines
        if "[ERROR]" in ln or "[WARN]" in ln or ln.lstrip().startswith("Traceback") or ln.lstrip().startswith("raise ")
    ]
    if important:
        digest["error_lines"] = important[-12:]

    digest["tail_lines"] = last_lines[-15:]
    return digest


def _format_failure_note(notes: list[dict[str, Any]]) -> str:
    if not notes:
        return ""
    lines = [
        "SYSTEM: One or more background jobs from this run finished with status=failed. "
        "Lead your reply by telling the user (in their language) what failed and why. "
        "Use the structured fields below to give a SPECIFIC diagnosis: cite the actual "
        "exception class, the host:port / URL that failed, the failing step, and a one- "
        "line guess at the likely root cause (e.g. 'remote service on 127.0.0.1:8002 is "
        "not running'). Then ask whether to retry or adjust. Do NOT silently restart "
        "the failed job.",
    ]
    for note in notes:
        kind = note.get("kind") or "job"
        bits = [f"- {kind} (job_id={note.get('job_id')})"]
        if note.get("code"):
            bits.append(f"code={note['code']}")
        if note.get("step"):
            bits.append(f"step={note['step']}")
        if note.get("retryable") is not None:
            bits.append(f"retryable={note['retryable']}")
        lines.append(" ".join(bits))
        if note.get("user_message"):
            lines.append(f"  user_message: {note['user_message']}")
        tail = note.get("technical_detail_tail")
        if tail:
            lines.append(f"  technical_detail (tail): {tail}")
        digest = note.get("log_digest")
        if isinstance(digest, dict):
            lines.append(f"  log_path: {digest.get('log_path')}")
            if digest.get("last_exception"):
                lines.append(f"  last_exception: {digest['last_exception']}")
            if digest.get("exit_code") is not None:
                lines.append(f"  exit_code: {digest['exit_code']}")
            if digest.get("hostports"):
                lines.append(f"  hostports_in_log: {', '.join(digest['hostports'])}")
            if digest.get("urls"):
                lines.append(f"  urls_in_log: {', '.join(digest['urls'])}")
            if digest.get("classified") and not note.get("code"):
                lines.append(f"  classified: {digest['classified']}")
            err_lines = digest.get("error_lines")
            if err_lines:
                lines.append("  error_lines (last):")
                for ln in err_lines:
                    lines.append(f"    | {ln[:300]}")
            tail_lines = digest.get("tail_lines")
            if tail_lines and not err_lines:
                lines.append("  tail_lines (last 15):")
                for ln in tail_lines:
                    lines.append(f"    | {ln[:300]}")
    return "\n".join(lines)


def handle_agent_loop_message(
    *,
    session_id: str | None,
    run_id: str | None,
    text: str | None,
    image_bytes: bytes | None = None,
    scene_service_url: str | None = None,
) -> dict[str, Any]:
    context = resolve_runtime_context(session_id=session_id, run_id=run_id, create=True)
    if context is None:
        context = create_session()
    context.ensure()

    has_uploaded_image = _save_input_image(context, image_bytes)
    state = _load_agent_state(context)
    state["current_run_id"] = context.run_id
    _ensure_run_state(state, context)

    failure_notes = _sync_inflight_jobs(state, context)
    if failure_notes:
        # sync_*_to_session writes to disk; reload so we see the latest state.
        state = _load_agent_state(context)
        _ensure_run_state(state, context)

    user_text = (text or "").strip()
    if not user_text and not has_uploaded_image:
        user_text = "(no text; the user clicked submit with no instruction.)"
    _append_agent_history(state, "user", user_text or f"[image:{has_uploaded_image}]")
    _save_agent_state(context, state)

    lctx = LoopContext(
        context,
        state,
        has_uploaded_image=has_uploaded_image,
        scene_service_url=scene_service_url or SCENE_SERVICE_URL,
    )

    input_messages: list[dict[str, Any]] = []
    failure_note_text = _format_failure_note(failure_notes)
    if failure_note_text:
        input_messages.append({"role": "system", "content": failure_note_text})
    input_messages.append({"role": "user", "content": user_text})

    client = _get_openai_client()
    final_text = ""
    last_response = None

    for turn in range(MAX_TOOL_TURNS):
        response = client.responses.create(
            model=AGENT_LOOP_MODEL,
            instructions=SYSTEM_PROMPT,
            input=input_messages,
            tools=TOOL_DEFINITIONS,
        )
        last_response = response
        output_items = list(getattr(response, "output", None) or [])
        function_calls: list[Any] = []
        for item in output_items:
            item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
            if item_type == "function_call":
                function_calls.append(item)
            elif item_type == "message":
                text_block = _extract_text_from_message_item(item)
                if text_block:
                    final_text = text_block

        if not function_calls:
            break

        for fc in function_calls:
            input_messages.append(_output_item_to_input(fc))
        for fc in function_calls:
            name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else None)
            arguments = getattr(fc, "arguments", None) or (fc.get("arguments") if isinstance(fc, dict) else "{}")
            call_id = getattr(fc, "call_id", None) or (fc.get("call_id") if isinstance(fc, dict) else None)
            result = _execute_tool(str(name or ""), str(arguments or "{}"), lctx)
            lctx.tool_steps.append(
                {
                    "tool": name,
                    "args": arguments,
                    "summary": _summarize_tool_result(result),
                }
            )
            input_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )
    else:
        final_text = final_text or (
            f"Reached the {MAX_TOOL_TURNS}-tool-call cap before finishing. "
            "Tell me how you want to proceed."
        )

    if not final_text:
        final_text = "Done."

    _append_agent_history(state, "assistant", final_text)

    if lctx.last_started_state == STATE_RUN_REAL2SIM:
        _set_current_state(state, STATE_RUN_REAL2SIM, last_intent=STATE_RUN_REAL2SIM)
    elif lctx.last_started_state == STATE_RUN_SCENE_ROBOT_COLLECT:
        _set_current_state(state, STATE_RUN_SCENE_ROBOT_COLLECT, last_intent=STATE_RUN_SCENE_ROBOT_COLLECT)
    elif lctx.scene_result is not None:
        _set_current_state(state, STATE_COMPLETED, last_completed_state="generate_scene")
    elif lctx.scene_graph_changed:
        _set_current_state(state, STATE_COMPLETED, last_completed_state="create_scene_graph")
    state["pending_question"] = None
    _save_agent_state(context, state)

    decision = {
        "intent": "agent_loop",
        "state": state.get("current_state"),
        "reason": "Tool-use loop completed.",
        "tool_steps": lctx.tool_steps,
    }

    extra: dict[str, Any] = {
        "tool_steps": lctx.tool_steps,
        "warnings": lctx.warnings,
    }
    if lctx.real2sim_job is not None:
        extra["real2sim_job"] = lctx.real2sim_job
    if lctx.scene_robot_job is not None:
        extra["scene_robot_job"] = lctx.scene_robot_job
    if lctx.scene_result is not None:
        extra["scene_result"] = lctx.scene_result

    scene_graph = None
    if context.scene_graph_path.exists():
        try:
            scene_graph = json.loads(context.scene_graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            scene_graph = None
    if scene_graph is not None:
        extra["scene_graph"] = scene_graph

    return _build_agent_response(
        context,
        state=str(state.get("current_state") or "agent_loop"),
        intent="agent_loop",
        message=final_text,
        reason="Tool-use loop completed.",
        decision=decision,
        outcome="started_job" if (lctx.real2sim_job or lctx.scene_robot_job) else "completed",
        session_state=_session_state_snapshot(state, context),
        **extra,
    )
