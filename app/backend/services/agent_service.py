from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..config import RUNTIME_DIR, SCENE_SERVICE_URL
from .instruction_service import apply_instruction, create_scene_graph_from_input
from .openai_service import react_agent_step, read_json_file, route_agent_request
from .pipeline_service import collect_scene_result_artifacts, start_real2sim_job
from .runtime_context import RuntimeContext, create_session, resolve_runtime_context


_AGENT_STATE_FILENAME = "agent_state.json"

STATE_UNDERSTAND_REQUEST = "understand_request"
STATE_NEEDS_CLARIFICATION = "needs_clarification"
STATE_CREATE_SCENE_GRAPH = "create_scene_graph"
STATE_EDIT_SCENE_GRAPH = "edit_scene_graph"
STATE_RUN_REAL2SIM = "run_real2sim"
STATE_AWAIT_LAYOUT_STRATEGY = "await_layout_strategy"
STATE_GENERATE_SCENE = "generate_scene"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"

TOP_LEVEL_ROUTE_CONFIDENCE_THRESHOLD = 0.7
REACT_MAX_STEPS = 4


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip()


def _load_scene_graph(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json_file(path)
    return payload if isinstance(payload, dict) else None


def _runtime_file_url(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        target = candidate.resolve()
    except FileNotFoundError:
        target = candidate
    runtime_root = RUNTIME_DIR.resolve()
    if not target.exists():
        return None
    if target != runtime_root and runtime_root not in target.parents:
        return None
    rel = target.relative_to(runtime_root).as_posix()
    ts = int(target.stat().st_mtime_ns)
    return f"/runtime_file/{rel}?ts={ts}"


def _resolve_context_path(context: RuntimeContext, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = context.run_root / candidate
    return candidate.resolve()


def _enrich_real2sim_artifacts(context: RuntimeContext, artifacts: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(artifacts or {})
    root_dir = _resolve_context_path(context, raw.get("real2sim_root_dir") or context.run_root)
    results_dir = _resolve_context_path(context, raw.get("scene_results_dir") or context.real2sim_scene_results_dir)
    enriched = dict(raw)
    enriched["real2sim_root_dir"] = str(root_dir) if root_dir is not None else str(context.run_root)
    enriched["scene_results_dir"] = str(results_dir) if results_dir is not None else str(context.real2sim_scene_results_dir)

    for key in ("assignment_json", "poses_json", "manifest_json", "scene_glb", "scene_usd"):
        resolved = _resolve_context_path(context, raw.get(key))
        enriched[f"{key}_path"] = str(resolved) if resolved is not None else None
        enriched[f"{key}_url"] = _runtime_file_url(resolved)

    object_glb_paths: list[str] = []
    object_glb_urls: list[str] = []
    for rel in raw.get("object_glbs", []) or []:
        resolved = _resolve_context_path(context, rel)
        if resolved is None:
            continue
        object_glb_paths.append(str(resolved))
        url = _runtime_file_url(resolved)
        if url:
            object_glb_urls.append(url)
    enriched["object_glb_paths"] = object_glb_paths
    enriched["object_glb_urls"] = object_glb_urls

    object_usd_paths: list[str] = []
    object_usd_urls: list[str] = []
    for rel in raw.get("object_usds", []) or []:
        resolved = _resolve_context_path(context, rel)
        if resolved is None:
            continue
        object_usd_paths.append(str(resolved))
        url = _runtime_file_url(resolved)
        if url:
            object_usd_urls.append(url)
    enriched["object_usd_paths"] = object_usd_paths
    enriched["object_usd_urls"] = object_usd_urls
    return enriched


def _collect_real2sim_artifacts_for_context(context: RuntimeContext) -> dict[str, Any]:
    artifacts = collect_scene_result_artifacts(str(context.run_root), str(context.real2sim_scene_results_dir))
    artifacts["real2sim_root_dir"] = str(context.run_root)
    artifacts["scene_results_dir"] = str(context.real2sim_scene_results_dir)
    return _enrich_real2sim_artifacts(context, artifacts)


def _has_real2sim_artifacts(artifacts: dict[str, Any] | None) -> bool:
    if not isinstance(artifacts, dict):
        return False
    return any(
        [
            artifacts.get("assignment_json_path"),
            artifacts.get("poses_json_path"),
            artifacts.get("manifest_json_path"),
            artifacts.get("scene_glb_path"),
            artifacts.get("scene_usd_path"),
            artifacts.get("object_glb_paths"),
            artifacts.get("object_usd_paths"),
        ]
    )


def _normalize_scene_result_for_agent(context: RuntimeContext, scene_result: dict[str, Any]) -> dict[str, Any]:
    result = dict(scene_result or {})
    saved_usd_path = _resolve_context_path(context, result.get("saved_usd") or context.scene_service_usd_path)
    placements_path = _resolve_context_path(context, result.get("placements_path") or context.default_placements_path)
    render_path = _resolve_context_path(context, result.get("screenshot_path") or context.render_path)

    result["session_id"] = context.session_id
    result["run_id"] = context.run_id
    result["saved_usd"] = str(saved_usd_path) if saved_usd_path is not None else None
    result["saved_usd_url"] = _runtime_file_url(saved_usd_path)
    result["placements_path"] = str(placements_path) if placements_path is not None else None
    result["placements_url"] = _runtime_file_url(placements_path)
    result["screenshot_path"] = str(render_path) if render_path is not None else None
    result["render_image_path"] = str(render_path) if render_path is not None else None
    result["render_image_url"] = _runtime_file_url(render_path)

    if "placements" not in result and placements_path is not None and placements_path.exists():
        payload = read_json_file(placements_path)
        if isinstance(payload, dict):
            result["placements"] = payload
    return result


def _scene_graph_exists(scene_graph: dict[str, Any] | None) -> bool:
    return isinstance(scene_graph, dict)


def _has_real2sim_objects(scene_graph: dict[str, Any] | None) -> bool:
    if not isinstance(scene_graph, dict):
        return False
    obj_map = scene_graph.get("obj")
    if not isinstance(obj_map, dict):
        return False
    return any(
        isinstance(meta, dict) and str(meta.get("source") or "").strip().lower() == "real2sim"
        for meta in obj_map.values()
    )


def _parse_layout_strategy(text: str | None) -> str | None:
    normalized = _normalize_text(text).lower()
    if not normalized:
        return None
    if "lock_real2sim" in normalized or "lock real2sim" in normalized:
        return "lock_real2sim"
    if "keep observed" in normalized or "keep real2sim" in normalized or "preserve real2sim" in normalized:
        return "lock_real2sim"
    if "joint" in normalized:
        return "joint"
    if "fresh layout" in normalized or "resample all" in normalized or "rebuild all" in normalized:
        return "joint"
    return None


def _resolve_scene_endpoint(text: str | None, explicit_value: str | None) -> str:
    normalized_explicit = _normalize_text(explicit_value).lower()
    if normalized_explicit in {"scene", "scene_new"}:
        return normalized_explicit
    return "scene_new"


def _normalize_scene_endpoint_value(value: str | None) -> str | None:
    normalized = _normalize_text(value).lower()
    if normalized in {"scene", "scene_new"}:
        return normalized
    return None


def _normalize_resample_mode_value(value: str | None) -> str | None:
    normalized = _normalize_text(value).lower()
    if normalized in {"joint", "lock_real2sim"}:
        return normalized
    return None


def _option(
    option_id: str,
    label: str,
    *,
    description: str,
    reply: str | None = None,
    action: str | None = None,
    resample_mode: str | None = None,
    scene_endpoint: str | None = None,
) -> dict[str, Any]:
    payload = {
        "id": option_id,
        "label": label,
        "description": description,
    }
    if reply:
        payload["reply"] = reply
    if action:
        payload["action"] = action
    if resample_mode:
        payload["resample_mode"] = resample_mode
    if scene_endpoint:
        payload["scene_endpoint"] = scene_endpoint
    return payload


def _layout_strategy_options() -> list[dict[str, Any]]:
    return [
        _option(
            "joint",
            "Joint Resample",
            description="Resample every object together for a fresh layout.",
            reply="joint",
            action="generate_scene",
            resample_mode="joint",
        ),
        _option(
            "lock_real2sim",
            "Lock Real2Sim",
            description="Keep observed Real2Sim support chains rigid while resampling the rest.",
            reply="lock_real2sim",
            action="generate_scene",
            resample_mode="lock_real2sim",
        ),
    ]


def _graph_mode_options() -> list[dict[str, Any]]:
    return [
        _option(
            "create_scene_graph",
            "New Graph",
            description="Replace the current scene graph with a new one from this prompt/image.",
            reply="Create a new scene graph from this input.",
            action="create_scene_graph",
        ),
        _option(
            "edit_scene_graph",
            "Edit Current",
            description="Keep the current scene graph and apply this instruction as an edit.",
            reply="Edit the current scene graph.",
            action="edit_scene_graph",
        ),
    ]


def _bootstrap_scene_graph_options(*, from_saved_image: bool) -> list[dict[str, Any]]:
    if not from_saved_image:
        return []
    return [
        _option(
            "create_from_current_image",
            "Create Graph",
            description="Build a new scene graph from the current run image before continuing.",
            reply="Create a new scene graph from the current image.",
            action="create_scene_graph",
        )
    ]


def _generic_intent_options(*, has_scene_graph: bool, has_saved_image: bool) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    if has_saved_image:
        options.append(
            _option(
                "create_scene_graph",
                "Build Graph",
                description="Create a new scene graph from the current image.",
                reply="Create a new scene graph from the current image.",
                action="create_scene_graph",
            )
        )
    if has_scene_graph:
        options.extend(
            [
                _option(
                    "edit_scene_graph",
                    "Edit Graph",
                    description="Modify the current scene graph in place.",
                    reply="Edit the current scene graph.",
                    action="edit_scene_graph",
                ),
                _option(
                    "run_real2sim",
                    "Run Real2Sim",
                    description="Extract observed objects from the current run image and graph.",
                    reply="Run Real2Sim on the current scene graph.",
                    action="run_real2sim",
                ),
                _option(
                    "generate_scene",
                    "Generate Scene",
                    description="Call the scene service using the current graph.",
                    reply="Generate the scene.",
                    action="generate_scene",
                ),
            ]
        )
    return options


def _question_payload(
    question_type: str,
    question: str,
    *,
    options: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    scene_endpoint: str | None = None,
) -> dict[str, Any]:
    payload = {
        "type": question_type,
        "question": question,
        "options": options or [],
    }
    if run_id is not None:
        payload["run_id"] = run_id
    if scene_endpoint is not None:
        payload["scene_endpoint"] = scene_endpoint
    return payload


def _decision_payload(
    *,
    intent: str,
    state: str,
    reason: str,
    signals: list[str] | None = None,
    requires_clarification: bool = False,
    question: str | None = None,
    options: list[dict[str, Any]] | None = None,
    scene_endpoint: str | None = None,
    resample_mode: str | None = None,
    router: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "intent": intent,
        "state": state,
        "reason": reason,
        "signals": signals or [],
        "requires_clarification": requires_clarification,
        "question": question,
        "options": options or [],
        "scene_endpoint": scene_endpoint,
        "resample_mode": resample_mode,
        "router": router or {},
    }


def _resolve_explicit_intent(action: str | None) -> str | None:
    normalized = _normalize_text(action).lower()
    if normalized in {"create_scene_graph", "create_graph"}:
        return STATE_CREATE_SCENE_GRAPH
    if normalized in {"edit_scene_graph", "edit_graph"}:
        return STATE_EDIT_SCENE_GRAPH
    if normalized in {"graph", "scene_graph"}:
        return "scene_graph"
    if normalized in {"run_real2sim", "real2sim"}:
        return STATE_RUN_REAL2SIM
    if normalized in {"generate_scene", "scene", "scene_new"}:
        return STATE_GENERATE_SCENE
    return None


def _resolve_explicit_scene_endpoint(action: str | None, explicit_value: str | None) -> str | None:
    normalized_explicit = _normalize_scene_endpoint_value(explicit_value)
    if normalized_explicit is not None:
        return normalized_explicit
    normalized_action = _normalize_text(action).lower()
    if normalized_action in {"scene", "scene_new"}:
        return normalized_action
    return None


def _decide_graph_intent(
    text: str,
    *,
    scene_graph: dict[str, Any] | None,
    explicit_intent: str | None,
    has_uploaded_image: bool,
    has_saved_image: bool,
) -> dict[str, Any]:
    scene_exists = _scene_graph_exists(scene_graph)
    normalized = _normalize_text(text)

    if explicit_intent == STATE_CREATE_SCENE_GRAPH:
        if not normalized and not has_uploaded_image and not has_saved_image:
            question = "Upload a reference image or describe the scene so I can create a new scene graph."
            return _decision_payload(
                intent=STATE_CREATE_SCENE_GRAPH,
                state=STATE_NEEDS_CLARIFICATION,
                reason="Explicit create_scene_graph action was requested without any scene description or image.",
                requires_clarification=True,
                question=question,
            )
        return _decision_payload(
            intent=STATE_CREATE_SCENE_GRAPH,
            state=STATE_CREATE_SCENE_GRAPH,
            reason="Explicit create_scene_graph action was provided.",
        )

    if explicit_intent == STATE_EDIT_SCENE_GRAPH:
        if not scene_exists:
            question = "There is no scene graph in this run yet. Should I create a new scene graph first?"
            options = _bootstrap_scene_graph_options(from_saved_image=has_saved_image or has_uploaded_image)
            return _decision_payload(
                intent=STATE_EDIT_SCENE_GRAPH,
                state=STATE_NEEDS_CLARIFICATION,
                reason="Explicit edit_scene_graph action was requested, but the current run has no scene graph.",
                requires_clarification=True,
                question=question,
                options=options,
            )
        if not normalized:
            question = "Tell me what to change in the current scene graph."
            return _decision_payload(
                intent=STATE_EDIT_SCENE_GRAPH,
                state=STATE_NEEDS_CLARIFICATION,
                reason="Editing an existing scene graph requires a text instruction.",
                requires_clarification=True,
                question=question,
            )
        return _decision_payload(
            intent=STATE_EDIT_SCENE_GRAPH,
            state=STATE_EDIT_SCENE_GRAPH,
            reason="Explicit edit_scene_graph action was provided.",
        )

    question = "Should I create a new scene graph from this input, or treat it as an edit to the current scene graph?"
    options = _graph_mode_options()
    return _decision_payload(
        intent=STATE_UNDERSTAND_REQUEST,
        state=STATE_NEEDS_CLARIFICATION,
        reason="The explicit scene-graph request does not clearly say whether to replace or edit the current graph.",
        requires_clarification=True,
        question=question,
        options=options,
    )


def _build_top_level_clarification(
    *,
    reason: str,
    scene_graph: dict[str, Any] | None,
    has_saved_image: bool,
    suggested_focus: str | None = None,
    router: dict[str, Any] | None = None,
) -> dict[str, Any]:
    has_scene_graph = _scene_graph_exists(scene_graph)
    focus = _normalize_text(suggested_focus).lower() or "intent"

    if focus == "graph_mode" and has_scene_graph:
        question = "Should I create a new scene graph from this input, or edit the current scene graph?"
        options = _graph_mode_options()
    else:
        question = "Tell me whether you want to create/edit the scene graph, run Real2Sim, or generate the scene."
        options = _generic_intent_options(
            has_scene_graph=has_scene_graph,
            has_saved_image=has_saved_image,
        )

    return _decision_payload(
        intent=STATE_UNDERSTAND_REQUEST,
        state=STATE_NEEDS_CLARIFICATION,
        reason=reason,
        requires_clarification=True,
        question=question,
        options=options,
        router=router,
    )


def _map_agent_route_intent(intent: str | None) -> str | None:
    normalized = _normalize_text(intent).lower()
    if normalized == STATE_CREATE_SCENE_GRAPH:
        return STATE_CREATE_SCENE_GRAPH
    if normalized == STATE_EDIT_SCENE_GRAPH:
        return STATE_EDIT_SCENE_GRAPH
    if normalized == STATE_RUN_REAL2SIM:
        return STATE_RUN_REAL2SIM
    if normalized == STATE_GENERATE_SCENE:
        return STATE_GENERATE_SCENE
    return None


def _decide_request(
    text: str,
    *,
    explicit_action: str | None,
    scene_endpoint: str | None,
    resample_mode: str | None,
    scene_graph: dict[str, Any] | None,
    has_uploaded_image: bool,
    has_saved_image: bool,
) -> dict[str, Any]:
    explicit_intent = _resolve_explicit_intent(explicit_action)

    if explicit_intent == STATE_RUN_REAL2SIM:
        return _decision_payload(
            intent=STATE_RUN_REAL2SIM,
            state=STATE_RUN_REAL2SIM,
            reason="Explicit Real2Sim action was provided.",
        )
    if explicit_intent == STATE_GENERATE_SCENE:
        return _decision_payload(
            intent=STATE_GENERATE_SCENE,
            state=STATE_GENERATE_SCENE,
            reason="Explicit scene-generation action was provided.",
            scene_endpoint=_resolve_explicit_scene_endpoint(explicit_action, scene_endpoint),
            resample_mode=_normalize_resample_mode_value(resample_mode),
        )
    if explicit_intent in {STATE_CREATE_SCENE_GRAPH, STATE_EDIT_SCENE_GRAPH}:
        return _decide_graph_intent(
            text,
            scene_graph=scene_graph,
            explicit_intent=explicit_intent,
            has_uploaded_image=has_uploaded_image,
            has_saved_image=has_saved_image,
        )

    normalized = _normalize_text(text)
    if not normalized and not has_uploaded_image:
        return _build_top_level_clarification(
            reason="The request did not include a text instruction or a new image upload.",
            scene_graph=scene_graph,
            has_saved_image=has_saved_image,
        )

    try:
        router = route_agent_request(
            normalized,
            scene_graph=scene_graph,
            has_uploaded_image=has_uploaded_image,
            has_saved_image=has_saved_image,
        )
    except Exception as exc:
        return _build_top_level_clarification(
            reason=f"Top-level LLM routing is unavailable right now: {exc}",
            scene_graph=scene_graph,
            has_saved_image=has_saved_image,
        )

    route_intent = _map_agent_route_intent(router.get("intent"))
    confidence = float(router.get("confidence") or 0.0)
    scene_endpoint = _normalize_scene_endpoint_value(router.get("scene_endpoint"))
    routed_resample_mode = _normalize_resample_mode_value(router.get("resample_mode"))
    clarification_focus = _normalize_text(router.get("clarification_focus")).lower() or "intent"
    route_reason = str(router.get("reason") or "The top-level router did not provide a reason.")

    if route_intent is None or router.get("intent") == "clarification":
        return _build_top_level_clarification(
            reason=route_reason,
            scene_graph=scene_graph,
            has_saved_image=has_saved_image,
            suggested_focus=clarification_focus,
            router=router,
        )

    if confidence < TOP_LEVEL_ROUTE_CONFIDENCE_THRESHOLD:
        return _build_top_level_clarification(
            reason=f"{route_reason} Confidence {confidence:.2f} is below the execution threshold.",
            scene_graph=scene_graph,
            has_saved_image=has_saved_image,
            suggested_focus=clarification_focus,
            router=router,
        )

    return _decision_payload(
        intent=route_intent,
        state=route_intent,
        reason=route_reason,
        scene_endpoint=scene_endpoint,
        resample_mode=routed_resample_mode,
        router=router,
    )


def _load_agent_state(context: RuntimeContext) -> dict[str, Any]:
    state_path = context.session_root / _AGENT_STATE_FILENAME
    if not state_path.exists():
        return {
            "session_id": context.session_id,
            "current_run_id": context.run_id,
            "current_state": STATE_UNDERSTAND_REQUEST,
            "pending_question": None,
            "last_intent": None,
            "last_layout_strategy": None,
            "last_completed_state": None,
            "last_decision": None,
            "latest_real2sim_run_id": None,
            "latest_scene_generation_run_id": None,
            "runs": {},
            "history": [],
        }
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("session_id", context.session_id)
    payload.setdefault("current_run_id", context.run_id)
    payload.setdefault("current_state", STATE_UNDERSTAND_REQUEST)
    payload.setdefault("pending_question", None)
    payload.setdefault("last_intent", None)
    payload.setdefault("last_layout_strategy", None)
    payload.setdefault("last_completed_state", None)
    payload.setdefault("last_decision", None)
    payload.setdefault("latest_real2sim_run_id", None)
    payload.setdefault("latest_scene_generation_run_id", None)
    payload.setdefault("runs", {})
    payload.setdefault("history", [])
    return payload


def _save_agent_state(context: RuntimeContext, state: dict[str, Any]) -> None:
    state_path = context.session_root / _AGENT_STATE_FILENAME
    context.ensure()
    state["session_id"] = context.session_id
    state["current_run_id"] = context.run_id
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_agent_history(state: dict[str, Any], role: str, content: str) -> None:
    history = state.setdefault("history", [])
    if not isinstance(history, list):
        history = []
        state["history"] = history
    history.append({"role": role, "content": content})
    if len(history) > 20:
        del history[:-20]


def _set_current_state(
    state: dict[str, Any],
    current_state: str,
    *,
    last_intent: str | None = None,
    last_completed_state: str | None = None,
    last_decision: dict[str, Any] | None = None,
) -> None:
    state["current_state"] = current_state
    if last_intent is not None:
        state["last_intent"] = last_intent
    if last_completed_state is not None:
        state["last_completed_state"] = last_completed_state
    if last_decision is not None:
        state["last_decision"] = last_decision


def _ensure_run_state(state: dict[str, Any], context: RuntimeContext) -> dict[str, Any]:
    runs = state.setdefault("runs", {})
    if not isinstance(runs, dict):
        runs = {}
        state["runs"] = runs
    run_state = runs.setdefault(context.run_id, {})
    if not isinstance(run_state, dict):
        run_state = {}
        runs[context.run_id] = run_state
    run_state.setdefault("run_id", context.run_id)
    run_state.setdefault("scene_graph_path", str(context.scene_graph_path))
    run_state.setdefault("latest_input_image", str(context.latest_input_image))
    run_state.setdefault("render_image", str(context.render_path))
    run_state.setdefault("placements_path", str(context.default_placements_path))
    return run_state


def _session_state_snapshot(state: dict[str, Any], context: RuntimeContext) -> dict[str, Any]:
    runs = state.get("runs") if isinstance(state.get("runs"), dict) else {}
    current_run = runs.get(context.run_id, {}) if isinstance(runs, dict) else {}
    history = state.get("history") if isinstance(state.get("history"), list) else []
    return {
        "session_id": context.session_id,
        "current_run_id": context.run_id,
        "current_state": state.get("current_state"),
        "last_intent": state.get("last_intent"),
        "last_completed_state": state.get("last_completed_state"),
        "last_layout_strategy": state.get("last_layout_strategy"),
        "latest_real2sim_run_id": state.get("latest_real2sim_run_id"),
        "latest_scene_generation_run_id": state.get("latest_scene_generation_run_id"),
        "history": history,
        "current_run": current_run,
    }


def _record_real2sim_state(
    state: dict[str, Any],
    context: RuntimeContext,
    *,
    status: str,
    job_id: str | None = None,
    artifacts: dict[str, Any] | None = None,
    error: str | None = None,
    error_info: dict[str, Any] | None = None,
    log_path: str | None = None,
    log_start_offset: int | None = None,
) -> dict[str, Any]:
    run_state = _ensure_run_state(state, context)
    real2sim_state = run_state.get("real2sim")
    if not isinstance(real2sim_state, dict):
        real2sim_state = {}
    real2sim_state.update(
        {
            "status": status,
            "job_id": job_id or real2sim_state.get("job_id"),
            "updated_at": _utcnow_iso(),
            "results_dir": str(context.real2sim_scene_results_dir),
        }
    )
    if artifacts is not None:
        real2sim_state["artifacts"] = artifacts
    if error is not None:
        real2sim_state["error"] = error
    elif "error" in real2sim_state and status != STATE_FAILED:
        real2sim_state.pop("error", None)
    if error_info is not None:
        real2sim_state["error_info"] = error_info
    elif "error_info" in real2sim_state and status != STATE_FAILED:
        real2sim_state.pop("error_info", None)
    if log_path is not None:
        real2sim_state["log_path"] = log_path
    if log_start_offset is not None:
        real2sim_state["log_start_offset"] = int(log_start_offset)
    run_state["real2sim"] = real2sim_state
    state["latest_real2sim_run_id"] = context.run_id
    return real2sim_state


def _record_scene_generation_state(
    state: dict[str, Any],
    context: RuntimeContext,
    *,
    scene_result: dict[str, Any],
    scene_endpoint: str,
    resample_mode: str,
) -> dict[str, Any]:
    run_state = _ensure_run_state(state, context)
    normalized = _normalize_scene_result_for_agent(context, scene_result)
    run_state["scene_generation"] = {
        "status": "succeeded",
        "updated_at": _utcnow_iso(),
        "scene_endpoint": scene_endpoint,
        "resample_mode": normalized.get("resample_mode") or resample_mode,
        "outputs": normalized,
    }
    state["latest_scene_generation_run_id"] = context.run_id
    return normalized


def sync_real2sim_job_to_session(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(job, dict):
        return None
    payload = job.get("payload")
    if not isinstance(payload, dict):
        return None
    session_id = payload.get("session_id")
    run_id = payload.get("run_id")
    if not isinstance(session_id, str) or not isinstance(run_id, str):
        return None

    try:
        context = resolve_runtime_context(session_id=session_id, run_id=run_id, create=True)
    except ValueError:
        return None
    if context is None:
        return None

    state = _load_agent_state(context)
    previous_real2sim = {}
    run_state = _ensure_run_state(state, context)
    if isinstance(run_state.get("real2sim"), dict):
        previous_real2sim = dict(run_state.get("real2sim") or {})
    artifacts = job.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    enriched_artifacts = _enrich_real2sim_artifacts(context, artifacts) if artifacts else _collect_real2sim_artifacts_for_context(context)
    status = str(job.get("status") or "unknown")
    error_info = job.get("error_info") if isinstance(job.get("error_info"), dict) else None
    _record_real2sim_state(
        state,
        context,
        status=status,
        job_id=str(job.get("job_id") or ""),
        artifacts=enriched_artifacts,
        error=str(job.get("error")) if job.get("error") else None,
        error_info=error_info,
        log_path=str(job.get("log_path") or context.real2sim_log_path),
        log_start_offset=int(job.get("log_start_offset") or 0),
    )
    previous_status = str(previous_real2sim.get("status") or "")
    if status == "succeeded" and previous_status != "succeeded":
        _append_agent_history(state, "assistant", "Real2Sim completed and saved the current run artifacts.")
        _set_current_state(
            state,
            STATE_COMPLETED,
            last_intent=STATE_RUN_REAL2SIM,
            last_completed_state=STATE_RUN_REAL2SIM,
            last_decision=state.get("last_decision") if isinstance(state.get("last_decision"), dict) else None,
        )
    if status == "failed" and previous_status != "failed":
        failure_message = None
        if isinstance(error_info, dict):
            failure_message = error_info.get("user_message")
        if not failure_message:
            failure_message = str(job.get("error") or "Real2Sim failed.")
        _append_agent_history(state, "assistant", f"Real2Sim failed: {failure_message}")
        _set_current_state(
            state,
            STATE_FAILED,
            last_intent=STATE_RUN_REAL2SIM,
            last_decision=state.get("last_decision") if isinstance(state.get("last_decision"), dict) else None,
        )
    _save_agent_state(context, state)
    return _session_state_snapshot(state, context)


def _agent_paths_payload(context: RuntimeContext) -> dict[str, str]:
    return {
        "scene_graph_path": str(context.scene_graph_path),
        "latest_input_image": str(context.latest_input_image),
        "render_image": str(context.render_path),
        "placements_path": str(context.default_placements_path),
        "real2sim_results_dir": str(context.real2sim_scene_results_dir),
        "real2sim_manifest_path": str(context.real2sim_manifest_path),
        "real2sim_log_path": str(context.real2sim_log_path),
        "scene_usd_path": str(context.scene_service_usd_path),
    }


def _agent_summary_from_state(state: dict[str, Any], context: RuntimeContext) -> dict[str, Any]:
    current_state = str(state.get("current_state") or STATE_UNDERSTAND_REQUEST)
    pending_question = state.get("pending_question") if isinstance(state.get("pending_question"), dict) else None
    last_decision = state.get("last_decision") if isinstance(state.get("last_decision"), dict) else {}
    current_run = _ensure_run_state(state, context)
    real2sim_state = current_run.get("real2sim") if isinstance(current_run.get("real2sim"), dict) else {}
    scene_state = current_run.get("scene_generation") if isinstance(current_run.get("scene_generation"), dict) else {}
    last_completed_state = str(state.get("last_completed_state") or "") or None
    intent = str(state.get("last_intent") or current_state or STATE_UNDERSTAND_REQUEST)

    reason = str(last_decision.get("reason") or "")
    question = str(pending_question.get("question") or "") if pending_question else None
    options = list(pending_question.get("options") or []) if pending_question else []
    message = "The agent is ready."
    outcome = "completed"

    if pending_question is not None:
        if current_state == STATE_AWAIT_LAYOUT_STRATEGY:
            message = "I need a layout strategy before generating the scene."
        else:
            message = "I need more information before continuing."
        outcome = "needs_user_input"
    elif current_state == STATE_RUN_REAL2SIM:
        status = str(real2sim_state.get("status") or "running")
        if status in {"queued", "running"}:
            message = "Real2Sim is running for the current run."
            outcome = "started_job"
        elif status == "failed":
            error_info = real2sim_state.get("error_info") if isinstance(real2sim_state.get("error_info"), dict) else {}
            message = str(error_info.get("user_message") or real2sim_state.get("error") or "Real2Sim failed for the current run.")
            outcome = "failed"
        elif status == "succeeded":
            message = "Real2Sim completed for the current run."
    elif current_state == STATE_COMPLETED:
        if last_completed_state == STATE_CREATE_SCENE_GRAPH:
            message = "Created a new scene graph for the current run."
        elif last_completed_state == STATE_EDIT_SCENE_GRAPH:
            message = "Updated the current scene graph."
        elif last_completed_state == STATE_RUN_REAL2SIM:
            message = "Real2Sim completed for the current run."
        elif last_completed_state == STATE_GENERATE_SCENE:
            if isinstance(scene_state.get("outputs"), dict):
                strategy = scene_state.get("resample_mode") or state.get("last_layout_strategy") or "joint"
                message = f"Scene generated with layout strategy '{strategy}'."
            else:
                message = "Scene generation completed."
        else:
            message = "The last agent action completed."
    elif current_state == STATE_FAILED:
        message = "The last agent action failed."
        outcome = "failed"

    return {
        "state": current_state,
        "intent": intent,
        "message": message,
        "question": question,
        "reason": reason or None,
        "decision": last_decision,
        "options": options,
        "pending_question": pending_question,
        "completed_state": last_completed_state,
        "outcome": outcome,
    }


def get_agent_state_response(*, session_id: str | None, run_id: str | None) -> dict[str, Any]:
    context = resolve_runtime_context(session_id=session_id, run_id=run_id, create=True)
    if context is None:
        context = create_session()
    context.ensure()

    state = _load_agent_state(context)
    state["current_run_id"] = context.run_id
    current_run = _ensure_run_state(state, context)
    _save_agent_state(context, state)

    agent_summary = _agent_summary_from_state(state, context)
    scene_graph = _load_scene_graph(context.scene_graph_path)
    scene_result = current_run.get("scene_generation", {}).get("outputs")
    scene_result_payload = dict(scene_result) if isinstance(scene_result, dict) else None
    if scene_result_payload is not None:
        scene_result_payload = _normalize_scene_result_for_agent(context, scene_result_payload)

    real2sim_state = current_run.get("real2sim") if isinstance(current_run.get("real2sim"), dict) else {}
    error_info = real2sim_state.get("error_info") if isinstance(real2sim_state.get("error_info"), dict) else None
    real2sim_job = None
    if str(real2sim_state.get("status") or "") in {"queued", "running"} and real2sim_state.get("job_id"):
        real2sim_job = {
            "job_id": str(real2sim_state.get("job_id")),
            "log_path": str(real2sim_state.get("log_path") or context.real2sim_log_path),
            "log_start_offset": int(real2sim_state.get("log_start_offset") or 0),
        }

    return _build_agent_response(
        context,
        state=agent_summary["state"],
        intent=agent_summary["intent"],
        message=agent_summary["message"],
        question=agent_summary["question"],
        reason=agent_summary["reason"],
        decision=agent_summary["decision"],
        options=agent_summary["options"],
        pending_question=agent_summary["pending_question"],
        completed_state=agent_summary["completed_state"],
        outcome=agent_summary["outcome"],
        session_state=_session_state_snapshot(state, context),
        scene_graph=scene_graph,
        scene_result=scene_result_payload,
        error_info=error_info,
        real2sim_job=real2sim_job,
    )


def _build_agent_response(
    context: RuntimeContext,
    *,
    state: str,
    intent: str,
    message: str,
    question: str | None = None,
    reason: str | None = None,
    decision: dict[str, Any] | None = None,
    options: list[dict[str, Any]] | None = None,
    pending_question: dict[str, Any] | None = None,
    completed_state: str | None = None,
    outcome: str | None = None,
    session_state: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "session_id": context.session_id,
        "run_id": context.run_id,
        "agent": {
            "state": state,
            "intent": intent,
            "message": message,
            "question": question,
            "reason": reason,
            "decision": decision or {},
            "options": options or [],
            "pending_question": pending_question,
            "completed_state": completed_state,
            "outcome": outcome or ("needs_user_input" if question else "completed"),
        },
        "paths": _agent_paths_payload(context),
        "session_state": session_state,
        **extra,
    }


def _graph_response_payload(
    context: RuntimeContext,
    result: dict[str, Any],
    *,
    completed_state: str,
    reason: str,
    decision: dict[str, Any],
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _build_agent_response(
        context,
        state=STATE_COMPLETED,
        intent=completed_state,
        message=result.get("assistant_message") or "Scene graph updated.",
        reason=reason,
        decision=decision,
        completed_state=completed_state,
        session_state=session_state,
        scene_graph=result.get("scene_graph"),
        placements=result.get("placements"),
        warnings=result.get("warnings") or [],
    )


def _clarification_response(
    context: RuntimeContext,
    *,
    state: str,
    intent: str,
    message: str,
    question: str,
    reason: str,
    decision: dict[str, Any],
    pending_question: dict[str, Any],
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _build_agent_response(
        context,
        state=state,
        intent=intent,
        message=message,
        question=question,
        reason=reason,
        decision=decision,
        options=pending_question.get("options") or [],
        pending_question=pending_question,
        outcome="needs_user_input",
        session_state=session_state,
    )


def _scene_service_payload(
    context: RuntimeContext,
    *,
    scene_endpoint: str,
    resample_mode: str,
    seed: int | None = None,
) -> dict[str, Any]:
    payload = {
        "session_id": context.session_id,
        "run_id": context.run_id,
        "camera_eye": [18.0, 0.0, 18.0],
        "camera_target": [0.0, 0.0, 1.0],
        "frames": 20,
        "capture_frame": 10,
        "resolution": [1280, 720],
        "use_default_ground": True,
        "default_ground_z_offset": -0.05,
        "generate_room": True,
        "room_include_back_wall": True,
        "room_include_left_wall": True,
        "room_include_right_wall": True,
        "room_include_front_wall": False,
        "resample_mode": resample_mode if scene_endpoint == "scene_new" else "joint",
    }
    if seed is not None:
        payload["seed"] = int(seed)
    return payload


def _run_scene_service(
    context: RuntimeContext,
    *,
    scene_endpoint: str,
    resample_mode: str,
    scene_service_url: str,
    seed: int | None = None,
) -> dict[str, Any]:
    payload = _scene_service_payload(
        context,
        scene_endpoint=scene_endpoint,
        resample_mode=resample_mode,
        seed=seed,
    )
    url = scene_service_url.rstrip("/") + f"/{scene_endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=(10, 600))
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Scene service is unavailable at {scene_service_url}. Start the service on port 8001.") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Scene service timed out while generating the scene.") from exc

    try:
        data = response.json()
    except ValueError:
        data = {"error": response.text[:500]}

    if not response.ok:
        detail = data.get("detail") if isinstance(data, dict) else data
        if isinstance(detail, dict):
            detail = json.dumps(detail, ensure_ascii=False)
        message = detail or (data.get("error") if isinstance(data, dict) else None) or f"HTTP {response.status_code}"
        raise RuntimeError(f"Scene service failed: {message}")
    if not isinstance(data, dict):
        raise RuntimeError("Scene service returned an invalid JSON payload.")
    return data


def _looks_like_layout_conflict_error(message: str | None) -> bool:
    normalized = _normalize_text(message).lower()
    return "collision-free layout" in normalized or "layout collision" in normalized


def _load_manifest_warnings(context: RuntimeContext) -> list[str]:
    manifest_path = context.real2sim_manifest_path
    if not manifest_path.exists():
        return []
    payload = read_json_file(manifest_path)
    if not isinstance(payload, dict):
        return []

    warnings: list[str] = []
    unmatched_scene_paths = [
        str(path)
        for path in payload.get("unmatched_scene_paths", [])
        if isinstance(path, str) and path
    ]
    unmatched_outputs = [
        str(output_name)
        for output_name in payload.get("unmatched_outputs", [])
        if isinstance(output_name, str) and output_name
    ]
    if unmatched_scene_paths:
        warnings.append(
            "Real2Sim manifest is still missing node bindings for: "
            + ", ".join(unmatched_scene_paths)
        )
    if unmatched_outputs:
        warnings.append(
            "Real2Sim produced outputs that are not bound to scene nodes: "
            + ", ".join(unmatched_outputs)
        )
    return warnings


def _run_scene_service_with_repair(
    context: RuntimeContext,
    *,
    scene_endpoint: str,
    resample_mode: str,
    scene_service_url: str,
    scene_graph: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    requested_mode = resample_mode

    try:
        return (
            _run_scene_service(
                context,
                scene_endpoint=scene_endpoint,
                resample_mode=requested_mode,
                scene_service_url=scene_service_url,
            ),
            warnings,
        )
    except RuntimeError as exc:
        if scene_endpoint != "scene_new" or not _looks_like_layout_conflict_error(str(exc)):
            raise

        retry_seed = random.SystemRandom().randrange(0, 2**32 - 1)
        try:
            result = _run_scene_service(
                context,
                scene_endpoint=scene_endpoint,
                resample_mode=requested_mode,
                scene_service_url=scene_service_url,
                seed=retry_seed,
            )
            warnings.append(
                "Scene generation hit layout conflicts once and retried automatically with a new seed."
            )
            return result, warnings
        except RuntimeError as retry_exc:
            if requested_mode != "lock_real2sim" or not _has_real2sim_objects(scene_graph):
                raise retry_exc

            fallback_seed = random.SystemRandom().randrange(0, 2**32 - 1)
            result = _run_scene_service(
                context,
                scene_endpoint=scene_endpoint,
                resample_mode="joint",
                scene_service_url=scene_service_url,
                seed=fallback_seed,
            )
            warnings.extend(
                [
                    "Lock Real2Sim layout conflicted repeatedly, so the agent retried automatically with joint resampling.",
                    "Review the latest Real2Sim manifest before trusting locked-relative placement behavior.",
                ]
            )
            return result, warnings


def _continue_pending_question(
    context: RuntimeContext,
    state: dict[str, Any],
    *,
    text: str,
    resample_mode: str | None,
    scene_service_url: str,
) -> dict[str, Any] | None:
    pending = state.get("pending_question")
    if not isinstance(pending, dict):
        return None
    if pending.get("run_id") not in {None, context.run_id}:
        state["pending_question"] = None
        _set_current_state(state, STATE_UNDERSTAND_REQUEST)
        _save_agent_state(context, state)
        return None

    if pending.get("type") != "layout_strategy":
        state["pending_question"] = None
        _set_current_state(state, STATE_UNDERSTAND_REQUEST)
        _save_agent_state(context, state)
        return None

    strategy = resample_mode or _parse_layout_strategy(text)
    decision = _decision_payload(
        intent=STATE_GENERATE_SCENE,
        state=STATE_AWAIT_LAYOUT_STRATEGY,
        reason="Scene generation is waiting for an explicit layout strategy because the scene contains real2sim objects.",
        requires_clarification=True,
        question=str(pending.get("question") or ""),
        options=pending.get("options") or _layout_strategy_options(),
    )
    if strategy is None:
        question = str(pending.get("question") or "Choose a layout strategy: joint or lock_real2sim.")
        message = "I still need a layout strategy before generating the scene."
        _append_agent_history(state, "assistant", question)
        _set_current_state(state, STATE_AWAIT_LAYOUT_STRATEGY, last_intent=STATE_GENERATE_SCENE, last_decision=decision)
        _save_agent_state(context, state)
        return _clarification_response(
            context,
            state=STATE_AWAIT_LAYOUT_STRATEGY,
            intent=STATE_GENERATE_SCENE,
            message=message,
            question=question,
            reason=decision["reason"],
            decision=decision,
            pending_question=pending,
            session_state=_session_state_snapshot(state, context),
        )

    scene_endpoint = str(pending.get("scene_endpoint") or "scene_new")
    result, repair_warnings = _run_scene_service_with_repair(
        context,
        scene_endpoint=scene_endpoint,
        resample_mode=strategy,
        scene_service_url=scene_service_url,
        scene_graph=_load_scene_graph(context.scene_graph_path),
    )
    repair_warnings.extend(_load_manifest_warnings(context))
    normalized_result = _record_scene_generation_state(
        state,
        context,
        scene_result=result,
        scene_endpoint=scene_endpoint,
        resample_mode=strategy,
    )
    effective_strategy = str(normalized_result.get("resample_mode") or strategy)
    state["pending_question"] = None
    state["last_layout_strategy"] = effective_strategy
    _append_agent_history(state, "assistant", f"Generated scene with strategy {effective_strategy}.")
    _set_current_state(
        state,
        STATE_COMPLETED,
        last_intent=STATE_GENERATE_SCENE,
        last_completed_state=STATE_GENERATE_SCENE,
        last_decision=decision,
    )
    _save_agent_state(context, state)
    return _build_agent_response(
        context,
        state=STATE_COMPLETED,
        intent=STATE_GENERATE_SCENE,
        message=f"Scene generated with layout strategy '{effective_strategy}'.",
        reason="Resolved the pending layout strategy question and completed scene generation.",
        decision=decision,
        completed_state=STATE_GENERATE_SCENE,
        session_state=_session_state_snapshot(state, context),
        scene_result=normalized_result,
        warnings=repair_warnings,
    )


def _normalize_react_action(value: str | None) -> str | None:
    normalized = _normalize_text(value).lower()
    if normalized in {
        STATE_CREATE_SCENE_GRAPH,
        STATE_EDIT_SCENE_GRAPH,
        STATE_RUN_REAL2SIM,
        STATE_GENERATE_SCENE,
        "ask_clarification",
        "finish",
    }:
        return normalized
    return None


def _react_trace_entry(step: int, *, reason: str, action: str, observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step,
        "reason": reason,
        "action": action,
        "observation": observation,
    }


def _react_finish_response(
    context: RuntimeContext,
    state: dict[str, Any],
    *,
    completed_state: str,
    decision: dict[str, Any],
    message: str,
    latest_graph_result: dict[str, Any] | None,
    latest_scene_result: dict[str, Any] | None,
    warnings: list[str],
    react_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    state["pending_question"] = None
    _append_agent_history(state, "assistant", message)
    _set_current_state(
        state,
        STATE_COMPLETED,
        last_intent=completed_state,
        last_completed_state=completed_state,
        last_decision=decision,
    )
    _save_agent_state(context, state)
    return _build_agent_response(
        context,
        state=STATE_COMPLETED,
        intent=completed_state,
        message=message,
        reason=decision["reason"],
        decision=decision,
        completed_state=completed_state,
        session_state=_session_state_snapshot(state, context),
        scene_graph=latest_graph_result.get("scene_graph") if latest_graph_result else None,
        placements=latest_graph_result.get("placements") if latest_graph_result else None,
        scene_result=latest_scene_result,
        warnings=warnings,
        react_trace=react_trace,
    )


def _react_clarification_response(
    context: RuntimeContext,
    state: dict[str, Any],
    *,
    question: str,
    clarification_kind: str,
    reason: str,
    react_trace: list[dict[str, Any]],
    scene_endpoint: str | None = None,
) -> dict[str, Any]:
    normalized_kind = _normalize_text(clarification_kind).lower()
    if normalized_kind == "layout_strategy":
        options = _layout_strategy_options()
        pending_type = "layout_strategy"
        agent_state = STATE_AWAIT_LAYOUT_STRATEGY
        intent = STATE_GENERATE_SCENE
    elif normalized_kind == "graph_mode":
        options = _graph_mode_options()
        pending_type = "intent_clarification"
        agent_state = STATE_NEEDS_CLARIFICATION
        intent = STATE_UNDERSTAND_REQUEST
    else:
        options = _generic_intent_options(
            has_scene_graph=context.scene_graph_path.exists(),
            has_saved_image=context.latest_input_image.exists(),
        )
        pending_type = "intent_clarification"
        agent_state = STATE_NEEDS_CLARIFICATION
        intent = STATE_UNDERSTAND_REQUEST

    decision = _decision_payload(
        intent=intent,
        state=agent_state,
        reason=reason,
        requires_clarification=True,
        question=question,
        options=options,
        router={"react_trace": react_trace},
        scene_endpoint=scene_endpoint,
    )
    pending_question = _question_payload(
        pending_type,
        question,
        options=options,
        run_id=context.run_id,
        scene_endpoint=scene_endpoint if pending_type == "layout_strategy" else None,
    )
    state["pending_question"] = pending_question
    _append_agent_history(state, "assistant", question)
    _set_current_state(state, agent_state, last_intent=intent, last_decision=decision)
    _save_agent_state(context, state)
    return _clarification_response(
        context,
        state=agent_state,
        intent=intent,
        message=reason,
        question=question,
        reason=reason,
        decision=decision,
        pending_question=pending_question,
        session_state=_session_state_snapshot(state, context),
    )


def _run_react_loop(
    context: RuntimeContext,
    state: dict[str, Any],
    *,
    instruction: str,
    image_bytes: bytes | None,
    class_names_raw: str,
    has_saved_image: bool,
    has_uploaded_image: bool,
    scene_service_url: str,
) -> dict[str, Any]:
    react_trace: list[dict[str, Any]] = []
    latest_graph_result: dict[str, Any] | None = None
    latest_scene_result: dict[str, Any] | None = None
    accumulated_warnings: list[str] = []
    last_generate_endpoint: str | None = None

    for step_index in range(1, REACT_MAX_STEPS + 1):
        current_scene_graph = _load_scene_graph(context.scene_graph_path) or (
            latest_graph_result.get("scene_graph") if latest_graph_result else None
        )
        try:
            reaction = react_agent_step(
                instruction,
                scene_graph=current_scene_graph,
                has_uploaded_image=has_uploaded_image,
                has_saved_image=has_saved_image,
                trace=react_trace,
            )
        except Exception as exc:
            return _react_clarification_response(
                context,
                state,
                question="I could not complete the reasoning loop. Tell me whether you want graph editing, Real2Sim, or scene generation.",
                clarification_kind="intent",
                reason=f"ReAct loop is unavailable right now: {exc}",
                react_trace=react_trace,
            )

        action = _normalize_react_action(reaction.get("action"))
        confidence = float(reaction.get("confidence") or 0.0)
        reason = str(reaction.get("reason") or "The ReAct loop did not provide a reason.")
        step_instruction = _normalize_text(reaction.get("instruction")) or instruction
        scene_endpoint = _normalize_scene_endpoint_value(reaction.get("scene_endpoint")) or "scene_new"
        resample_mode = _normalize_resample_mode_value(reaction.get("resample_mode"))
        question = _normalize_text(reaction.get("question"))
        clarification_kind = _normalize_text(reaction.get("clarification_kind")).lower() or "intent"

        if action is None or confidence < TOP_LEVEL_ROUTE_CONFIDENCE_THRESHOLD:
            return _react_clarification_response(
                context,
                state,
                question=question or "Tell me whether you want to edit the graph, run Real2Sim, or generate the scene.",
                clarification_kind=clarification_kind,
                reason=f"{reason} Confidence {confidence:.2f} is below the execution threshold.",
                react_trace=react_trace,
                scene_endpoint=last_generate_endpoint,
            )

        if action == "ask_clarification":
            return _react_clarification_response(
                context,
                state,
                question=question or "I need one detail before continuing.",
                clarification_kind=clarification_kind,
                reason=reason,
                react_trace=react_trace,
                scene_endpoint=last_generate_endpoint,
            )

        if action == "finish":
            completed_state = STATE_GENERATE_SCENE if latest_scene_result else (
                STATE_EDIT_SCENE_GRAPH if latest_graph_result else STATE_UNDERSTAND_REQUEST
            )
            return _react_finish_response(
                context,
                state,
                completed_state=completed_state,
                decision=_decision_payload(intent=completed_state, state=completed_state, reason=reason, router=reaction),
                message="Finished the ReAct loop for this request.",
                latest_graph_result=latest_graph_result,
                latest_scene_result=latest_scene_result,
                warnings=accumulated_warnings,
                react_trace=react_trace,
            )

        if action == STATE_CREATE_SCENE_GRAPH:
            latest_graph_result = create_scene_graph_from_input(
                step_instruction,
                class_names_raw=class_names_raw,
                image_bytes=image_bytes,
                scene_graph_path=context.scene_graph_path,
                placements_path=context.default_placements_path,
                latest_input_image_path=context.latest_input_image,
                reuse_saved_image=image_bytes is None,
            )
            current_scene_graph = latest_graph_result.get("scene_graph") or _load_scene_graph(context.scene_graph_path)
            observation = {
                "status": "ok",
                "code": "scene_graph_created",
                "message": latest_graph_result.get("assistant_message") or "Scene graph created.",
                "warning_count": len(latest_graph_result.get("warnings") or []),
            }
            accumulated_warnings.extend(latest_graph_result.get("warnings") or [])
            react_trace.append(_react_trace_entry(step_index, reason=reason, action=action, observation=observation))
            continue

        if action == STATE_EDIT_SCENE_GRAPH:
            if current_scene_graph is None:
                react_trace.append(
                    _react_trace_entry(
                        step_index,
                        reason=reason,
                        action=action,
                        observation={
                            "status": "blocked",
                            "code": "missing_scene_graph",
                            "message": "Editing requires a scene graph first.",
                        },
                    )
                )
                continue
            latest_graph_result = apply_instruction(
                step_instruction,
                class_names_raw=class_names_raw,
                image_bytes=image_bytes,
                scene_graph_path=context.scene_graph_path,
                placements_path=context.default_placements_path,
                latest_input_image_path=context.latest_input_image,
                render_path=context.render_path,
            )
            observation = {
                "status": "ok",
                "code": "scene_graph_updated",
                "message": latest_graph_result.get("assistant_message") or "Scene graph updated.",
                "warning_count": len(latest_graph_result.get("warnings") or []),
            }
            accumulated_warnings.extend(latest_graph_result.get("warnings") or [])
            react_trace.append(_react_trace_entry(step_index, reason=reason, action=action, observation=observation))
            continue

        if action == STATE_RUN_REAL2SIM:
            if current_scene_graph is None:
                react_trace.append(
                    _react_trace_entry(
                        step_index,
                        reason=reason,
                        action=action,
                        observation={
                            "status": "blocked",
                            "code": "missing_scene_graph",
                            "message": "Real2Sim requires a scene graph first.",
                        },
                    )
                )
                continue
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
                state,
                context,
                status="running",
                job_id=str(job.get("job_id") or ""),
                artifacts=_collect_real2sim_artifacts_for_context(context),
                log_path=str(job.get("log_path") or context.real2sim_log_path),
                log_start_offset=int(job.get("log_start_offset") or 0),
            )
            react_trace.append(
                _react_trace_entry(
                    step_index,
                    reason=reason,
                    action=action,
                    observation={"status": "started_job", "code": "real2sim_started", "job_id": job.get("job_id")},
                )
            )
            state["pending_question"] = None
            _append_agent_history(state, "assistant", f"Started ReAct Real2Sim job {job['job_id']}.")
            decision = _decision_payload(intent=STATE_RUN_REAL2SIM, state=STATE_RUN_REAL2SIM, reason=reason, router=reaction)
            _set_current_state(state, STATE_RUN_REAL2SIM, last_intent=STATE_RUN_REAL2SIM, last_decision=decision)
            _save_agent_state(context, state)
            return _build_agent_response(
                context,
                state=STATE_RUN_REAL2SIM,
                intent=STATE_RUN_REAL2SIM,
                message="Started Real2Sim from the ReAct loop.",
                reason=reason,
                decision=decision,
                outcome="started_job",
                session_state=_session_state_snapshot(state, context),
                real2sim_job=job,
                scene_graph=latest_graph_result.get("scene_graph") if latest_graph_result else None,
                placements=latest_graph_result.get("placements") if latest_graph_result else None,
                warnings=accumulated_warnings,
                react_trace=react_trace,
            )

        if action == STATE_GENERATE_SCENE:
            if current_scene_graph is None:
                react_trace.append(
                    _react_trace_entry(
                        step_index,
                        reason=reason,
                        action=action,
                        observation={
                            "status": "blocked",
                            "code": "missing_scene_graph",
                            "message": "Scene generation requires a scene graph first.",
                        },
                    )
                )
                continue
            last_generate_endpoint = scene_endpoint
            if _has_real2sim_objects(current_scene_graph) and resample_mode is None:
                react_trace.append(
                    _react_trace_entry(
                        step_index,
                        reason=reason,
                        action=action,
                        observation={
                            "status": "blocked",
                            "code": "missing_layout_strategy",
                            "message": "Scene generation needs joint or lock_real2sim for graphs with real2sim objects.",
                            "scene_endpoint": scene_endpoint,
                        },
                    )
                )
                continue
            strategy = str(resample_mode or "joint")
            result, repair_warnings = _run_scene_service_with_repair(
                context,
                scene_endpoint=scene_endpoint,
                resample_mode=strategy,
                scene_service_url=scene_service_url,
                scene_graph=current_scene_graph,
            )
            accumulated_warnings.extend(repair_warnings)
            accumulated_warnings.extend(_load_manifest_warnings(context))
            latest_scene_result = _record_scene_generation_state(
                state,
                context,
                scene_result=result,
                scene_endpoint=scene_endpoint,
                resample_mode=strategy,
            )
            state["last_layout_strategy"] = str(latest_scene_result.get("resample_mode") or strategy)
            react_trace.append(
                _react_trace_entry(
                    step_index,
                    reason=reason,
                    action=action,
                    observation={"status": "completed", "code": "scene_generated", "scene_endpoint": scene_endpoint},
                )
            )
            return _react_finish_response(
                context,
                state,
                completed_state=STATE_GENERATE_SCENE,
                decision=_decision_payload(
                    intent=STATE_GENERATE_SCENE,
                    state=STATE_GENERATE_SCENE,
                    reason=reason,
                    scene_endpoint=scene_endpoint,
                    resample_mode=strategy,
                    router=reaction,
                ),
                message="Generated the scene from the ReAct loop.",
                latest_graph_result=latest_graph_result,
                latest_scene_result=latest_scene_result,
                warnings=accumulated_warnings,
                react_trace=react_trace,
            )

    return _react_clarification_response(
        context,
        state,
        question="I hit the reasoning step limit. Tell me the single next thing you want: create/edit graph, run Real2Sim, or generate scene.",
        clarification_kind="intent",
        reason=f"The ReAct loop hit its step limit of {REACT_MAX_STEPS}.",
        react_trace=react_trace,
        scene_endpoint=last_generate_endpoint,
    )


def handle_agent_message(
    *,
    session_id: str | None,
    run_id: str | None,
    text: str | None,
    image_bytes: bytes | None = None,
    class_names_raw: str = "",
    action: str | None = None,
    resample_mode: str | None = None,
    scene_endpoint: str | None = None,
    scene_service_url: str | None = None,
) -> dict[str, Any]:
    context = resolve_runtime_context(session_id=session_id, run_id=run_id, create=True)
    if context is None:
        context = create_session()
    context.ensure()

    normalized_text = _normalize_text(text)
    scene_graph = _load_scene_graph(context.scene_graph_path)
    has_uploaded_image = image_bytes is not None
    has_saved_image = context.latest_input_image.exists()

    state = _load_agent_state(context)
    state["current_run_id"] = context.run_id
    _ensure_run_state(state, context)
    _append_agent_history(state, "user", normalized_text or f"[image:{bool(image_bytes)} action:{action or ''}]")
    _set_current_state(state, STATE_UNDERSTAND_REQUEST)
    _save_agent_state(context, state)

    pending_result = _continue_pending_question(
        context,
        state,
        text=normalized_text,
        resample_mode=resample_mode,
        scene_service_url=scene_service_url or SCENE_SERVICE_URL,
    )
    if pending_result is not None:
        return pending_result

    if action is None and (normalized_text or has_uploaded_image):
        return _run_react_loop(
            context,
            state,
            instruction=normalized_text,
            image_bytes=image_bytes,
            class_names_raw=class_names_raw,
            has_saved_image=has_saved_image,
            has_uploaded_image=has_uploaded_image,
            scene_service_url=scene_service_url or SCENE_SERVICE_URL,
        )

    decision = _decide_request(
        normalized_text,
        explicit_action=action,
        scene_endpoint=scene_endpoint,
        resample_mode=resample_mode,
        scene_graph=scene_graph,
        has_uploaded_image=has_uploaded_image,
        has_saved_image=has_saved_image,
    )
    _set_current_state(
        state,
        decision["state"],
        last_intent=decision["intent"],
        last_decision=decision,
    )
    _save_agent_state(context, state)

    if decision["requires_clarification"]:
        question = str(decision["question"] or "I need more information before continuing.")
        pending_question = _question_payload(
            "layout_strategy" if decision["state"] == STATE_AWAIT_LAYOUT_STRATEGY else "intent_clarification",
            question,
            options=decision.get("options") or [],
            run_id=context.run_id,
        )
        state["pending_question"] = pending_question
        _append_agent_history(state, "assistant", question)
        _set_current_state(state, decision["state"], last_intent=decision["intent"], last_decision=decision)
        _save_agent_state(context, state)
        return _clarification_response(
            context,
            state=decision["state"],
            intent=decision["intent"],
            message=decision["reason"],
            question=question,
            reason=decision["reason"],
            decision=decision,
            pending_question=pending_question,
            session_state=_session_state_snapshot(state, context),
        )

    intent = decision["intent"]

    if intent == STATE_CREATE_SCENE_GRAPH:
        result = create_scene_graph_from_input(
            normalized_text,
            class_names_raw=class_names_raw,
            image_bytes=image_bytes,
            scene_graph_path=context.scene_graph_path,
            placements_path=context.default_placements_path,
            latest_input_image_path=context.latest_input_image,
            reuse_saved_image=image_bytes is None,
        )
        state["pending_question"] = None
        _append_agent_history(state, "assistant", str(result.get("assistant_message") or "Scene graph created."))
        _set_current_state(
            state,
            STATE_COMPLETED,
            last_intent=STATE_CREATE_SCENE_GRAPH,
            last_completed_state=STATE_CREATE_SCENE_GRAPH,
            last_decision=decision,
        )
        _save_agent_state(context, state)
        return _graph_response_payload(
            context,
            result,
            completed_state=STATE_CREATE_SCENE_GRAPH,
            reason=decision["reason"],
            decision=decision,
            session_state=_session_state_snapshot(state, context),
        )

    if intent == STATE_EDIT_SCENE_GRAPH:
        result = apply_instruction(
            normalized_text,
            class_names_raw=class_names_raw,
            image_bytes=image_bytes,
            scene_graph_path=context.scene_graph_path,
            placements_path=context.default_placements_path,
            latest_input_image_path=context.latest_input_image,
            render_path=context.render_path,
        )
        state["pending_question"] = None
        _append_agent_history(state, "assistant", str(result.get("assistant_message") or "Scene graph updated."))
        _set_current_state(
            state,
            STATE_COMPLETED,
            last_intent=STATE_EDIT_SCENE_GRAPH,
            last_completed_state=STATE_EDIT_SCENE_GRAPH,
            last_decision=decision,
        )
        _save_agent_state(context, state)
        return _graph_response_payload(
            context,
            result,
            completed_state=STATE_EDIT_SCENE_GRAPH,
            reason=decision["reason"],
            decision=decision,
            session_state=_session_state_snapshot(state, context),
        )

    if intent == STATE_RUN_REAL2SIM:
        if scene_graph is None:
            question = (
                "I need a scene graph before Real2Sim. "
                "Should I create one from the current image first?"
                if has_saved_image or has_uploaded_image
                else "I need a scene graph before Real2Sim. Upload a reference image or describe the scene first."
            )
            options = _bootstrap_scene_graph_options(from_saved_image=has_saved_image or has_uploaded_image)
            pending_question = _question_payload("real2sim_bootstrap", question, options=options, run_id=context.run_id)
            state["pending_question"] = pending_question
            _append_agent_history(state, "assistant", question)
            clarification_decision = _decision_payload(
                intent=STATE_RUN_REAL2SIM,
                state=STATE_NEEDS_CLARIFICATION,
                reason="Real2Sim requires an existing scene graph in the current run.",
                requires_clarification=True,
                question=question,
                options=options,
            )
            _set_current_state(state, STATE_NEEDS_CLARIFICATION, last_intent=STATE_RUN_REAL2SIM, last_decision=clarification_decision)
            _save_agent_state(context, state)
            return _clarification_response(
                context,
                state=STATE_NEEDS_CLARIFICATION,
                intent=STATE_RUN_REAL2SIM,
                message="I need a scene graph before Real2Sim.",
                question=question,
                reason=clarification_decision["reason"],
                decision=clarification_decision,
                pending_question=pending_question,
                session_state=_session_state_snapshot(state, context),
            )

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
            state,
            context,
            status="running",
            job_id=str(job.get("job_id") or ""),
            artifacts=_collect_real2sim_artifacts_for_context(context),
            log_path=str(job.get("log_path") or context.real2sim_log_path),
            log_start_offset=int(job.get("log_start_offset") or 0),
        )
        state["pending_question"] = None
        _append_agent_history(state, "assistant", f"Started Real2Sim job {job['job_id']}.")
        _set_current_state(state, STATE_RUN_REAL2SIM, last_intent=STATE_RUN_REAL2SIM, last_decision=decision)
        _save_agent_state(context, state)
        return _build_agent_response(
            context,
            state=STATE_RUN_REAL2SIM,
            intent=STATE_RUN_REAL2SIM,
            message="Started Real2Sim for the current run.",
            reason=decision["reason"],
            decision=decision,
            outcome="started_job",
            session_state=_session_state_snapshot(state, context),
            real2sim_job=job,
        )

    if intent == STATE_GENERATE_SCENE:
        if scene_graph is None:
            question = (
                "I need a scene graph before generating the scene. "
                "Should I create one from the current image first?"
                if has_saved_image or has_uploaded_image
                else "I need a scene graph before generating the scene. Upload a reference image or describe the scene first."
            )
            options = _bootstrap_scene_graph_options(from_saved_image=has_saved_image or has_uploaded_image)
            pending_question = _question_payload("scene_bootstrap", question, options=options, run_id=context.run_id)
            state["pending_question"] = pending_question
            _append_agent_history(state, "assistant", question)
            clarification_decision = _decision_payload(
                intent=STATE_GENERATE_SCENE,
                state=STATE_NEEDS_CLARIFICATION,
                reason="Scene generation requires an existing scene graph in the current run.",
                requires_clarification=True,
                question=question,
                options=options,
            )
            _set_current_state(state, STATE_NEEDS_CLARIFICATION, last_intent=STATE_GENERATE_SCENE, last_decision=clarification_decision)
            _save_agent_state(context, state)
            return _clarification_response(
                context,
                state=STATE_NEEDS_CLARIFICATION,
                intent=STATE_GENERATE_SCENE,
                message="I need a scene graph before generating the scene.",
                question=question,
                reason=clarification_decision["reason"],
                decision=clarification_decision,
                pending_question=pending_question,
                session_state=_session_state_snapshot(state, context),
            )

        resolved_endpoint = (
            _normalize_scene_endpoint_value(scene_endpoint)
            or _normalize_scene_endpoint_value(decision.get("scene_endpoint"))
            or _resolve_scene_endpoint(None, None)
        )
        strategy = (
            _normalize_resample_mode_value(resample_mode)
            or _normalize_resample_mode_value(decision.get("resample_mode"))
        )
        if _has_real2sim_objects(scene_graph) and strategy is None:
            question = (
                "This scene contains real2sim objects. Choose a layout strategy: "
                "`joint` to resample everything, or `lock_real2sim` to keep observed support chains rigid."
            )
            options = _layout_strategy_options()
            pending_question = _question_payload(
                "layout_strategy",
                question,
                options=options,
                run_id=context.run_id,
                scene_endpoint=resolved_endpoint,
            )
            layout_decision = _decision_payload(
                intent=STATE_GENERATE_SCENE,
                state=STATE_AWAIT_LAYOUT_STRATEGY,
                reason="Scene generation needs an explicit layout strategy because the current graph includes real2sim objects.",
                requires_clarification=True,
                question=question,
                options=options,
            )
            state["pending_question"] = pending_question
            _append_agent_history(state, "assistant", question)
            _set_current_state(state, STATE_AWAIT_LAYOUT_STRATEGY, last_intent=STATE_GENERATE_SCENE, last_decision=layout_decision)
            _save_agent_state(context, state)
            return _clarification_response(
                context,
                state=STATE_AWAIT_LAYOUT_STRATEGY,
                intent=STATE_GENERATE_SCENE,
                message="I need a layout strategy before generating the scene.",
                question=question,
                reason=layout_decision["reason"],
                decision=layout_decision,
                pending_question=pending_question,
                session_state=_session_state_snapshot(state, context),
            )

        strategy = strategy or "joint"
        result, repair_warnings = _run_scene_service_with_repair(
            context,
            scene_endpoint=resolved_endpoint,
            resample_mode=strategy,
            scene_service_url=scene_service_url or SCENE_SERVICE_URL,
            scene_graph=scene_graph,
        )
        repair_warnings.extend(_load_manifest_warnings(context))
        normalized_result = _record_scene_generation_state(
            state,
            context,
            scene_result=result,
            scene_endpoint=resolved_endpoint,
            resample_mode=strategy,
        )
        effective_strategy = str(normalized_result.get("resample_mode") or strategy)
        state["pending_question"] = None
        state["last_layout_strategy"] = effective_strategy
        _append_agent_history(state, "assistant", f"Generated scene with strategy {effective_strategy}.")
        _set_current_state(
            state,
            STATE_COMPLETED,
            last_intent=STATE_GENERATE_SCENE,
            last_completed_state=STATE_GENERATE_SCENE,
            last_decision=decision,
        )
        _save_agent_state(context, state)
        return _build_agent_response(
            context,
            state=STATE_COMPLETED,
            intent=STATE_GENERATE_SCENE,
            message=f"Scene generated with layout strategy '{effective_strategy}'.",
            reason=decision["reason"],
            decision=decision,
            completed_state=STATE_GENERATE_SCENE,
            session_state=_session_state_snapshot(state, context),
            scene_result=normalized_result,
            warnings=repair_warnings,
        )

    raise ValueError(f"Unsupported agent intent: {intent}")
