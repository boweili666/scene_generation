"""Plan -> Execute -> Reflect agent loop.

Three phases the user can drive separately:

1. `propose_plan`: an LLM call that emits a structured Plan (a list of
   tool steps + reasoning). The plan is persisted as `state["active_plan"]`
   but NOT executed. The UI can show it for human review / approval.

2. `execute_plan`: walks the active_plan step by step. "Instant" tools
   (inspect_state, create_scene_graph, generate_scene) run synchronously.
   "Long" tools (run_real2sim, run_scene_robot_collect) start their job,
   record a pause marker on the plan, and return — the loop does NOT
   poll. The UI's existing job monitors stream those jobs' logs; when
   the user nudges the plan again, execution resumes after the pause.

3. `reflect_plan`: an LLM call that summarizes what happened (using the
   plan + observations) and proposes whether the goal is done, what the
   next plan should be, or whether the user must answer something.

Tool definitions are reused verbatim from `agent_loop.TOOL_HANDLERS`;
this module only adds the planning / executing / reflecting layer on
top.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from ..config import SCENE_SERVICE_URL
from .agent_loop import LoopContext, TOOL_DEFINITIONS, TOOL_HANDLERS
from .agent_service import (
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_RUN_REAL2SIM,
    STATE_RUN_SCENE_ROBOT_COLLECT,
    _build_agent_response,
    _ensure_run_state,
    _load_agent_state,
    _save_agent_state,
    _session_state_snapshot,
    _set_current_state,
)
from .openai_service import _get_openai_client, _json_schema_format
from .runtime_context import (
    RuntimeContext,
    create_session,
    resolve_runtime_context,
)


PLANNER_MODEL = os.getenv(
    "AGENT_PLANNER_MODEL", os.getenv("AGENT_LOOP_MODEL", os.getenv("AGENT_ROUTER_MODEL", "gpt-5.4-mini"))
)
REFLECTOR_MODEL = os.getenv("AGENT_REFLECTOR_MODEL", PLANNER_MODEL)
MAX_PLAN_STEPS = int(os.getenv("AGENT_PLAN_MAX_STEPS", "10"))
MAX_PLAN_HISTORY = int(os.getenv("AGENT_PLAN_MAX_HISTORY", "20"))


LONG_RUNNING_TOOLS = {"run_real2sim", "run_scene_robot_collect"}
TOOL_NAMES = sorted(TOOL_HANDLERS.keys())


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": TOOL_NAMES,
                    },
                    "args_json": {
                        "type": "string",
                        "description": "JSON-encoded argument object for the tool. Empty object {} when no args.",
                    },
                    "why": {"type": "string"},
                },
                "required": ["tool", "args_json", "why"],
                "additionalProperties": False,
            },
        },
        "ask_user": {
            "type": ["string", "null"],
            "description": "Set to a question only when essential info is missing. When null, the plan is ready to execute.",
        },
    },
    "required": ["goal", "steps", "ask_user"],
    "additionalProperties": False,
}


REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "goal_complete": {"type": "boolean"},
        "next_action": {
            "type": "string",
            "enum": ["done", "wait_for_long_job", "ask_user", "follow_up_plan"],
        },
        "ask_user": {"type": ["string", "null"]},
        "follow_up_plan": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": TOOL_NAMES},
                    "args_json": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["tool", "args_json", "why"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "goal_complete", "next_action", "ask_user", "follow_up_plan"],
    "additionalProperties": False,
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tool_catalog_text() -> str:
    rows: list[str] = []
    for tool in TOOL_DEFINITIONS:
        params = tool.get("parameters", {}) or {}
        properties = params.get("properties") or {}
        if properties:
            arg_keys = ", ".join(sorted(properties.keys()))
        else:
            arg_keys = "(no args)"
        long_marker = " [LONG-RUNNING]" if tool["name"] in LONG_RUNNING_TOOLS else ""
        rows.append(f"- {tool['name']}{long_marker}: {tool['description']}\n    args: {arg_keys}")
    return "\n".join(rows)


PLANNER_PROMPT = """
You are the planner for a 3D scene + robot pipeline. Given the user's
request and the current run state, you produce a STRUCTURED PLAN that a
deterministic executor will run.

You DO NOT execute anything yourself. You only emit a plan.

Available tools (each step in the plan picks one):
{tool_catalog}

Pipeline order convention:
  scene_graph -> (optional) run_real2sim -> generate_scene -> run_scene_robot_collect

Planning rules:
1. Always start with one `inspect_state` step so the executor (and the
   reflector after) has a clear baseline observation.
2. Long-running tools (run_real2sim, run_scene_robot_collect) MUST be
   the LAST step of the plan, OR be followed only by tools that depend
   purely on the long job's success (rare). The executor pauses on
   long-running steps; later steps in the same plan only run if the
   long job has finished by the time the user resumes the plan.
3. Use args_json carefully — it must be valid JSON encoding the tool's
   arguments. Empty: "{{}}".
4. Each step must include a one-sentence `why` so a human reviewing the
   plan can judge it.
5. If essential information is missing (no scene graph, no image, no
   target object), set ask_user to a single short question and leave
   steps as []. Do not invent missing inputs.
6. Cap the plan at {max_steps} steps.

Return JSON only, conforming to the schema.
""".strip()


REFLECTOR_PROMPT = """
You are the reflector. You receive a plan, the observations the executor
collected, and the current run state. You must:

1. Summarize what happened in one or two short sentences.
2. Decide goal_complete (true only if the user's original goal is now
   satisfied; long jobs that just started do NOT count as complete).
3. Pick a next_action:
   - "done" when goal_complete is true and nothing else is required.
   - "wait_for_long_job" when a real2sim or scene_robot job is still
     running and the user just needs to wait before resuming.
   - "ask_user" when the next decision needs human input. Put the
     question in ask_user.
   - "follow_up_plan" when there is an obvious deterministic next step
     the executor should run after the user resumes. Provide
     follow_up_plan as a list of steps in the same shape as a plan.
4. follow_up_plan can be null when next_action is not "follow_up_plan".

Be conservative with follow_up_plan — only propose it when the next
steps are obvious from the observations.

Return JSON only.
""".strip()


# --------- Plan persistence ---------


def _new_plan_id() -> str:
    return f"plan_{uuid.uuid4().hex[:12]}"


def _store_active_plan(state: dict[str, Any], plan: dict[str, Any]) -> None:
    state["active_plan"] = plan


def _get_active_plan(state: dict[str, Any]) -> dict[str, Any] | None:
    plan = state.get("active_plan")
    return plan if isinstance(plan, dict) else None


def _clear_active_plan(state: dict[str, Any]) -> None:
    state["active_plan"] = None


def _archive_active_plan(state: dict[str, Any]) -> dict[str, Any] | None:
    """Move state['active_plan'] into state['plan_history'] and clear it.

    Returns the archived plan (now stamped with `archived_at`) or None if
    there was no active plan to archive. Caps history at MAX_PLAN_HISTORY.
    """
    plan = state.get("active_plan")
    if not isinstance(plan, dict):
        return None
    history = state.setdefault("plan_history", [])
    if not isinstance(history, list):
        history = []
        state["plan_history"] = history
    archived = dict(plan)
    archived["archived_at"] = _utcnow_iso()
    history.append(archived)
    if len(history) > MAX_PLAN_HISTORY:
        del history[: len(history) - MAX_PLAN_HISTORY]
    state["active_plan"] = None
    return archived


# --------- Plan proposing ---------


def _build_planner_input(text: str, snapshot: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "Current run state:",
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            f'User request: "{text}"',
            "Return JSON only, matching the plan schema.",
        ]
    )


def _state_snapshot_for_planner(context: RuntimeContext, lctx: LoopContext) -> dict[str, Any]:
    return TOOL_HANDLERS["inspect_state"]({}, lctx)


def _save_input_image(context: RuntimeContext, image_bytes: bytes | None) -> bool:
    if not image_bytes:
        return False
    target = context.latest_input_image
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(image_bytes)
    return True


def _coerce_plan_steps(raw_steps: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_steps, list):
        return []
    coerced: list[dict[str, Any]] = []
    for raw in raw_steps[:MAX_PLAN_STEPS]:
        if not isinstance(raw, dict):
            continue
        tool = raw.get("tool")
        if tool not in TOOL_HANDLERS:
            continue
        args_json = raw.get("args_json")
        if not isinstance(args_json, str):
            args_json = "{}"
        try:
            args = json.loads(args_json)
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError:
            args = {}
        coerced.append(
            {
                "id": f"step_{len(coerced) + 1}",
                "tool": tool,
                "args": args,
                "why": str(raw.get("why") or "").strip(),
                "status": "pending",
                "result": None,
                "started_at": None,
                "finished_at": None,
            }
        )
    return coerced


def propose_plan(
    *,
    session_id: str | None,
    run_id: str | None,
    text: str | None,
    image_bytes: bytes | None = None,
) -> dict[str, Any]:
    context = resolve_runtime_context(session_id=session_id, run_id=run_id, create=True)
    if context is None:
        context = create_session()
    context.ensure()

    has_uploaded_image = _save_input_image(context, image_bytes)
    state = _load_agent_state(context)
    state["current_run_id"] = context.run_id
    _ensure_run_state(state, context)

    user_text = (text or "").strip()
    if not user_text and not has_uploaded_image:
        user_text = "(no text; the user clicked submit with no instruction.)"

    lctx = LoopContext(
        context,
        state,
        has_uploaded_image=has_uploaded_image,
        scene_service_url=SCENE_SERVICE_URL,
    )
    snapshot = _state_snapshot_for_planner(context, lctx)
    prompt = _build_planner_input(user_text, snapshot)
    instructions = PLANNER_PROMPT.format(
        tool_catalog=_tool_catalog_text(),
        max_steps=MAX_PLAN_STEPS,
    )

    response = _get_openai_client().responses.create(
        model=PLANNER_MODEL,
        instructions=instructions,
        input=prompt,
        text=_json_schema_format("plan", PLAN_SCHEMA),
    )
    try:
        parsed = json.loads(response.output_text)
    except (AttributeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Planner returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Planner returned a non-object payload.")

    ask_user = parsed.get("ask_user")
    steps = _coerce_plan_steps(parsed.get("steps"))
    plan = {
        "id": _new_plan_id(),
        "goal": str(parsed.get("goal") or user_text),
        "user_request": user_text,
        "steps": steps,
        "ask_user": ask_user if isinstance(ask_user, str) and ask_user.strip() else None,
        "status": "needs_user" if isinstance(ask_user, str) and ask_user.strip() else "proposed",
        "current_step_index": 0,
        "paused_for_job": None,
        "snapshot_before": snapshot,
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
    }

    _archive_active_plan(state)
    _store_active_plan(state, plan)
    _save_agent_state(context, state)

    decision = {
        "intent": "agent_plan",
        "state": "plan_proposed" if plan["status"] == "proposed" else "plan_needs_user",
        "reason": "Plan proposed; awaiting user approval to execute.",
    }
    return _build_agent_response(
        context,
        state=str(plan["status"]),
        intent="agent_plan",
        message=plan["ask_user"] or f"Proposed a {len(steps)}-step plan.",
        reason="Planner produced a structured plan.",
        decision=decision,
        outcome="needs_user_input" if plan["ask_user"] else "completed",
        session_state=_session_state_snapshot(state, context),
        active_plan=plan,
    )


# --------- Plan executing ---------


def _job_status_from_state(state: dict[str, Any], context: RuntimeContext, kind: str) -> str:
    run_state = _ensure_run_state(state, context)
    section = run_state.get(kind) if isinstance(run_state.get(kind), dict) else {}
    return str(section.get("status") or "")


def _resume_check(plan: dict[str, Any], state: dict[str, Any], context: RuntimeContext) -> tuple[bool, str | None]:
    """Return (can_resume, paused_status_text) for a plan paused on a long job."""
    paused = plan.get("paused_for_job")
    if not isinstance(paused, dict):
        return True, None
    kind = paused.get("kind")
    if kind not in {"real2sim", "scene_robot"}:
        return True, None
    state_key = "real2sim" if kind == "real2sim" else "scene_robot"
    job_status = _job_status_from_state(state, context, state_key)
    if job_status == "succeeded":
        return True, "succeeded"
    if job_status == "failed":
        return True, "failed"
    return False, job_status or "running"


def _step_starts_long_job(step: dict[str, Any]) -> bool:
    return step.get("tool") in LONG_RUNNING_TOOLS


def execute_plan(
    *,
    session_id: str | None,
    run_id: str | None,
) -> dict[str, Any]:
    context = resolve_runtime_context(session_id=session_id, run_id=run_id, create=True)
    if context is None:
        return _empty_plan_response("No active session.")

    state = _load_agent_state(context)
    plan = _get_active_plan(state)
    if plan is None:
        return _empty_plan_response("No active plan to execute.")

    if plan.get("status") in {"completed", "cancelled"}:
        return _build_plan_response(context, state, plan, message="Plan already finished.")

    if plan.get("status") == "needs_user" and plan.get("ask_user"):
        return _build_plan_response(
            context,
            state,
            plan,
            message=str(plan["ask_user"]),
            outcome="needs_user_input",
        )

    can_resume, status_text = _resume_check(plan, state, context)
    if not can_resume:
        return _build_plan_response(
            context,
            state,
            plan,
            message=f"Plan is still paused on a long-running job ({status_text}).",
            outcome="started_job",
        )
    if status_text == "succeeded":
        plan["paused_for_job"] = None
        plan["status"] = "running"
    elif status_text == "failed":
        plan["paused_for_job"] = None
        plan["status"] = "failed"
        plan["updated_at"] = _utcnow_iso()
        _store_active_plan(state, plan)
        _save_agent_state(context, state)
        reflection = reflect_plan(state=state, context=context, plan=plan)
        return _build_plan_response(
            context,
            state,
            plan,
            message="A long-running job failed; plan stopped.",
            outcome="failed",
            reflection=reflection,
        )

    plan["status"] = "running"
    plan["updated_at"] = _utcnow_iso()
    _store_active_plan(state, plan)
    _save_agent_state(context, state)

    lctx = LoopContext(
        context,
        state,
        has_uploaded_image=False,
        scene_service_url=SCENE_SERVICE_URL,
    )

    paused = False
    while plan["current_step_index"] < len(plan["steps"]):
        idx = plan["current_step_index"]
        step = plan["steps"][idx]
        step["started_at"] = _utcnow_iso()
        handler = TOOL_HANDLERS.get(step["tool"])
        if handler is None:
            step["status"] = "failed"
            step["result"] = {"ok": False, "error": f"Unknown tool: {step['tool']}"}
            step["finished_at"] = _utcnow_iso()
            plan["status"] = "failed"
            plan["current_step_index"] = idx + 1
            break
        try:
            result = handler(step.get("args") or {}, lctx)
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        step["status"] = "ok" if result.get("ok", True) and not result.get("error") else "failed"
        step["result"] = result
        step["finished_at"] = _utcnow_iso()
        plan["updated_at"] = _utcnow_iso()
        plan["current_step_index"] = idx + 1

        if step["status"] == "failed":
            plan["status"] = "failed"
            break

        if _step_starts_long_job(step):
            kind = "real2sim" if step["tool"] == "run_real2sim" else "scene_robot"
            plan["paused_for_job"] = {
                "kind": kind,
                "job_id": result.get("job_id"),
                "tool": step["tool"],
            }
            plan["status"] = "paused"
            paused = True
            break

    if plan["status"] == "running" and plan["current_step_index"] >= len(plan["steps"]):
        plan["status"] = "completed"

    if lctx.last_started_state == STATE_RUN_REAL2SIM:
        _set_current_state(state, STATE_RUN_REAL2SIM, last_intent=STATE_RUN_REAL2SIM)
    elif lctx.last_started_state == STATE_RUN_SCENE_ROBOT_COLLECT:
        _set_current_state(state, STATE_RUN_SCENE_ROBOT_COLLECT, last_intent=STATE_RUN_SCENE_ROBOT_COLLECT)
    elif plan["status"] == "completed":
        _set_current_state(state, STATE_COMPLETED, last_completed_state="agent_plan")
    elif plan["status"] == "failed":
        _set_current_state(state, STATE_FAILED, last_intent="agent_plan")

    _store_active_plan(state, plan)
    _save_agent_state(context, state)

    reflection = reflect_plan(state=state, context=context, plan=plan)

    if reflection.get("next_action") == "follow_up_plan" and isinstance(reflection.get("follow_up_plan"), list):
        steps = _coerce_plan_steps(reflection["follow_up_plan"])
        if steps:
            plan["follow_up_plan"] = steps

    plan["updated_at"] = _utcnow_iso()
    _store_active_plan(state, plan)
    _save_agent_state(context, state)

    if plan["status"] == "completed":
        message = reflection.get("summary") or "Plan completed."
        outcome = "completed"
    elif plan["status"] == "paused":
        message = reflection.get("summary") or "Plan paused on a long-running job."
        outcome = "started_job"
    elif plan["status"] == "failed":
        message = reflection.get("summary") or "Plan failed."
        outcome = "failed"
    else:
        message = reflection.get("summary") or "Plan finished."
        outcome = "completed"

    extra: dict[str, Any] = {
        "active_plan": plan,
        "reflection": reflection,
        "warnings": list(lctx.warnings),
    }
    if lctx.real2sim_job is not None:
        extra["real2sim_job"] = lctx.real2sim_job
    if lctx.scene_robot_job is not None:
        extra["scene_robot_job"] = lctx.scene_robot_job
    if lctx.scene_result is not None:
        extra["scene_result"] = lctx.scene_result

    return _build_agent_response(
        context,
        state=str(plan["status"]),
        intent="agent_plan",
        message=message,
        reason="Plan executor finished a pass.",
        decision={"intent": "agent_plan", "state": plan["status"], "reason": "Executor pass."},
        outcome=outcome,
        session_state=_session_state_snapshot(state, context),
        **extra,
    )


# --------- Reflection ---------


def _build_reflector_input(plan: dict[str, Any], snapshot: dict[str, Any]) -> str:
    observations = [
        {
            "step_id": step.get("id"),
            "tool": step.get("tool"),
            "args": step.get("args"),
            "status": step.get("status"),
            "result": step.get("result"),
        }
        for step in plan.get("steps", [])
    ]
    return "\n\n".join(
        [
            "Plan goal:",
            str(plan.get("goal") or ""),
            "Plan status:",
            str(plan.get("status") or ""),
            "Paused for job:",
            json.dumps(plan.get("paused_for_job"), ensure_ascii=False),
            "Step observations (only completed/failed have result):",
            json.dumps(observations, ensure_ascii=False, indent=2),
            "Current run state:",
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            "Return JSON only, matching the reflection schema.",
        ]
    )


def reflect_plan(
    *,
    state: dict[str, Any],
    context: RuntimeContext,
    plan: dict[str, Any],
) -> dict[str, Any]:
    lctx = LoopContext(
        context,
        state,
        has_uploaded_image=False,
        scene_service_url=SCENE_SERVICE_URL,
    )
    snapshot = TOOL_HANDLERS["inspect_state"]({}, lctx)
    prompt = _build_reflector_input(plan, snapshot)

    response = _get_openai_client().responses.create(
        model=REFLECTOR_MODEL,
        instructions=REFLECTOR_PROMPT,
        input=prompt,
        text=_json_schema_format("reflection", REFLECT_SCHEMA),
    )
    try:
        parsed = json.loads(response.output_text)
    except (AttributeError, json.JSONDecodeError):
        return {
            "summary": "Reflection unavailable.",
            "goal_complete": plan.get("status") == "completed",
            "next_action": "done" if plan.get("status") == "completed" else "ask_user",
            "ask_user": None,
            "follow_up_plan": None,
        }
    if not isinstance(parsed, dict):
        return {
            "summary": "Reflection unavailable.",
            "goal_complete": plan.get("status") == "completed",
            "next_action": "done" if plan.get("status") == "completed" else "ask_user",
            "ask_user": None,
            "follow_up_plan": None,
        }
    return parsed


# --------- Cancel / read ---------


EDITABLE_PLAN_STATUSES = {"proposed", "needs_user", "paused"}


def _normalize_edit_step(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    tool = raw.get("tool")
    if tool not in TOOL_HANDLERS:
        return None
    args = raw.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        args = {}
    return {
        "tool": tool,
        "args_json": json.dumps(args, ensure_ascii=False),
        "why": str(raw.get("why") or "").strip(),
    }


def update_plan(
    *,
    session_id: str | None,
    run_id: str | None,
    steps: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    context = resolve_runtime_context(session_id=session_id, run_id=run_id, create=True)
    if context is None:
        return _empty_plan_response("No active session.")

    state = _load_agent_state(context)
    plan = _get_active_plan(state)
    if plan is None:
        return _empty_plan_response("No active plan to edit.")

    status = str(plan.get("status") or "")
    if status not in EDITABLE_PLAN_STATUSES:
        raise ValueError(f"Plan in status '{status}' cannot be edited.")

    raw_list = list(steps or [])
    if not raw_list:
        raise ValueError("Plan must have at least one step.")
    if len(raw_list) > MAX_PLAN_STEPS:
        raise ValueError(f"Plan exceeds max steps ({MAX_PLAN_STEPS}).")

    normalized: list[dict[str, Any]] = []
    for raw in raw_list:
        coerced = _normalize_edit_step(raw)
        if coerced is None:
            raise ValueError(
                "Invalid step: each step needs a known `tool`, an object/JSON-string `args`, and a `why`."
            )
        normalized.append(coerced)

    new_pending_steps = _coerce_plan_steps(normalized)
    if not new_pending_steps:
        raise ValueError("Plan must have at least one valid step.")

    if status == "paused":
        keep_count = int(plan.get("current_step_index") or 0)
        completed_steps = list(plan.get("steps") or [])[:keep_count]
        for offset, step in enumerate(new_pending_steps):
            step["id"] = f"step_{keep_count + offset + 1}"
        plan["steps"] = completed_steps + new_pending_steps
        plan["status"] = "proposed"
    else:
        plan["steps"] = new_pending_steps
        plan["current_step_index"] = 0
        plan["status"] = "proposed"

    plan["paused_for_job"] = None
    plan["updated_at"] = _utcnow_iso()
    _store_active_plan(state, plan)
    _save_agent_state(context, state)

    return _build_plan_response(
        context,
        state,
        plan,
        message=f"Plan updated ({len(new_pending_steps)} pending step(s)).",
    )


def update_follow_up_plan(
    *,
    session_id: str | None,
    run_id: str | None,
    steps: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Replace `plan["follow_up_plan"]` on the active plan after user edits.

    `steps` empty list clears the follow-up. Steps are validated via the
    same `_normalize_edit_step` path as `update_plan`.
    """
    context = resolve_runtime_context(session_id=session_id, run_id=run_id, create=True)
    if context is None:
        return _empty_plan_response("No active session.")

    state = _load_agent_state(context)
    plan = _get_active_plan(state)
    if plan is None:
        raise ValueError("No active plan to attach a follow-up to.")

    raw_list = list(steps or [])
    if not raw_list:
        plan["follow_up_plan"] = []
        plan["updated_at"] = _utcnow_iso()
        _store_active_plan(state, plan)
        _save_agent_state(context, state)
        return _build_plan_response(
            context,
            state,
            plan,
            message="Follow-up plan cleared.",
        )

    if len(raw_list) > MAX_PLAN_STEPS:
        raise ValueError(f"Follow-up plan exceeds max steps ({MAX_PLAN_STEPS}).")

    normalized: list[dict[str, Any]] = []
    for raw in raw_list:
        coerced = _normalize_edit_step(raw)
        if coerced is None:
            raise ValueError(
                "Invalid follow-up step: each step needs a known `tool`, an object/JSON-string `args`, and a `why`."
            )
        normalized.append(coerced)

    new_follow_up = _coerce_plan_steps(normalized)
    if not new_follow_up:
        raise ValueError("Follow-up plan has no valid steps.")

    plan["follow_up_plan"] = new_follow_up
    plan["updated_at"] = _utcnow_iso()
    _store_active_plan(state, plan)
    _save_agent_state(context, state)

    return _build_plan_response(
        context,
        state,
        plan,
        message=f"Follow-up plan updated ({len(new_follow_up)} step(s)).",
    )


def _followup_to_plan_steps(follow_up: list[Any]) -> list[dict[str, Any]]:
    """Convert reflector follow_up_plan items into fresh pending plan steps.

    `plan["follow_up_plan"]` is already shaped like plan steps (ids,
    args dict, status etc.) when it was set in `execute_plan`. We
    re-stamp ids and reset status here so the new active_plan looks
    fresh.
    """
    fresh: list[dict[str, Any]] = []
    for raw in follow_up:
        if not isinstance(raw, dict):
            continue
        tool = raw.get("tool")
        if tool not in TOOL_HANDLERS:
            continue
        args = raw.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        fresh.append(
            {
                "id": f"step_{len(fresh) + 1}",
                "tool": tool,
                "args": args,
                "why": str(raw.get("why") or "").strip(),
                "status": "pending",
                "result": None,
                "started_at": None,
                "finished_at": None,
            }
        )
    return fresh


def accept_follow_up_plan(*, session_id: str | None, run_id: str | None) -> dict[str, Any]:
    context = resolve_runtime_context(session_id=session_id, run_id=run_id, create=True)
    if context is None:
        return _empty_plan_response("No active session.")

    state = _load_agent_state(context)
    plan = _get_active_plan(state)
    if plan is None:
        raise ValueError("No active plan to derive a follow-up from.")
    follow_up = plan.get("follow_up_plan")
    if not isinstance(follow_up, list) or not follow_up:
        raise ValueError("Active plan has no follow-up plan to accept.")

    fresh_steps = _followup_to_plan_steps(follow_up)
    if not fresh_steps:
        raise ValueError("Follow-up plan has no valid steps.")

    new_plan = {
        "id": _new_plan_id(),
        "goal": str(plan.get("goal") or ""),
        "user_request": str(plan.get("user_request") or ""),
        "steps": fresh_steps,
        "ask_user": None,
        "status": "proposed",
        "current_step_index": 0,
        "paused_for_job": None,
        "snapshot_before": plan.get("snapshot_before") or {},
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
        "derived_from": plan.get("id"),
    }
    _archive_active_plan(state)
    _store_active_plan(state, new_plan)
    _save_agent_state(context, state)

    return _build_plan_response(
        context,
        state,
        new_plan,
        message=f"Accepted follow-up plan ({len(fresh_steps)} step(s)). Review and run when ready.",
    )


def cancel_plan(*, session_id: str | None, run_id: str | None) -> dict[str, Any]:
    context = resolve_runtime_context(session_id=session_id, run_id=run_id, create=True)
    if context is None:
        return _empty_plan_response("No active session.")

    state = _load_agent_state(context)
    plan = _get_active_plan(state)
    archived = None
    if plan is not None:
        plan["status"] = "cancelled"
        plan["updated_at"] = _utcnow_iso()
        _store_active_plan(state, plan)
        archived = _archive_active_plan(state)
    _save_agent_state(context, state)
    return _build_agent_response(
        context,
        state="cancelled",
        intent="agent_plan",
        message="Plan cancelled.",
        reason="User requested cancel.",
        decision={"intent": "agent_plan", "state": "cancelled", "reason": "User cancel."},
        outcome="completed",
        session_state=_session_state_snapshot(state, context),
        active_plan=None,
        archived_plan=archived,
    )


# --------- Helpers ---------


def get_plan_history(
    *,
    session_id: str | None,
    run_id: str | None,
    limit: int | None = None,
) -> dict[str, Any]:
    context = resolve_runtime_context(session_id=session_id, run_id=run_id, create=True)
    if context is None:
        return {"status": "ok", "plans": []}
    state = _load_agent_state(context)
    history = state.get("plan_history") or []
    if not isinstance(history, list):
        history = []
    cap = MAX_PLAN_HISTORY if limit is None else max(1, min(int(limit), MAX_PLAN_HISTORY))
    plans = list(reversed(history))[:cap]
    return {
        "status": "ok",
        "session_id": context.session_id,
        "run_id": context.run_id,
        "plans": plans,
        "session_state": _session_state_snapshot(state, context),
    }


def _empty_plan_response(message: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "agent": {
            "state": "idle",
            "intent": "agent_plan",
            "message": message,
            "outcome": "completed",
        },
        "active_plan": None,
    }


def _build_plan_response(
    context: RuntimeContext,
    state: dict[str, Any],
    plan: dict[str, Any],
    *,
    message: str,
    outcome: str = "completed",
    reflection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {"active_plan": plan}
    if reflection is not None:
        extra["reflection"] = reflection
    return _build_agent_response(
        context,
        state=str(plan.get("status") or "idle"),
        intent="agent_plan",
        message=message,
        reason="Plan dispatcher.",
        decision={"intent": "agent_plan", "state": plan.get("status"), "reason": "Plan dispatcher."},
        outcome=outcome,
        session_state=_session_state_snapshot(state, context),
        **extra,
    )
