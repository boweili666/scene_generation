from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import RUNTIME_DIR, SESSIONS_DIR


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SESSION_METADATA_FILENAME = "session.json"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_runtime_identifier(value: str | None, *, label: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        raise ValueError(f"{label} is required")
    if not _IDENTIFIER_RE.fullmatch(candidate):
        raise ValueError(
            f"Invalid {label} '{candidate}'. Use 1-128 chars: letters, digits, '_' or '-'."
        )
    return candidate


def generate_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class RuntimeContext:
    session_id: str
    runtime_root: Path
    sessions_root: Path
    session_root: Path
    uploads_dir: Path
    latest_input_image: Path
    scene_graph_dir: Path
    scene_graph_path: Path
    renders_dir: Path
    render_path: Path
    logs_dir: Path
    real2sim_log_path: Path
    real2sim_runtime_dir: Path
    real2sim_masks_dir: Path
    real2sim_meshes_dir: Path
    real2sim_scene_results_dir: Path
    real2sim_objects_dir: Path
    real2sim_object_usd_dir: Path
    real2sim_assignment_path: Path
    real2sim_poses_path: Path
    real2sim_manifest_path: Path
    scene_service_runtime_dir: Path
    scene_service_placements_dir: Path
    default_placements_path: Path
    scene_service_usd_dir: Path
    scene_service_usd_path: Path
    scene_service_room_usd_path: Path
    robot_placement_dir: Path
    scene_robot_log_path: Path
    scene_robot_convert_log_path: Path
    scene_robot_train_log_path: Path
    scene_robot_eval_log_path: Path

    def ensure(self) -> "RuntimeContext":
        directories = (
            self.runtime_root,
            self.sessions_root,
            self.session_root,
            self.uploads_dir,
            self.scene_graph_dir,
            self.renders_dir,
            self.logs_dir,
            self.real2sim_runtime_dir,
            self.real2sim_masks_dir,
            self.real2sim_meshes_dir,
            self.real2sim_scene_results_dir,
            self.real2sim_objects_dir,
            self.real2sim_object_usd_dir,
            self.scene_service_runtime_dir,
            self.scene_service_placements_dir,
            self.scene_service_usd_dir,
            self.robot_placement_dir,
        )
        for path in directories:
            path.mkdir(parents=True, exist_ok=True)

        session_meta = self.session_root / _SESSION_METADATA_FILENAME
        if not session_meta.exists():
            session_meta.write_text(
                json.dumps({"session_id": self.session_id, "created_at": _utcnow_iso()}, indent=2),
                encoding="utf-8",
            )

        if not self.default_placements_path.exists():
            self.default_placements_path.write_text("{}\n", encoding="utf-8")

        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def build_runtime_context(session_id: str) -> RuntimeContext:
    normalized_session_id = normalize_runtime_identifier(session_id, label="session_id")
    session_root = SESSIONS_DIR / normalized_session_id
    return RuntimeContext(
        session_id=normalized_session_id,
        runtime_root=RUNTIME_DIR,
        sessions_root=SESSIONS_DIR,
        session_root=session_root,
        uploads_dir=session_root / "uploads",
        latest_input_image=session_root / "uploads" / "latest_input.jpg",
        scene_graph_dir=session_root / "scene_graph",
        scene_graph_path=session_root / "scene_graph" / "current_scene_graph.json",
        renders_dir=session_root / "renders",
        render_path=session_root / "renders" / "render.png",
        logs_dir=session_root / "logs",
        real2sim_log_path=session_root / "logs" / "real2sim.log",
        real2sim_runtime_dir=session_root / "real2sim",
        real2sim_masks_dir=session_root / "real2sim" / "masks",
        real2sim_meshes_dir=session_root / "real2sim" / "meshes",
        real2sim_scene_results_dir=session_root / "real2sim" / "scene_results",
        real2sim_objects_dir=session_root / "real2sim" / "scene_results" / "objects",
        real2sim_object_usd_dir=session_root / "real2sim" / "scene_results" / "usd_objects",
        real2sim_assignment_path=session_root / "real2sim" / "scene_results" / "assignment.json",
        real2sim_poses_path=session_root / "real2sim" / "scene_results" / "poses.json",
        real2sim_manifest_path=session_root / "real2sim" / "scene_results" / "real2sim_asset_manifest.json",
        scene_service_runtime_dir=session_root / "scene_service",
        scene_service_placements_dir=session_root / "scene_service" / "placements",
        default_placements_path=session_root / "scene_service" / "placements" / "placements_default.json",
        scene_service_usd_dir=session_root / "scene_service" / "usd",
        scene_service_usd_path=session_root / "scene_service" / "usd" / "scene_latest.usd",
        scene_service_room_usd_path=session_root / "scene_service" / "usd" / "generated_room.scene_service.usd",
        robot_placement_dir=session_root / "robot_placement",
        scene_robot_log_path=session_root / "logs" / "scene_robot.log",
        scene_robot_convert_log_path=session_root / "logs" / "scene_robot_convert.log",
        scene_robot_train_log_path=session_root / "logs" / "scene_robot_train.log",
        scene_robot_eval_log_path=session_root / "logs" / "scene_robot_eval.log",
    )


def resolve_runtime_context(
    session_id: str | None = None,
    *,
    create: bool = False,
) -> RuntimeContext | None:
    if session_id is None:
        return None
    normalized_session_id = normalize_runtime_identifier(session_id, label="session_id")
    context = build_runtime_context(normalized_session_id)
    return context.ensure() if create else context


def create_session(session_id: str | None = None) -> RuntimeContext:
    normalized_session_id = normalize_runtime_identifier(
        session_id or generate_session_id(),
        label="session_id",
    )
    return build_runtime_context(normalized_session_id).ensure()


def _interrupt_and_cancel_jobs(session_id: str) -> list[dict]:
    from . import agent_interrupt as _agent_interrupt
    from .pipeline_service import cancel_real2sim_jobs_for_session
    from .scene_robot_service import cancel_scene_robot_jobs_for_session

    _agent_interrupt.request_interrupt(session_id)
    results: list[dict] = []
    for fn in (cancel_real2sim_jobs_for_session, cancel_scene_robot_jobs_for_session):
        try:
            results.append(fn(session_id))
        except Exception as exc:  # noqa: BLE001 - best-effort teardown
            results.append({"service": getattr(fn, "__name__", "?"), "error": str(exc)})
    return results


def delete_session(session_id: str) -> dict:
    normalized_session_id = normalize_runtime_identifier(session_id, label="session_id")
    session_root = SESSIONS_DIR / normalized_session_id
    if not session_root.exists() or not session_root.is_dir():
        raise FileNotFoundError(f"Session not found: {normalized_session_id}")

    cancelled = _interrupt_and_cancel_jobs(normalized_session_id)
    shutil.rmtree(session_root, ignore_errors=False)
    return {
        "session_id": normalized_session_id,
        "cancelled_jobs": cancelled,
    }
