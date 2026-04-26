from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import RUNTIME_DIR, SESSIONS_DIR


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_CURRENT_RUN_FILENAME = "current_run.txt"
_SESSION_METADATA_FILENAME = "session.json"
_RUN_METADATA_FILENAME = "run.json"


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


def generate_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class RuntimeContext:
    session_id: str
    run_id: str
    runtime_root: Path
    sessions_root: Path
    session_root: Path
    runs_root: Path
    run_root: Path
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

    def ensure(self) -> "RuntimeContext":
        directories = (
            self.runtime_root,
            self.sessions_root,
            self.session_root,
            self.runs_root,
            self.run_root,
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

        run_meta = self.run_root / _RUN_METADATA_FILENAME
        if not run_meta.exists():
            run_meta.write_text(
                json.dumps(
                    {
                        "session_id": self.session_id,
                        "run_id": self.run_id,
                        "created_at": _utcnow_iso(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        current_run_path = self.session_root / _CURRENT_RUN_FILENAME
        current_run_path.write_text(self.run_id + "\n", encoding="utf-8")

        if not self.default_placements_path.exists():
            self.default_placements_path.write_text("{}\n", encoding="utf-8")

        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def build_runtime_context(session_id: str, run_id: str) -> RuntimeContext:
    normalized_session_id = normalize_runtime_identifier(session_id, label="session_id")
    normalized_run_id = normalize_runtime_identifier(run_id, label="run_id")

    session_root = SESSIONS_DIR / normalized_session_id
    run_root = session_root / "runs" / normalized_run_id
    return RuntimeContext(
        session_id=normalized_session_id,
        run_id=normalized_run_id,
        runtime_root=RUNTIME_DIR,
        sessions_root=SESSIONS_DIR,
        session_root=session_root,
        runs_root=session_root / "runs",
        run_root=run_root,
        uploads_dir=run_root / "uploads",
        latest_input_image=run_root / "uploads" / "latest_input.jpg",
        scene_graph_dir=run_root / "scene_graph",
        scene_graph_path=run_root / "scene_graph" / "current_scene_graph.json",
        renders_dir=run_root / "renders",
        render_path=run_root / "renders" / "render.png",
        logs_dir=run_root / "logs",
        real2sim_log_path=run_root / "logs" / "real2sim.log",
        real2sim_runtime_dir=run_root / "real2sim",
        real2sim_masks_dir=run_root / "real2sim" / "masks",
        real2sim_meshes_dir=run_root / "real2sim" / "meshes",
        real2sim_scene_results_dir=run_root / "real2sim" / "scene_results",
        real2sim_objects_dir=run_root / "real2sim" / "scene_results" / "objects",
        real2sim_object_usd_dir=run_root / "real2sim" / "scene_results" / "usd_objects",
        real2sim_assignment_path=run_root / "real2sim" / "scene_results" / "assignment.json",
        real2sim_poses_path=run_root / "real2sim" / "scene_results" / "poses.json",
        real2sim_manifest_path=run_root / "real2sim" / "scene_results" / "real2sim_asset_manifest.json",
        scene_service_runtime_dir=run_root / "scene_service",
        scene_service_placements_dir=run_root / "scene_service" / "placements",
        default_placements_path=run_root / "scene_service" / "placements" / "placements_default.json",
        scene_service_usd_dir=run_root / "scene_service" / "usd",
        scene_service_usd_path=run_root / "scene_service" / "usd" / "scene_latest.usd",
        scene_service_room_usd_path=run_root / "scene_service" / "usd" / "generated_room.scene_service.usd",
        robot_placement_dir=run_root / "robot_placement",
        scene_robot_log_path=run_root / "logs" / "scene_robot.log",
    )


def get_current_run_id(session_id: str) -> str | None:
    normalized_session_id = normalize_runtime_identifier(session_id, label="session_id")
    current_run_path = SESSIONS_DIR / normalized_session_id / _CURRENT_RUN_FILENAME
    if not current_run_path.exists():
        return None
    value = current_run_path.read_text(encoding="utf-8").strip()
    return normalize_runtime_identifier(value, label="run_id") if value else None


def resolve_runtime_context(
    session_id: str | None = None,
    run_id: str | None = None,
    *,
    create: bool = False,
) -> RuntimeContext | None:
    if session_id is None and run_id is None:
        return None

    normalized_session_id = normalize_runtime_identifier(
        session_id or "default",
        label="session_id",
    )
    resolved_run_id = run_id
    if resolved_run_id is None:
        resolved_run_id = get_current_run_id(normalized_session_id)
    if resolved_run_id is None:
        if not create:
            raise ValueError(f"No active run found for session '{normalized_session_id}'.")
        resolved_run_id = generate_run_id()

    context = build_runtime_context(normalized_session_id, resolved_run_id)
    return context.ensure() if create else context


def create_session(session_id: str | None = None, *, run_id: str | None = None) -> RuntimeContext:
    normalized_session_id = normalize_runtime_identifier(
        session_id or generate_session_id(),
        label="session_id",
    )
    normalized_run_id = normalize_runtime_identifier(run_id or generate_run_id(), label="run_id")
    return build_runtime_context(normalized_session_id, normalized_run_id).ensure()


def create_run(session_id: str, run_id: str | None = None) -> RuntimeContext:
    normalized_session_id = normalize_runtime_identifier(session_id, label="session_id")
    normalized_run_id = normalize_runtime_identifier(run_id or generate_run_id(), label="run_id")
    return build_runtime_context(normalized_session_id, normalized_run_id).ensure()
