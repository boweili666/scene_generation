#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCENE_ROBOT_SRC))


DEFAULT_SCENE_USD_PATH = (
    PROJECT_ROOT
    / "runtime"
    / "sessions"
    / "sess_37daed605d8c"
    / "runs"
    / "run_ab267fae7ae8"
    / "scene_service"
    / "usd"
    / "scene_latest.usd"
)
DEFAULT_SELECTED_PATH = PROJECT_ROOT / "runtime" / "robot_placement" / "selected_grasp_proposal.json"
DEFAULT_SHORTLIST_PATH = PROJECT_ROOT / "runtime" / "robot_placement" / "grasp_candidate_shortlist.json"


def _maybe_reexec_with_conda_libstdcpp() -> None:
    if os.environ.get("SCENE_GRASP_ISAAC_LD_READY") == "1":
        return

    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    conda_lib = Path(conda_prefix) / "lib"
    libstdcpp = conda_lib / "libstdc++.so.6"
    if not libstdcpp.exists():
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in existing.split(":") if part]
    conda_lib_str = str(conda_lib)
    if parts and parts[0] == conda_lib_str:
        os.environ["SCENE_GRASP_ISAAC_LD_READY"] = "1"
        return

    merged = [conda_lib_str]
    merged.extend(part for part in parts if part != conda_lib_str)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(merged)
    env["SCENE_GRASP_ISAAC_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_maybe_reexec_with_conda_libstdcpp()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize grasp poses directly in Isaac Sim without running robot execution.")
parser.add_argument("--target", type=str, default="/World/bolt_2", help="Target prim path, e.g. /World/bolt_2")
parser.add_argument("--scene-usd-path", type=Path, default=DEFAULT_SCENE_USD_PATH)
parser.add_argument("--scene-grasp-proposals-path", type=Path, default=None)
parser.add_argument("--selected-proposal-path", type=Path, default=DEFAULT_SELECTED_PATH)
parser.add_argument("--shortlist-path", type=Path, default=DEFAULT_SHORTLIST_PATH)
parser.add_argument("--selected-only", action="store_true", help="Only draw the selected grasp pose.")
parser.add_argument("--max-grasps", type=int, default=12, help="Maximum shortlist grasp frames to draw.")
parser.add_argument("--camera-scale", type=float, default=1.0, help="Camera distance scale relative to target bbox.")
parser.add_argument("--settle-steps", type=int, default=80, help="Physics steps to run before the first grasp visualization.")
parser.add_argument("--refresh-every", type=int, default=1, help="Rebuild grasp poses from the live USD stage every N frames.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_default_scene_grasp_proposals_path(scene_usd_path: Path, scene_grasp_proposals_path: Path | None) -> Path:
    if scene_grasp_proposals_path is not None:
        return scene_grasp_proposals_path.resolve()
    suffix = scene_usd_path.suffix or ".usd"
    return scene_usd_path.resolve().with_suffix(f"{suffix}.grasp_proposals.json")


def _candidate_from_payload(payload: dict[str, Any]):
    from scene_robot_apps.grasp_execution import FilteredGraspExecution, GraspExecutionPose

    grasp_payload = payload.get("grasp", payload)
    grasp = GraspExecutionPose(
        object_prim=str(grasp_payload.get("object_prim", "")),
        primitive_type=str(grasp_payload.get("primitive_type", "")),
        candidate_id=str(grasp_payload.get("candidate_id", "")),
        score=float(grasp_payload.get("score", payload.get("score", 0.0))),
        position_world=tuple(float(v) for v in grasp_payload.get("position_world", [0.0, 0.0, 0.0])),
        quat_wxyz_world=tuple(float(v) for v in grasp_payload.get("quat_wxyz_world", [1.0, 0.0, 0.0, 0.0])),
        approach_axis_world=tuple(float(v) for v in grasp_payload.get("approach_axis_world", [1.0, 0.0, 0.0])),
        closing_axis_world=tuple(float(v) for v in grasp_payload.get("closing_axis_world", [0.0, 1.0, 0.0])),
        width=None if grasp_payload.get("width") is None else float(grasp_payload.get("width")),
        source_branch=grasp_payload.get("source_branch"),
        source_primitive_index=int(grasp_payload.get("source_primitive_index", 0)),
    )
    return FilteredGraspExecution(
        grasp=grasp,
        arm_side=str(payload.get("arm_side", "left")),
        pre_grasp_pos_world=tuple(float(v) for v in payload.get("pre_grasp_pos_world", grasp.position_world)),
        pre_grasp_quat_world=tuple(float(v) for v in payload.get("pre_grasp_quat_world", grasp.quat_wxyz_world)),
        lift_pos_world=tuple(float(v) for v in payload.get("lift_pos_world", grasp.position_world)),
        lift_quat_world=tuple(float(v) for v in payload.get("lift_quat_world", grasp.quat_wxyz_world)),
        retreat_pos_world=tuple(float(v) for v in payload.get("retreat_pos_world", grasp.position_world)),
        retreat_quat_world=tuple(float(v) for v in payload.get("retreat_quat_world", grasp.quat_wxyz_world)),
        base_frame_xy=tuple(float(v) for v in payload.get("base_frame_xy", [0.0, 0.0])),
        pre_grasp_base_frame_xy=tuple(float(v) for v in payload.get("pre_grasp_base_frame_xy", [0.0, 0.0])),
        workspace_margin_xy=tuple(float(v) for v in payload.get("workspace_margin_xy", [0.0, 0.0])),
        support_clearance=float(payload.get("support_clearance", 0.0)),
        score=float(payload.get("score", grasp.score)),
        start_pose_position_error=(
            None if payload.get("start_pose_position_error") is None else float(payload.get("start_pose_position_error"))
        ),
        start_pose_rotation_error_deg=(
            None
            if payload.get("start_pose_rotation_error_deg") is None
            else float(payload.get("start_pose_rotation_error_deg"))
        ),
        ranking_score=None if payload.get("ranking_score") is None else float(payload.get("ranking_score")),
    )


def _load_selected_candidate(target_prim: str, selected_path: Path | None):
    if selected_path is None or not selected_path.exists():
        return None
    payload = _load_json(selected_path)
    if str(payload.get("target_prim")) != str(target_prim):
        return None
    selected = payload.get("selected_grasp_proposal")
    if not isinstance(selected, dict):
        return None
    return _candidate_from_payload(selected)


def _load_shortlist_candidates(target_prim: str, shortlist_path: Path | None, max_grasps: int):
    if shortlist_path is None or not shortlist_path.exists():
        return []
    payload = _load_json(shortlist_path)
    if str(payload.get("target_prim")) != str(target_prim):
        return []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [_candidate_from_payload(entry) for entry in candidates[: max(0, int(max_grasps))] if isinstance(entry, dict)]


def _expand_scene_candidates(scene_grasp_payload: dict[str, Any], target_prim: str, max_grasps: int):
    from scene_robot_apps.grasp_execution import FilteredGraspExecution, expand_grasp_candidates

    expanded = expand_grasp_candidates(scene_grasp_payload, target_prim=target_prim)
    converted: list[FilteredGraspExecution] = []
    for grasp in expanded[: max(0, int(max_grasps))]:
        converted.append(
            FilteredGraspExecution(
                grasp=grasp,
                arm_side="left",
                pre_grasp_pos_world=grasp.position_world,
                pre_grasp_quat_world=grasp.quat_wxyz_world,
                lift_pos_world=grasp.position_world,
                lift_quat_world=grasp.quat_wxyz_world,
                retreat_pos_world=grasp.position_world,
                retreat_quat_world=grasp.quat_wxyz_world,
                base_frame_xy=(0.0, 0.0),
                pre_grasp_base_frame_xy=(0.0, 0.0),
                workspace_margin_xy=(0.0, 0.0),
                support_clearance=0.0,
                score=float(grasp.score),
            )
        )
    return converted


def _resolve_live_grasp_payload(stage, base_payload: dict[str, Any], target_prim: str) -> dict[str, Any]:
    from app.backend.services.grasp_scene_adapter import build_stage_grasp_proposals

    manifest_path = base_payload.get("manifest_path")
    annotation_root = base_payload.get("annotation_root")
    if not manifest_path or not annotation_root:
        return base_payload
    try:
        return build_stage_grasp_proposals(
            stage,
            manifest_path,
            annotation_root=annotation_root,
            output_path=None,
            target_prim_paths=[target_prim],
        )
    except Exception:
        return base_payload


def _match_selected_candidate(
    selected_candidate,
    live_candidates: list,
):
    if selected_candidate is None:
        return live_candidates[0] if live_candidates else None
    selected_id = selected_candidate.grasp.candidate_id
    for candidate in live_candidates:
        if candidate.grasp.candidate_id == selected_id:
            return candidate
    return live_candidates[0] if live_candidates else selected_candidate


def _focus_camera_on_target(sim, stage, target_prim: str, camera_scale: float) -> None:
    from pxr import Usd, UsdGeom

    prim = _resolve_visual_target_prim(stage, target_prim)
    if not prim.IsValid():
        return
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["render", "default"], useExtentsHint=True)
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if aligned.IsEmpty():
        return
    center = aligned.GetMidpoint()
    size = aligned.GetSize()
    diag = max(0.05, float((size[0] ** 2 + size[1] ** 2 + size[2] ** 2) ** 0.5))
    scale = float(max(0.25, camera_scale)) * diag
    eye = (
        float(center[0] + 2.5 * scale),
        float(center[1] - 2.0 * scale),
        float(center[2] + 1.8 * scale),
    )
    target = (float(center[0]), float(center[1]), float(center[2]))
    sim.set_camera_view(eye, target)


def _resolve_visual_target_prim(stage, target_prim: str):
    prim = stage.GetPrimAtPath(target_prim)
    if not prim.IsValid():
        return prim
    asset_ref = prim.GetChild("AssetRef")
    if asset_ref.IsValid():
        return asset_ref
    return prim


def _target_live_center(stage, target_prim: str) -> tuple[float, float, float] | None:
    from pxr import Usd, UsdGeom

    prim = _resolve_visual_target_prim(stage, target_prim)
    if not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["render", "default"], useExtentsHint=True)
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if aligned.IsEmpty():
        return None
    center = aligned.GetMidpoint()
    return (
        float(center[0]),
        float(center[1]),
        float(center[2]),
    )


def _format_vec3(value: tuple[float, float, float] | None) -> str:
    if value is None:
        return "None"
    return f"({value[0]:.4f}, {value[1]:.4f}, {value[2]:.4f})"


def main() -> None:
    import omni.usd
    import isaaclab.sim as sim_utils

    from scene_robot_apps.grasp_visualization import add_grasp_candidates_visuals_to_stage

    scene_usd_path = args_cli.scene_usd_path.resolve()
    scene_grasp_path = _resolve_default_scene_grasp_proposals_path(scene_usd_path, args_cli.scene_grasp_proposals_path)
    if not scene_usd_path.exists():
        raise FileNotFoundError(f"Scene USD not found: {scene_usd_path}")
    if not scene_grasp_path.exists():
        raise FileNotFoundError(f"Scene grasp proposals JSON not found: {scene_grasp_path}")

    opened = omni.usd.get_context().open_stage(str(scene_usd_path))
    if not opened:
        raise RuntimeError(f"Failed to open USD stage in Isaac Sim: {scene_usd_path}")
    for _ in range(10):
        simulation_app.update()

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Failed to acquire live USD stage from omni.usd context.")
    sim.reset()

    scene_grasp_payload = _load_json(scene_grasp_path)
    selected_candidate = _load_selected_candidate(args_cli.target, args_cli.selected_proposal_path)
    shortlist_candidates = [] if args_cli.selected_only else _load_shortlist_candidates(args_cli.target, args_cli.shortlist_path, args_cli.max_grasps)

    def _refresh_visuals() -> dict[str, Any]:
        live_payload = _resolve_live_grasp_payload(stage, scene_grasp_payload, args_cli.target)
        live_candidates = _expand_scene_candidates(live_payload, args_cli.target, args_cli.max_grasps)
        matched_selected = _match_selected_candidate(selected_candidate, live_candidates)
        ranked_candidates = shortlist_candidates if shortlist_candidates else live_candidates
        if shortlist_candidates and live_candidates:
            live_by_id = {candidate.grasp.candidate_id: candidate for candidate in live_candidates}
            ranked_candidates = [live_by_id.get(candidate.grasp.candidate_id, candidate) for candidate in shortlist_candidates]
        visual_summary = add_grasp_candidates_visuals_to_stage(
            stage,
            root_prim_path="/Visuals/IsaacGraspDebug",
            ranked_candidates=ranked_candidates,
            selected_candidate=matched_selected,
            max_candidates=0 if args_cli.selected_only else args_cli.max_grasps,
            parent_prim_path=args_cli.target,
        )
        visual_summary["target_live_center"] = _target_live_center(stage, args_cli.target)
        visual_summary["selected_live_grasp_pos"] = (
            None if matched_selected is None else tuple(float(v) for v in matched_selected.grasp.position_world)
        )
        visual_summary["selected_live_grasp_id"] = (
            None if matched_selected is None else matched_selected.grasp.candidate_id
        )
        return visual_summary

    print(f"[INFO] Settling live scene for {max(0, int(args_cli.settle_steps))} steps before drawing grasp frames.")
    for _ in range(max(0, int(args_cli.settle_steps))):
        sim.step()
        simulation_app.update()

    visual_summary = _refresh_visuals()
    _focus_camera_on_target(sim, stage, args_cli.target, args_cli.camera_scale)

    print(f"[INFO] Scene USD: {scene_usd_path}")
    print(f"[INFO] Scene grasp proposals: {scene_grasp_path}")
    print(f"[INFO] Target: {args_cli.target}")
    print(f"[INFO] Visual root: {visual_summary['visual_root_path']}")
    print(f"[INFO] Rendered shortlist count: {visual_summary['candidate_count']}")
    if selected_candidate is not None:
        print(f"[INFO] Selected grasp: {selected_candidate.grasp.candidate_id}")
    print(f"[INFO] Target live center: {_format_vec3(visual_summary.get('target_live_center'))}")
    print(f"[INFO] Selected grasp live pos: {_format_vec3(visual_summary.get('selected_live_grasp_pos'))}")
    print("[INFO] Close the Isaac Sim window to exit.")

    step_idx = 0
    while simulation_app.is_running():
        if step_idx % max(1, int(args_cli.refresh_every)) == 0:
            visual_summary = _refresh_visuals()
            print(
                "[LIVE] "
                f"target_center={_format_vec3(visual_summary.get('target_live_center'))} "
                f"selected_grasp={visual_summary.get('selected_live_grasp_id')} "
                f"selected_pos={_format_vec3(visual_summary.get('selected_live_grasp_pos'))}"
            )
        step_idx += 1
        sim.step()
        simulation_app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
