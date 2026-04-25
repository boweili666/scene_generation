"""Auto grasp data collection: scene boot → ranking → preview → rollout.

This used to be a single 1743-line module that bundled the args, the
HDF5 writer, the preview UI + visual cleanup, JSON helpers, the
proposals loader, every pose-math helper, the phase runners, the
candidate ranker, the episode rollout, and the orchestrator. Most of
that lives in focused modules now:

* `auto_grasp_writer.py` — `SceneAutoGraspEpisodeWriter`,
  `_recover_corrupt_hdf5_file`, `PHASE_NAME_TO_ID`.
* `auto_grasp_preview.py` — `AutoGraspPreviewUI`, `_clear_grasp_visuals`,
  `_wait_for_run_request`, `_refresh_preview_markers`,
  `EE_MARKER_DEBUG_TRANSLATION_LOCAL`.
* `grasp_proposals_io.py` — manifest discovery + scene grasp payload
  load/build (`_load_or_build_scene_grasp_payload`,
  `_world_bbox_payload`).
* `grasp_target_state.py` — physx target-body snapshot/restore +
  per-episode randomization (`_snapshot_target_rigid_body_state`,
  `_restore_target_rigid_body_state`, `_robot_forward_xy_world`,
  `_shifted_target_snapshot`, `_shifted_candidate`,
  `_quat_wxyz_to_rotvec`).
* `phase_runner.py` — `_run_target_phase`, `_run_hold_phase` plus the
  pose-math helpers (`_build_action_tensor`, `_world_pose_to_base`,
  `_semantic_to_controller_pose_world`).
* `grasp_ranking.py` — geometric filter + EE-pose ranking
  (`_filter_candidates`, `_rank_candidates_by_current_pose`,
  `_prepare_ranked_grasp_candidates`, `_candidate_attempt_payload`,
  `_world_to_base_xy_simple`).

What stays here:

* `SceneAutoGraspCollectArgs` — CLI args dataclass.
* JSON helpers (`_load_json`, `_write_json`,
  `_selection_output_path`, `_shortlist_output_path`).
* `_build_scene_mouse_collect_args` / `_build_preview_selection_output` —
  small adapters from auto-grasp args to the generic scene-builder args.
* `_refresh_live_target_grasp_payload` /
  `_refresh_candidate_world_pose_after_reset` — re-poll the live stage
  to refresh grasp poses after physics settle / scene reset.
* `_attempt_recorded_grasp` — one-episode rollout (reset → pre_grasp →
  approach → close → lift → retreat).
* `_run_preview_gate` — the Run/Close interactive gate.
* `_run_episode_loop` — `--num-episodes` rollouts with optional
  per-episode target XY randomization.
* `run_scene_auto_grasp_collect` — top-level orchestrator.

Names that used to be defined here are re-exported below so existing
`from .scene_auto_grasp_collect import _restore_target_rigid_body_state`
(etc.) callers — notably `scene_eval_policy.py` — keep working
unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils

from ..ui.auto_grasp_preview import (
    EE_MARKER_DEBUG_TRANSLATION_LOCAL,
    AutoGraspPreviewUI,
    _clear_grasp_visuals,
    _refresh_preview_markers,
    _wait_for_run_request,
)
from ..record.auto_grasp_writer import (
    PHASE_NAME_TO_ID,
    SceneAutoGraspEpisodeWriter,
    _recover_corrupt_hdf5_file,
)
from ..grasp.execution import FilteredGraspExecution, expand_grasp_candidates
from ..grasp.proposals_io import (
    _auto_manifest_path,
    _load_or_build_scene_grasp_payload,
    _maybe_lazy_build_target_annotation,
    _resolve_adjacent_path,
    _resolve_manifest_and_annotation_root,
    _world_bbox_payload,
)
from ..grasp.ranking import (
    _candidate_attempt_payload,
    _filter_candidates,
    _prepare_ranked_grasp_candidates,
    _rank_candidates_by_current_pose,
    _world_to_base_xy_simple,
)
from ..grasp.target_state import (
    _quat_wxyz_to_rotvec,
    _restore_target_rigid_body_state,
    _robot_forward_xy_world,
    _shifted_candidate,
    _shifted_target_snapshot,
    _snapshot_target_rigid_body_state,
)
from ..control.phase_runner import (
    _build_action_tensor,
    _run_hold_phase,
    _run_target_phase,
    _semantic_to_controller_pose_world,
    _world_pose_to_base,
)
from .scene_mouse_collect import (
    SceneMouseCollectArgs,
    SceneTeleopEpisodeWriter,
    _build_scene_mouse_collect,
    _compute_world_prim_max_z,
    _get_rigid_body_view,
    _plan_camera_pose,
    _planned_base_height,
    _read_rigid_body_world_position_z,
    _reset_scene_to_plan,
    _settle_dynamic_scene,
)


DEFAULT_SELECTION_FILENAME = "selected_grasp_proposal.json"
DEFAULT_SHORTLIST_FILENAME = "grasp_candidate_shortlist.json"


# Backward-compat re-export surface. Every name below either lives in a
# sibling module that this file imports from, or is defined further down
# in the orchestrator. External callers (notably `scene_eval_policy.py`)
# import these by name from this module, so the move into focused
# modules has to stay invisible at the import surface.
__all__ = [
    "AutoGraspPreviewUI",
    "DEFAULT_SELECTION_FILENAME",
    "DEFAULT_SHORTLIST_FILENAME",
    "EE_MARKER_DEBUG_TRANSLATION_LOCAL",
    "FilteredGraspExecution",
    "PHASE_NAME_TO_ID",
    "SceneAutoGraspCollectArgs",
    "SceneAutoGraspEpisodeWriter",
    "SceneMouseCollectArgs",
    "SceneTeleopEpisodeWriter",
    "_attempt_recorded_grasp",
    "_auto_manifest_path",
    "_build_action_tensor",
    "_build_preview_selection_output",
    "_build_scene_mouse_collect",
    "_build_scene_mouse_collect_args",
    "_candidate_attempt_payload",
    "_clear_grasp_visuals",
    "_compute_world_prim_max_z",
    "_filter_candidates",
    "_get_rigid_body_view",
    "_load_json",
    "_load_or_build_scene_grasp_payload",
    "_maybe_lazy_build_target_annotation",
    "_plan_camera_pose",
    "_planned_base_height",
    "_prepare_ranked_grasp_candidates",
    "_quat_wxyz_to_rotvec",
    "_rank_candidates_by_current_pose",
    "_read_rigid_body_world_position_z",
    "_recover_corrupt_hdf5_file",
    "_refresh_candidate_world_pose_after_reset",
    "_refresh_live_target_grasp_payload",
    "_refresh_preview_markers",
    "_reset_scene_to_plan",
    "_resolve_adjacent_path",
    "_resolve_manifest_and_annotation_root",
    "_restore_target_rigid_body_state",
    "_robot_forward_xy_world",
    "_run_episode_loop",
    "_run_hold_phase",
    "_run_preview_gate",
    "_run_target_phase",
    "_selection_output_path",
    "_semantic_to_controller_pose_world",
    "_settle_dynamic_scene",
    "_shifted_candidate",
    "_shifted_target_snapshot",
    "_shortlist_output_path",
    "_snapshot_target_rigid_body_state",
    "_wait_for_run_request",
    "_world_bbox_payload",
    "_world_pose_to_base",
    "_world_to_base_xy_simple",
    "_write_json",
    "run_scene_auto_grasp_collect",
]


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class SceneAutoGraspCollectArgs:
    device: str
    num_envs: int
    dataset_file: str
    capture_hz: float
    append: bool
    scene_usd_path: str
    scene_graph_path: str
    placements_path: str
    target: str | None
    support: str | None
    plan_output_dir: str
    base_z_bias: float
    arm_side_preference: str
    wait_for_run_request: bool
    manifest_path: str | None
    annotation_root: str | None
    scene_grasp_proposals_path: str | None
    lazy_build_target_annotation: bool
    axis_band_slide_samples: int
    axis_band_ring_samples: int
    max_geom_candidates: int
    workspace_margin: float
    body_clearance_margin: float
    pre_grasp_distance: float
    lift_height: float
    retreat_distance: float
    approach_clearance: float
    pre_grasp_steps: int
    approach_steps: int
    close_steps: int
    lift_steps: int
    retreat_steps: int
    pos_tol: float
    grasp_pos_tol: float
    rot_tol_deg: float
    success_lift_delta: float
    start_pose_distance_weight: float
    start_pose_rotation_weight: float
    agibot_ee_frame_remap: str
    num_episodes: int
    fingertip_distance: float
    # Straight-line interpolation speeds used inside `_run_target_phase`. When
    # `phase_linear_speed > 0` (m/s), the commanded target ramps from the
    # phase's starting EE pose toward the final target at this linear speed
    # instead of jumping directly to the final pose. Same for angular speed
    # (deg/s). Both zero preserves the original "command final every step"
    # behaviour that relies on the controller's own velocity limits.
    phase_linear_speed: float = 0.0
    phase_angular_speed_deg: float = 0.0
    # Per-episode randomization: shift the target rigid body along the
    # robot's forward axis (world-frame projection of the base +x direction)
    # by a uniform sample in [-range, +range] meters before each rollout.
    # The cached grasp candidate's waypoints are shifted by the same vector
    # so the planned pre-grasp / grasp / lift / retreat poses track the
    # moved target. 0 disables randomization.
    target_forward_randomization: float = 0.0


# =============================================================================
# JSON / output path helpers
# =============================================================================
def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _selection_output_path(args: SceneAutoGraspCollectArgs) -> Path:
    return Path(args.plan_output_dir).resolve() / DEFAULT_SELECTION_FILENAME


def _shortlist_output_path(args: SceneAutoGraspCollectArgs) -> Path:
    return Path(args.plan_output_dir).resolve() / DEFAULT_SHORTLIST_FILENAME


# =============================================================================
# Live-stage helpers (refresh proposals after physics settle / reset)
# =============================================================================
def _refresh_live_target_grasp_payload(
    scene,
    controller,
    sync_cameras,
    args: SceneAutoGraspCollectArgs,
    *,
    target_prim: str,
    settle_steps: int = 20,
) -> tuple[dict[str, Any], Path]:
    _settle_dynamic_scene(scene, controller, sync_cameras, settle_steps=max(0, int(settle_steps)))
    return _load_or_build_scene_grasp_payload(
        scene.stage,
        args,
        target_prim=target_prim,
    )


def _refresh_candidate_world_pose_after_reset(
    stage,
    args: SceneAutoGraspCollectArgs,
    candidate: FilteredGraspExecution,
) -> FilteredGraspExecution:
    # After `_reset_scene_to_plan` the target prim's world transform may have
    # changed (scene reset + physics settle). The cached `candidate.grasp.*_world`
    # fields still hold the pre-reset world coordinates, so re-expand the grasp
    # payload against the current stage state and splice the refreshed pose back
    # in. Match by `candidate_id` so we stay on the same sampled grasp.
    target_prim = candidate.grasp.object_prim
    try:
        refreshed_payload, _refreshed_path = _load_or_build_scene_grasp_payload(
            stage,
            args,
            target_prim=target_prim,
        )
    except Exception:
        return candidate

    refreshed_grasps = expand_grasp_candidates(
        refreshed_payload,
        target_prim=target_prim,
        axis_band_slide_samples=int(args.axis_band_slide_samples),
        axis_band_ring_samples=int(args.axis_band_ring_samples),
    )
    original_id = candidate.grasp.candidate_id
    match = next((g for g in refreshed_grasps if g.candidate_id == original_id), None)
    if match is None:
        return candidate

    return replace(
        candidate,
        grasp=match,
        pre_grasp_pos_world=match.position_world,
        pre_grasp_quat_world=match.quat_wxyz_world,
        lift_pos_world=match.position_world,
        lift_quat_world=match.quat_wxyz_world,
        retreat_pos_world=match.position_world,
        retreat_quat_world=match.quat_wxyz_world,
    )


# =============================================================================
# Args / selection-output adapters
# =============================================================================
def _build_scene_mouse_collect_args(args: SceneAutoGraspCollectArgs, initial_arm_side: str) -> SceneMouseCollectArgs:
    # Translate auto-grasp CLI args into the generic scene-builder args used by
    # the teleop pipeline. Collision-approx / decomposition knobs aren't part
    # of auto-grasp's own CLI, so bake in the teleop defaults.
    return SceneMouseCollectArgs(
        device=args.device,
        num_envs=args.num_envs,
        dataset_file=args.dataset_file,
        capture_hz=args.capture_hz,
        append=args.append,
        lin_step=0.015,
        ang_step=0.10,
        scene_usd_path=args.scene_usd_path,
        scene_graph_path=args.scene_graph_path,
        placements_path=args.placements_path,
        target=args.target,
        support=args.support,
        object_collision_approx="convex_decomposition",
        target_collision_approx="convex_decomposition",
        convex_decomp_voxel_resolution=1_000_000,
        convex_decomp_max_convex_hulls=64,
        convex_decomp_error_percentage=2.0,
        convex_decomp_shrink_wrap=True,
        plan_output_dir=args.plan_output_dir,
        base_z_bias=args.base_z_bias,
        arm_side=initial_arm_side,
        show_workspace=False,
    )


def _build_preview_selection_output(
    args: SceneAutoGraspCollectArgs,
    proposals_path: Path,
    plan,
    candidate: FilteredGraspExecution,
    selection_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scene_usd_path": str(Path(args.scene_usd_path).resolve()),
        "scene_grasp_proposals_path": str(proposals_path),
        "target_prim": plan.target_prim,
        "support_prim": plan.support_prim,
        "agibot_ee_frame_remap": args.agibot_ee_frame_remap,
        "selected_grasp_candidate_id": candidate.grasp.candidate_id,
        "selected_grasp_proposal": selection_payload,
        "robot_filter_attempts": [],
    }


# =============================================================================
# One-episode rollout: reset → pre_grasp → approach → close → lift → retreat
# =============================================================================
def _attempt_recorded_grasp(
    scene,
    controller,
    cameras: dict[str, object],
    sync_cameras,
    writer: SceneAutoGraspEpisodeWriter,
    stage,
    scene_root_path: str,
    plan,
    robot_name: str,
    base_z: float,
    candidate: FilteredGraspExecution,
    selection_payload: dict[str, Any],
    args: SceneAutoGraspCollectArgs,
    target_state_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_live_prim_path = f"{scene_root_path}/{Path(plan.target_prim).name}"

    _reset_scene_to_plan(scene, controller, plan, base_z, sync_cameras)
    # `_reset_scene_to_plan` only resets isaaclab-tracked assets (robot,
    # cubes). The target lives inside an untracked `AssetBaseCfg` USD
    # reference, so we teleport it back through the physx tensor view the
    # same way isaaclab's `RigidObject.write_root_link_pose_to_sim` does.
    if target_state_snapshot is not None:
        _restore_target_rigid_body_state(target_state_snapshot)
        scene.write_data_to_sim()
        controller.sim.step()
        scene.update(controller.sim.get_physics_dt())
    if controller.arm_switch_supported:
        controller.switch_arm_side(candidate.arm_side)

    # Capture the bolt's pre-rollout world Z from the physx backend. This is
    # the "on the table, at rest" baseline for the lift check below. Reading
    # via `SingleXFormPrim.get_world_pose()` bypasses the classic USD stage,
    # so it reflects the true physics state even on a GPU scene where
    # `UsdGeom.BBoxCache` is stale.
    baseline_target_z = _read_rigid_body_world_position_z(target_live_prim_path)

    def _phase_diag(label: str, target_pos_world) -> None:
        ee_pos_w, _ee_quat_w = controller.current_ee_pose_world()
        ee_xyz = tuple(round(float(v), 4) for v in ee_pos_w[0].detach().cpu().tolist())
        target_xyz = tuple(round(float(v), 4) for v in target_pos_world)
        bolt_z = _read_rigid_body_world_position_z(target_live_prim_path)
        bolt_z_str = f"{bolt_z:.4f}" if bolt_z is not None else "None"
        print(f"[DIAG] after {label}: ee_world={ee_xyz}, target_world={target_xyz}, bolt_z={bolt_z_str}")

    writer.set_selected_grasp(selection_payload)
    sim_time = 0.0
    writer.start_recording(sim_time)
    _phase_diag("reset", candidate.pre_grasp_pos_world)

    sim_time, pre_result = _run_target_phase(
        scene,
        controller,
        cameras,
        sync_cameras,
        writer,
        sim_time,
        robot_name=robot_name,
        ee_frame_remap=args.agibot_ee_frame_remap,
        fingertip_distance=args.fingertip_distance,
        phase_linear_speed=args.phase_linear_speed,
        phase_angular_speed_deg=args.phase_angular_speed_deg,
        phase_name="pre_grasp",
        target_pos_world=candidate.pre_grasp_pos_world,
        target_quat_world=candidate.pre_grasp_quat_world,
        gripper_closed=False,
        max_steps=args.pre_grasp_steps,
        pos_tol=args.pos_tol,
        rot_tol_deg=args.rot_tol_deg,
    )
    _phase_diag("pre_grasp", candidate.pre_grasp_pos_world)
    if not pre_result["success"]:
        # Soft warning: the robot didn't hit `pos_tol` at pre-grasp but may still
        # be close enough for the approach/close/lift pipeline to produce a real
        # grab. Keep recording and let `_target_lift_success` decide.
        print(
            f"[WARN] pre_grasp did not converge to tolerance "
            f"(pos_err={pre_result['position_error']:.4f}m, rot_err={pre_result['rotation_error_deg']:.2f}deg); "
            f"continuing anyway."
        )

    sim_time, approach_result = _run_target_phase(
        scene,
        controller,
        cameras,
        sync_cameras,
        writer,
        sim_time,
        robot_name=robot_name,
        ee_frame_remap=args.agibot_ee_frame_remap,
        fingertip_distance=args.fingertip_distance,
        phase_linear_speed=args.phase_linear_speed,
        phase_angular_speed_deg=args.phase_angular_speed_deg,
        phase_name="approach",
        target_pos_world=candidate.grasp.position_world,
        target_quat_world=candidate.grasp.quat_wxyz_world,
        gripper_closed=False,
        max_steps=args.approach_steps,
        pos_tol=args.grasp_pos_tol,
        rot_tol_deg=args.rot_tol_deg,
    )
    _phase_diag("approach", candidate.grasp.position_world)
    # Do NOT discard here if `approach` failed to hit its tolerance: fingers may
    # have bumped the target mid-motion and the wrist just couldn't reach the
    # nominal grasp pose. Still try to close the gripper — `_target_lift_success`
    # after the lift phase is the authoritative "did the robot actually grab it"
    # check, so let that decide the episode outcome.

    sim_time, close_result = _run_hold_phase(
        scene,
        controller,
        cameras,
        sync_cameras,
        writer,
        sim_time,
        robot_name=robot_name,
        ee_frame_remap=args.agibot_ee_frame_remap,
        fingertip_distance=args.fingertip_distance,
        phase_name="close",
        hold_pos_world=candidate.grasp.position_world,
        hold_quat_world=candidate.grasp.quat_wxyz_world,
        gripper_closed=True,
        steps=args.close_steps,
    )
    sim_time, lift_result = _run_target_phase(
        scene,
        controller,
        cameras,
        sync_cameras,
        writer,
        sim_time,
        robot_name=robot_name,
        ee_frame_remap=args.agibot_ee_frame_remap,
        fingertip_distance=args.fingertip_distance,
        phase_linear_speed=args.phase_linear_speed,
        phase_angular_speed_deg=args.phase_angular_speed_deg,
        phase_name="lift",
        target_pos_world=candidate.lift_pos_world,
        target_quat_world=candidate.lift_quat_world,
        gripper_closed=True,
        max_steps=args.lift_steps,
        pos_tol=args.pos_tol,
        rot_tol_deg=args.rot_tol_deg,
    )
    if not lift_result["success"]:
        print(
            f"[WARN] lift did not converge to tolerance "
            f"(pos_err={lift_result['position_error']:.4f}m, rot_err={lift_result['rotation_error_deg']:.2f}deg); "
            f"continuing to physical lift check."
        )

    # Relative lift test: did the bolt's physx-backed root Z rise by at least
    # `success_lift_delta` metres compared to the per-episode baseline we
    # captured right after reset. This avoids all of the USD BBoxCache / Fabric
    # staleness problems — `_read_rigid_body_world_position_z` reads from the
    # physx scene directly.
    current_target_z = _read_rigid_body_world_position_z(target_live_prim_path)
    if baseline_target_z is None or current_target_z is None:
        rise = None
        lift_success = False
    else:
        rise = float(current_target_z) - float(baseline_target_z)
        lift_success = rise >= float(args.success_lift_delta)
    print(
        f"[DEBUG] lift_object check: baseline_z={baseline_target_z}, "
        f"current_z={current_target_z}, rise={rise}, "
        f"threshold={args.success_lift_delta} (m); passed={lift_success}"
    )
    if not lift_success:
        # The only fatal check: physics says the bolt never rose enough off
        # its resting position. Discard the episode.
        writer.stop_and_discard()
        return {
            "success": False,
            "failed_phase": "lift_object",
            "lift": lift_result,
            "baseline_target_z": baseline_target_z,
            "current_target_z": current_target_z,
            "rise": rise,
        }

    sim_time, retreat_result = _run_target_phase(
        scene,
        controller,
        cameras,
        sync_cameras,
        writer,
        sim_time,
        robot_name=robot_name,
        ee_frame_remap=args.agibot_ee_frame_remap,
        fingertip_distance=args.fingertip_distance,
        phase_linear_speed=args.phase_linear_speed,
        phase_angular_speed_deg=args.phase_angular_speed_deg,
        phase_name="retreat",
        target_pos_world=candidate.retreat_pos_world,
        target_quat_world=candidate.retreat_quat_world,
        gripper_closed=True,
        max_steps=args.retreat_steps,
        pos_tol=args.pos_tol,
        rot_tol_deg=args.rot_tol_deg,
    )
    if not retreat_result["success"]:
        # The bolt was already lifted above the support (we passed the fatal
        # `_target_lift_success` check), so the episode is still a valid grasp
        # even if the retreat motion didn't hit its nominal pose exactly.
        print(
            f"[WARN] retreat did not converge to tolerance "
            f"(pos_err={retreat_result['position_error']:.4f}m, rot_err={retreat_result['rotation_error_deg']:.2f}deg); "
            f"saving episode anyway since the lift check already passed."
        )

    writer.stop_and_save()
    return {
        "success": True,
        "pre_grasp": pre_result,
        "approach": approach_result,
        "close": close_result,
        "lift": lift_result,
        "retreat": retreat_result,
    }


# =============================================================================
# Preview gate (optional Run/Close UI before the rollout loop)
# =============================================================================
def _run_preview_gate(
    *,
    simulation_app,
    scene,
    controller,
    sync_cameras,
    plan,
    robot_name: str,
    args: SceneAutoGraspCollectArgs,
    candidate: FilteredGraspExecution,
) -> bool:
    # Draws the selected grasp's axis marker + the robot's current gripper
    # frame, then blocks on the `Run Selected Grasp` / `Close Preview` UI
    # until the user decides. Returns True on Run, False on Close. All
    # preview visuals are wiped before returning either way so the recorded
    # rollout starts from a clean scene.
    preview_state = {"candidate": candidate}

    def _refresh_preview() -> None:
        refreshed_candidate = _refresh_candidate_world_pose_after_reset(
            scene.stage,
            args,
            preview_state["candidate"],
        )
        preview_state["candidate"] = refreshed_candidate
        _refresh_preview_markers(
            scene.stage,
            controller,
            refreshed_candidate,
            robot_name=robot_name,
            ee_frame_remap=args.agibot_ee_frame_remap,
        )

    # Scene was just built + settled; no reset needed before drawing.
    _refresh_preview()

    if not args.wait_for_run_request:
        _clear_grasp_visuals(scene.stage)
        return True

    print("[INFO] Waiting for Run Selected Grasp button. The robot will stay still until you click it.")
    should_run = _wait_for_run_request(
        simulation_app,
        scene,
        controller,
        sync_cameras,
        title=f"{robot_name} Auto Grasp Preview",
        refresh_preview=_refresh_preview,
    )
    _clear_grasp_visuals(scene.stage)
    if not should_run:
        print("[INFO] Auto grasp preview closed before execution.")
        return False
    return True


# =============================================================================
# Multi-episode rollout loop
# =============================================================================
def _run_episode_loop(
    *,
    scene,
    controller,
    cameras,
    sync_cameras,
    writer: SceneAutoGraspEpisodeWriter,
    plan,
    robot_name: str,
    base_z: float,
    candidate: FilteredGraspExecution,
    selection_payload: dict[str, Any],
    selection_output: dict[str, Any],
    args: SceneAutoGraspCollectArgs,
    target_state_snapshot: dict[str, Any] | None,
    scene_root_path: str,
) -> None:
    num_episodes = max(1, int(args.num_episodes))
    episode_results: list[dict[str, Any]] = []
    success_count = 0
    import random

    randomization_range = float(args.target_forward_randomization)
    fwd_x, fwd_y = _robot_forward_xy_world(controller) if randomization_range > 0.0 else (0.0, 0.0)
    for episode_idx in range(num_episodes):
        print(f"[INFO] === Episode {episode_idx + 1}/{num_episodes} ===")
        if randomization_range > 0.0:
            delta = random.uniform(-randomization_range, randomization_range)
            offset_xy = (fwd_x * delta, fwd_y * delta)
            episode_snapshot = _shifted_target_snapshot(target_state_snapshot, offset_xy)
            episode_candidate = _shifted_candidate(candidate, offset_xy)
            print(
                f"[INFO] Target forward randomization: delta={delta:+.4f}m "
                f"(offset_xy=({offset_xy[0]:+.4f}, {offset_xy[1]:+.4f}))"
            )
        else:
            episode_snapshot = target_state_snapshot
            episode_candidate = candidate
        try:
            rollout_result = _attempt_recorded_grasp(
                scene,
                controller,
                cameras,
                sync_cameras,
                writer,
                scene.stage,
                scene_root_path,
                plan,
                robot_name,
                base_z,
                episode_candidate,
                selection_payload,
                args,
                target_state_snapshot=episode_snapshot,
            )
        except Exception as exc:
            print(f"[WARN] Episode {episode_idx + 1}: rollout raised: {exc}")
            episode_results.append({
                "episode": episode_idx,
                "success": False,
                "failed_stage": "rollout_exception",
                "error": str(exc),
            })
            continue

        rollout_result["episode"] = episode_idx
        selection_output["rollout_result"] = rollout_result
        _write_json(_selection_output_path(args), selection_output)
        episode_results.append(rollout_result)
        if rollout_result.get("success"):
            success_count += 1
            print(f"[INFO] Episode {episode_idx + 1}: SUCCESS")
        else:
            print(f"[WARN] Episode {episode_idx + 1}: failed at phase={rollout_result.get('failed_phase')}")

    selection_output["episode_results"] = episode_results
    selection_output["episode_success_count"] = int(success_count)
    selection_output["episode_total"] = int(num_episodes)
    _write_json(_selection_output_path(args), selection_output)
    print(f"[INFO] Finished {num_episodes} episodes — {success_count} successful, {num_episodes - success_count} failed.")


# =============================================================================
# Main entry: orchestrate the full pipeline
# =============================================================================
def run_scene_auto_grasp_collect(simulation_app, robot_name: str, args: SceneAutoGraspCollectArgs) -> None:
    """Drive the full auto-grasp collection pipeline end-to-end.

    Stages:
      1. Build the isaaclab scene + controller + cameras (via scene_mouse_collect).
      2. Load / build the scene-grasp proposals cache, refresh against live stage.
      3. Snapshot the target rigid body's physx pose so we can reset it between
         episodes (isaaclab's own `scene.reset()` doesn't track it).
      4. Filter + rank grasp candidates by "closest to current EE pose".
      5. Pick `ranked_candidates[0]` as the grasp to execute.
      6. Optionally show preview markers + block on the Run/Close UI.
      7. Build the HDF5 episode writer.
      8. Loop `--num-episodes` rollouts of `_attempt_recorded_grasp` and write
         each to the HDF5 dataset.
    """
    if args.num_envs != 1:
        raise ValueError("Scene auto grasp collection only supports --num_envs 1.")

    initial_arm_side = "left" if args.arm_side_preference == "auto" else args.arm_side_preference
    build_args = _build_scene_mouse_collect_args(args, initial_arm_side)
    env_name = f"Isaac-{robot_name.capitalize()}-SceneAutoGraspCollect-v0"
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
    writer: SceneAutoGraspEpisodeWriter | None = None
    try:
        # --- Stage 1: build scene + robot ---
        (
            scene,
            controller,
            cameras,
            sync_cameras,
            camera_aliases,
            plan,
            effective_base_z_bias,
            aligned_base_z,
            physics_rebind_summary,
            floor_realign_summary,
            workspace_visual_summary,
        ) = _build_scene_mouse_collect(sim, robot_name, build_args)
        planned_eye, planned_target = _plan_camera_pose(plan)
        sim.set_camera_view(planned_eye, planned_target)
        del aligned_base_z, physics_rebind_summary, floor_realign_summary, workspace_visual_summary

        scene_root_path = f"{scene.env_prim_paths[0]}/GeneratedScene"
        base_z = _planned_base_height(robot_name) + float(effective_base_z_bias)

        # --- Stage 2: refresh scene grasp payload against live stage ---
        proposals_payload, proposals_path = _refresh_live_target_grasp_payload(
            scene, controller, sync_cameras, args, target_prim=plan.target_prim
        )

        # --- Stage 3: snapshot target rigid body pose for per-episode reset ---
        target_live_prim_path = f"{scene_root_path}/{Path(plan.target_prim).name}"
        target_state_snapshot = _snapshot_target_rigid_body_state(target_live_prim_path)
        if target_state_snapshot is None:
            print(
                f"[WARN] Could not snapshot target prim state for {target_live_prim_path}; "
                "bolt will not be reset between episodes."
            )

        # --- Stage 4+5: filter, rank, pick top candidate ---
        ranked_candidates = _prepare_ranked_grasp_candidates(
            scene, controller, sync_cameras, scene.stage, scene_root_path,
            plan, robot_name, base_z, proposals_payload, args,
            shortlist_path=_shortlist_output_path(args),
        )
        candidate = ranked_candidates[0]
        selection_payload = _candidate_attempt_payload(
            candidate,
            evaluation={
                "success": False,
                "preview": True,
                "reason": "Preview candidate taken from geometric ranking before execution filtering.",
            },
        )
        selection_output = _build_preview_selection_output(
            args, proposals_path, plan, candidate, selection_payload
        )
        _write_json(_selection_output_path(args), selection_output)

        print(f"[INFO] {robot_name} scene auto grasp collection ready.")
        print(f"[INFO] Scene USD: {os.path.abspath(args.scene_usd_path)}")
        print(f"[INFO] Scene grasp proposals: {proposals_path}")
        print(f"[INFO] Target object: {plan.target_prim}")
        print(f"[INFO] Support object: {plan.support_prim}")
        print(f"[INFO] Preview grasp: {candidate.grasp.candidate_id} arm={selection_payload['arm_side']}")

        # --- Stage 6: preview + Run/Close gate ---
        should_run = _run_preview_gate(
            simulation_app=simulation_app,
            scene=scene,
            controller=controller,
            sync_cameras=sync_cameras,
            plan=plan,
            robot_name=robot_name,
            args=args,
            candidate=candidate,
        )
        if not should_run:
            return

        # --- Stage 7: build HDF5 writer ---
        writer = SceneAutoGraspEpisodeWriter(
            args.dataset_file,
            args.capture_hz,
            args.append,
            env_name,
            camera_aliases,
            plan,
            args.scene_usd_path,
            args.scene_graph_path,
            args.placements_path,
            initial_arm_side=initial_arm_side,
            arm_switch_supported=controller.arm_switch_supported,
        )
        print(f"[INFO] Dataset: {os.path.abspath(writer.dataset_file)}")
        print(f"[INFO] Executing grasp: {candidate.grasp.candidate_id} arm={selection_payload['arm_side']}")

        # --- Stage 8: episode rollout loop ---
        _run_episode_loop(
            scene=scene,
            controller=controller,
            cameras=cameras,
            sync_cameras=sync_cameras,
            writer=writer,
            plan=plan,
            robot_name=robot_name,
            base_z=base_z,
            candidate=candidate,
            selection_payload=selection_payload,
            selection_output=selection_output,
            args=args,
            target_state_snapshot=target_state_snapshot,
            scene_root_path=scene_root_path,
        )
    finally:
        if writer is not None:
            writer.close()
