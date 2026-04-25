"""Geometric filter + start-pose ranking for grasp candidates.

The scene grasp proposals payload contains many candidate poses per
target (axis-band slides + ring rotations). Auto grasp collection runs
two stages on top of that:

1. **Geometric filter** (`_filter_candidates`) — `expand_grasp_candidates`
   materialises every sample, then `filter_grasp_candidates_geometry`
   discards anything that lands outside the planned base's workspace,
   collides with the robot body envelope, or floats above a support
   that has fallen out of the configured clearance.

2. **Start-pose ranking** (`_rank_candidates_by_current_pose`) — for
   each arm side present in the surviving candidates, read the live EE
   pose, optionally remap it through the agibot EE-frame convention,
   and reorder candidates by weighted (position, rotation) distance to
   the EE. The cheapest grasp to reach from the current pose wins.

`_prepare_ranked_grasp_candidates` chains both stages and writes the
shortlist JSON next to the run output. `_candidate_attempt_payload`
serialises a single candidate for the writer's per-episode metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .ee_frame_remap import apply_local_ee_frame_remap_to_world_quat
from .grasp_execution import (
    FilteredGraspExecution,
    expand_grasp_candidates,
    filter_grasp_candidates_geometry,
    rank_filtered_grasp_candidates_by_start_pose,
)
from .grasp_proposals_io import _world_bbox_payload
from .scene_physics import _compute_world_prim_max_z


DEFAULT_SHORTLIST_FILENAME = "grasp_candidate_shortlist.json"


def _candidate_attempt_payload(candidate: FilteredGraspExecution, evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = candidate.to_payload()
    if evaluation is not None:
        payload["evaluation"] = evaluation
    payload["score"] = float(candidate.score)
    return payload


def _filter_candidates(
    stage,
    scene_root_path: str,
    robot_name: str,
    plan,
    proposals_payload: dict[str, Any],
    args,
) -> list[FilteredGraspExecution]:
    target_live_prim_path = f"{scene_root_path}/{Path(plan.target_prim).name}"
    support_live_prim_path = f"{scene_root_path}/{Path(plan.support_prim).name}"
    target_bbox_world = _world_bbox_payload(stage, target_live_prim_path)
    support_top_z = _compute_world_prim_max_z(stage, support_live_prim_path)
    if support_top_z is None:
        raise RuntimeError(f"Failed to resolve support top z for {support_live_prim_path}")

    candidates = expand_grasp_candidates(
        proposals_payload,
        target_prim=plan.target_prim,
        axis_band_slide_samples=args.axis_band_slide_samples,
        axis_band_ring_samples=args.axis_band_ring_samples,
    )
    filtered = filter_grasp_candidates_geometry(
        candidates,
        robot=robot_name,
        base_pose=plan.base_pose,
        support_center_xy=plan.support_center_xy,
        support_half_extents_xy=plan.support_half_extents_xy,
        support_yaw_deg=plan.support_yaw_deg,
        support_top_z=float(support_top_z),
        target_bbox_world=target_bbox_world,
        preferred_arm_side=None if args.arm_side_preference == "auto" else args.arm_side_preference,
        workspace_margin=args.workspace_margin,
        body_clearance_margin=args.body_clearance_margin,
        pre_grasp_distance=args.pre_grasp_distance,
        lift_height=args.lift_height,
        retreat_distance=args.retreat_distance,
        approach_clearance=args.approach_clearance,
    )
    # `max_geom_candidates` still exists as a safety cap in case geometric
    # filtering lets through an absurd number, but the default is raised so
    # ranking sees every survivor and can pick the globally best grasp by
    # start-pose distance. Set `--max-geom-candidates` explicitly to override.
    cap = max(1, int(args.max_geom_candidates))
    return filtered if cap >= len(filtered) else filtered[:cap]


def _rank_candidates_by_current_pose(
    scene,
    controller,
    sync_cameras: Callable[[], None] | None,
    plan,
    robot_name: str,
    base_z: float,
    candidates: list[FilteredGraspExecution],
    args,
) -> list[FilteredGraspExecution]:
    if not candidates:
        return []

    use_grasp_orientation = robot_name == "agibot" and str(args.agibot_ee_frame_remap or "none").strip().lower() not in {
        "",
        "none",
    }
    # Read each arm's current EE pose without a scene reset — the initial
    # `_build_scene_mouse_collect` already leaves the robot in its reset
    # configuration, and any subsequent settling is small enough to not
    # matter for ranking-distance purposes.
    arm_sides_present = sorted({candidate.arm_side for candidate in candidates})
    current_pose_by_arm: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
    for arm_side in arm_sides_present:
        if controller.arm_switch_supported:
            controller.switch_arm_side(arm_side)
        ee_pos_w, ee_quat_w = controller.current_ee_pose_world()
        current_quat_wxyz = tuple(float(value) for value in ee_quat_w[0].detach().cpu().tolist())
        if use_grasp_orientation:
            current_quat_wxyz = apply_local_ee_frame_remap_to_world_quat(current_quat_wxyz, args.agibot_ee_frame_remap)
        current_pose_by_arm[arm_side] = (
            tuple(float(value) for value in ee_pos_w[0].detach().cpu().tolist()),
            current_quat_wxyz,
        )

    ranked: list[FilteredGraspExecution] = []
    for arm_side, pose in current_pose_by_arm.items():
        arm_candidates = [candidate for candidate in candidates if candidate.arm_side == arm_side]
        ranked.extend(
            rank_filtered_grasp_candidates_by_start_pose(
                arm_candidates,
                current_pos_world=pose[0],
                current_quat_wxyz=pose[1],
                position_weight=args.start_pose_distance_weight,
                rotation_weight=args.start_pose_rotation_weight,
                use_grasp_orientation=use_grasp_orientation,
            )
        )
    ranked.sort(
        key=lambda item: (
            float(item.ranking_score if item.ranking_score is not None else item.score),
            float(item.score),
            float(item.support_clearance),
            float(item.base_frame_xy[0]),
        ),
        reverse=True,
    )
    return ranked


def _world_to_base_xy_simple(
    base_pose: tuple[float, float, float, float],
    point_world: tuple[float, float, float],
) -> tuple[float, float]:
    base_x, base_y, _base_z, base_yaw_deg = (float(value) for value in base_pose)
    dx = float(point_world[0]) - base_x
    dy = float(point_world[1]) - base_y
    yaw_rad = np.radians(-float(base_yaw_deg))
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    return (
        (cos_yaw * dx) - (sin_yaw * dy),
        (sin_yaw * dx) + (cos_yaw * dy),
    )


def _prepare_ranked_grasp_candidates(
    scene,
    controller,
    sync_cameras: Callable[[], None] | None,
    stage,
    scene_root_path: str,
    plan,
    robot_name: str,
    base_z: float,
    proposals_payload: dict[str, Any],
    args,
    *,
    shortlist_path: Path | None = None,
) -> list[FilteredGraspExecution]:
    filtered = _filter_candidates(stage, scene_root_path, robot_name, plan, proposals_payload, args)
    ranking_source = filtered
    ranking_note = "geometric_filter"
    ranked = _rank_candidates_by_current_pose(
        scene,
        controller,
        sync_cameras,
        plan,
        robot_name,
        base_z,
        ranking_source,
        args,
    )
    shortlist_payload = [candidate.to_payload() for candidate in ranked]
    if shortlist_path is None:
        shortlist_path = Path(args.plan_output_dir).resolve() / DEFAULT_SHORTLIST_FILENAME
    shortlist_path.parent.mkdir(parents=True, exist_ok=True)
    shortlist_path.write_text(
        json.dumps(
            {
                "target_prim": plan.target_prim,
                "support_prim": plan.support_prim,
                "candidate_count": len(ranked),
                "candidate_source": ranking_note,
                "ranking_metric": {
                    "target_pose": "pre_grasp",
                    "position_weight": float(args.start_pose_distance_weight),
                    "rotation_weight": float(args.start_pose_rotation_weight),
                    "orientation_reference": (
                        "remapped_current_ee_vs_grasp_orientation"
                        if robot_name == "agibot"
                        and str(args.agibot_ee_frame_remap or "none").strip().lower() not in {"", "none"}
                        else "raw_current_ee_vs_pre_grasp_orientation"
                    ),
                },
                "candidates": shortlist_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not ranked:
        raise RuntimeError(f"No grasp candidates survived geometric filtering for {plan.target_prim}")
    return ranked
