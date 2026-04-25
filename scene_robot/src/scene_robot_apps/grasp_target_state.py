"""PhysX target rigid-body snapshot/restore + per-episode randomization.

The auto-grasp pipeline reuses a single ranked grasp candidate across
many episodes. Between episodes we need to:

* Read the target rigid body's pre-rollout pose so the lift check can
  judge a relative Z rise from a known baseline.
* Teleport the body back to that pose at the start of every episode
  (Isaac Lab's `scene.reset()` doesn't track our `AssetBaseCfg`-imported
  target).
* Optionally jitter the target along the robot's forward axis before
  each rollout (and shift the cached candidate's waypoints by the same
  delta so the EE keeps tracking the moved object).

All of those reads/writes have to go through PhysX tensor views — classic
USD `xformOp` reads are stale on GPU scenes. `_get_rigid_body_view` from
`scene_physics` is the entry point for that, and these helpers wrap it.

`scene_auto_grasp_collect.py` re-exports the public names below so the
existing call sites in `scene_eval_policy.py` (`_snapshot_*`,
`_restore_*`, `_robot_forward_xy_world`, `_shifted_target_snapshot`)
keep working unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .grasp_execution import FilteredGraspExecution
from .scene_physics import _get_rigid_body_view


def _quat_wxyz_to_rotvec(current_quat_wxyz: torch.Tensor, target_quat_wxyz: torch.Tensor) -> torch.Tensor:
    current = current_quat_wxyz.detach().cpu().numpy().reshape(4)
    target = target_quat_wxyz.detach().cpu().numpy().reshape(4)
    current_xyzw = np.array([current[1], current[2], current[3], current[0]], dtype=float)
    target_xyzw = np.array([target[1], target[2], target[3], target[0]], dtype=float)
    delta = R.from_quat(target_xyzw) * R.from_quat(current_xyzw).inv()
    return torch.tensor(delta.as_rotvec(), dtype=torch.float32)


def _snapshot_target_rigid_body_state(prim_path: str) -> dict[str, Any] | None:
    # Capture the target rigid body's current pose directly from the physx
    # tensor view. This is the only reading path that sees the true physics
    # state on a GPU scene — classic USD xformOps and `Usd.Stage` reads are
    # disconnected from physx here. `transforms` is an (N, 7) tensor laid
    # out as `[pos_xyz, quat_xyzw]`, same as what `view.set_transforms()`
    # expects, so the snapshot is written back verbatim during restore.
    view = _get_rigid_body_view(prim_path)
    if view is None:
        return None
    transforms = view.get_transforms()
    if transforms is None or transforms.shape[0] == 0:
        return None
    return {"prim_path": prim_path, "transforms": transforms.clone()}


def _restore_target_rigid_body_state(snapshot: dict[str, Any] | None) -> None:
    # Teleport the rigid body back to the snapshot pose AND zero its linear
    # and angular velocity, all through the physx tensor view (the same API
    # isaaclab's `RigidObject.write_root_link_pose_to_sim` /
    # `write_root_com_velocity_to_sim` use at `rigid_object.py:282, 362`).
    # Writes via USD xformOps don't round-trip to physx on GPU scenes, hence
    # the previous episode's pose "leaked" into the next one.
    if snapshot is None:
        return
    prim_path = snapshot["prim_path"]
    view = _get_rigid_body_view(prim_path)
    if view is None:
        return
    pose = snapshot["transforms"]
    view.set_transforms(pose, indices=torch.arange(pose.shape[0], device=pose.device))
    velocities = torch.zeros((pose.shape[0], 6), device=pose.device, dtype=pose.dtype)
    view.set_velocities(velocities, indices=torch.arange(pose.shape[0], device=pose.device))


def _robot_forward_xy_world(controller) -> tuple[float, float]:
    # World-frame projection of the robot base's +x axis, ignoring Z. Uses
    # the same (root quat wxyz -> 2D forward) derivation we use for the
    # world camera override in scene_mouse_collect — kept inline here so
    # this module can sample a randomization direction without importing
    # from scene_mouse_collect.
    root_pose_w = controller.robot.data.root_pose_w  # (N, 7) xyz + wxyz
    quat = root_pose_w[0, 3:7]
    qw, qx, qy, qz = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    norm = (fwd_x * fwd_x + fwd_y * fwd_y) ** 0.5
    if norm < 1e-8:
        return 1.0, 0.0
    return fwd_x / norm, fwd_y / norm


def _shifted_target_snapshot(
    snapshot: dict[str, Any] | None,
    offset_xy: tuple[float, float],
) -> dict[str, Any] | None:
    # Clone the physx target-body snapshot and add `offset_xy` to the (x, y)
    # position columns, leaving z and the quaternion untouched. The clone
    # isolates per-episode randomization from the canonical snapshot captured
    # once at scene build time.
    if snapshot is None:
        return None
    transforms = snapshot["transforms"].clone()
    transforms[:, 0] += float(offset_xy[0])
    transforms[:, 1] += float(offset_xy[1])
    return {"prim_path": snapshot["prim_path"], "transforms": transforms}


def _shifted_candidate(
    candidate: FilteredGraspExecution,
    offset_xy: tuple[float, float],
) -> FilteredGraspExecution:
    # Shift every world-frame waypoint on a grasp candidate by `offset_xy`.
    # The planner caches a single best candidate and we reuse it across
    # episodes, so when we randomize the target's XY pose we must co-move
    # the grasp/pre-grasp/lift/retreat points by the same delta — otherwise
    # the EE would chase the *original* bolt position while physics shows
    # the bolt shifted, and every episode would miss.
    dx, dy = float(offset_xy[0]), float(offset_xy[1])

    def _shift(pos: tuple[float, float, float]) -> tuple[float, float, float]:
        return (pos[0] + dx, pos[1] + dy, pos[2])

    shifted_grasp = dataclasses.replace(
        candidate.grasp,
        position_world=_shift(candidate.grasp.position_world),
    )
    return dataclasses.replace(
        candidate,
        grasp=shifted_grasp,
        pre_grasp_pos_world=_shift(candidate.pre_grasp_pos_world),
        lift_pos_world=_shift(candidate.lift_pos_world),
        retreat_pos_world=_shift(candidate.retreat_pos_world),
    )
