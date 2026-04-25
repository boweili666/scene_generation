"""Phase runners + small pose-math helpers used by auto grasp rollouts.

A grasp rollout is broken into named phases (`pre_grasp`, `approach`,
`close`, `lift`, `retreat`); each phase either tracks an EE pose target
until convergence (`_run_target_phase`) or holds a pose for a fixed
number of steps (`_run_hold_phase`). Both runners stream observations
into the auto-grasp HDF5 writer per physics step.

The pose-math helpers here are tightly coupled to the runners:

* `_world_pose_to_base` — convert (world pos, world quat) to base-frame
  for `controller.step_pose_target`.
* `_build_action_tensor` — assemble the 7-D `[pos_delta(3), rot_delta(3),
  gripper(1)]` row that gets written into the dataset each step.
* `_semantic_to_controller_pose_world` — apply the agibot EE-frame remap
  + fingertip-distance offset so the wrist target reaches the grasp
  point in the controller's own EE convention.

`scene_auto_grasp_collect.py` re-exports these for backwards-compat.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp

from isaaclab.utils.math import subtract_frame_transforms

from .ee_frame_remap import (
    apply_inverse_local_ee_frame_remap_to_world_quat,
    quat_wxyz_to_matrix,
)
from .grasp_execution import pose_error_metrics
from .grasp_target_state import _quat_wxyz_to_rotvec


def _build_action_tensor(controller, target_pos_b: torch.Tensor, target_quat_b: torch.Tensor, *, gripper_closed: bool) -> torch.Tensor:
    ee_pos_b, ee_quat_b = controller.current_ee_pose_base()
    pos_delta = (target_pos_b[0] - ee_pos_b[0]).detach().cpu().to(dtype=torch.float32)
    rot_delta = _quat_wxyz_to_rotvec(ee_quat_b[0], target_quat_b[0])
    return torch.cat(
        [
            pos_delta,
            rot_delta,
            torch.tensor([1.0 if gripper_closed else 0.0], dtype=torch.float32),
        ],
        dim=0,
    )


def _world_pose_to_base(
    controller,
    position_world: tuple[float, float, float],
    quat_wxyz_world: tuple[float, float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    root_pose_w = controller.robot.data.root_pose_w[:, 0:7]
    pos_w = torch.tensor(position_world, device=controller.sim.device, dtype=torch.float32).unsqueeze(0)
    quat_w = torch.tensor(quat_wxyz_world, device=controller.sim.device, dtype=torch.float32).unsqueeze(0)
    pos_b, quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        pos_w,
        quat_w,
    )
    return pos_b, quat_b


def _semantic_to_controller_pose_world(
    robot_name: str,
    ee_frame_remap: str,
    fingertip_distance: float,
    target_pos_world: tuple[float, float, float],
    target_quat_world: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    # Convert a `(pos, quat)` target expressed in the grasp annotator's semantic
    # EE frame (X=approach, Y=closing, Z=approach×closing) into the controller's
    # wrist-frame target. Applies two corrections:
    #   - rotation: `ee_frame_remap` maps semantic EE axes onto the controller's
    #     own EE-axis convention (agibot-specific; a no-op otherwise).
    #   - translation: when `fingertip_distance > 0`, the wrist is pulled back
    #     along the semantic +X axis (i.e. along -approach in world) so that a
    #     gripper whose fingertips extend `fingertip_distance` ahead of the
    #     wrist places its tips exactly on the original target position.
    pos_out = (
        float(target_pos_world[0]),
        float(target_pos_world[1]),
        float(target_pos_world[2]),
    )
    if float(fingertip_distance) > 0.0:
        rotation_world = quat_wxyz_to_matrix(target_quat_world)
        approach_axis_world = np.asarray(rotation_world[:, 0], dtype=float).reshape(3)
        approach_norm = float(np.linalg.norm(approach_axis_world))
        if approach_norm > 1.0e-8:
            approach_unit = approach_axis_world / approach_norm
            scale = float(fingertip_distance)
            pos_out = (
                pos_out[0] - float(approach_unit[0]) * scale,
                pos_out[1] - float(approach_unit[1]) * scale,
                pos_out[2] - float(approach_unit[2]) * scale,
            )

    quat_out: tuple[float, float, float, float] = (
        float(target_quat_world[0]),
        float(target_quat_world[1]),
        float(target_quat_world[2]),
        float(target_quat_world[3]),
    )
    if robot_name == "agibot" and str(ee_frame_remap or "none").strip().lower() not in {"", "none"}:
        quat_out = apply_inverse_local_ee_frame_remap_to_world_quat(quat_out, ee_frame_remap)
    return pos_out, quat_out


def _run_target_phase(
    scene,
    controller,
    cameras: dict[str, object],
    sync_cameras: Callable[[], None] | None,
    writer,
    sim_time: float,
    *,
    robot_name: str,
    ee_frame_remap: str,
    fingertip_distance: float = 0.0,
    phase_linear_speed: float = 0.0,
    phase_angular_speed_deg: float = 0.0,
    phase_name: str,
    target_pos_world: tuple[float, float, float],
    target_quat_world: tuple[float, float, float, float],
    gripper_closed: bool,
    max_steps: int,
    pos_tol: float,
    rot_tol_deg: float,
) -> tuple[float, dict[str, Any]]:
    reached_steps = 0
    max_position_error = 0.0
    max_rotation_error_deg = 0.0
    last_position_error = float("inf")
    last_rotation_error_deg = float("inf")
    controller_target_pos_world, controller_target_quat_world = _semantic_to_controller_pose_world(
        robot_name, ee_frame_remap, fingertip_distance, target_pos_world, target_quat_world
    )

    # Snapshot current controller EE pose (world frame) and set up speed-limited
    # interpolation. If both speeds are <= 0, `alpha` jumps to 1.0 on the first
    # step and the commanded target is always the final target (original
    # behaviour). Otherwise the commanded target slides from `start` to `final`
    # along a straight line + slerp at the requested linear/angular speeds.
    start_pos_w_tensor, start_quat_w_tensor = controller.current_ee_pose_world()
    start_pos_w = np.asarray(
        [float(v) for v in start_pos_w_tensor[0].detach().cpu().tolist()], dtype=float
    )
    start_quat_w_tuple = tuple(float(v) for v in start_quat_w_tensor[0].detach().cpu().tolist())
    final_pos_w = np.asarray(controller_target_pos_world, dtype=float)
    distance = float(np.linalg.norm(final_pos_w - start_pos_w))
    start_rot = R.from_quat(
        [start_quat_w_tuple[1], start_quat_w_tuple[2], start_quat_w_tuple[3], start_quat_w_tuple[0]]
    )
    final_rot = R.from_quat(
        [
            controller_target_quat_world[1],
            controller_target_quat_world[2],
            controller_target_quat_world[3],
            controller_target_quat_world[0],
        ]
    )
    rot_delta_deg = float(np.degrees((final_rot * start_rot.inv()).magnitude()))
    dt = float(controller.sim.get_physics_dt())
    linear_rate = 1.0
    if float(phase_linear_speed) > 0.0 and distance > 1.0e-6:
        linear_rate = float(phase_linear_speed) * dt / distance
    angular_rate = 1.0
    if float(phase_angular_speed_deg) > 0.0 and rot_delta_deg > 1.0e-3:
        angular_rate = float(phase_angular_speed_deg) * dt / rot_delta_deg
    ramp_rate = float(min(linear_rate, angular_rate))
    slerp = None
    # When speed-limited interpolation is active, auto-extend the phase's step
    # budget so there's enough time to finish the ramp AND let the controller
    # converge to the final target afterwards. Without this, a slow linear
    # speed over a long approach distance never reaches alpha=1 within the
    # fixed `max_steps`, so the phase is reported as failed even though the
    # robot was moving correctly.
    effective_max_steps = max(1, int(max_steps))
    if ramp_rate < 1.0:
        required_ramp_steps = int(np.ceil(1.0 / ramp_rate))
        effective_max_steps = max(effective_max_steps, required_ramp_steps + 30)
        slerp = Slerp([0.0, 1.0], R.concatenate([start_rot, final_rot]))

    for step_index in range(effective_max_steps):
        alpha = min(1.0, ramp_rate * float(step_index + 1))
        if alpha >= 1.0:
            commanded_pos_w = controller_target_pos_world
            commanded_quat_w = controller_target_quat_world
        else:
            interp_pos = start_pos_w + (final_pos_w - start_pos_w) * alpha
            commanded_pos_w = (
                float(interp_pos[0]),
                float(interp_pos[1]),
                float(interp_pos[2]),
            )
            interp_quat_xyzw = slerp([alpha]).as_quat()[0]  # type: ignore[union-attr]
            commanded_quat_w = (
                float(interp_quat_xyzw[3]),
                float(interp_quat_xyzw[0]),
                float(interp_quat_xyzw[1]),
                float(interp_quat_xyzw[2]),
            )
        target_pos_b, target_quat_b = _world_pose_to_base(controller, commanded_pos_w, commanded_quat_w)
        action = _build_action_tensor(controller, target_pos_b, target_quat_b, gripper_closed=gripper_closed)
        controller.step_pose_target(target_pos_b, target_quat_b, gripper_closed)
        scene.write_data_to_sim()
        controller.sim.step()
        sim_time += controller.sim.get_physics_dt()
        scene.update(controller.sim.get_physics_dt())
        if sync_cameras is not None:
            sync_cameras()
        if writer is not None:
            writer.maybe_record_auto_frame(sim_time, action, controller, cameras, phase_name=phase_name)

        ee_pos_b, ee_quat_b = controller.current_ee_pose_base()
        # Measure pose error against the FINAL target (not the interpolated
        # intermediate) so `success` only trips once we've both finished the
        # ramp and the EE has converged to the actual goal.
        final_target_pos_b, final_target_quat_b = _world_pose_to_base(
            controller, controller_target_pos_world, controller_target_quat_world
        )
        last_position_error, last_rotation_error_deg = pose_error_metrics(
            ee_pos_b[0].detach().cpu().numpy(),
            ee_quat_b[0].detach().cpu().numpy(),
            final_target_pos_b[0].detach().cpu().numpy(),
            final_target_quat_b[0].detach().cpu().numpy(),
        )
        max_position_error = max(max_position_error, last_position_error)
        max_rotation_error_deg = max(max_rotation_error_deg, last_rotation_error_deg)
        if (
            alpha >= 1.0
            and last_position_error <= float(pos_tol)
            and last_rotation_error_deg <= float(rot_tol_deg)
        ):
            reached_steps += 1
        else:
            reached_steps = 0
        if reached_steps >= 3:
            break

    return sim_time, {
        "phase_name": phase_name,
        "success": reached_steps >= 3,
        "position_error": float(last_position_error),
        "rotation_error_deg": float(last_rotation_error_deg),
        "max_position_error": float(max_position_error),
        "max_rotation_error_deg": float(max_rotation_error_deg),
    }


def _run_hold_phase(
    scene,
    controller,
    cameras: dict[str, object],
    sync_cameras: Callable[[], None] | None,
    writer,
    sim_time: float,
    *,
    robot_name: str,
    ee_frame_remap: str,
    fingertip_distance: float = 0.0,
    phase_name: str,
    hold_pos_world: tuple[float, float, float],
    hold_quat_world: tuple[float, float, float, float],
    gripper_closed: bool,
    steps: int,
) -> tuple[float, dict[str, Any]]:
    controller_hold_pos_world, controller_hold_quat_world = _semantic_to_controller_pose_world(
        robot_name, ee_frame_remap, fingertip_distance, hold_pos_world, hold_quat_world
    )
    for _ in range(max(1, int(steps))):
        target_pos_b, target_quat_b = _world_pose_to_base(controller, controller_hold_pos_world, controller_hold_quat_world)
        action = _build_action_tensor(controller, target_pos_b, target_quat_b, gripper_closed=gripper_closed)
        controller.step_pose_target(target_pos_b, target_quat_b, gripper_closed)
        scene.write_data_to_sim()
        controller.sim.step()
        sim_time += controller.sim.get_physics_dt()
        scene.update(controller.sim.get_physics_dt())
        if sync_cameras is not None:
            sync_cameras()
        if writer is not None:
            writer.maybe_record_auto_frame(sim_time, action, controller, cameras, phase_name=phase_name)
    return sim_time, {
        "phase_name": phase_name,
        "success": True,
        "steps": int(steps),
    }
