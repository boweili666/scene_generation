"""RobotController — single-arm + whole-body dual-arm execution helper.

This used to live inside `stack_cube.py` alongside the stack-cube task
specs and the demo runner, but in practice the controller is the central
class used by every pipeline in this package (auto grasp collection,
mouse teleop record, scene mouse collect, scene eval) regardless of
whether they touch the stack-cube task at all. Pulling it out keeps
`stack_cube.py` focused on the stack-cube task definitions and shrinks
the file from ~1200 to ~600 lines without changing any imports — every
existing `from .stack_cube import RobotController` keeps working
because `stack_cube.py` re-exports it from here.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import apply_delta_pose, combine_frame_transforms, subtract_frame_transforms

from controller import IsaacLabBimanualPoseController, IsaacLabPoseController

from .stack_cube import (
    MouseClickTeleopUI,
    Phase,
    RobotStackSpec,
    SWITCHABLE_ARM_SIDE_ROBOTS,
    _arm_config_fields,
    _arm_title_label,
    normalize_arm_side,
)


class RobotController:
    def __init__(self, sim: sim_utils.SimulationContext, scene: InteractiveScene, spec: RobotStackSpec, asset_prefix: str = ""):
        self.sim = sim
        self.scene = scene
        self.spec = spec
        self.asset_prefix = asset_prefix
        self.robot_name = f"{asset_prefix}robot"
        self.base_name = f"{asset_prefix}cube_base"
        self.pick_name = f"{asset_prefix}cube_pick"

        self.robot = scene[self.robot_name]
        self.cube_base = scene[self.base_name]
        self.cube_pick = scene[self.pick_name]

        self.use_whole_body = spec.name in {"agibot", "r1lite"}
        self.arm_switch_supported = bool(spec.arm_switch_supported and spec.name in SWITCHABLE_ARM_SIDE_ROBOTS)
        self.active_arm_side = normalize_arm_side(spec.name, spec.arm_side if self.arm_switch_supported else "left")
        self.torso_joint_ids: torch.Tensor | None = None
        self.left_arm_joint_ids: torch.Tensor | None = None
        self.right_arm_joint_ids: torch.Tensor | None = None
        self.bimanual_ik_controller: IsaacLabBimanualPoseController | None = None
        self._arm_controller_cache: dict[str, tuple[IsaacLabPoseController, IsaacLabPoseController]] = {}
        if self.use_whole_body:
            if spec.name == "agibot":
                torso_patterns = ("idx01_body_joint1", "idx02_body_joint2")
                left_patterns = ("idx2[1-7]_arm_l_joint[1-7]",)
                right_patterns = ("idx6[1-7]_arm_r_joint[1-7]",)
            else:
                torso_patterns = ("torso_joint[1-3]",)
                left_patterns = ("left_arm_joint[1-6]",)
                right_patterns = ("right_arm_joint[1-6]",)
            torso_ids, _ = self.robot.find_joints(list(torso_patterns))
            left_ids, _ = self.robot.find_joints(list(left_patterns))
            right_ids, _ = self.robot.find_joints(list(right_patterns))
            if len(torso_ids) == 0 or len(left_ids) == 0 or len(right_ids) == 0:
                raise RuntimeError(
                    f"{spec.name} whole-body joint groups not found: torso={len(torso_ids)}, left={len(left_ids)}, right={len(right_ids)}"
                )
            self.torso_joint_ids = torch.tensor(torso_ids, dtype=torch.long, device=sim.device)
            self.left_arm_joint_ids = torch.tensor(left_ids, dtype=torch.long, device=sim.device)
            self.right_arm_joint_ids = torch.tensor(right_ids, dtype=torch.long, device=sim.device)

        hold_joint_ids: list[int] = []
        for pattern in spec.hold_default_patterns:
            joint_ids, _ = self.robot.find_joints([pattern])
            hold_joint_ids.extend(joint_ids)
        if self.use_whole_body:
            controlled = set(int(i) for i in self.torso_joint_ids.tolist())
            controlled.update(int(i) for i in self.left_arm_joint_ids.tolist())
            controlled.update(int(i) for i in self.right_arm_joint_ids.tolist())
            hold_joint_ids = [jid for jid in hold_joint_ids if int(jid) not in controlled]
        self.hold_joint_ids = torch.tensor(hold_joint_ids, dtype=torch.long, device=sim.device) if hold_joint_ids else None

        self.reset_joint_pos = self.robot.data.default_joint_pos.clone()
        self.reset_joint_vel = self.robot.data.default_joint_vel.clone()
        init_joint_pos = getattr(self.spec.robot_cfg.init_state, "joint_pos", None)
        if isinstance(init_joint_pos, dict):
            for joint_name, joint_value in init_joint_pos.items():
                joint_ids, _ = self.robot.find_joints([joint_name])
                if len(joint_ids) != 1:
                    continue
                self.reset_joint_pos[:, int(joint_ids[0])] = float(joint_value)

        if self.use_whole_body:
            self.bimanual_ik_controller = IsaacLabBimanualPoseController(
                robot_name=spec.name,
                num_envs=scene.num_envs,
                device=sim.device,
                robot=self.robot,
            )
        if self.arm_switch_supported:
            for side in ("left", "right"):
                self._arm_controller_cache[side] = self._build_arm_controllers(side)
        else:
            self._arm_controller_cache[self.active_arm_side] = self._build_arm_controllers(self.active_arm_side)
        self._configure_active_arm(self.active_arm_side)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        self.ee_marker = VisualizationMarkers(marker_cfg.replace(prim_path=f"/Visuals/{spec.name}_{asset_prefix}ee_current"))
        self.goal_marker = VisualizationMarkers(marker_cfg.replace(prim_path=f"/Visuals/{spec.name}_{asset_prefix}ee_goal"))

        num_envs = scene.num_envs
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=sim.device)
        self.phase_time = torch.zeros(num_envs, device=sim.device)
        self.gripper_alpha = torch.zeros(num_envs, 1, device=sim.device)
        self.waypoint_idx = torch.zeros(num_envs, dtype=torch.long, device=sim.device)
        self.phase_start_pos = torch.zeros(num_envs, 3, dtype=torch.float32, device=sim.device)
        self.phase_start_initialized = torch.zeros(num_envs, dtype=torch.bool, device=sim.device)
        self.pick_pos_b_init = torch.zeros(num_envs, 3, dtype=torch.float32, device=sim.device)
        self.home_pos_b_init = torch.zeros(num_envs, 3, dtype=torch.float32, device=sim.device)
        self.ui_target_pos = None
        self.ui_target_quat = None
        self.target_pos = None
        self.target_quat = None
        self.inactive_hold_pos_b: torch.Tensor | None = None
        self.inactive_hold_quat_b: torch.Tensor | None = None

    def _active_arm_fields(self, arm_side: str) -> dict[str, object]:
        if self.spec.name in SWITCHABLE_ARM_SIDE_ROBOTS:
            return _arm_config_fields(self.spec.name, arm_side)
        return {
            "ee_joint_patterns": self.spec.ee_joint_patterns,
            "ee_body_name": self.spec.ee_body_name,
            "gripper_joint_patterns": self.spec.gripper.joint_patterns,
            "hold_open_patterns": self.spec.gripper.hold_open_patterns,
        }

    def _build_arm_controllers(self, arm_side: str) -> tuple[IsaacLabPoseController, IsaacLabPoseController]:
        side = normalize_arm_side(self.spec.name, arm_side if self.arm_switch_supported else "left")
        return (
            IsaacLabPoseController(
                robot_name=self.spec.name,
                num_envs=self.scene.num_envs,
                device=self.sim.device,
                robot=self.robot,
                arm_side=side,
            ),
            IsaacLabPoseController(
                robot_name=self.spec.name,
                num_envs=self.scene.num_envs,
                device=self.sim.device,
                robot=self.robot,
                arm_side=side,
                use_relative_mode=True,
            ),
        )

    def _get_arm_controllers(self, arm_side: str) -> tuple[IsaacLabPoseController, IsaacLabPoseController]:
        side = normalize_arm_side(self.spec.name, arm_side if self.arm_switch_supported else "left")
        controllers = self._arm_controller_cache.get(side)
        if controllers is None:
            controllers = self._build_arm_controllers(side)
            self._arm_controller_cache[side] = controllers
        return controllers

    def _configure_active_arm(self, arm_side: str) -> None:
        side = normalize_arm_side(self.spec.name, arm_side if self.arm_switch_supported else "left")
        fields = self._active_arm_fields(side)
        entity_cfg = SceneEntityCfg(
            self.robot_name,
            joint_names=list(fields["ee_joint_patterns"]),
            body_names=[str(fields["ee_body_name"])],
        )
        entity_cfg.resolve(self.scene)
        self.arm_joint_ids = entity_cfg.joint_ids
        self.ee_body_id = entity_cfg.body_ids[0]
        self.ee_jacobi_idx = self.ee_body_id - 1 if self.robot.is_fixed_base else self.ee_body_id

        gripper_joint_ids, _ = self.robot.find_joints(list(fields["gripper_joint_patterns"]))
        self.gripper_joint_ids = torch.tensor(gripper_joint_ids, dtype=torch.long, device=self.sim.device)

        hold_open_joint_ids: list[int] = []
        for pattern in fields["hold_open_patterns"]:
            joint_ids, _ = self.robot.find_joints([pattern])
            hold_open_joint_ids.extend(joint_ids)
        self.hold_open_joint_ids = (
            torch.tensor(hold_open_joint_ids, dtype=torch.long, device=self.sim.device) if hold_open_joint_ids else None
        )

        self.ik_controller, self.teleop_ik_controller = self._get_arm_controllers(side)
        self.active_arm_side = side
        self.ui_target_pos = None
        self.ui_target_quat = None
        self.target_pos = None
        self.target_quat = None
        self.inactive_hold_pos_b = None
        self.inactive_hold_quat_b = None

    def current_window_title(self) -> str:
        if self.spec.name == "agibot":
            return f"Agibot {_arm_title_label(self.active_arm_side)} Teleop"
        if self.spec.name == "r1lite":
            return f"R1Lite {_arm_title_label(self.active_arm_side)} Teleop"
        return self.spec.window_title

    def switch_arm_side(self, arm_side: str | None = None) -> str:
        if not self.arm_switch_supported:
            return self.active_arm_side
        next_side = normalize_arm_side(
            self.spec.name,
            arm_side or ("right" if self.active_arm_side == "left" else "left"),
        )
        if next_side != self.active_arm_side:
            self._configure_active_arm(next_side)
        return self.active_arm_side

    def _apply_reset_joint_state(self):
        joint_pos = self.reset_joint_pos.clone()
        joint_vel = self.reset_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.set_joint_position_target(joint_pos)
        self.robot.set_joint_velocity_target(joint_vel)

    def _apply_arm_command(self, arm_joint_pos_des: torch.Tensor):
        if self.use_whole_body:
            if self.inactive_hold_pos_b is None or self.inactive_hold_quat_b is None:
                inactive_pos, inactive_quat = self._current_inactive_ee_pose()
                self.inactive_hold_pos_b = inactive_pos.clone()
                self.inactive_hold_quat_b = inactive_quat.clone()
            inactive_pos, inactive_quat = self.inactive_hold_pos_b, self.inactive_hold_quat_b
            if self.active_arm_side == "left":
                left_pos, left_quat = self.target_pos, self.target_quat
                right_pos, right_quat = inactive_pos, inactive_quat
            else:
                left_pos, left_quat = inactive_pos, inactive_quat
                right_pos, right_quat = self.target_pos, self.target_quat
            self.bimanual_ik_controller.set_command(
                torch.cat([left_pos, left_quat], dim=1), torch.cat([right_pos, right_quat], dim=1)
            )
            full_q_des = self.bimanual_ik_controller.compute()
            torso_dof = int(self.torso_joint_ids.numel())
            left_dof = int(self.left_arm_joint_ids.numel())
            right_dof = int(self.right_arm_joint_ids.numel())
            self.robot.set_joint_position_target(full_q_des[:, 0:torso_dof], joint_ids=self.torso_joint_ids)
            self.robot.set_joint_position_target(full_q_des[:, torso_dof : torso_dof + left_dof], joint_ids=self.left_arm_joint_ids)
            self.robot.set_joint_position_target(
                full_q_des[:, torso_dof + left_dof : torso_dof + left_dof + right_dof], joint_ids=self.right_arm_joint_ids
            )
        else:
            self.robot.set_joint_position_target(arm_joint_pos_des, joint_ids=self.arm_joint_ids)

    def _current_inactive_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        left_pos_b, left_quat_b, right_pos_b, right_quat_b = self.bimanual_ik_controller.current_ee_poses_base()
        if self.active_arm_side == "left":
            return right_pos_b, right_quat_b
        return left_pos_b, left_quat_b

    def reset(self):
        self.phase[:] = Phase.ABOVE_PICK
        self.phase_time[:] = 0.0
        self.waypoint_idx[:] = 0
        self.phase_start_initialized[:] = False
        self.gripper_alpha.zero_()
        self.ui_target_pos = None
        self.ui_target_quat = None
        self.inactive_hold_pos_b = None
        self.inactive_hold_quat_b = None

        root_state = self.robot.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins
        if self.spec.root_z_zero:
            root_state[:, 2] = 0.0
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])
        self._apply_reset_joint_state()
        self.robot.reset()

        pick_state = self.cube_pick.data.default_root_state.clone()
        pick_state[:, :3] += self.scene.env_origins
        self.cube_pick.write_root_pose_to_sim(pick_state[:, :7])
        self.cube_pick.write_root_velocity_to_sim(pick_state[:, 7:])
        root_pose_w = self.robot.data.default_root_state.clone()
        self.pick_pos_b_init[:], _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], pick_state[:, :3], pick_state[:, 3:7]
        )

        base_state = self.cube_base.data.default_root_state.clone()
        base_state[:, :3] += self.scene.env_origins
        self.cube_base.write_root_pose_to_sim(base_state[:, :7])
        self.cube_base.write_root_velocity_to_sim(base_state[:, 7:])

    def _current_ee(self):
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.arm_joint_ids]
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_id]
        root_pose_w = self.robot.data.root_pose_w
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        return jacobian, ee_pose_w, arm_joint_pos, ee_pos_b, ee_quat_b

    def _update_hold_joints(self):
        if self.hold_joint_ids is not None and self.hold_joint_ids.numel() > 0:
            self.robot.set_joint_position_target(self.reset_joint_pos[:, self.hold_joint_ids], joint_ids=self.hold_joint_ids)

    def _apply_gripper(self, sim_dt: float, closed: torch.Tensor | bool):
        close_rate = 1.0 / max(self.spec.gripper.close_time, 1.0e-3)
        open_rate = 1.0 / max(self.spec.gripper.open_time, 1.0e-3)
        if isinstance(closed, bool):
            if closed:
                self.gripper_alpha = torch.clamp(self.gripper_alpha + close_rate * sim_dt, 0.0, 1.0)
            else:
                self.gripper_alpha = torch.clamp(self.gripper_alpha - open_rate * sim_dt, 0.0, 1.0)
        else:
            self.gripper_alpha[closed] = torch.clamp(self.gripper_alpha[closed] + close_rate * sim_dt, 0.0, 1.0)
            self.gripper_alpha[~closed] = torch.clamp(self.gripper_alpha[~closed] - open_rate * sim_dt, 0.0, 1.0)

        if self.spec.gripper.mode == "simple":
            limits = self.robot.data.soft_joint_pos_limits[:, self.gripper_joint_ids, :]
            gripper_open = limits[..., 0]
            gripper_close = limits[..., 1]
            close_soft = gripper_open + (gripper_close - gripper_open) * self.spec.gripper.close_ratio
            cmd = gripper_open + (close_soft - gripper_open) * self.gripper_alpha
            self.robot.set_joint_position_target(cmd, joint_ids=self.gripper_joint_ids)
        elif self.spec.gripper.mode == "signed_limit":
            limits = self.robot.data.soft_joint_pos_limits[:, self.gripper_joint_ids, :]
            low = limits[..., 0]
            high = limits[..., 1]
            gripper_open = torch.where(torch.abs(low) > torch.abs(high), low, high)
            gripper_close = torch.where(torch.abs(low) <= torch.abs(high), low, high)
            close_soft = gripper_open + (gripper_close - gripper_open) * self.spec.gripper.close_ratio
            cmd = gripper_open + (close_soft - gripper_open) * self.gripper_alpha
            self.robot.set_joint_position_target(cmd, joint_ids=self.gripper_joint_ids)
        elif self.spec.gripper.mode == "omnipicker":
            soft_limits = self.robot.data.soft_joint_pos_limits[:, self.gripper_joint_ids, :]
            hard_limits = self.robot.data.joint_pos_limits[:, self.gripper_joint_ids, :]
            soft_low = soft_limits[..., 0]
            soft_high = soft_limits[..., 1]
            hard_low = hard_limits[..., 0]
            hard_high = hard_limits[..., 1]
            gripper_close = torch.where(torch.abs(soft_low) <= torch.abs(soft_high), soft_low, soft_high)
            candidates = torch.stack((soft_low, soft_high, hard_low, hard_high), dim=-1)
            open_idx = torch.argmax(torch.abs(candidates - gripper_close.unsqueeze(-1)), dim=-1, keepdim=True)
            gripper_open = torch.gather(candidates, dim=-1, index=open_idx).squeeze(-1) * self.spec.gripper.open_scale
            close_soft = gripper_open + (gripper_close - gripper_open) * self.spec.gripper.close_ratio
            cmd = gripper_open + (close_soft - gripper_open) * self.gripper_alpha
            self.robot.set_joint_position_target(cmd, joint_ids=self.gripper_joint_ids)
        else:
            raise ValueError(f"Unsupported gripper mode: {self.spec.gripper.mode}")

        if self.hold_open_joint_ids is not None and self.hold_open_joint_ids.numel() > 0:
            if self.spec.gripper.mode == "omnipicker":
                soft_limits = self.robot.data.soft_joint_pos_limits[:, self.hold_open_joint_ids, :]
                hard_limits = self.robot.data.joint_pos_limits[:, self.hold_open_joint_ids, :]
                soft_low = soft_limits[..., 0]
                soft_high = soft_limits[..., 1]
                hard_low = hard_limits[..., 0]
                hard_high = hard_limits[..., 1]
                close_target = torch.where(torch.abs(soft_low) <= torch.abs(soft_high), soft_low, soft_high)
                candidates = torch.stack((soft_low, soft_high, hard_low, hard_high), dim=-1)
                open_idx = torch.argmax(torch.abs(candidates - close_target.unsqueeze(-1)), dim=-1, keepdim=True)
                open_target = torch.gather(candidates, dim=-1, index=open_idx).squeeze(-1) * self.spec.gripper.open_scale
            elif self.spec.gripper.mode == "signed_limit":
                limits = self.robot.data.soft_joint_pos_limits[:, self.hold_open_joint_ids, :]
                low = limits[..., 0]
                high = limits[..., 1]
                open_target = torch.where(torch.abs(low) > torch.abs(high), low, high)
            else:
                limits = self.robot.data.soft_joint_pos_limits[:, self.hold_open_joint_ids, :]
                open_target = limits[..., 0]
            self.robot.set_joint_position_target(open_target, joint_ids=self.hold_open_joint_ids)

    def _stack_goal_quat(self, ee_quat_b: torch.Tensor) -> torch.Tensor:
        if self.spec.stack.stack_quat_goal is not None:
            return torch.tensor(self.spec.stack.stack_quat_goal, device=self.sim.device).repeat(self.scene.num_envs, 1)
        if self.spec.fixed_ee_quat_goal is not None:
            return torch.tensor(self.spec.fixed_ee_quat_goal, device=self.sim.device).repeat(self.scene.num_envs, 1)
        return ee_quat_b

    def step_stack(self):
        sim_dt = self.sim.get_physics_dt()
        jacobian, ee_pose_w, arm_joint_pos, ee_pos_b, ee_quat_b = self._current_ee()
        root_pose_w = self.robot.data.root_pose_w

        if not self.phase_start_initialized.all():
            init_mask = ~self.phase_start_initialized
            if self.spec.stack.home_pose_override is None:
                self.home_pos_b_init[init_mask] = ee_pos_b[init_mask]
            else:
                self.home_pos_b_init[init_mask] = torch.tensor(
                    self.spec.stack.home_pose_override, device=self.sim.device
                ).repeat(int(init_mask.sum().item()), 1)

        base_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            self.cube_base.data.root_pos_w,
            self.cube_base.data.root_quat_w,
        )
        pick_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            self.cube_pick.data.root_pos_w,
            self.cube_pick.data.root_quat_w,
        )
        fallen_mask = pick_pos_b[:, 2] < self.spec.stack.fallen_reset_z
        if fallen_mask.any():
            env_ids = torch.nonzero(fallen_mask, as_tuple=False).squeeze(-1)
            pick_state = self.cube_pick.data.default_root_state[env_ids].clone()
            pick_state[:, :3] += self.scene.env_origins[env_ids]
            self.cube_pick.write_root_pose_to_sim(pick_state[:, :7], env_ids=env_ids)
            self.cube_pick.write_root_velocity_to_sim(pick_state[:, 7:], env_ids=env_ids)
            self.phase[env_ids] = Phase.ABOVE_PICK
            self.phase_time[env_ids] = 0.0
            self.waypoint_idx[env_ids] = 0
            self.phase_start_initialized[env_ids] = False
            pick_pos_b_init, _ = subtract_frame_transforms(
                root_pose_w[env_ids, 0:3], root_pose_w[env_ids, 3:7], pick_state[:, :3], pick_state[:, 3:7]
            )
            self.pick_pos_b_init[env_ids] = pick_pos_b_init

        init_mask = ~self.phase_start_initialized
        if init_mask.any():
            if self.spec.stack.home_pose_override is None:
                self.phase_start_pos[init_mask] = ee_pos_b[init_mask]
            else:
                self.phase_start_pos[init_mask] = self.home_pos_b_init[init_mask]
            self.phase_start_initialized[init_mask] = True
            self.waypoint_idx[init_mask] = 0

        z_offset = torch.tensor([0.0, 0.0, self.spec.stack.ee_pick_z_offset], device=self.sim.device)
        pick_anchor = self.pick_pos_b_init + z_offset
        stack_anchor = base_pos_b + z_offset

        above_pick = pick_anchor + torch.tensor(self.spec.stack.above_pick_delta, device=self.sim.device)
        at_pick = pick_anchor
        lifted = pick_anchor + torch.tensor(self.spec.stack.lift_delta, device=self.sim.device)
        above_stack = stack_anchor + torch.tensor(self.spec.stack.above_stack_delta, device=self.sim.device)
        at_stack = stack_anchor + torch.tensor(self.spec.stack.place_delta, device=self.sim.device)
        retreat = stack_anchor + torch.tensor(self.spec.stack.retreat_delta, device=self.sim.device)

        phase_goal_pos = ee_pos_b.clone()
        phase_goal_pos[self.phase == Phase.ABOVE_PICK] = above_pick[self.phase == Phase.ABOVE_PICK]
        phase_goal_pos[self.phase == Phase.DESCEND_PICK] = at_pick[self.phase == Phase.DESCEND_PICK]
        phase_goal_pos[self.phase == Phase.CLOSE] = at_pick[self.phase == Phase.CLOSE]
        phase_goal_pos[self.phase == Phase.LIFT] = lifted[self.phase == Phase.LIFT]
        phase_goal_pos[self.phase == Phase.ABOVE_STACK] = above_stack[self.phase == Phase.ABOVE_STACK]
        phase_goal_pos[self.phase == Phase.PLACE] = at_stack[self.phase == Phase.PLACE]
        phase_goal_pos[self.phase == Phase.OPEN] = at_stack[self.phase == Phase.OPEN]
        phase_goal_pos[self.phase == Phase.RETREAT] = retreat[self.phase == Phase.RETREAT]
        phase_goal_pos[self.phase == Phase.HOME] = self.home_pos_b_init[self.phase == Phase.HOME]

        move_mask = (self.phase != Phase.CLOSE) & (self.phase != Phase.OPEN)
        phase_vec = phase_goal_pos - self.phase_start_pos
        phase_dist = torch.norm(phase_vec, dim=1)
        waypoint_count = torch.clamp(torch.ceil(phase_dist / max(1.0e-4, self.spec.stack.waypoint_step)).to(torch.long), min=1)

        if self.spec.stack.waypoint_hold_time is not None:
            waypoint_hold_time = max(1.0e-3, self.spec.stack.waypoint_hold_time)
            self.waypoint_idx = torch.floor(self.phase_time / waypoint_hold_time).to(dtype=torch.long)
            self.waypoint_idx = torch.minimum(self.waypoint_idx, waypoint_count - 1)
            waypoint_alpha = (self.waypoint_idx.to(dtype=ee_pos_b.dtype) + 1.0) / waypoint_count.to(dtype=ee_pos_b.dtype)
            waypoint_alpha = torch.clamp(waypoint_alpha, max=1.0)
            waypoint_target = self.phase_start_pos + phase_vec * waypoint_alpha.unsqueeze(-1)
            reached = move_mask & (self.phase_time >= waypoint_hold_time * waypoint_count.to(dtype=self.phase_time.dtype))
        else:
            waypoint_alpha = (self.waypoint_idx.to(dtype=ee_pos_b.dtype) + 1.0) / waypoint_count.to(dtype=ee_pos_b.dtype)
            waypoint_alpha = torch.clamp(waypoint_alpha, max=1.0)
            waypoint_target = self.phase_start_pos + phase_vec * waypoint_alpha.unsqueeze(-1)
            waypoint_reached = torch.norm(waypoint_target - ee_pos_b, dim=1) < max(0.001, self.spec.stack.waypoint_thresh or 0.006)
            final_waypoint = self.waypoint_idx >= (waypoint_count - 1)
            reached = move_mask & waypoint_reached & final_waypoint
            advance_waypoint = move_mask & waypoint_reached & (~final_waypoint)
            self.waypoint_idx[advance_waypoint] += 1

        self.target_pos = phase_goal_pos.clone()
        self.target_pos[move_mask] = waypoint_target[move_mask]
        self.target_quat = self._stack_goal_quat(ee_quat_b)

        if self.use_whole_body:
            arm_joint_pos_des = arm_joint_pos
        else:
            self.ik_controller.set_command(torch.cat([self.target_pos, self.target_quat], dim=1))
            arm_joint_pos_des_raw = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, arm_joint_pos)
            motion_scale = max(0.01, min(self.spec.stack.motion_scale, 1.0))
            arm_joint_pos_des = arm_joint_pos + motion_scale * (arm_joint_pos_des_raw - arm_joint_pos)

        self._apply_arm_command(arm_joint_pos_des)
        self._update_hold_joints()

        close_mask = (self.phase == Phase.CLOSE) | (self.phase == Phase.LIFT) | (self.phase == Phase.ABOVE_STACK) | (self.phase == Phase.PLACE)
        self._apply_gripper(sim_dt, close_mask)

        to_descend = (self.phase == Phase.ABOVE_PICK) & reached
        to_close = (self.phase == Phase.DESCEND_PICK) & reached
        to_lift = (self.phase == Phase.CLOSE) & (self.phase_time > 0.30) & (
            self.gripper_alpha.squeeze(-1) >= self.spec.gripper.min_alpha_for_lift
        )
        to_above_stack = (self.phase == Phase.LIFT) & reached
        to_place = (self.phase == Phase.ABOVE_STACK) & reached
        to_open = (self.phase == Phase.PLACE) & reached
        to_retreat = (self.phase == Phase.OPEN) & (self.phase_time > 0.25)
        to_home = (self.phase == Phase.RETREAT) & reached

        self.phase[to_descend] = Phase.DESCEND_PICK
        self.phase[to_close] = Phase.CLOSE
        self.phase[to_lift] = Phase.LIFT
        self.phase[to_above_stack] = Phase.ABOVE_STACK
        self.phase[to_place] = Phase.PLACE
        self.phase[to_open] = Phase.OPEN
        self.phase[to_retreat] = Phase.RETREAT
        self.phase[to_home] = Phase.HOME

        switched = to_descend | to_close | to_lift | to_above_stack | to_place | to_open | to_retreat | to_home
        self.waypoint_idx[switched] = 0
        self.phase_start_pos[switched] = ee_pos_b[switched]
        self.phase_start_initialized[switched] = True
        self.phase_time[switched] = 0.0
        self.phase_time += sim_dt

        target_pos_w, target_quat_w = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], self.target_pos, self.target_quat
        )
        self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        self.goal_marker.visualize(target_pos_w, target_quat_w)

    def step_click_delta(self, delta_cmd: torch.Tensor, gripper_closed: bool):
        sim_dt = self.sim.get_physics_dt()
        jacobian, ee_pose_w, arm_joint_pos, ee_pos_b, ee_quat_b = self._current_ee()
        root_pose_w = self.robot.data.root_pose_w
        if self.ui_target_pos is None:
            self.ui_target_pos = ee_pos_b.clone()
            self.ui_target_quat = ee_quat_b.clone()
        delta = delta_cmd.unsqueeze(0).repeat(self.scene.num_envs, 1)
        self.ui_target_pos, self.ui_target_quat = apply_delta_pose(self.ui_target_pos, self.ui_target_quat, delta)
        self.target_pos = self.ui_target_pos
        self.target_quat = self.ui_target_quat

        if self.use_whole_body:
            arm_joint_pos_des = arm_joint_pos
        else:
            self.ik_controller.set_command(torch.cat([self.target_pos, self.target_quat], dim=1))
            arm_joint_pos_des_raw = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, arm_joint_pos)
            motion_scale = max(0.01, min(self.spec.stack.motion_scale, 1.0))
            arm_joint_pos_des = arm_joint_pos + motion_scale * (arm_joint_pos_des_raw - arm_joint_pos)
        self._apply_arm_command(arm_joint_pos_des)
        self._update_hold_joints()
        self._apply_gripper(sim_dt, gripper_closed)
        target_pos_w, target_quat_w = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], self.target_pos, self.target_quat
        )
        self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        self.goal_marker.visualize(target_pos_w, target_quat_w)

    def current_ee_pose_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        _jacobian, _ee_pose_w, _arm_joint_pos, ee_pos_b, ee_quat_b = self._current_ee()
        return ee_pos_b.clone(), ee_quat_b.clone()

    def current_ee_pose_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        _jacobian, ee_pose_w, _arm_joint_pos, _ee_pos_b, _ee_quat_b = self._current_ee()
        return ee_pose_w[:, 0:3].clone(), ee_pose_w[:, 3:7].clone()

    def step_pose_target(
        self,
        target_pos_b: torch.Tensor,
        target_quat_b: torch.Tensor,
        gripper_closed: bool,
    ) -> None:
        sim_dt = self.sim.get_physics_dt()
        jacobian, ee_pose_w, arm_joint_pos, ee_pos_b, ee_quat_b = self._current_ee()
        root_pose_w = self.robot.data.root_pose_w

        if target_pos_b.ndim == 1:
            target_pos_b = target_pos_b.unsqueeze(0)
        if target_quat_b.ndim == 1:
            target_quat_b = target_quat_b.unsqueeze(0)

        self.target_pos = target_pos_b.clone()
        self.target_quat = target_quat_b.clone()
        self.ui_target_pos = self.target_pos.clone()
        self.ui_target_quat = self.target_quat.clone()

        if self.use_whole_body:
            arm_joint_pos_des = arm_joint_pos
        else:
            self.ik_controller.set_command(torch.cat([self.target_pos, self.target_quat], dim=1))
            arm_joint_pos_des_raw = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, arm_joint_pos)
            motion_scale = max(0.01, min(self.spec.stack.motion_scale, 1.0))
            arm_joint_pos_des = arm_joint_pos + motion_scale * (arm_joint_pos_des_raw - arm_joint_pos)

        self._apply_arm_command(arm_joint_pos_des)
        self._update_hold_joints()
        self._apply_gripper(sim_dt, gripper_closed)
        target_pos_w, target_quat_w = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], self.target_pos, self.target_quat
        )
        self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        self.goal_marker.visualize(target_pos_w, target_quat_w)

    def step_click_teleop(self, teleop_ui: MouseClickTeleopUI):
        delta_cmd = teleop_ui.consume_delta(self.sim.device)
        self.step_click_delta(delta_cmd, teleop_ui.gripper_closed)

    def step_keyboard_teleop(self, delta_cmd: torch.Tensor, gripper_closed: bool):
        sim_dt = self.sim.get_physics_dt()
        jacobian, ee_pose_w, arm_joint_pos, ee_pos_b, ee_quat_b = self._current_ee()
        root_pose_w = self.robot.data.root_pose_w
        delta_cmd = delta_cmd.unsqueeze(0).repeat(self.scene.num_envs, 1)
        self.teleop_ik_controller.set_command(delta_cmd, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        if self.use_whole_body:
            arm_joint_pos_des = arm_joint_pos
        else:
            arm_joint_pos_des = self.teleop_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, arm_joint_pos)
        self.target_pos = self.teleop_ik_controller.ee_pos_des
        self.target_quat = self.teleop_ik_controller.ee_quat_des
        self._apply_arm_command(arm_joint_pos_des)
        self._update_hold_joints()
        self._apply_gripper(sim_dt, gripper_closed)
        target_pos_w, target_quat_w = combine_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            self.teleop_ik_controller.ee_pos_des,
            self.teleop_ik_controller.ee_quat_des,
        )
        self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        self.goal_marker.visualize(target_pos_w, target_quat_w)
