from __future__ import annotations

from dataclasses import dataclass, replace

import carb
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import apply_delta_pose, combine_frame_transforms, subtract_frame_transforms

from controller import IsaacLabBimanualPoseController, IsaacLabPoseController

from scene_robot_assets import (
    AGIBOT_G1_OMNIPICKER_STACK_CUBE_DIFF_IK_CFG,
    GEN3_7DOF_VISION_ROBOTIQ_2F85_STACK_CUBE_DIFF_IK_CFG,
    R1LITE_STACK_CUBE_DIFF_IK_CFG,
)


class Phase:
    ABOVE_PICK = 0
    DESCEND_PICK = 1
    CLOSE = 2
    LIFT = 3
    ABOVE_STACK = 4
    PLACE = 5
    OPEN = 6
    RETREAT = 7
    HOME = 8


@dataclass(frozen=True)
class CuboidSpec:
    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    color: tuple[float, float, float]
    kinematic: bool = False
    mass: float | None = None
    friction: tuple[float, float] | None = None


@dataclass(frozen=True)
class TableSpec:
    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    color: tuple[float, float, float] = (0.45, 0.45, 0.45)


@dataclass(frozen=True)
class GripperSpec:
    mode: str
    joint_patterns: tuple[str, ...]
    hold_open_patterns: tuple[str, ...] = ()
    close_ratio: float = 0.85
    close_time: float = 0.45
    open_time: float = 0.35
    min_alpha_for_lift: float = 0.9
    open_scale: float = 1.0


@dataclass(frozen=True)
class StackMotionSpec:
    ee_pick_z_offset: float
    above_pick_delta: tuple[float, float, float]
    lift_delta: tuple[float, float, float]
    above_stack_delta: tuple[float, float, float]
    place_delta: tuple[float, float, float]
    retreat_delta: tuple[float, float, float]
    waypoint_step: float
    motion_scale: float
    fallen_reset_z: float
    waypoint_hold_time: float | None = None
    waypoint_thresh: float | None = None
    stack_quat_goal: tuple[float, float, float, float] | None = None
    home_pose_override: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class RobotStackSpec:
    name: str
    window_title: str
    robot_cfg: object
    ee_joint_patterns: tuple[str, ...]
    ee_body_name: str
    arm_side: str = "left"
    arm_switch_supported: bool = False
    hold_default_patterns: tuple[str, ...] = ()
    table: TableSpec | None = None
    cube_base: CuboidSpec | None = None
    cube_pick: CuboidSpec | None = None
    stack: StackMotionSpec | None = None
    gripper: GripperSpec | None = None
    fixed_ee_quat_goal: tuple[float, float, float, float] | None = None
    ground_z: float = 0.0
    root_z_zero: bool = True
    light_intensity: float = 3000.0
    env_spacing: float = 3.0
    camera_eye: tuple[float, float, float] = (2.5, 2.5, 2.5)
    camera_target: tuple[float, float, float] = (0.5, 0.0, 0.5)


KINOVA_STACK_SPEC = RobotStackSpec(
    name="kinova",
    window_title="Kinova Teleop",
    robot_cfg=GEN3_7DOF_VISION_ROBOTIQ_2F85_STACK_CUBE_DIFF_IK_CFG,
    arm_side="left",
    arm_switch_supported=False,
    ee_joint_patterns=("joint_[1-7]",),
    ee_body_name="bracelet_link",
    cube_base=CuboidSpec(
        size=(0.05, 0.05, 0.05),
        pos=(0.55, 0.0, 0.025),
        color=(0.8, 0.2, 0.2),
        kinematic=True,
        friction=(1.8, 1.5),
    ),
    cube_pick=CuboidSpec(
        size=(0.05, 0.05, 0.05),
        pos=(0.45, 0.18, 0.025),
        color=(0.2, 0.2, 0.9),
        mass=0.08,
        friction=(1.8, 1.5),
    ),
    stack=StackMotionSpec(
        ee_pick_z_offset=0.2,
        above_pick_delta=(0.0, 0.0, 0.05),
        lift_delta=(0.0, 0.0, 0.03),
        above_stack_delta=(0.0, 0.0, 0.08),
        place_delta=(0.0, 0.0, 0.07),
        retreat_delta=(0.0, 0.0, 0.2),
        waypoint_step=0.005,
        waypoint_thresh=0.006,
        motion_scale=0.12,
        fallen_reset_z=0.01,
    ),
    gripper=GripperSpec(
        mode="simple",
        joint_patterns=("finger_joint",),
        close_ratio=1.0,
        close_time=0.45,
        open_time=0.35,
        min_alpha_for_lift=0.9,
    ),
    fixed_ee_quat_goal=(1.0, 0.0, 0.0, 0.0),
    ground_z=0.0,
    env_spacing=2.5,
    camera_eye=(2.6, 2.2, 2.2),
    camera_target=(0.5, 0.0, 0.2),
)


AGIBOT_STACK_SPEC = RobotStackSpec(
    name="agibot",
    window_title="Agibot LeftArm Teleop",
    robot_cfg=AGIBOT_G1_OMNIPICKER_STACK_CUBE_DIFF_IK_CFG,
    arm_side="left",
    arm_switch_supported=True,
    ee_joint_patterns=("idx2[1-7]_arm_l_joint[1-7]",),
    ee_body_name="arm_l_end_link",
    hold_default_patterns=("idx01_body_joint1", "idx02_body_joint2", "idx11_head_joint1", "idx12_head_joint2", "idx6[1-7]_arm_r_joint[1-7]"),
    table=TableSpec(size=(0.90, 0.60, 0.05), pos=(0.68, 0.0, 0.60)),
    cube_base=CuboidSpec(size=(0.05, 0.05, 0.05), pos=(0.48, 0.08, 0.70), color=(0.8, 0.2, 0.2), kinematic=True),
    cube_pick=CuboidSpec(size=(0.05, 0.05, 0.05), pos=(0.46, 0.20, 0.70), color=(0.2, 0.2, 0.9), mass=0.08),
    stack=StackMotionSpec(
        ee_pick_z_offset=0.16,
        above_pick_delta=(0.0, 0.0, 0.06),
        lift_delta=(0.0, 0.0, 0.08),
        above_stack_delta=(0.0, 0.0, 0.08),
        place_delta=(0.0, 0.0, 0.07),
        retreat_delta=(0.0, 0.0, 0.07),
        waypoint_step=0.005,
        waypoint_hold_time=0.05,
        motion_scale=0.08,
        fallen_reset_z=0.40,
        home_pose_override=(0.415999, 0.51890045, 0.9534997),
    ),
    gripper=GripperSpec(
        mode="omnipicker",
        joint_patterns=("idx41_gripper_l_outer_joint1",),
        hold_open_patterns=("idx81_gripper_r_outer_joint1",),
        close_ratio=1.0,
        close_time=0.2,
        open_time=0.35,
        min_alpha_for_lift=0.9,
        open_scale=1.6,
    ),
    fixed_ee_quat_goal=(0.0, 1.0, 0.0, 0.0),
    ground_z=-1.05,
    camera_eye=(3.2, 3.0, 2.4),
    camera_target=(0.6, 0.0, 0.9),
)


R1LITE_STACK_SPEC = RobotStackSpec(
    name="r1lite",
    window_title="R1Lite LeftArm Teleop",
    robot_cfg=R1LITE_STACK_CUBE_DIFF_IK_CFG,
    arm_side="left",
    arm_switch_supported=True,
    ee_joint_patterns=("left_arm_joint[1-6]",),
    ee_body_name="left_arm_link6",
    hold_default_patterns=("torso_joint[1-3]", "right_arm_joint[1-6]"),
    table=TableSpec(size=(0.90, 0.60, 0.05), pos=(0.62, 0.0, 0.85)),
    cube_base=CuboidSpec(size=(0.05, 0.05, 0.05), pos=(0.46, 0.08, 0.90), color=(0.8, 0.2, 0.2), kinematic=True),
    cube_pick=CuboidSpec(size=(0.05, 0.05, 0.05), pos=(0.42, 0.20, 0.90), color=(0.2, 0.2, 0.9), mass=0.08),
    stack=StackMotionSpec(
        ee_pick_z_offset=0.14,
        above_pick_delta=(0.0, 0.0, 0.06),
        lift_delta=(0.0, 0.0, 0.08),
        above_stack_delta=(0.0, 0.0, 0.07),
        place_delta=(0.0, 0.0, 0.07),
        retreat_delta=(0.0, 0.0, 0.07),
        waypoint_step=0.005,
        waypoint_hold_time=0.05,
        motion_scale=0.08,
        fallen_reset_z=0.40,
        stack_quat_goal=(1.0, 0.0, 0.0, 0.0),
    ),
    gripper=GripperSpec(
        mode="signed_limit",
        joint_patterns=("left_gripper_finger_joint[1-2]",),
        hold_open_patterns=("right_gripper_finger_joint[1-2]",),
        close_ratio=1.0,
        close_time=0.45,
        open_time=0.35,
        min_alpha_for_lift=0.9,
    ),
    ground_z=-1.05,
    camera_eye=(3.0, 2.5, 2.2),
    camera_target=(0.55, 0.0, 0.8),
)


STACK_SPECS = {
    "kinova": KINOVA_STACK_SPEC,
    "agibot": AGIBOT_STACK_SPEC,
    "r1lite": R1LITE_STACK_SPEC,
}

SWITCHABLE_ARM_SIDE_ROBOTS = {"agibot", "r1lite"}


def normalize_arm_side(robot_name: str, arm_side: str | None = None) -> str:
    side = str(arm_side or "left").strip().lower()
    if robot_name not in SWITCHABLE_ARM_SIDE_ROBOTS:
        if side != "left":
            raise ValueError(f"Robot '{robot_name}' does not support arm_side='{side}'.")
        return "left"
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported {robot_name} arm side: {side}")
    return side


def _arm_title_label(arm_side: str) -> str:
    return "LeftArm" if normalize_arm_side("agibot", arm_side) == "left" else "RightArm"


def _mirror_y_pose(pos: tuple[float, float, float], arm_side: str) -> tuple[float, float, float]:
    if arm_side == "left":
        return pos
    return (pos[0], -pos[1], pos[2])


def _arm_config_fields(robot_name: str, arm_side: str) -> dict[str, object]:
    side = normalize_arm_side(robot_name, arm_side)
    if robot_name == "agibot":
        return {
            "ee_joint_patterns": ("idx2[1-7]_arm_l_joint[1-7]",) if side == "left" else ("idx6[1-7]_arm_r_joint[1-7]",),
            "ee_body_name": "arm_l_end_link" if side == "left" else "arm_r_end_link",
            "gripper_joint_patterns": ("idx41_gripper_l_outer_joint1",) if side == "left" else ("idx81_gripper_r_outer_joint1",),
            "hold_open_patterns": ("idx81_gripper_r_outer_joint1",) if side == "left" else ("idx41_gripper_l_outer_joint1",),
            "hold_default_patterns": (
                "idx01_body_joint1",
                "idx02_body_joint2",
                "idx11_head_joint1",
                "idx12_head_joint2",
                "idx6[1-7]_arm_r_joint[1-7]" if side == "left" else "idx2[1-7]_arm_l_joint[1-7]",
            ),
        }
    if robot_name == "r1lite":
        return {
            "ee_joint_patterns": ("left_arm_joint[1-6]",) if side == "left" else ("right_arm_joint[1-6]",),
            "ee_body_name": "left_arm_link6" if side == "left" else "right_arm_link6",
            "gripper_joint_patterns": (
                "left_gripper_finger_joint[1-2]",
            )
            if side == "left"
            else ("right_gripper_finger_joint[1-2]",),
            "hold_open_patterns": (
                "right_gripper_finger_joint[1-2]",
            )
            if side == "left"
            else ("left_gripper_finger_joint[1-2]",),
            "hold_default_patterns": (
                "torso_joint[1-3]",
                "right_arm_joint[1-6]" if side == "left" else "left_arm_joint[1-6]",
            ),
        }
    return {
        "ee_joint_patterns": STACK_SPECS[robot_name].ee_joint_patterns,
        "ee_body_name": STACK_SPECS[robot_name].ee_body_name,
        "gripper_joint_patterns": STACK_SPECS[robot_name].gripper.joint_patterns,
        "hold_open_patterns": STACK_SPECS[robot_name].gripper.hold_open_patterns,
        "hold_default_patterns": STACK_SPECS[robot_name].hold_default_patterns,
    }


def resolve_stack_spec(robot_name: str, arm_side: str | None = None) -> RobotStackSpec:
    if robot_name not in STACK_SPECS:
        raise ValueError(f"Unsupported robot '{robot_name}'. Choose from: {sorted(STACK_SPECS)}")
    base = STACK_SPECS[robot_name]
    side = normalize_arm_side(robot_name, arm_side)
    if robot_name not in SWITCHABLE_ARM_SIDE_ROBOTS:
        return replace(base, arm_side="left", arm_switch_supported=False)

    fields = _arm_config_fields(robot_name, side)
    stack = base.stack
    if stack is not None and stack.home_pose_override is not None:
        stack = replace(stack, home_pose_override=_mirror_y_pose(stack.home_pose_override, side))
    return replace(
        base,
        arm_side=side,
        arm_switch_supported=True,
        window_title=f"{'Agibot' if robot_name == 'agibot' else 'R1Lite'} {_arm_title_label(side)} Teleop",
        ee_joint_patterns=fields["ee_joint_patterns"],
        ee_body_name=fields["ee_body_name"],
        hold_default_patterns=fields["hold_default_patterns"],
        cube_base=None if base.cube_base is None else replace(base.cube_base, pos=_mirror_y_pose(base.cube_base.pos, side)),
        cube_pick=None if base.cube_pick is None else replace(base.cube_pick, pos=_mirror_y_pose(base.cube_pick.pos, side)),
        stack=stack,
        gripper=replace(
            base.gripper,
            joint_patterns=fields["gripper_joint_patterns"],
            hold_open_patterns=fields["hold_open_patterns"],
        ),
    )


def _add_offset(pos: tuple[float, float, float], offset: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(pos, offset, strict=True))


def shifted_spec(spec: RobotStackSpec, offset: tuple[float, float, float]) -> RobotStackSpec:
    robot_cfg = spec.robot_cfg.replace(
        init_state=spec.robot_cfg.init_state.replace(pos=_add_offset(spec.robot_cfg.init_state.pos, offset))
    )
    return replace(
        spec,
        robot_cfg=robot_cfg,
        table=None if spec.table is None else replace(spec.table, pos=_add_offset(spec.table.pos, offset)),
        cube_base=None if spec.cube_base is None else replace(spec.cube_base, pos=_add_offset(spec.cube_base.pos, offset)),
        cube_pick=None if spec.cube_pick is None else replace(spec.cube_pick, pos=_add_offset(spec.cube_pick.pos, offset)),
        camera_eye=_add_offset(spec.camera_eye, offset),
        camera_target=_add_offset(spec.camera_target, offset),
    )


def _make_material(spec: CuboidSpec):
    physics_material = None
    if spec.friction is not None:
        physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=spec.friction[0],
            dynamic_friction=spec.friction[1],
            restitution=0.0,
        )
    return physics_material


def _make_cube_cfg(prim_path: str, spec: CuboidSpec) -> RigidObjectCfg:
    if spec.kinematic:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    else:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
            max_depenetration_velocity=0.8,
        )
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=spec.size,
            rigid_props=rigid_props,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_make_material(spec),
            mass_props=None if spec.mass is None else sim_utils.MassPropertiesCfg(mass=spec.mass),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.color, metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=spec.pos),
    )


def build_single_robot_scene_cfg(spec: RobotStackSpec):
    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, spec.ground_z)),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=spec.light_intensity, color=(0.75, 0.75, 0.75)),
        )
        robot = spec.robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if spec.table is not None:
            table = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=spec.table.size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.table.color, metallic=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=spec.table.pos),
            )

        if spec.cube_base is not None:
            cube_base = _make_cube_cfg("{ENV_REGEX_NS}/CubeBase", spec.cube_base)
        if spec.cube_pick is not None:
            cube_pick = _make_cube_cfg("{ENV_REGEX_NS}/CubePick", spec.cube_pick)

    return _SceneCfg


def build_merged_scene_cfg(spec_items: list[tuple[str, RobotStackSpec]]):
    attrs: dict[str, object] = {
        "ground": AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        ),
        "dome_light": AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        ),
    }
    for prefix, spec in spec_items:
        attrs[f"{prefix}_robot"] = spec.robot_cfg.replace(prim_path=f"/World/{prefix}/Robot")
        if spec.table is not None:
            attrs[f"{prefix}_table"] = RigidObjectCfg(
                prim_path=f"/World/{prefix}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=spec.table.size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.table.color, metallic=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=spec.table.pos),
            )
        attrs[f"{prefix}_cube_base"] = _make_cube_cfg(f"/World/{prefix}/CubeBase", spec.cube_base)
        attrs[f"{prefix}_cube_pick"] = _make_cube_cfg(f"/World/{prefix}/CubePick", spec.cube_pick)
    return configclass(type("MergedStackSceneCfg", (InteractiveSceneCfg,), attrs))


class MouseClickTeleopUI:
    def __init__(self, title: str, lin_step: float = 0.015, ang_step: float = 0.10):
        import omni.ui as ui

        self.lin_step = float(lin_step)
        self.ang_step = float(ang_step)
        self.gripper_closed = False
        self._pending_delta = [0.0] * 6
        self._window = ui.Window(title, width=380, height=320)
        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Click to move one step (xyzrpy)", height=20)
                with ui.HStack(spacing=6):
                    ui.Button("+X", clicked_fn=lambda: self._add_delta(0, +self.lin_step))
                    ui.Button("-X", clicked_fn=lambda: self._add_delta(0, -self.lin_step))
                    ui.Button("+Y", clicked_fn=lambda: self._add_delta(1, +self.lin_step))
                    ui.Button("-Y", clicked_fn=lambda: self._add_delta(1, -self.lin_step))
                    ui.Button("+Z", clicked_fn=lambda: self._add_delta(2, +self.lin_step))
                    ui.Button("-Z", clicked_fn=lambda: self._add_delta(2, -self.lin_step))
                with ui.HStack(spacing=6):
                    ui.Button("+R", clicked_fn=lambda: self._add_delta(3, +self.ang_step))
                    ui.Button("-R", clicked_fn=lambda: self._add_delta(3, -self.ang_step))
                    ui.Button("+P", clicked_fn=lambda: self._add_delta(4, +self.ang_step))
                    ui.Button("-P", clicked_fn=lambda: self._add_delta(4, -self.ang_step))
                    ui.Button("+Y", clicked_fn=lambda: self._add_delta(5, +self.ang_step))
                    ui.Button("-Y", clicked_fn=lambda: self._add_delta(5, -self.ang_step))
                with ui.HStack(spacing=6):
                    ui.Button("Gripper Toggle", clicked_fn=self._toggle_gripper)
                    ui.Button("Clear Step Queue", clicked_fn=self._clear_delta)

    def _add_delta(self, idx: int, value: float):
        self._pending_delta[idx] += value

    def _toggle_gripper(self):
        self.gripper_closed = not self.gripper_closed

    def _clear_delta(self):
        self._pending_delta = [0.0] * 6

    def consume_delta(self, device: str) -> torch.Tensor:
        delta = torch.tensor(self._pending_delta, device=device, dtype=torch.float32)
        self._pending_delta = [0.0] * 6
        return delta


class KeyboardTeleop:
    def __init__(self):
        import omni.appwindow

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self.gripper_closed = False
        self._active = {key: False for key in ("W", "S", "A", "D", "R", "F", "I", "K", "J", "L", "U", "O")}

    def _on_keyboard_event(self, event):
        key = event.input if isinstance(event.input, str) else event.input.name
        if key in self._active:
            if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
                self._active[key] = True
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self._active[key] = False
        elif key == "G" and event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.gripper_closed = not self.gripper_closed
        return True

    def compute_delta_cmd(self, dt: float, lin_speed: float, ang_speed: float, device: str) -> torch.Tensor:
        cmd = torch.zeros(6, device=device)
        if self._active["W"]:
            cmd[0] += lin_speed * dt
        if self._active["S"]:
            cmd[0] -= lin_speed * dt
        if self._active["A"]:
            cmd[1] += lin_speed * dt
        if self._active["D"]:
            cmd[1] -= lin_speed * dt
        if self._active["R"]:
            cmd[2] += lin_speed * dt
        if self._active["F"]:
            cmd[2] -= lin_speed * dt
        if self._active["I"]:
            cmd[3] += ang_speed * dt
        if self._active["K"]:
            cmd[3] -= ang_speed * dt
        if self._active["J"]:
            cmd[4] += ang_speed * dt
        if self._active["L"]:
            cmd[4] -= ang_speed * dt
        if self._active["U"]:
            cmd[5] += ang_speed * dt
        if self._active["O"]:
            cmd[5] -= ang_speed * dt
        return cmd


# RobotController lives in robot_controller.py. Importing it at the top of
# this module would create a circular import (robot_controller imports
# RobotStackSpec / Phase / MouseClickTeleopUI / etc. from here), so we
# import lazily inside build_stack_scene below. With
# `from __future__ import annotations` enabled, the type-hint reference
# in build_stack_scene's signature stays as a string and doesn't need a
# real symbol at module load time.


def build_stack_scene(
    sim: sim_utils.SimulationContext,
    robot_name: str,
    num_envs: int = 1,
    arm_side: str = "left",
) -> "tuple[InteractiveScene, RobotController]":
    from .robot_controller import RobotController

    spec = resolve_stack_spec(robot_name, arm_side)
    scene_cfg = build_single_robot_scene_cfg(spec)(num_envs=num_envs, env_spacing=spec.env_spacing)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    controller = RobotController(sim, scene, spec)
    controller.reset()
    scene.reset()
    return scene, controller


def run_stack_cube_demo(
    simulation_app,
    robot_name: str,
    device: str,
    num_envs: int = 1,
    arm_side: str = "left",
) -> None:
    spec = resolve_stack_spec(robot_name, arm_side)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device))
    sim.set_camera_view(spec.camera_eye, spec.camera_target)

    scene, controller = build_stack_scene(sim, robot_name, num_envs=num_envs, arm_side=arm_side)
    print(f"[INFO] {robot_name} stack-cube demo ready.")

    while simulation_app.is_running():
        controller.step_stack()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
