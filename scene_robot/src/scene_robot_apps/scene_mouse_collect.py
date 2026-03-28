from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab.utils.math import combine_frame_transforms

from app.backend.config.settings import DEFAULT_PLACEMENTS_PATH, SCENE_GRAPH_PATH
from app.backend.services.robot_placement import (
    DEFAULT_OUTPUT_DIR,
    RobotPlacementPlan,
    load_scene_state,
    plan_robot_base_pose,
    plan_to_payload,
    save_plan_outputs,
)
from app.backend.services.robot_scene import compute_robot_floor_offset_z, resolve_robot_asset_path
from .mouse_teleop_record import (
    _AGIBOT_CAMERA_SPECS,
    _AGIBOT_MOUNT_POSES,
    _R1LITE_CAMERA_SPECS,
    _configure_world_camera,
    _make_pinhole_camera_cfg,
)
from .stack_cube import CuboidSpec, RobotStackController, STACK_SPECS, _make_cube_cfg
from .stack_cube import resolve_stack_spec


DEFAULT_SCENE_USD_PATH = Path(__file__).resolve().parents[3] / "runtime" / "scene_service" / "usd" / "scene_latest.usd"
_DYNAMIC_MESH_COLLISION_APPROX = "convexHull"
_COLLISION_APPROX_ALIASES = {
    "default": None,
    "triangle_mesh": "none",
    "convex_hull": "convexHull",
    "convex_decomposition": "convexDecomposition",
    "mesh_simplification": "meshSimplification",
    "bounding_cube": "boundingCube",
    "bounding_sphere": "boundingSphere",
    "sdf": "sdf",
    "sphere_fill": "sphereFill",
}


@dataclass(frozen=True)
class SceneMouseCollectArgs:
    device: str
    num_envs: int
    dataset_file: str
    capture_hz: float
    append: bool
    lin_step: float
    ang_step: float
    scene_usd_path: str
    scene_graph_path: str
    placements_path: str
    target: str | None
    support: str | None
    object_collision_approx: str
    target_collision_approx: str
    convex_decomp_voxel_resolution: int
    convex_decomp_max_convex_hulls: int
    convex_decomp_error_percentage: float
    convex_decomp_shrink_wrap: bool
    plan_output_dir: str
    base_z_bias: float
    arm_side: str
    collision_only: bool


@dataclass(frozen=True)
class ConvexDecompositionSettings:
    voxel_resolution: int
    max_convex_hulls: int
    error_percentage: float
    shrink_wrap: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "voxel_resolution": self.voxel_resolution,
            "max_convex_hulls": self.max_convex_hulls,
            "error_percentage": self.error_percentage,
            "shrink_wrap": self.shrink_wrap,
        }


class CollisionOverlayController:
    """Temporarily enable viewport + PhysX collision overlays for the current app session."""

    def __init__(self, enabled: bool):
        self.enabled = False
        self._settings = None
        self._physxui = None
        self._setting_display_colliders: str | None = None
        self._setting_display_collider_normals: str | None = None
        self._setting_visualization_collision_mesh: str | None = None
        self._saved_display_colliders: int | None = None
        self._saved_display_collider_normals: bool | None = None
        self._saved_visualization_collision_mesh: bool | None = None
        if enabled:
            self._enable()

    def _enable(self) -> None:
        try:
            import carb.settings
            from omni.physx.bindings._physx import (
                SETTING_DISPLAY_COLLIDERS,
                SETTING_DISPLAY_COLLIDER_NORMALS,
                SETTING_VISUALIZATION_COLLISION_MESH,
            )
            from omni.physxui import get_physxui_interface
        except Exception as exc:
            print(f"[WARN] Collision overlay unavailable: {exc}")
            return

        settings = carb.settings.get_settings()
        self._settings = settings
        self._physxui = get_physxui_interface()
        self._setting_display_colliders = SETTING_DISPLAY_COLLIDERS
        self._setting_display_collider_normals = SETTING_DISPLAY_COLLIDER_NORMALS
        self._setting_visualization_collision_mesh = SETTING_VISUALIZATION_COLLISION_MESH
        self._saved_display_colliders = settings.get_as_int(SETTING_DISPLAY_COLLIDERS)
        self._saved_display_collider_normals = settings.get_as_bool(SETTING_DISPLAY_COLLIDER_NORMALS)
        self._saved_visualization_collision_mesh = settings.get_as_bool(SETTING_VISUALIZATION_COLLISION_MESH)

        # Viewport colliders draw wireframes for every collider, while the PhysX UI
        # overlay adds the solid collision-mesh approximation on top of the graphics.
        settings.set(SETTING_DISPLAY_COLLIDERS, 2)
        settings.set(SETTING_DISPLAY_COLLIDER_NORMALS, False)
        settings.set(SETTING_VISUALIZATION_COLLISION_MESH, True)

        self._apply_physxui_state()
        self.enabled = True

    def _apply_physxui_state(self) -> None:
        if self._physxui is None:
            return
        self._physxui.enable_collision_mesh_visualization(True)
        self._physxui.set_collision_mesh_type("both")
        self._physxui.explode_view_distance(0.0)

    def refresh(self) -> None:
        if not self.enabled:
            return
        try:
            self._apply_physxui_state()
        except Exception as exc:
            print(f"[WARN] Failed to refresh collision overlay: {exc}")
            self.enabled = False

    def close(self) -> None:
        if self._settings is None:
            return
        try:
            if self._physxui is not None and self._saved_visualization_collision_mesh is not None:
                self._physxui.enable_collision_mesh_visualization(bool(self._saved_visualization_collision_mesh))
            if self._setting_display_colliders is not None and self._saved_display_colliders is not None:
                self._settings.set(self._setting_display_colliders, int(self._saved_display_colliders))
            if self._setting_display_collider_normals is not None and self._saved_display_collider_normals is not None:
                self._settings.set(self._setting_display_collider_normals, bool(self._saved_display_collider_normals))
            if (
                self._setting_visualization_collision_mesh is not None
                and self._saved_visualization_collision_mesh is not None
            ):
                self._settings.set(
                    self._setting_visualization_collision_mesh,
                    bool(self._saved_visualization_collision_mesh),
                )
        except Exception as exc:
            print(f"[WARN] Failed to restore collision overlay settings: {exc}")
        finally:
            self.enabled = False


def _resolve_collision_approx(option: str | None) -> str | None:
    key = str(option or "default").strip().lower()
    if key not in _COLLISION_APPROX_ALIASES:
        raise ValueError(
            f"Unsupported collision approximation '{option}'. "
            f"Expected one of: {', '.join(sorted(_COLLISION_APPROX_ALIASES))}."
        )
    return _COLLISION_APPROX_ALIASES[key]


def _resolve_convex_decomposition_settings(args: SceneMouseCollectArgs) -> ConvexDecompositionSettings:
    voxel_resolution = int(args.convex_decomp_voxel_resolution)
    max_convex_hulls = int(args.convex_decomp_max_convex_hulls)
    error_percentage = float(args.convex_decomp_error_percentage)
    shrink_wrap = bool(args.convex_decomp_shrink_wrap)

    if not 50000 <= voxel_resolution <= 5000000:
        raise ValueError("--convex_decomp_voxel_resolution must be in [50000, 5000000].")
    if not 1 <= max_convex_hulls <= 2048:
        raise ValueError("--convex_decomp_max_convex_hulls must be in [1, 2048].")
    if not 0.0 <= error_percentage <= 20.0:
        raise ValueError("--convex_decomp_error_percentage must be in [0, 20].")

    return ConvexDecompositionSettings(
        voxel_resolution=voxel_resolution,
        max_convex_hulls=max_convex_hulls,
        error_percentage=error_percentage,
        shrink_wrap=shrink_wrap,
    )


def _apply_convex_decomposition_settings(prim, settings: ConvexDecompositionSettings) -> None:
    from pxr import PhysxSchema

    api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    api.CreateVoxelResolutionAttr().Set(int(settings.voxel_resolution))
    api.CreateMaxConvexHullsAttr().Set(int(settings.max_convex_hulls))
    api.CreateErrorPercentageAttr().Set(float(settings.error_percentage))
    api.CreateShrinkWrapAttr().Set(bool(settings.shrink_wrap))


class MouseCommandCollectUI:
    _WINDOW_WIDTH = 460
    _WINDOW_HEIGHT = 430
    _POSE_BUTTON_WIDTH = 68
    _POSE_BUTTON_HEIGHT = 76
    _ACTION_BUTTON_WIDTH = 106
    _ACTION_BUTTON_HEIGHT = 64
    _EPISODE_BUTTON_WIDTH = 144
    _EPISODE_BUTTON_HEIGHT = 64

    def __init__(
        self,
        title: str,
        lin_step: float = 0.015,
        ang_step: float = 0.10,
        *,
        allow_arm_switch: bool = False,
    ):
        import omni.ui as ui

        self.lin_step = float(lin_step)
        self.ang_step = float(ang_step)
        self.gripper_closed = False
        self._pending_delta = [0.0] * 6
        self._start_requested = False
        self._stop_save_requested = False
        self._stop_discard_requested = False
        self._reset_requested = False
        self._switch_arm_requested = False
        self._window = ui.Window(title, width=self._WINDOW_WIDTH, height=self._WINDOW_HEIGHT)
        self._apply_window_size()
        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Mouse teleop + episode control", height=20)
                with ui.HStack(spacing=6):
                    self._make_pose_button(ui, "+X", lambda: self._add_delta(0, +self.lin_step))
                    self._make_pose_button(ui, "-X", lambda: self._add_delta(0, -self.lin_step))
                    self._make_pose_button(ui, "+Y", lambda: self._add_delta(1, +self.lin_step))
                    self._make_pose_button(ui, "-Y", lambda: self._add_delta(1, -self.lin_step))
                    self._make_pose_button(ui, "+Z", lambda: self._add_delta(2, +self.lin_step))
                    self._make_pose_button(ui, "-Z", lambda: self._add_delta(2, -self.lin_step))
                with ui.HStack(spacing=6):
                    self._make_pose_button(ui, "+R", lambda: self._add_delta(3, +self.ang_step))
                    self._make_pose_button(ui, "-R", lambda: self._add_delta(3, -self.ang_step))
                    self._make_pose_button(ui, "+P", lambda: self._add_delta(4, +self.ang_step))
                    self._make_pose_button(ui, "-P", lambda: self._add_delta(4, -self.ang_step))
                    self._make_pose_button(ui, "+Yaw", lambda: self._add_delta(5, +self.ang_step))
                    self._make_pose_button(ui, "-Yaw", lambda: self._add_delta(5, -self.ang_step))
                with ui.HStack(spacing=6):
                    self._make_action_button(ui, "Gripper Toggle", self._toggle_gripper)
                    self._make_action_button(ui, "Clear Step Queue", self._clear_delta)
                    self._make_action_button(ui, "Reset Robot", self._request_reset)
                    if allow_arm_switch:
                        self._make_action_button(ui, "Switch Arm", self._request_switch_arm)
                ui.Spacer(height=6)
                with ui.HStack(spacing=6):
                    self._make_episode_button(ui, "Start Recording", self._request_start)
                    self._make_episode_button(ui, "Stop + Save", self._request_stop_save)
                    self._make_episode_button(ui, "Stop + Discard", self._request_stop_discard)

    def _apply_window_size(self) -> None:
        for attr, value in (("width", self._WINDOW_WIDTH), ("height", self._WINDOW_HEIGHT)):
            try:
                setattr(self._window, attr, value)
            except Exception:
                pass

    def _make_pose_button(self, ui, label: str, clicked_fn: Callable[[], None]) -> None:
        ui.Button(label, clicked_fn=clicked_fn, width=self._POSE_BUTTON_WIDTH, height=self._POSE_BUTTON_HEIGHT)

    def _make_action_button(self, ui, label: str, clicked_fn: Callable[[], None]) -> None:
        ui.Button(label, clicked_fn=clicked_fn, width=self._ACTION_BUTTON_WIDTH, height=self._ACTION_BUTTON_HEIGHT)

    def _make_episode_button(self, ui, label: str, clicked_fn: Callable[[], None]) -> None:
        ui.Button(label, clicked_fn=clicked_fn, width=self._EPISODE_BUTTON_WIDTH, height=self._EPISODE_BUTTON_HEIGHT)

    def _add_delta(self, idx: int, value: float):
        self._pending_delta[idx] += value

    def _toggle_gripper(self):
        self.gripper_closed = not self.gripper_closed

    def _clear_delta(self):
        self._pending_delta = [0.0] * 6

    def _request_start(self):
        self._start_requested = True

    def _request_stop_save(self):
        self._stop_save_requested = True

    def _request_stop_discard(self):
        self._stop_discard_requested = True

    def _request_reset(self):
        self._reset_requested = True

    def _request_switch_arm(self):
        self._switch_arm_requested = True

    def consume_delta(self, device: str) -> torch.Tensor:
        delta = torch.tensor(self._pending_delta, device=device, dtype=torch.float32)
        self._pending_delta = [0.0] * 6
        return delta

    def consume_start_request(self) -> bool:
        requested = self._start_requested
        self._start_requested = False
        return requested

    def consume_stop_save_request(self) -> bool:
        requested = self._stop_save_requested
        self._stop_save_requested = False
        return requested

    def consume_stop_discard_request(self) -> bool:
        requested = self._stop_discard_requested
        self._stop_discard_requested = False
        return requested

    def consume_reset_request(self) -> bool:
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def consume_switch_arm_request(self) -> bool:
        requested = self._switch_arm_requested
        self._switch_arm_requested = False
        return requested

    def set_title(self, title: str) -> None:
        try:
            self._window.title = title
            self._apply_window_size()
        except Exception:
            pass


class SceneTeleopEpisodeWriter:
    def __init__(
        self,
        dataset_file: str,
        capture_hz: float,
        append: bool,
        env_name: str,
        camera_aliases: dict[str, dict[str, object]],
        plan: RobotPlacementPlan,
        scene_usd_path: str,
        scene_graph_path: str,
        placements_path: str,
        *,
        initial_arm_side: str = "left",
        arm_switch_supported: bool = False,
    ):
        self.dataset_file = self._normalize_dataset_file(dataset_file)
        self.capture_period = 1.0 / max(capture_hz, 1.0e-6)
        self.file_handler = HDF5DatasetFileHandler()
        self.recording = False
        plan_payload = json.loads(json.dumps(plan_to_payload(plan)))

        if append and os.path.exists(self.dataset_file):
            self.file_handler.open(self.dataset_file, mode="a")
        else:
            self.file_handler.create(self.dataset_file, env_name=env_name)

        self.file_handler.add_env_args(
            {
                "camera_aliases": camera_aliases,
                "capture_hz": capture_hz,
                "scene_usd_path": scene_usd_path,
                "scene_graph_path": scene_graph_path,
                "placements_path": placements_path,
                "placement_plan": plan_payload,
                "teleop_ui": {
                    "type": "mouse_click_scene_collect",
                    "pose_step_buttons": ["+X", "-X", "+Y", "-Y", "+Z", "-Z", "+R", "-R", "+P", "-P", "+Yaw", "-Yaw"],
                    "gripper_toggle": "Gripper Toggle",
                    "switch_arm": "Switch Arm" if arm_switch_supported else None,
                    "start_recording": "Start Recording",
                    "stop_and_save": "Stop + Save",
                    "stop_and_discard": "Stop + Discard",
                    "reset_robot": "Reset Robot",
                },
                "initial_arm_side": initial_arm_side,
            }
        )
        self.reset_episode()

    @staticmethod
    def _normalize_dataset_file(dataset_file: str) -> str:
        if dataset_file.endswith(".hdf5"):
            return dataset_file
        return f"{dataset_file}.hdf5"

    def reset_episode(self):
        self.episode = EpisodeData()
        self.frame_count = 0
        self.episode_start_time = None
        self.next_capture_time = 0.0

    def start_recording(self, sim_time: float):
        if self.recording:
            print("[WARN] Recording is already active.")
            return
        self.reset_episode()
        self.episode_start_time = sim_time
        self.next_capture_time = sim_time
        self.recording = True
        print("[INFO] Recording started.")

    def stop_and_save(self):
        if not self.recording:
            print("[WARN] Recording is not active.")
            return
        if self.frame_count == 0:
            print("[WARN] Recording stopped with zero frames. Nothing was saved.")
            self.recording = False
            self.reset_episode()
            return
        self.episode.success = True
        self.episode.pre_export()
        self.file_handler.write_episode(self.episode)
        self.file_handler.flush()
        saved_episode_idx = self.file_handler.demo_count - 1
        print(
            f"[INFO] Saved episode demo_{saved_episode_idx} with {self.frame_count} frames to "
            f"{os.path.abspath(self.dataset_file)}"
        )
        self.recording = False
        self.reset_episode()

    def stop_and_discard(self):
        if not self.recording:
            print("[WARN] Recording is not active.")
            return
        print(f"[INFO] Discarded current recording ({self.frame_count} captured frames).")
        self.recording = False
        self.reset_episode()

    def close(self):
        self.file_handler.close()

    def maybe_record_frame(self, sim_time: float, action: torch.Tensor, controller: RobotStackController, cameras: dict[str, object]):
        if not self.recording:
            return False
        if self.episode_start_time is None:
            self.episode_start_time = sim_time
            self.next_capture_time = sim_time
        if self.frame_count > 0 and sim_time + 1.0e-9 < self.next_capture_time:
            return False
        while self.next_capture_time <= sim_time + 1.0e-9:
            self.next_capture_time += self.capture_period
        self._record_frame(sim_time, action, controller, cameras)
        self.frame_count += 1
        return True

    def _record_frame(self, sim_time: float, action: torch.Tensor, controller: RobotStackController, cameras: dict[str, object]):
        robot = controller.robot
        _, _, _, ee_pos_b, ee_quat_b = controller._current_ee()
        target_pos = controller.target_pos if controller.target_pos is not None else ee_pos_b
        target_quat = controller.target_quat if controller.target_quat is not None else ee_quat_b

        self.episode.add("actions", action.to(dtype=torch.float32, device="cpu"))
        self.episode.add("timestamps", torch.tensor(sim_time - self.episode_start_time, dtype=torch.float32))
        self.episode.add("obs/joint_pos", robot.data.joint_pos[0].detach().cpu().to(dtype=torch.float32))
        self.episode.add("obs/joint_vel", robot.data.joint_vel[0].detach().cpu().to(dtype=torch.float32))
        self.episode.add("obs/root_pose", robot.data.root_pose_w[0].detach().cpu().to(dtype=torch.float32))
        self.episode.add("obs/ee_pos", ee_pos_b[0].detach().cpu().to(dtype=torch.float32))
        self.episode.add("obs/ee_quat", ee_quat_b[0].detach().cpu().to(dtype=torch.float32))
        self.episode.add("obs/target_ee_pos", target_pos[0].detach().cpu().to(dtype=torch.float32))
        self.episode.add("obs/target_ee_quat", target_quat[0].detach().cpu().to(dtype=torch.float32))
        self.episode.add(
            "obs/gripper_joint_pos",
            robot.data.joint_pos[0, controller.gripper_joint_ids].detach().cpu().to(dtype=torch.float32),
        )
        self.episode.add(
            "obs/active_arm_side",
            torch.tensor(1 if controller.active_arm_side == "right" else 0, dtype=torch.int64),
        )
        for camera_name, camera in cameras.items():
            rgb = camera.data.output["rgb"][0]
            if rgb.shape[-1] > 3:
                rgb = rgb[..., :3]
            rgb = rgb.detach().cpu()
            if rgb.dtype != torch.uint8:
                rgb = torch.clamp(rgb, 0, 255).to(dtype=torch.uint8)
            self.episode.add(f"obs/{camera_name}", rgb.contiguous())


def _yaw_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(yaw_deg) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _planned_base_height(robot_name: str) -> float:
    try:
        return float(compute_robot_floor_offset_z(robot_name))
    except Exception:
        spec = STACK_SPECS[robot_name]
        return max(0.0, -float(spec.ground_z))


ROBOT_BASE_Z_BIAS = {
    "kinova": 0.0,
    "agibot": 0.0,
    "r1lite": 0.0,
}


def _plan_camera_pose(plan: RobotPlacementPlan) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    support_x, support_y = plan.support_center_xy
    support_z = plan.support_z
    eye = (
        support_x + 2.3,
        support_y + 2.0,
        max(2.1, support_z + 1.8),
    )
    target = (
        support_x,
        support_y,
        max(0.45, support_z + 0.55),
    )
    return eye, target


def _make_dummy_cube_specs() -> tuple[CuboidSpec, CuboidSpec]:
    cube_base = CuboidSpec(size=(0.01, 0.01, 0.01), pos=(100.0, 100.0, -10.0), color=(0.1, 0.1, 0.1), kinematic=True)
    cube_pick = CuboidSpec(size=(0.01, 0.01, 0.01), pos=(100.2, 100.0, -10.0), color=(0.1, 0.1, 0.1), kinematic=True)
    return cube_base, cube_pick


def _make_spec_for_scene_collect(robot_name: str, plan: RobotPlacementPlan, base_z_bias: float, arm_side: str):
    spec = resolve_stack_spec(robot_name, arm_side)
    base_z = _planned_base_height(robot_name) + float(base_z_bias)
    camera_eye, camera_target = _plan_camera_pose(plan)
    cube_base, cube_pick = _make_dummy_cube_specs()
    robot_cfg = spec.robot_cfg.replace(
        init_state=spec.robot_cfg.init_state.replace(pos=(plan.base_pose[0], plan.base_pose[1], base_z))
    )
    return replace(
        spec,
        robot_cfg=robot_cfg,
        root_z_zero=False,
        cube_base=cube_base,
        cube_pick=cube_pick,
        camera_eye=camera_eye,
        camera_target=camera_target,
    )


def _apply_root_pose(controller: RobotStackController, plan: RobotPlacementPlan, base_z: float) -> None:
    root_state = controller.robot.data.default_root_state.clone()
    root_state[:, 0] = float(plan.base_pose[0])
    root_state[:, 1] = float(plan.base_pose[1])
    root_state[:, 2] = float(base_z)
    quat = torch.tensor(_yaw_quat_wxyz(plan.base_pose[3]), device=controller.sim.device, dtype=torch.float32)
    root_state[:, 3:7] = quat.unsqueeze(0).repeat(controller.scene.num_envs, 1)
    controller.robot.write_root_pose_to_sim(root_state[:, :7])
    controller.robot.write_root_velocity_to_sim(root_state[:, 7:])
    controller.robot.reset()


def _compute_world_prim_min_z(stage, prim_path: str) -> float | None:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
    aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if aligned.IsEmpty():
        return None
    return float(aligned.GetMin()[2])


def _align_robot_root_to_floor(
    scene: InteractiveScene,
    controller: RobotStackController,
    *,
    floor_z: float = 0.0,
    max_passes: int = 3,
) -> float:
    robot_prim_path = f"{scene.env_prim_paths[0]}/Robot"
    applied_root_z = float(controller.robot.data.root_pose_w[0, 2].item())

    for _ in range(max_passes):
        scene.write_data_to_sim()
        controller.sim.step()
        scene.update(controller.sim.get_physics_dt())
        min_z = _compute_world_prim_min_z(scene.stage, robot_prim_path)
        if min_z is None:
            return applied_root_z
        delta_z = float(floor_z) - float(min_z)
        if abs(delta_z) <= 1.0e-3:
            return applied_root_z
        root_pose = controller.robot.data.root_pose_w.clone()
        root_pose[:, 2] += delta_z
        root_velocity = controller.robot.data.root_vel_w.clone()
        controller.robot.write_root_pose_to_sim(root_pose)
        controller.robot.write_root_velocity_to_sim(root_velocity)
        controller.robot.reset()
        applied_root_z = float(root_pose[0, 2].item())

    return applied_root_z


def _assign_simulation_owner(api_schema, physics_scene_path: str | None) -> None:
    if not physics_scene_path:
        return
    rel = api_schema.CreateSimulationOwnerRel()
    rel.SetTargets([physics_scene_path])


def _set_disable_gravity_attr(prim, enabled: bool) -> None:
    from pxr import Sdf

    attr = prim.GetAttribute("physxRigidBody:disableGravity")
    if not attr.IsValid():
        attr = prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, False)
    attr.Set(not bool(enabled))


def _iter_collision_prims(root_prim):
    from pxr import Usd, UsdGeom

    if not root_prim or not root_prim.IsValid():
        return []
    return [prim for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdGeom.Gprim)]


def _is_proxy_collision_prim(prim) -> bool:
    return bool(prim and prim.IsValid() and prim.GetName().startswith("CollisionProxy"))


def _iter_visual_collision_prims(root_prim):
    from pxr import UsdGeom

    visual_prims = []
    for prim in _iter_collision_prims(root_prim):
        if _is_proxy_collision_prim(prim):
            continue
        purpose = UsdGeom.Imageable(prim).GetPurposeAttr().Get()
        if purpose == UsdGeom.Tokens.guide:
            continue
        visual_prims.append(prim)
    return visual_prims


def _set_collision_enabled(prim, enabled: bool) -> None:
    from pxr import UsdPhysics

    if not prim or not prim.IsValid():
        return
    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr(bool(enabled))


def _remove_collision_proxy_prims(root_prim) -> None:
    if not root_prim or not root_prim.IsValid():
        return
    stage = root_prim.GetStage()
    to_remove = []
    root_prefix = str(root_prim.GetPath()) + "/"
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(root_prefix):
            continue
        if prim.GetName().startswith("CollisionProxy"):
            to_remove.append(prim_path)
    for prim_path in sorted(to_remove, key=len, reverse=True):
        stage.RemovePrim(prim_path)


def _create_collision_cube(stage, prim_path: str, center, size, physics_scene_path: str) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.CreateVisibilityAttr("invisible")
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xform.AddScaleOp().Set(Gf.Vec3f(*[max(1e-3, float(v)) for v in size]))

    collision_api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)
    _assign_simulation_owner(collision_api, physics_scene_path)


def _upsert_collision_proxy(root_prim, center, size, physics_scene_path: str) -> None:
    if not root_prim or not root_prim.IsValid():
        return
    _remove_collision_proxy_prims(root_prim)
    _create_collision_cube(root_prim.GetStage(), str(root_prim.GetPath().AppendChild("CollisionProxy")), center, size, physics_scene_path)


def _supported_scene_object_paths(scene_graph: dict) -> set[str]:
    supported_paths: set[str] = set()
    for edge in (scene_graph.get("edges") or {}).get("obj-obj", []):
        if not isinstance(edge, dict):
            continue
        relation_tokens = {token.strip().lower() for token in str(edge.get("relation") or "").split(",") if token.strip()}
        source = edge.get("source")
        target = edge.get("target")
        if "supported by" in relation_tokens and isinstance(source, str):
            supported_paths.add(source)
        if "supports" in relation_tokens and isinstance(target, str):
            supported_paths.add(target)
    return supported_paths


def _compute_bounds_for_prims(stage, prims) -> object | None:
    from pxr import Gf, Usd, UsdGeom

    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
    combined = Gf.Range3d()
    found = False
    for prim in prims:
        if not prim or not prim.IsValid():
            continue
        rng = bounds_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        if not found:
            combined = Gf.Range3d(rng.GetMin(), rng.GetMax())
            found = True
            continue
        combined.UnionWith(rng)
    return combined if found and not combined.IsEmpty() else None


def _compute_local_bbox_info_for_prim(root_prim) -> dict[str, tuple[float, float, float]] | None:
    from pxr import Usd, UsdGeom

    if not root_prim or not root_prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
    bbox = bbox_cache.ComputeLocalBound(root_prim)
    rng = bbox.ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    bmin = rng.GetMin()
    bmax = rng.GetMax()
    return {
        "center": (
            float((bmin[0] + bmax[0]) * 0.5),
            float((bmin[1] + bmax[1]) * 0.5),
            float((bmin[2] + bmax[2]) * 0.5),
        ),
        "size": (
            float(bmax[0] - bmin[0]),
            float(bmax[1] - bmin[1]),
            float(bmax[2] - bmin[2]),
        ),
    }


def _realign_generated_scene_floor_objects(
    stage,
    scene_root_path: str,
    scene_graph: dict,
    *,
    floor_z: float = 0.0,
) -> dict[str, int]:
    from pxr import Gf, UsdGeom

    supported_paths = _supported_scene_object_paths(scene_graph)
    grounded_roots = 0
    visual_roots = 0
    for prim_path in (scene_graph.get("obj") or {}):
        if prim_path in supported_paths:
            continue
        prim_name = Path(str(prim_path)).name
        live_prim = stage.GetPrimAtPath(f"{scene_root_path}/{prim_name}")
        if not live_prim.IsValid():
            continue

        visual_prims = _iter_visual_collision_prims(live_prim)
        if not visual_prims:
            continue
        visual_roots += 1

        rng = _compute_bounds_for_prims(stage, visual_prims)
        if rng is None:
            continue
        delta_z = float(floor_z) - float(rng.GetMin()[2])
        if abs(delta_z) <= 1.0e-4:
            continue

        xform = UsdGeom.Xformable(live_prim)
        ops = xform.GetOrderedXformOps()
        if not ops:
            continue

        mat = Gf.Matrix4d(ops[0].Get())
        translate = mat.ExtractTranslation()
        mat.SetTranslateOnly(translate + Gf.Vec3d(0.0, 0.0, delta_z))
        ops[0].Set(mat)
        grounded_roots += 1

    return {"visual_roots": visual_roots, "grounded_roots": grounded_roots}


def _find_live_physics_scene_path(stage, scene_root_path: str) -> str | None:
    preferred = None
    fallback = None
    scene_root_prefix = scene_root_path.rstrip("/") + "/"
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        prim_path = str(prim.GetPath())
        if fallback is None:
            fallback = prim_path
        if not prim_path.startswith(scene_root_prefix):
            preferred = prim_path
            break
    return preferred or fallback


def _remove_nested_physics_scenes(stage, scene_root_path: str) -> list[str]:
    removed_paths: list[str] = []
    scene_root_prefix = scene_root_path.rstrip("/") + "/"
    to_remove: list[str] = []
    for prim in stage.Traverse():
        prim_name = prim.GetName().lower()
        if prim.GetTypeName() != "PhysicsScene" and prim_name != "physicsscene":
            continue
        prim_path = str(prim.GetPath())
        if prim_path.startswith(scene_root_prefix):
            to_remove.append(prim_path)
    for prim_path in sorted(to_remove, key=len, reverse=True):
        stage.RemovePrim(prim_path)
        removed_paths.append(prim_path)
    return removed_paths


def _prepare_scene_usd_without_physics_scenes(scene_usd_path: str) -> str:
    from pxr import Usd

    source_path = Path(scene_usd_path).resolve()
    stage = Usd.Stage.Open(str(source_path))
    if stage is None:
        return str(source_path)

    to_remove: list[str] = []
    for prim in stage.Traverse():
        prim_name = prim.GetName().lower()
        if prim.GetTypeName() == "PhysicsScene" or prim_name == "physicsscene":
            to_remove.append(str(prim.GetPath()))

    if not to_remove:
        return str(source_path)

    for prim_path in sorted(to_remove, key=len, reverse=True):
        stage.RemovePrim(prim_path)

    sanitized_path = Path(tempfile.gettempdir()) / f"{source_path.stem}.scene_collect_nophysics.usd"
    stage.GetRootLayer().Export(str(sanitized_path))
    return str(sanitized_path)


def _rebind_generated_scene_physics(
    stage,
    scene_root_path: str,
    scene_graph: dict,
    *,
    object_collision_approx: str | None = None,
    target_prim: str | None = None,
    target_collision_approx: str | None = None,
    convex_decomposition_settings: ConvexDecompositionSettings | None = None,
) -> dict[str, object]:
    from pxr import UsdGeom, UsdPhysics

    physics_scene_path = _find_live_physics_scene_path(stage, scene_root_path)
    if physics_scene_path is None:
        return {
            "physics_scene_path": None,
            "dynamic_roots": 0,
            "room_colliders": 0,
            "object_colliders": 0,
            "visual_object_colliders": 0,
            "proxy_object_colliders": 0,
            "object_collision_approx": None,
            "target_collision_approx": None,
            "convex_decomposition_prim_count": 0,
            "convex_decomposition_settings": None,
        }

    room_colliders = 0
    room_root = stage.GetPrimAtPath(f"{scene_root_path}/GeneratedRoom/Room")
    for prim in _iter_collision_prims(room_root):
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr(True)
        _assign_simulation_owner(collision_api, physics_scene_path)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)
        room_colliders += 1

    dynamic_roots = 0
    object_colliders = 0
    visual_object_colliders = 0
    proxy_object_colliders = 0
    applied_object_collision_approx = None
    applied_target_collision_approx = None
    convex_decomposition_prim_count = 0
    for prim_path in (scene_graph.get("obj") or {}):
        prim_name = Path(str(prim_path)).name
        live_prim = stage.GetPrimAtPath(f"{scene_root_path}/{prim_name}")
        if not live_prim.IsValid():
            continue

        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(live_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        rigid_body_api.CreateKinematicEnabledAttr(False)
        rigid_body_api.CreateStartsAsleepAttr(False)
        _assign_simulation_owner(rigid_body_api, physics_scene_path)
        _set_disable_gravity_attr(live_prim, True)

        mass_api = UsdPhysics.MassAPI.Apply(live_prim)
        mass_api.CreateMassAttr(1.0)

        proxy_collision_prims = [prim for prim in _iter_collision_prims(live_prim) if _is_proxy_collision_prim(prim)]
        visual_collision_prims = _iter_visual_collision_prims(live_prim)
        if not visual_collision_prims and not proxy_collision_prims:
            bbox_info = _compute_local_bbox_info_for_prim(live_prim)
            if bbox_info is not None:
                _upsert_collision_proxy(
                    live_prim,
                    bbox_info["center"],
                    bbox_info["size"],
                    physics_scene_path,
                )
                proxy_collision_prims = [prim for prim in _iter_collision_prims(live_prim) if _is_proxy_collision_prim(prim)]

        active_collision_prims = (
            visual_collision_prims if visual_collision_prims else proxy_collision_prims
        )

        for proxy_prim in proxy_collision_prims:
            _set_collision_enabled(proxy_prim, not visual_collision_prims)
            _assign_simulation_owner(UsdPhysics.CollisionAPI.Apply(proxy_prim), physics_scene_path)

        for collision_prim in active_collision_prims:
            collision_api = UsdPhysics.CollisionAPI.Apply(collision_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            _assign_simulation_owner(collision_api, physics_scene_path)
            if collision_prim.IsA(UsdGeom.Mesh):
                is_target_prim = str(prim_path) == str(target_prim)
                approximation = (
                    object_collision_approx
                    if object_collision_approx is not None
                    else (
                        target_collision_approx
                        if target_collision_approx is not None and is_target_prim
                        else (
                            "convexHull"
                            if _is_proxy_collision_prim(collision_prim)
                            else _DYNAMIC_MESH_COLLISION_APPROX
                        )
                    )
                )
                UsdPhysics.MeshCollisionAPI.Apply(collision_prim).CreateApproximationAttr(
                    approximation
                )
                if approximation == "convexDecomposition" and convex_decomposition_settings is not None:
                    _apply_convex_decomposition_settings(collision_prim, convex_decomposition_settings)
                    convex_decomposition_prim_count += 1
                if object_collision_approx is not None:
                    applied_object_collision_approx = approximation
                elif target_collision_approx is not None and is_target_prim:
                    applied_target_collision_approx = approximation
            object_colliders += 1
        visual_object_colliders += len(visual_collision_prims)
        if not visual_collision_prims:
            proxy_object_colliders += len(proxy_collision_prims)
        dynamic_roots += 1

    return {
        "physics_scene_path": physics_scene_path,
        "dynamic_roots": dynamic_roots,
        "room_colliders": room_colliders,
        "object_colliders": object_colliders,
        "visual_object_colliders": visual_object_colliders,
        "proxy_object_colliders": proxy_object_colliders,
        "object_collision_approx": applied_object_collision_approx,
        "target_collision_approx": applied_target_collision_approx,
        "convex_decomposition_prim_count": convex_decomposition_prim_count,
        "convex_decomposition_settings": (
            None if convex_decomposition_prim_count == 0 or convex_decomposition_settings is None
            else convex_decomposition_settings.as_dict()
        ),
    }


def _settle_dynamic_scene(
    scene: InteractiveScene,
    controller: RobotStackController,
    sync_cameras: Callable[[], None] | None,
    *,
    settle_steps: int = 90,
) -> None:
    for _ in range(max(0, settle_steps)):
        scene.write_data_to_sim()
        controller.sim.step()
        scene.update(controller.sim.get_physics_dt())
        if sync_cameras is not None:
            sync_cameras()


def _reset_scene_to_plan(
    scene: InteractiveScene,
    controller: RobotStackController,
    plan: RobotPlacementPlan,
    base_z: float,
    sync_cameras: Callable[[], None] | None,
    *,
    settle_steps: int = 90,
) -> float:
    controller.reset()
    scene.reset()
    _apply_root_pose(controller, plan, base_z)
    if hasattr(controller, "_apply_reset_joint_state"):
        controller._apply_reset_joint_state()
    scene.write_data_to_sim()
    controller.sim.step()
    scene.update(controller.sim.get_physics_dt())
    aligned_base_z = _align_robot_root_to_floor(scene, controller, floor_z=0.0)
    _settle_dynamic_scene(scene, controller, sync_cameras, settle_steps=settle_steps)
    aligned_base_z = _align_robot_root_to_floor(scene, controller, floor_z=0.0)
    if sync_cameras is not None:
        sync_cameras()
    return aligned_base_z


def _build_scene_collect_cfg(spec, scene_usd_path: str, robot_name: str):
    scene_usd = _prepare_scene_usd_without_physics_scenes(scene_usd_path)

    attrs: dict[str, object] = {
        "generated_scene": AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GeneratedScene",
            spawn=sim_utils.UsdFileCfg(usd_path=scene_usd),
        ),
        "robot": spec.robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        "cube_base": _make_cube_cfg("{ENV_REGEX_NS}/CubeBase", spec.cube_base),
        "cube_pick": _make_cube_cfg("{ENV_REGEX_NS}/CubePick", spec.cube_pick),
    }

    if robot_name == "agibot":
        attrs["head_camera"] = CameraCfg(
            prim_path=_AGIBOT_CAMERA_SPECS["head"]["prim_path"],
            update_period=0.0,
            width=_AGIBOT_CAMERA_SPECS["head"]["width"],
            height=_AGIBOT_CAMERA_SPECS["head"]["height"],
            data_types=["rgb"],
            spawn=_AGIBOT_CAMERA_SPECS["head"]["spawn"],
        )
        attrs["left_hand_camera"] = CameraCfg(
            prim_path=_AGIBOT_CAMERA_SPECS["left_hand"]["prim_path"],
            update_period=0.0,
            width=_AGIBOT_CAMERA_SPECS["left_hand"]["width"],
            height=_AGIBOT_CAMERA_SPECS["left_hand"]["height"],
            data_types=["rgb"],
            spawn=_AGIBOT_CAMERA_SPECS["left_hand"]["spawn"],
        )
        attrs["right_hand_camera"] = CameraCfg(
            prim_path=_AGIBOT_CAMERA_SPECS["right_hand"]["prim_path"],
            update_period=0.0,
            width=_AGIBOT_CAMERA_SPECS["right_hand"]["width"],
            height=_AGIBOT_CAMERA_SPECS["right_hand"]["height"],
            data_types=["rgb"],
            spawn=_AGIBOT_CAMERA_SPECS["right_hand"]["spawn"],
        )

    if robot_name == "r1lite":
        attrs["head_camera"] = CameraCfg(
            prim_path=_R1LITE_CAMERA_SPECS["head"]["prim_path"],
            update_period=0.0,
            width=_R1LITE_CAMERA_SPECS["head"]["width"],
            height=_R1LITE_CAMERA_SPECS["head"]["height"],
            data_types=["rgb"],
            spawn=_make_pinhole_camera_cfg(
                _R1LITE_CAMERA_SPECS["head"]["fovx_deg"],
                _R1LITE_CAMERA_SPECS["head"]["fovy_deg"],
            ),
            offset=CameraCfg.OffsetCfg(
                pos=_R1LITE_CAMERA_SPECS["head"]["pos"],
                rot=_R1LITE_CAMERA_SPECS["head"]["quat"],
                convention="world",
            ),
        )
        attrs["left_hand_camera"] = CameraCfg(
            prim_path=_R1LITE_CAMERA_SPECS["left_hand"]["prim_path"],
            update_period=0.0,
            width=_R1LITE_CAMERA_SPECS["left_hand"]["width"],
            height=_R1LITE_CAMERA_SPECS["left_hand"]["height"],
            data_types=["rgb"],
            spawn=_make_pinhole_camera_cfg(
                _R1LITE_CAMERA_SPECS["left_hand"]["fovx_deg"],
                _R1LITE_CAMERA_SPECS["left_hand"]["fovy_deg"],
            ),
            offset=CameraCfg.OffsetCfg(
                pos=_R1LITE_CAMERA_SPECS["left_hand"]["pos"],
                rot=_R1LITE_CAMERA_SPECS["left_hand"]["quat"],
                convention="world",
            ),
        )
        attrs["right_hand_camera"] = CameraCfg(
            prim_path=_R1LITE_CAMERA_SPECS["right_hand"]["prim_path"],
            update_period=0.0,
            width=_R1LITE_CAMERA_SPECS["right_hand"]["width"],
            height=_R1LITE_CAMERA_SPECS["right_hand"]["height"],
            data_types=["rgb"],
            spawn=_make_pinhole_camera_cfg(
                _R1LITE_CAMERA_SPECS["right_hand"]["fovx_deg"],
                _R1LITE_CAMERA_SPECS["right_hand"]["fovy_deg"],
            ),
            offset=CameraCfg.OffsetCfg(
                pos=_R1LITE_CAMERA_SPECS["right_hand"]["pos"],
                rot=_R1LITE_CAMERA_SPECS["right_hand"]["quat"],
                convention="world",
            ),
        )

    attrs["world_camera"] = CameraCfg(
        prim_path="{ENV_REGEX_NS}/WorldCamera",
        update_period=0.0,
        width=848,
        height=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 6.0),
        ),
    )

    return configclass(type("SceneCollectCfg", (InteractiveSceneCfg,), attrs))


def _build_scene_mouse_collect(
    sim: sim_utils.SimulationContext,
    robot_name: str,
    args: SceneMouseCollectArgs,
):
    scene_graph, placements = load_scene_state(args.scene_graph_path, args.placements_path)
    plan = plan_robot_base_pose(
        scene_graph,
        placements,
        target_prim=args.target,
        support_prim=args.support,
        robot=robot_name,
    )
    save_plan_outputs(scene_graph, placements, plan, output_dir=args.plan_output_dir)

    effective_base_z_bias = ROBOT_BASE_Z_BIAS.get(robot_name, 0.0) + float(args.base_z_bias)
    spec = _make_spec_for_scene_collect(robot_name, plan, effective_base_z_bias, args.arm_side)
    scene_cfg = _build_scene_collect_cfg(spec, args.scene_usd_path, robot_name)(
        num_envs=args.num_envs,
        env_spacing=spec.env_spacing,
    )
    scene = InteractiveScene(scene_cfg)
    scene_root_path = f"{scene.env_prim_paths[0]}/GeneratedScene"
    removed_physics_scenes = _remove_nested_physics_scenes(scene.stage, scene_root_path)
    if removed_physics_scenes:
        print(f"[INFO] Removed nested physics scenes from generated scene: {removed_physics_scenes}")
    sim.reset()
    _configure_world_camera(scene, spec, sim)
    controller = RobotStackController(sim, scene, spec)
    controller.ee_marker.set_visibility(False)
    controller.goal_marker.set_visibility(False)
    if hasattr(controller, "_apply_reset_joint_state"):
        controller._apply_reset_joint_state()
    _settle_dynamic_scene(scene, controller, None, settle_steps=20)
    resolved_convex_decomposition_settings = _resolve_convex_decomposition_settings(args)
    resolved_object_collision_approx = _resolve_collision_approx(args.object_collision_approx)
    resolved_target_collision_approx = _resolve_collision_approx(args.target_collision_approx)
    physics_rebind_summary = _rebind_generated_scene_physics(
        scene.stage,
        scene_root_path,
        scene_graph,
        object_collision_approx=resolved_object_collision_approx,
        target_prim=plan.target_prim,
        target_collision_approx=resolved_target_collision_approx,
        convex_decomposition_settings=resolved_convex_decomposition_settings,
    )
    floor_realign_summary = _realign_generated_scene_floor_objects(scene.stage, scene_root_path, scene_graph)
    sim.reset()

    cameras = {"world": scene["world_camera"]}
    camera_aliases = {"world": {"prim_path": "{ENV_REGEX_NS}/WorldCamera", "width": 848, "height": 480}}
    sync_cameras: Callable[[], None] | None = None

    if robot_name == "agibot":
        cameras.update(
            {
                "head": scene["head_camera"],
                "left_hand": scene["left_hand_camera"],
                "right_hand": scene["right_hand_camera"],
            }
        )
        camera_aliases.update(
            {
                name: {
                    "prim_path": spec_cfg["prim_path"],
                    "width": spec_cfg["width"],
                    "height": spec_cfg["height"],
                }
                for name, spec_cfg in _AGIBOT_CAMERA_SPECS.items()
                if name != "world"
            }
        )

        body_ids: dict[str, int] = {}
        for camera_name, mount in _AGIBOT_MOUNT_POSES.items():
            body_id_list, _ = controller.robot.find_bodies([mount["body_name"]])
            if len(body_id_list) != 1:
                raise RuntimeError(f"Unable to resolve body for camera {camera_name}: {mount['body_name']}")
            body_ids[camera_name] = int(body_id_list[0])

        def _sync() -> None:
            body_pose_w = controller.robot.data.body_pose_w
            device = body_pose_w.device
            for camera_name, body_id in body_ids.items():
                mount = _AGIBOT_MOUNT_POSES[camera_name]
                mount_pos = torch.tensor(mount["pos"], device=device, dtype=torch.float32).repeat(scene.num_envs, 1)
                mount_quat = torch.tensor(mount["quat"], device=device, dtype=torch.float32).repeat(scene.num_envs, 1)
                cam_pos_w, cam_quat_w = combine_frame_transforms(
                    body_pose_w[:, body_id, 0:3],
                    body_pose_w[:, body_id, 3:7],
                    mount_pos,
                    mount_quat,
                )
                cameras[camera_name].set_world_poses(cam_pos_w, cam_quat_w, convention="opengl")

        sync_cameras = _sync

    if robot_name == "r1lite":
        cameras.update(
            {
                "head": scene["head_camera"],
                "left_hand": scene["left_hand_camera"],
                "right_hand": scene["right_hand_camera"],
            }
        )
        camera_aliases.update(
            {
                name: {
                    "prim_path": spec_cfg["prim_path"],
                    "width": spec_cfg["width"],
                    "height": spec_cfg["height"],
                }
                for name, spec_cfg in _R1LITE_CAMERA_SPECS.items()
                if name != "world"
            }
        )

    aligned_base_z = _reset_scene_to_plan(
        scene,
        controller,
        plan,
        _planned_base_height(robot_name) + effective_base_z_bias,
        sync_cameras,
    )
    return (
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
    )


def run_scene_mouse_collect(simulation_app, robot_name: str, args: SceneMouseCollectArgs) -> None:
    if args.num_envs != 1:
        raise ValueError("Scene mouse collection only supports --num_envs 1.")

    spec = resolve_stack_spec(robot_name, args.arm_side)
    env_name = f"Isaac-{robot_name.capitalize()}-SceneMouseCollect-v0"
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
    sim.set_camera_view(spec.camera_eye, spec.camera_target)
    collision_overlay = CollisionOverlayController(args.collision_only)
    writer: SceneTeleopEpisodeWriter | None = None

    try:
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
        ) = _build_scene_mouse_collect(
            sim,
            robot_name,
            args,
        )
        collision_overlay.refresh()
        planned_eye, planned_target = _plan_camera_pose(plan)
        sim.set_camera_view(planned_eye, planned_target)
        ui = MouseCommandCollectUI(
            f"{spec.window_title} Scene Collect",
            lin_step=args.lin_step,
            ang_step=args.ang_step,
            allow_arm_switch=spec.arm_switch_supported,
        )
        writer = SceneTeleopEpisodeWriter(
            args.dataset_file,
            args.capture_hz,
            args.append,
            env_name,
            camera_aliases,
            plan,
            args.scene_usd_path,
            args.scene_graph_path,
            args.placements_path,
            initial_arm_side=spec.arm_side,
            arm_switch_supported=spec.arm_switch_supported,
        )

        print(f"[INFO] {robot_name} scene teleop collection ready.")
        print(f"[INFO] Scene USD: {os.path.abspath(args.scene_usd_path)}")
        print(f"[INFO] Target object: {plan.target_prim}")
        print(f"[INFO] Support object: {plan.support_prim}")
        print(f"[INFO] Planned base pose: {plan.base_pose}")
        print(f"[INFO] Robot asset: {resolve_robot_asset_path(robot_name)}")
        print(f"[INFO] Static floor offset z: {_planned_base_height(robot_name):.4f}")
        print(f"[INFO] Effective base z bias: {effective_base_z_bias:.4f}")
        print(f"[INFO] Runtime aligned base z: {aligned_base_z:.4f}")
        print(f"[INFO] Active arm: {controller.active_arm_side}")
        print(
            "[INFO] Scene physics rebind: "
            f"scene={physics_rebind_summary['physics_scene_path']} "
            f"dynamic_roots={physics_rebind_summary['dynamic_roots']} "
            f"room_colliders={physics_rebind_summary['room_colliders']} "
            f"object_colliders={physics_rebind_summary['object_colliders']} "
            f"visual_object_colliders={physics_rebind_summary['visual_object_colliders']} "
            f"proxy_object_colliders={physics_rebind_summary['proxy_object_colliders']}"
        )
        if physics_rebind_summary["object_collision_approx"] is not None:
            print(
                "[INFO] Global object collision approximation override: "
                f"all_object_colliders -> {physics_rebind_summary['object_collision_approx']}"
            )
        if physics_rebind_summary["target_collision_approx"] is not None:
            print(
                "[INFO] Target collision approximation override: "
                f"{plan.target_prim} -> {physics_rebind_summary['target_collision_approx']}"
            )
        if physics_rebind_summary["convex_decomposition_settings"] is not None:
            convex_decomp_settings = physics_rebind_summary["convex_decomposition_settings"]
            print(
                "[INFO] Convex decomposition tuning: "
                f"prims={physics_rebind_summary['convex_decomposition_prim_count']} "
                f"voxel_resolution={convex_decomp_settings['voxel_resolution']} "
                f"max_convex_hulls={convex_decomp_settings['max_convex_hulls']} "
                f"error_percentage={convex_decomp_settings['error_percentage']} "
                f"shrink_wrap={convex_decomp_settings['shrink_wrap']}"
            )
        print(
            "[INFO] Scene floor realign: "
            f"visual_roots={floor_realign_summary['visual_roots']} "
            f"grounded_roots={floor_realign_summary['grounded_roots']}"
        )
        print(f"[INFO] Dataset: {os.path.abspath(writer.dataset_file)}")
        if collision_overlay.enabled:
            print(
                "[INFO] Collision overlay enabled: "
                "viewport_colliders=All solid_collision_mesh=True collision_mesh_type=both"
            )

        sim_time = 0.0
        while simulation_app.is_running():
            if ui.consume_start_request():
                writer.start_recording(sim_time)

            if ui.consume_switch_arm_request():
                next_side = controller.switch_arm_side()
                ui.set_title(f"{controller.current_window_title()} Scene Collect")
                print(f"[INFO] Switched active arm to: {next_side}")
                if sync_cameras is not None:
                    sync_cameras()

            delta_cmd = ui.consume_delta(sim.device)
            controller.step_click_delta(delta_cmd, ui.gripper_closed)
            scene.write_data_to_sim()
            sim.step()
            sim_time += sim.get_physics_dt()
            scene.update(sim.get_physics_dt())

            if sync_cameras is not None:
                sync_cameras()

            action = torch.cat(
                [
                    delta_cmd.detach().cpu().to(dtype=torch.float32),
                    torch.tensor([1.0 if ui.gripper_closed else 0.0], dtype=torch.float32),
                ],
                dim=0,
            )
            writer.maybe_record_frame(sim_time, action, controller, cameras)

            if ui.consume_stop_save_request():
                writer.stop_and_save()
                _reset_scene_to_plan(
                    scene,
                    controller,
                    plan,
                    _planned_base_height(robot_name) + effective_base_z_bias,
                    sync_cameras,
                )
                collision_overlay.refresh()

            if ui.consume_stop_discard_request():
                writer.stop_and_discard()
                _reset_scene_to_plan(
                    scene,
                    controller,
                    plan,
                    _planned_base_height(robot_name) + effective_base_z_bias,
                    sync_cameras,
                )
                collision_overlay.refresh()

            if ui.consume_reset_request():
                _reset_scene_to_plan(
                    scene,
                    controller,
                    plan,
                    _planned_base_height(robot_name) + effective_base_z_bias,
                    sync_cameras,
                )
                collision_overlay.refresh()

        if writer.recording and writer.frame_count > 0:
            print(
                f"[INFO] Exiting with an active unsaved recording containing {writer.frame_count} frames. "
                "Use Stop + Save before closing if you want to keep it."
            )
    finally:
        if writer is not None:
            writer.close()
        collision_overlay.close()
