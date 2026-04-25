"""Mouse-driven scene-based teleop data collection.

This used to be a single 1342-line module that bundled the args, the
Omni UI, the HDF5 episode writer, every physics-rebind helper, the
scene config builder, and the orchestrator. Most of that lives in
focused modules now:

* `episode_writer.SceneTeleopEpisodeWriter` — HDF5 writer.
* `mouse_collect_ui.MouseCommandCollectUI` — Omni UI window.
* `scene_physics.*` — collision / BBox / rigid-body / physics-rebind /
  reset / settle helpers + `ConvexDecompositionSettings`.

What stays here:

* `SceneMouseCollectArgs` — CLI args dataclass.
* `_resolve_convex_decomposition_settings(args)` — args→settings adapter
  (keeps the args dependency local; `ConvexDecompositionSettings` itself
  lives in `scene_physics`).
* Spec helpers: `_planned_base_height`, `_plan_camera_pose`,
  `_make_dummy_cube_specs`, `_make_spec_for_scene_collect`.
* `_build_scene_collect_cfg` — `InteractiveSceneCfg` factory (cameras +
  robot + cubes + generated scene reference).
* `_build_scene_mouse_collect` — full scene boot (planner → InteractiveScene
  → controller → physics rebind → camera mount).
* `run_scene_mouse_collect` — top-level orchestrator (event loop +
  episode lifecycle).

Names that used to be defined here are re-exported below so existing
`from .scene_mouse_collect import _reset_scene_to_plan` (etc.) callers
keep working unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms

from app.backend.config.settings import DEFAULT_PLACEMENTS_PATH, SCENE_GRAPH_PATH  # noqa: F401  (re-exported)
from app.backend.services.robot_placement import (
    DEFAULT_OUTPUT_DIR,  # noqa: F401  (re-exported)
    RobotPlacementPlan,
    load_scene_state,
    plan_robot_base_pose,
    save_plan_outputs,
)
from app.backend.services.robot_scene import compute_robot_floor_offset_z, resolve_robot_asset_path

from ..record.episode_writer import SceneTeleopEpisodeWriter
from ..ui.mouse_collect_ui import MouseCommandCollectUI
from .mouse_teleop_record import (
    _AGIBOT_CAMERA_SPECS,
    _AGIBOT_MOUNT_POSES,
    _R1LITE_CAMERA_SPECS,
    _make_pinhole_camera_cfg,
)
from ..control.robot_controller import RobotController
from ..scene.physics import (
    ConvexDecompositionSettings,
    _align_robot_root_to_floor,
    _apply_root_pose,
    _clear_rigid_body_view_cache,
    _compute_bounds_for_prims,
    _compute_world_prim_max_z,
    _compute_world_prim_min_z,
    _find_live_physics_scene_path,
    _get_rigid_body_view,
    _iter_collision_prims,
    _iter_visual_collision_prims,
    _prepare_scene_usd_without_physics_scenes,
    _read_rigid_body_world_position_z,
    _realign_generated_scene_floor_objects,
    _rebind_generated_scene_physics,
    _remove_nested_physics_scenes,
    _reset_scene_to_plan,
    _resolve_collision_approx,
    _settle_dynamic_scene,
    _supported_scene_object_paths,
    _yaw_quat_wxyz,
)
from ..control.robot_spec import CuboidSpec, ROBOT_SPECS, resolve_robot_spec
from ..control.scene_cfg import _make_cube_cfg


DEFAULT_SCENE_USD_PATH = Path(__file__).resolve().parents[3] / "runtime" / "scene_service" / "usd" / "scene_latest.usd"


# Backward-compat re-exports: every symbol below used to be defined in this
# file and is imported from here by sibling modules / scripts. Keep this
# namespace stable so the move into `scene_physics` / `episode_writer` /
# `mouse_collect_ui` is invisible to external callers.
__all__ = [
    "ConvexDecompositionSettings",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PLACEMENTS_PATH",
    "DEFAULT_SCENE_USD_PATH",
    "MouseCommandCollectUI",
    "ROBOT_BASE_Z_BIAS",
    "RobotController",
    "SCENE_GRAPH_PATH",
    "SceneMouseCollectArgs",
    "SceneTeleopEpisodeWriter",
    "_align_robot_root_to_floor",
    "_apply_root_pose",
    "_build_scene_collect_cfg",
    "_build_scene_mouse_collect",
    "_clear_rigid_body_view_cache",
    "_compute_bounds_for_prims",
    "_compute_world_prim_max_z",
    "_compute_world_prim_min_z",
    "_find_live_physics_scene_path",
    "_get_rigid_body_view",
    "_iter_collision_prims",
    "_iter_visual_collision_prims",
    "_make_dummy_cube_specs",
    "_make_spec_for_scene_collect",
    "_plan_camera_pose",
    "_planned_base_height",
    "_prepare_scene_usd_without_physics_scenes",
    "_read_rigid_body_world_position_z",
    "_realign_generated_scene_floor_objects",
    "_rebind_generated_scene_physics",
    "_remove_nested_physics_scenes",
    "_reset_scene_to_plan",
    "_resolve_collision_approx",
    "_resolve_convex_decomposition_settings",
    "_settle_dynamic_scene",
    "_supported_scene_object_paths",
    "_yaw_quat_wxyz",
    "run_scene_mouse_collect",
]


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
    show_workspace: bool


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


# =============================================================================
# Per-robot static spec helpers (base height, planner camera, dummy cubes)
# =============================================================================
def _planned_base_height(robot_name: str) -> float:
    try:
        return float(compute_robot_floor_offset_z(robot_name))
    except Exception:
        spec = ROBOT_SPECS[robot_name]
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
    spec = resolve_robot_spec(robot_name, arm_side)
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


# =============================================================================
# InteractiveScene cfg + builder
# =============================================================================
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

    # World camera is intentionally omitted: scene auto grasp collection
    # only records head/left_hand/right_hand onboard cameras, so spawning
    # an additional render sensor would just waste GPU time each step.

    return configclass(type("SceneCollectCfg", (InteractiveSceneCfg,), attrs))


def _build_scene_mouse_collect(
    sim: sim_utils.SimulationContext,
    robot_name: str,
    args: SceneMouseCollectArgs,
):
    from ..scene.workspaces import add_projected_workspace_visual_to_stage, project_workspace_box_to_support

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
    workspace_visual_summary = None
    scene_root_path = f"{scene.env_prim_paths[0]}/GeneratedScene"
    removed_physics_scenes = _remove_nested_physics_scenes(scene.stage, scene_root_path)
    if removed_physics_scenes:
        print(f"[INFO] Removed nested physics scenes from generated scene: {removed_physics_scenes}")
    sim.reset()
    controller = RobotController(sim, scene, spec)
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
    if args.show_workspace:
        support_live_prim_path = f"{scene_root_path}/{Path(plan.support_prim).name}"
        support_top_z = _compute_world_prim_max_z(scene.stage, support_live_prim_path)
        projected_workspace = project_workspace_box_to_support(
            robot=robot_name,
            base_pose=plan.base_pose,
            support_center_xy=plan.support_center_xy,
            support_half_extents_xy=plan.support_half_extents_xy,
            support_yaw_deg=plan.support_yaw_deg,
        )
        if support_top_z is not None and projected_workspace is not None:
            workspace_visual_summary = add_projected_workspace_visual_to_stage(
                scene.stage,
                robot=robot_name,
                root_prim_path=f"{scene_root_path}/TaskWorkspaceVisuals",
                projected_workspace=projected_workspace,
                top_z=support_top_z,
            )
    sim.reset()

    # World camera is intentionally not recorded anymore: episodes only
    # capture the robot's own three cameras (head, left_hand, right_hand).
    cameras: dict = {}
    camera_aliases: dict = {}
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
        workspace_visual_summary,
    )


# =============================================================================
# Top-level orchestrator
# =============================================================================
def run_scene_mouse_collect(simulation_app, robot_name: str, args: SceneMouseCollectArgs) -> None:
    if args.num_envs != 1:
        raise ValueError("Scene mouse collection only supports --num_envs 1.")

    spec = resolve_robot_spec(robot_name, args.arm_side)
    env_name = f"Isaac-{robot_name.capitalize()}-SceneMouseCollect-v0"
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
    sim.set_camera_view(spec.camera_eye, spec.camera_target)
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
            workspace_visual_summary,
        ) = _build_scene_mouse_collect(
            sim,
            robot_name,
            args,
        )
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
            f"object_colliders={physics_rebind_summary['object_colliders']}"
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
        if workspace_visual_summary is not None:
            print(f"[INFO] Workspace visuals: {workspace_visual_summary['workspace_root_path']}")
        print(f"[INFO] Dataset: {os.path.abspath(writer.dataset_file)}")

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

            if ui.consume_stop_discard_request():
                writer.stop_and_discard()
                _reset_scene_to_plan(
                    scene,
                    controller,
                    plan,
                    _planned_base_height(robot_name) + effective_base_z_bias,
                    sync_cameras,
                )

            if ui.consume_reset_request():
                _reset_scene_to_plan(
                    scene,
                    controller,
                    plan,
                    _planned_base_height(robot_name) + effective_base_z_bias,
                    sync_cameras,
                )

        if writer.recording and writer.frame_count > 0:
            print(
                f"[INFO] Exiting with an active unsaved recording containing {writer.frame_count} frames. "
                "Use Stop + Save before closing if you want to keep it."
            )
    finally:
        if writer is not None:
            writer.close()
