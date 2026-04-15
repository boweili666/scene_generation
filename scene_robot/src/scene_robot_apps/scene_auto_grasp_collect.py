from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import subtract_frame_transforms

from app.backend.services.grasp_asset_cache import (
    cache_entry_path_for_prim,
    default_grasp_annotation_root,
    ensure_asset_grasp_cache_for_prim,
)
from app.backend.services.grasp_scene_adapter import build_stage_grasp_proposals, default_scene_grasp_proposals_path
from .ee_frame_remap import (
    apply_inverse_local_ee_frame_remap_to_world_quat,
    apply_local_ee_frame_remap_to_world_quat,
    apply_local_translation_to_world_pos,
    quat_wxyz_to_matrix,
)
from .grasp_execution import (
    FilteredGraspExecution,
    expand_grasp_candidates,
    filter_grasp_candidates_geometry,
    infer_arm_side,
    pose_error_metrics,
    rank_filtered_grasp_candidates_by_start_pose,
)
from .grasp_visualization import add_grasp_candidates_visuals_to_stage
from .grasp_visualization import add_pose_frames_to_stage
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
EE_MARKER_DEBUG_TRANSLATION_LOCAL = (0.12, 0.0, 0.0)
PHASE_NAME_TO_ID = {
    "pre_grasp": 0,
    "approach": 1,
    "close": 2,
    "lift": 3,
    "retreat": 4,
}


@dataclass(frozen=True)
# =============================================================================
# Configuration
# =============================================================================
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
    manifest_path: str | None
    annotation_root: str | None
    scene_grasp_proposals_path: str | None
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
    lazy_build_target_annotation: bool
    show_grasp_poses: bool
    wait_for_run_request: bool
    agibot_ee_frame_remap: str
    num_episodes: int = 1
    # Distance (in metres, along +approach_axis_world) from the controller EE
    # frame origin (wrist) to the fingertip the annotator's grasp point should
    # land at. When non-zero, every phase's world-space target is shifted
    # backwards along approach so that after IK settles, the fingertip reaches
    # the original grasp point instead of the wrist.
    fingertip_distance: float = 0.0
    # Straight-line interpolation speeds used inside `_run_target_phase`. When
    # `phase_linear_speed > 0` (m/s), the commanded target ramps from the
    # phase's starting EE pose toward the final target at this linear speed
    # instead of jumping directly to the final pose. Same for angular speed
    # (deg/s). Both zero preserves the original "command final every step"
    # behaviour that relies on the controller's own velocity limits.
    phase_linear_speed: float = 0.0
    phase_angular_speed_deg: float = 0.0


# =============================================================================
# HDF5 episode writer (+ corruption recovery)
# =============================================================================
def _recover_corrupt_hdf5_file(dataset_file: str, append: bool) -> bool:
    # `h5py.File(path, mode="a")` raises OSError on truncated/corrupt files
    # (typical after a Ctrl-C mid-write). Detect that case up front and delete
    # the dead file so the writer can create a fresh one. Returns True when a
    # corrupt file was removed.
    path = Path(dataset_file)
    if not path.exists():
        return False
    if not append:
        # Non-append mode will overwrite anyway; no cleanup needed.
        return False
    try:
        import h5py

        with h5py.File(str(path), "r"):
            return False  # file is readable, nothing to do
    except OSError as exc:
        print(f"[WARN] Dataset file {path} is unreadable ({exc}); removing to start fresh.")
        try:
            path.unlink()
        except OSError as unlink_exc:
            raise RuntimeError(f"Failed to remove corrupt dataset file {path}: {unlink_exc}") from unlink_exc
        return True


class SceneAutoGraspEpisodeWriter(SceneTeleopEpisodeWriter):
    def __init__(
        self,
        dataset_file: str,
        capture_hz: float,
        append: bool,
        env_name: str,
        camera_aliases: dict[str, dict[str, object]],
        plan,
        scene_usd_path: str,
        scene_graph_path: str,
        placements_path: str,
        *,
        initial_arm_side: str = "left",
        arm_switch_supported: bool = False,
    ):
        _recover_corrupt_hdf5_file(dataset_file, append)
        super().__init__(
            dataset_file,
            capture_hz,
            append,
            env_name,
            camera_aliases,
            plan,
            scene_usd_path,
            scene_graph_path,
            placements_path,
            initial_arm_side=initial_arm_side,
            arm_switch_supported=arm_switch_supported,
        )
        self._selected_grasp_payload: dict[str, Any] | None = None
        self.file_handler.add_env_args(
            {
                "autonomous_grasp": {
                    "phase_name_to_id": PHASE_NAME_TO_ID,
                }
            }
        )

    def set_selected_grasp(self, payload: dict[str, Any]) -> None:
        self._selected_grasp_payload = json.loads(json.dumps(payload))

    def maybe_record_auto_frame(
        self,
        sim_time: float,
        action: torch.Tensor,
        controller,
        cameras: dict[str, object],
        *,
        phase_name: str,
    ) -> bool:
        if not self.recording:
            return False
        if self.episode_start_time is None:
            self.episode_start_time = sim_time
            self.next_capture_time = sim_time
        if self.frame_count > 0 and sim_time + 1.0e-9 < self.next_capture_time:
            return False
        while self.next_capture_time <= sim_time + 1.0e-9:
            self.next_capture_time += self.capture_period
        self._record_auto_frame(sim_time, action, controller, cameras, phase_name=phase_name)
        self.frame_count += 1
        return True

    def _record_auto_frame(
        self,
        sim_time: float,
        action: torch.Tensor,
        controller,
        cameras: dict[str, object],
        *,
        phase_name: str,
    ) -> None:
        super()._record_frame(sim_time, action, controller, cameras)
        self.episode.add("obs/phase_id", torch.tensor(int(PHASE_NAME_TO_ID[phase_name]), dtype=torch.int64))
        selection = self._selected_grasp_payload or {}
        if selection:
            self.episode.add(
                "obs/selected_arm_side",
                torch.tensor(1 if str(selection.get("arm_side")) == "right" else 0, dtype=torch.int64),
            )
            self.episode.add(
                "obs/selected_grasp_score",
                torch.tensor(float(selection.get("ranking_score", selection.get("score", 0.0))), dtype=torch.float32),
            )
            grasp_payload = selection.get("grasp", {})
            grasp_pos = grasp_payload.get("position_world")
            if isinstance(grasp_pos, (list, tuple)):
                self.episode.add("obs/grasp_pos_world", torch.tensor(grasp_pos, dtype=torch.float32))
            grasp_quat = grasp_payload.get("quat_wxyz_world")
            if isinstance(grasp_quat, (list, tuple)):
                self.episode.add("obs/grasp_quat_world", torch.tensor(grasp_quat, dtype=torch.float32))
            for key in (
                "pre_grasp_pos_world",
                "pre_grasp_quat_world",
                "lift_pos_world",
                "lift_quat_world",
                "retreat_pos_world",
                "retreat_quat_world",
            ):
                value = selection.get(key)
                if isinstance(value, (list, tuple)):
                    self.episode.add(f"obs/{key}", torch.tensor(value, dtype=torch.float32))


# =============================================================================
# Preview UI (Run / Close gate)
# =============================================================================
class AutoGraspPreviewUI:
    _WINDOW_WIDTH = 360
    _WINDOW_HEIGHT = 170
    _BUTTON_WIDTH = 160
    _BUTTON_HEIGHT = 56

    def __init__(self, title: str):
        import omni.ui as ui

        self._run_requested = False
        self._close_requested = False
        self._window = ui.Window(title, width=self._WINDOW_WIDTH, height=self._WINDOW_HEIGHT)
        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("All grasp candidates are previewed.", height=20)
                ui.Label("Click Run to clear previews and execute.", height=20)
                with ui.HStack(spacing=8):
                    ui.Button(
                        "Run Selected Grasp",
                        clicked_fn=self._request_run,
                        width=self._BUTTON_WIDTH,
                        height=self._BUTTON_HEIGHT,
                    )
                    ui.Button(
                        "Close Preview",
                        clicked_fn=self._request_close,
                        width=self._BUTTON_WIDTH,
                        height=self._BUTTON_HEIGHT,
                    )

    def _request_run(self) -> None:
        self._run_requested = True

    def _request_close(self) -> None:
        self._close_requested = True

    def consume_run_request(self) -> bool:
        requested = self._run_requested
        self._run_requested = False
        return requested

    def consume_close_request(self) -> bool:
        requested = self._close_requested
        self._close_requested = False
        return requested

    def close(self) -> None:
        try:
            self._window.visible = False
        except Exception:
            pass


# =============================================================================
# JSON / path helpers
# =============================================================================
def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_adjacent_path(scene_usd_path: Path, relative_parts: tuple[str, ...]) -> Path | None:
    for parent in (scene_usd_path.parent, *scene_usd_path.parents):
        candidate = parent.joinpath(*relative_parts)
        if candidate.exists():
            return candidate.resolve()
    return None


def _auto_manifest_path(scene_usd_path: str | Path) -> Path | None:
    return _resolve_adjacent_path(Path(scene_usd_path).resolve(), ("real2sim", "scene_results", "real2sim_asset_manifest.json"))


def _selection_output_path(args: SceneAutoGraspCollectArgs) -> Path:
    return Path(args.plan_output_dir).resolve() / DEFAULT_SELECTION_FILENAME


def _shortlist_output_path(args: SceneAutoGraspCollectArgs) -> Path:
    return Path(args.plan_output_dir).resolve() / DEFAULT_SHORTLIST_FILENAME


def _resolve_manifest_and_annotation_root(args: SceneAutoGraspCollectArgs) -> tuple[Path, Path]:
    scene_usd_path = Path(args.scene_usd_path).resolve()
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else _auto_manifest_path(scene_usd_path)
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(
            "No scene grasp proposals found and no Real2Sim manifest could be resolved. "
            "Run build_grasp_asset_cache.py / export_scene_grasp_proposals.py first or pass --manifest-path."
        )
    annotation_root = Path(args.annotation_root).resolve() if args.annotation_root else default_grasp_annotation_root(manifest_path)
    return manifest_path, annotation_root


# =============================================================================
# Scene-grasp proposals I/O (annotation cache → world-space grasp payload)
# =============================================================================
def _maybe_lazy_build_target_annotation(
    args: SceneAutoGraspCollectArgs,
    manifest_path: Path,
    annotation_root: Path,
    *,
    target_prim: str | None,
) -> bool:
    if not args.lazy_build_target_annotation or not target_prim:
        return False
    target_cache_path = cache_entry_path_for_prim(manifest_path, target_prim, output_root=annotation_root)
    if target_cache_path is None:
        raise ValueError(f"Target prim '{target_prim}' was not found in manifest {manifest_path}.")
    if target_cache_path.exists():
        return False
    print(f"[INFO] Missing grasp annotation cache for target {target_prim}; building it now.")
    ensure_asset_grasp_cache_for_prim(
        manifest_path,
        target_prim,
        output_root=annotation_root,
        resume=True,
    )
    return True


def _load_or_build_scene_grasp_payload(
    stage,
    args: SceneAutoGraspCollectArgs,
    *,
    target_prim: str | None = None,
) -> tuple[dict[str, Any], Path]:
    scene_usd_path = Path(args.scene_usd_path).resolve()
    proposals_path = (
        Path(args.scene_grasp_proposals_path).resolve()
        if args.scene_grasp_proposals_path
        else default_scene_grasp_proposals_path(scene_usd_path)
    )
    manifest_path, annotation_root = _resolve_manifest_and_annotation_root(args)
    _maybe_lazy_build_target_annotation(
        args,
        manifest_path,
        annotation_root,
        target_prim=target_prim,
    )
    payload = build_stage_grasp_proposals(
        stage,
        manifest_path,
        annotation_root=annotation_root,
        output_path=proposals_path,
        target_prim_paths=None if target_prim is None else [target_prim],
    )
    objects = payload.get("objects", {})
    if target_prim is not None and (not isinstance(objects, dict) or target_prim not in objects):
        target_cache_path = cache_entry_path_for_prim(manifest_path, target_prim, output_root=annotation_root)
        missing_cache_note = ""
        if target_cache_path is not None and not target_cache_path.exists():
            missing_cache_note = f" Missing target cache: {target_cache_path}."
        raise RuntimeError(
            f"Failed to resolve scene grasp proposals for target '{target_prim}'."
            f"{missing_cache_note}"
        )
    return payload, proposals_path


def _world_bbox_payload(stage, prim_path: str) -> dict[str, list[float]] | None:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "guide", "render"], useExtentsHint=True)
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if aligned.IsEmpty():
        return None
    bmin = aligned.GetMin()
    bmax = aligned.GetMax()
    min_xyz = [float(bmin[0]), float(bmin[1]), float(bmin[2])]
    max_xyz = [float(bmax[0]), float(bmax[1]), float(bmax[2])]
    return {
        "min": min_xyz,
        "max": max_xyz,
        "center": [
            float((min_xyz[0] + max_xyz[0]) * 0.5),
            float((min_xyz[1] + max_xyz[1]) * 0.5),
            float((min_xyz[2] + max_xyz[2]) * 0.5),
        ],
        "size": [
            float(max_xyz[0] - min_xyz[0]),
            float(max_xyz[1] - min_xyz[1]),
            float(max_xyz[2] - min_xyz[2]),
        ],
    }


# =============================================================================
# Pose math + rigid-body physx view snapshot/restore
# =============================================================================
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
    import torch

    prim_path = snapshot["prim_path"]
    view = _get_rigid_body_view(prim_path)
    if view is None:
        return
    pose = snapshot["transforms"]
    view.set_transforms(pose, indices=torch.arange(pose.shape[0], device=pose.device))
    velocities = torch.zeros((pose.shape[0], 6), device=pose.device, dtype=pose.dtype)
    view.set_velocities(velocities, indices=torch.arange(pose.shape[0], device=pose.device))


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


def _world_pose_to_base(controller, position_world: tuple[float, float, float], quat_wxyz_world: tuple[float, float, float, float]) -> tuple[torch.Tensor, torch.Tensor]:
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


# =============================================================================
# Phase runners (target-tracking + hold)
# =============================================================================
def _run_target_phase(
    scene,
    controller,
    cameras: dict[str, object],
    sync_cameras: Callable[[], None] | None,
    writer: SceneAutoGraspEpisodeWriter | None,
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
    writer: SceneAutoGraspEpisodeWriter | None,
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


# =============================================================================
# Candidate selection: geometric filter → EE-pose ranking → top pick
# =============================================================================
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
    args: SceneAutoGraspCollectArgs,
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
    args: SceneAutoGraspCollectArgs,
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
    args: SceneAutoGraspCollectArgs,
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
    _write_json(
        _shortlist_output_path(args),
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
                    if robot_name == "agibot" and str(args.agibot_ee_frame_remap or "none").strip().lower() not in {"", "none"}
                    else "raw_current_ee_vs_pre_grasp_orientation"
                ),
            },
            "candidates": shortlist_payload,
        },
    )

    if not ranked:
        raise RuntimeError(f"No grasp candidates survived geometric filtering for {plan.target_prim}")
    return ranked


# =============================================================================
# One-episode rollout: reset → pre_grasp → approach → close → lift → retreat
# =============================================================================
def _attempt_recorded_grasp(
    scene,
    controller,
    cameras: dict[str, object],
    sync_cameras: Callable[[], None] | None,
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
# Visuals / preview refresh helpers
# =============================================================================
def _clear_grasp_visuals(stage, *, root_prim_path: str = "/Visuals/AutoGraspVisuals") -> None:
    stage.RemovePrim(root_prim_path)
    if root_prim_path != "/Visuals/AutoGraspPreviewFrames":
        stage.RemovePrim("/Visuals/AutoGraspPreviewFrames")


def _wait_for_run_request(
    simulation_app,
    scene,
    controller,
    sync_cameras: Callable[[], None] | None,
    *,
    title: str,
    refresh_preview: Callable[[], None] | None = None,
) -> bool:
    # Blocks while the preview UI is shown. Returns True if the user clicked
    # "Run Selected Grasp", False if they clicked "Close Preview" (or closed
    # the app). `refresh_preview` is invoked each tick so live grasp visuals
    # can follow the target if physics moves it.
    ui = AutoGraspPreviewUI(title)
    try:
        while simulation_app.is_running():
            if ui.consume_run_request():
                return True
            if ui.consume_close_request():
                return False
            if refresh_preview is not None:
                refresh_preview()
            scene.write_data_to_sim()
            controller.sim.step()
            scene.update(controller.sim.get_physics_dt())
            if sync_cameras is not None:
                sync_cameras()
        return False
    finally:
        ui.close()


def _refresh_preview_markers(
    stage,
    controller,
    selected_candidate: FilteredGraspExecution | None,
    *,
    robot_name: str = "agibot",
    ee_frame_remap: str = "none",
) -> None:
    if selected_candidate is not None and controller.arm_switch_supported:
        controller.switch_arm_side(selected_candidate.arm_side)
    ee_pos_w, ee_quat_w = controller.current_ee_pose_world()
    display_ee_pos = tuple(float(v) for v in ee_pos_w[0].detach().cpu().tolist())
    display_ee_quat = tuple(float(v) for v in ee_quat_w[0].detach().cpu().tolist())
    if robot_name == "agibot" and str(ee_frame_remap or "none").strip().lower() not in {"", "none"}:
        display_ee_quat = apply_local_ee_frame_remap_to_world_quat(display_ee_quat, ee_frame_remap)
        display_ee_pos = apply_local_translation_to_world_pos(
            display_ee_pos,
            display_ee_quat,
            EE_MARKER_DEBUG_TRANSLATION_LOCAL,
        )
    pose_frames = [
        {
            "name": "CurrentGripper",
            "position_world": display_ee_pos,
            "quat_wxyz_world": display_ee_quat,
            "axis_length": 0.18,
            "axis_thickness": 0.012,
            "opacity": 0.92,
        }
    ]
    if selected_candidate is not None:
        pose_frames.append(
            {
                "name": "GraspTarget",
                "position_world": selected_candidate.grasp.position_world,
                "quat_wxyz_world": selected_candidate.grasp.quat_wxyz_world,
                "axis_length": 0.18,
                "axis_thickness": 0.012,
                "opacity": 0.92,
                # Re-parent the GraspTarget under the target object's prim so the
                # marker tracks the visual mesh via the USD hierarchy regardless
                # of runtime physics / scale handling. The gripper frame stays in
                # world coordinates because it follows the robot, not the object.
                "parent_prim_path": selected_candidate.grasp.object_prim,
            }
        )
    add_pose_frames_to_stage(
        stage,
        root_prim_path="/Visuals/AutoGraspPreviewFrames",
        pose_frames=pose_frames,
    )
    controller.ee_marker.set_visibility(False)
    controller.goal_marker.set_visibility(False)


def _refresh_live_target_grasp_payload(
    scene,
    controller,
    sync_cameras: Callable[[], None] | None,
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
    from dataclasses import replace

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
    # Optionally draws the selected grasp's axis marker + the robot's current
    # gripper frame, then blocks on the `Run Selected Grasp` / `Close Preview`
    # UI until the user decides. Returns True on Run, False on Close.
    preview_state = {"candidate": candidate}

    def _refresh_preview() -> None:
        if not args.show_grasp_poses:
            return
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

    if args.show_grasp_poses:
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
        refresh_preview=_refresh_preview if args.show_grasp_poses else None,
    )
    if not should_run:
        print("[INFO] Auto grasp preview closed before execution.")
        return False
    _clear_grasp_visuals(scene.stage)
    return True


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
    for episode_idx in range(num_episodes):
        print(f"[INFO] === Episode {episode_idx + 1}/{num_episodes} ===")
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
                candidate,
                selection_payload,
                args,
                target_state_snapshot=target_state_snapshot,
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
