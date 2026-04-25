"""Preview UI + visual cleanup for the Run/Close gate.

When `--wait-for-run-request` is on, auto grasp collection blocks before
running the rollout so a human can verify the planned grasp. This
module owns:

* `AutoGraspPreviewUI` — the small Omniverse Kit window with the two
  buttons.
* `_clear_grasp_visuals` — strip every preview marker prim from the
  stage, including re-parented copies under the target's prim subtree
  (the live grasp visuals are parented to the target so they track the
  visual mesh through physics; that's why a shallow `RemovePrim` at
  stage root misses them).
* `_wait_for_run_request` — main-loop blocker that ticks the sim while
  showing the preview, calling a refresh callback every step so the
  markers can follow the target if it settles under physics.
* `_refresh_preview_markers` — rebuild the gripper-frame + grasp-target
  axis markers each tick.

`scene_auto_grasp_collect.py` re-exports these names so existing
imports keep working.
"""

from __future__ import annotations

from typing import Callable

from .ee_frame_remap import apply_local_ee_frame_remap_to_world_quat, apply_local_translation_to_world_pos
from .grasp_execution import FilteredGraspExecution
from .grasp_visualization import add_pose_frames_to_stage


EE_MARKER_DEBUG_TRANSLATION_LOCAL = (0.12, 0.0, 0.0)
_PREVIEW_LEAF_NAMES = ("AutoGraspVisuals", "AutoGraspPreviewFrames")


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


def _clear_grasp_visuals(stage, *, root_prim_path: str | None = None) -> None:
    # Remove *all* copies of the grasp-preview prim subtrees, including the
    # re-parented ones under `/World/envs/env_N/GeneratedScene/<target>/...`
    # that get created when `parent_prim_path` is used in
    # `add_pose_frames_to_stage` / `add_grasp_candidates_visuals_to_stage`.
    # The old implementation only removed paths at stage root, missing the
    # re-parented copies so the preview markers persisted after Run.
    leaf_names = set(_PREVIEW_LEAF_NAMES)
    if root_prim_path:
        leaf_names.add(root_prim_path.rsplit("/", 1)[-1])
    to_remove: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        if prim.GetName() in leaf_names:
            to_remove.append(str(prim.GetPath()))
    # Remove from longest-first so children are gone before parents.
    for path in sorted(set(to_remove), key=len, reverse=True):
        stage.RemovePrim(path)


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
