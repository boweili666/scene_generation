"""HDF5 episode writer for scene auto grasp collection.

`SceneAutoGraspEpisodeWriter` extends the generic teleop episode writer
with auto-grasp-specific per-frame columns: `obs/phase_id`, the selected
grasp's `arm_side` / `score`, and the planned grasp / pre-grasp / lift /
retreat poses (so the policy can see what waypoint each frame was
chasing). `_recover_corrupt_hdf5_file` deletes a truncated dataset file
left by a Ctrl-C'd run, so `--append` doesn't choke on it.

`scene_auto_grasp_collect.py` re-exports these names for backward
compatibility with existing imports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .episode_writer import SceneTeleopEpisodeWriter


PHASE_NAME_TO_ID = {
    "pre_grasp": 0,
    "approach": 1,
    "close": 2,
    "lift": 3,
    "retreat": 4,
}


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
