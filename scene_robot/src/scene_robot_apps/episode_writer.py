"""HDF5 episode writer for scene-mouse / scene-auto-grasp pipelines.

`SceneTeleopEpisodeWriter` wraps Isaac Lab's `HDF5DatasetFileHandler` /
`EpisodeData` with the dataset layout this repo standardises on:

    actions, timestamps,
    obs/{joint_pos, joint_vel, root_pose,
         ee_pos, ee_quat, target_ee_pos, target_ee_quat,
         gripper_joint_pos, active_arm_side},
    obs/<camera_name> for every camera dict entry passed in.

Originally lived inside `scene_mouse_collect.py`; moved here so the
mouse-collect orchestrator stays focused on scene assembly + episode
loop control. `scene_mouse_collect.py` re-exports `SceneTeleopEpisodeWriter`
so all existing `from .scene_mouse_collect import SceneTeleopEpisodeWriter`
callers (e.g. scene_auto_grasp_collect) keep working unchanged.
"""

from __future__ import annotations

import json
import os

import torch

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

from app.backend.services.robot_placement import RobotPlacementPlan, plan_to_payload

from .robot_controller import RobotController


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

    def maybe_record_frame(self, sim_time: float, action: torch.Tensor, controller: RobotController, cameras: dict[str, object]):
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

    def _record_frame(self, sim_time: float, action: torch.Tensor, controller: RobotController, cameras: dict[str, object]):
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
