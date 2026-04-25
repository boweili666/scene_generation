from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Callable

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab.utils.math import combine_frame_transforms

from .robot_controller import RobotController
from .stack_cube import build_single_robot_scene_cfg, resolve_stack_spec


@dataclass(frozen=True)
class MouseTeleopRecordArgs:
    device: str
    num_envs: int
    dataset_file: str
    capture_hz: float
    append: bool
    lin_step: float
    ang_step: float
    arm_side: str


class MouseRecordTeleopUI:
    _WINDOW_WIDTH = 420
    _WINDOW_HEIGHT = 360
    _POSE_BUTTON_WIDTH = 56
    _POSE_BUTTON_HEIGHT = 48
    _ACTION_BUTTON_WIDTH = 98
    _ACTION_BUTTON_HEIGHT = 38
    _EPISODE_BUTTON_WIDTH = 122
    _EPISODE_BUTTON_HEIGHT = 38

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
        self._save_requested = False
        self._discard_requested = False
        self._switch_arm_requested = False
        self._window = ui.Window(title, width=self._WINDOW_WIDTH, height=self._WINDOW_HEIGHT)
        self._apply_window_size()
        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Click to move one step (xyzrpy)", height=20)
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
                    if allow_arm_switch:
                        self._make_action_button(ui, "Switch Arm", self._request_switch_arm)
                with ui.HStack(spacing=6):
                    self._make_episode_button(ui, "Save Episode", self._request_save)
                    self._make_episode_button(ui, "Discard Episode", self._request_discard)

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

    def _request_save(self):
        self._save_requested = True

    def _request_discard(self):
        self._discard_requested = True

    def _request_switch_arm(self):
        self._switch_arm_requested = True

    def consume_delta(self, device: str) -> torch.Tensor:
        delta = torch.tensor(self._pending_delta, device=device, dtype=torch.float32)
        self._pending_delta = [0.0] * 6
        return delta

    def consume_save_request(self) -> bool:
        requested = self._save_requested
        self._save_requested = False
        return requested

    def consume_discard_request(self) -> bool:
        requested = self._discard_requested
        self._discard_requested = False
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


class MouseTeleopEpisodeWriter:
    def __init__(
        self,
        dataset_file: str,
        capture_hz: float,
        append: bool,
        env_name: str,
        camera_aliases: dict[str, dict[str, object]],
        *,
        initial_arm_side: str = "left",
        arm_switch_supported: bool = False,
    ):
        self.dataset_file = self._normalize_dataset_file(dataset_file)
        self.capture_period = 1.0 / max(capture_hz, 1.0e-6)
        self.file_handler = HDF5DatasetFileHandler()

        if append and os.path.exists(self.dataset_file):
            self.file_handler.open(self.dataset_file, mode="a")
        else:
            self.file_handler.create(self.dataset_file, env_name=env_name)

        self.file_handler.add_env_args(
            {
                "camera_aliases": camera_aliases,
                "capture_hz": capture_hz,
                "teleop_ui": {
                    "type": "mouse_click",
                    "pose_step_buttons": ["+X", "-X", "+Y", "-Y", "+Z", "-Z", "+R", "-R", "+P", "-P", "+Yaw", "-Yaw"],
                    "gripper_toggle": "Gripper Toggle",
                    "switch_arm": "Switch Arm" if arm_switch_supported else None,
                    "save_episode": "Save Episode",
                    "discard_episode": "Discard Episode",
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

    def discard_episode(self):
        discarded_frames = self.frame_count
        self.reset_episode()
        print(f"[INFO] Discarded current episode ({discarded_frames} recorded frames).")

    def save_episode(self):
        if self.frame_count == 0:
            print("[WARN] Current episode is empty. Nothing was saved.")
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
        self.reset_episode()

    def close(self):
        self.file_handler.close()

    def maybe_record_frame(self, sim_time: float, action: torch.Tensor, controller: RobotController, cameras: dict[str, object]):
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
        self.episode.add("obs/cube_pick_pos", controller.cube_pick.data.root_pos_w[0].detach().cpu().to(dtype=torch.float32))
        self.episode.add("obs/cube_base_pos", controller.cube_base.data.root_pos_w[0].detach().cpu().to(dtype=torch.float32))

        for camera_name, camera in cameras.items():
            rgb = camera.data.output["rgb"][0]
            if rgb.shape[-1] > 3:
                rgb = rgb[..., :3]
            rgb = rgb.detach().cpu()
            if rgb.dtype != torch.uint8:
                rgb = torch.clamp(rgb, 0, 255).to(dtype=torch.uint8)
            self.episode.add(f"obs/{camera_name}", rgb.contiguous())


def _make_pinhole_camera_cfg(fovx_deg: float, fovy_deg: float, clipping_range=(0.01, 1000.0)) -> sim_utils.PinholeCameraCfg:
    focal_length = 1.0
    horizontal_aperture = 2.0 * focal_length * math.tan(math.radians(fovx_deg) * 0.5)
    vertical_aperture = 2.0 * focal_length * math.tan(math.radians(fovy_deg) * 0.5)
    return sim_utils.PinholeCameraCfg(
        focal_length=focal_length,
        focus_distance=400.0,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
        clipping_range=clipping_range,
    )


def _configure_world_camera(scene: InteractiveScene, spec, sim: sim_utils.SimulationContext) -> None:
    world_camera = scene["world_camera"]
    eye = torch.tensor(spec.camera_eye, device=sim.device, dtype=torch.float32).repeat(scene.num_envs, 1)
    target = torch.tensor(spec.camera_target, device=sim.device, dtype=torch.float32).repeat(scene.num_envs, 1)
    eye += scene.env_origins
    target += scene.env_origins
    world_camera.set_world_poses_from_view(eye, target)


def _build_kinova_record_scene(sim: sim_utils.SimulationContext, num_envs: int, arm_side: str):
    spec = resolve_stack_spec("kinova", arm_side)

    @configclass
    class _SceneCfg(build_single_robot_scene_cfg(spec)):
        world_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/WorldCamera",
            update_period=0.0,
            width=848,
            height=480,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 5.0),
            ),
        )

    scene = InteractiveScene(_SceneCfg(num_envs=num_envs, env_spacing=spec.env_spacing))
    sim.reset()
    _configure_world_camera(scene, spec, sim)
    controller = RobotController(sim, scene, spec)
    controller.ee_marker.set_visibility(False)
    controller.goal_marker.set_visibility(False)
    controller.reset()
    scene.reset()
    cameras = {"world": scene["world_camera"]}
    aliases = {"world": {"prim_path": "{ENV_REGEX_NS}/WorldCamera", "width": 848, "height": 480}}
    return scene, controller, cameras, None, aliases


_AGIBOT_CAMERA_SPECS = {
    "head": {
        "prim_path": "{ENV_REGEX_NS}/HeadRecordCamera",
        "width": 640,
        "height": 480,
        "spawn": sim_utils.PinholeCameraCfg(
            focal_length=1.93,
            focus_distance=400.0,
            horizontal_aperture=3.896,
            vertical_aperture=2.453,
            clipping_range=(0.01, 1000.0),
        ),
    },
    "left_hand": {
        "prim_path": "{ENV_REGEX_NS}/LeftHandRecordCamera",
        "width": 640,
        "height": 480,
        "spawn": sim_utils.PinholeCameraCfg(
            focal_length=1.93,
            focus_distance=400.0,
            horizontal_aperture=3.7601311,
            vertical_aperture=2.1307867,
            clipping_range=(0.01, 1000.0),
        ),
    },
    "right_hand": {
        "prim_path": "{ENV_REGEX_NS}/RightHandRecordCamera",
        "width": 640,
        "height": 480,
        "spawn": sim_utils.PinholeCameraCfg(
            focal_length=1.93,
            focus_distance=400.0,
            horizontal_aperture=3.7601311,
            vertical_aperture=2.1307867,
            clipping_range=(0.01, 1000.0),
        ),
    },
    "world": {
        "prim_path": "{ENV_REGEX_NS}/WorldCamera",
        "width": 848,
        "height": 480,
        "spawn": sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
    },
}

_AGIBOT_MOUNT_POSES = {
    "head": {
        "body_name": "head_link2",
        "pos": (0.0858, -0.04119, 0.0),
        "quat": (6.123233995736766e-17, 0.7071067811865476, -4.329780281177467e-17, -0.7071067811865476),
    },
    "left_hand": {
        "body_name": "gripper_l_base_link",
        "pos": (-0.07593376119019446, 0.009, 0.0205),
        "quat": (0.14219926877000713, 0.6868383037850345, -0.6989317357080853, -0.13973883223733854),
    },
    "right_hand": {
        "body_name": "gripper_r_base_link",
        "pos": (0.07593, -0.009, 0.021),
        "quat": (-0.1397388322373386, -0.6989317357080854, -0.6868383037850344, -0.14219926877000738),
    },
}


def _build_agibot_record_scene(sim: sim_utils.SimulationContext, num_envs: int, arm_side: str):
    spec = resolve_stack_spec("agibot", arm_side)

    @configclass
    class _SceneCfg(build_single_robot_scene_cfg(spec)):
        head_camera = CameraCfg(
            prim_path=_AGIBOT_CAMERA_SPECS["head"]["prim_path"],
            update_period=0.0,
            width=_AGIBOT_CAMERA_SPECS["head"]["width"],
            height=_AGIBOT_CAMERA_SPECS["head"]["height"],
            data_types=["rgb"],
            spawn=_AGIBOT_CAMERA_SPECS["head"]["spawn"],
        )
        left_hand_camera = CameraCfg(
            prim_path=_AGIBOT_CAMERA_SPECS["left_hand"]["prim_path"],
            update_period=0.0,
            width=_AGIBOT_CAMERA_SPECS["left_hand"]["width"],
            height=_AGIBOT_CAMERA_SPECS["left_hand"]["height"],
            data_types=["rgb"],
            spawn=_AGIBOT_CAMERA_SPECS["left_hand"]["spawn"],
        )
        right_hand_camera = CameraCfg(
            prim_path=_AGIBOT_CAMERA_SPECS["right_hand"]["prim_path"],
            update_period=0.0,
            width=_AGIBOT_CAMERA_SPECS["right_hand"]["width"],
            height=_AGIBOT_CAMERA_SPECS["right_hand"]["height"],
            data_types=["rgb"],
            spawn=_AGIBOT_CAMERA_SPECS["right_hand"]["spawn"],
        )
        world_camera = CameraCfg(
            prim_path=_AGIBOT_CAMERA_SPECS["world"]["prim_path"],
            update_period=0.0,
            width=_AGIBOT_CAMERA_SPECS["world"]["width"],
            height=_AGIBOT_CAMERA_SPECS["world"]["height"],
            data_types=["rgb"],
            spawn=_AGIBOT_CAMERA_SPECS["world"]["spawn"],
        )

    scene = InteractiveScene(_SceneCfg(num_envs=num_envs, env_spacing=spec.env_spacing))
    sim.reset()
    _configure_world_camera(scene, spec, sim)
    controller = RobotController(sim, scene, spec)
    controller.ee_marker.set_visibility(False)
    controller.goal_marker.set_visibility(False)
    controller.reset()
    scene.reset()
    cameras = {
        "head": scene["head_camera"],
        "left_hand": scene["left_hand_camera"],
        "right_hand": scene["right_hand_camera"],
        "world": scene["world_camera"],
    }

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

    aliases = {
        name: {
            "prim_path": spec_cfg["prim_path"],
            "width": spec_cfg["width"],
            "height": spec_cfg["height"],
        }
        for name, spec_cfg in _AGIBOT_CAMERA_SPECS.items()
    }
    return scene, controller, cameras, _sync, aliases


_R1LITE_CAMERA_SPECS = {
    "head": {
        "prim_path": "{ENV_REGEX_NS}/Robot/torso_link3/HeadRecordCamera",
        "width": 1280,
        "height": 720,
        "fovx_deg": 106.09,
        "fovy_deg": 73.51,
        "body_name": "torso_link3",
        "pos": (0.054947, 0.03149, 0.63214),
        "quat": (0.9658215854254747, 3.370518140419221e-05, 0.2592077583691833, -4.462517564976509e-05),
    },
    "left_hand": {
        "prim_path": "{ENV_REGEX_NS}/Robot/left_D405_link/LeftHandRecordCamera",
        "width": 320,
        "height": 240,
        "fovx_deg": 87.0,
        "fovy_deg": 58.0,
        "body_name": "left_D405_link",
        "pos": (0.0, 0.0, 0.0),
        "quat": (0.5, 0.5, -0.5, 0.5),
    },
    "right_hand": {
        "prim_path": "{ENV_REGEX_NS}/Robot/right_D405_link/RightHandRecordCamera",
        "width": 320,
        "height": 240,
        "fovx_deg": 87.0,
        "fovy_deg": 58.0,
        "body_name": "right_D405_link",
        "pos": (0.0, 0.0, 0.0),
        "quat": (0.5, 0.5, -0.5, 0.5),
    },
    "world": {
        "prim_path": "{ENV_REGEX_NS}/WorldCamera",
        "width": 848,
        "height": 480,
    },
}


def _build_r1lite_record_scene(sim: sim_utils.SimulationContext, num_envs: int, arm_side: str):
    spec = resolve_stack_spec("r1lite", arm_side)

    @configclass
    class _SceneCfg(build_single_robot_scene_cfg(spec)):
        head_camera = CameraCfg(
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
        left_hand_camera = CameraCfg(
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
        right_hand_camera = CameraCfg(
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
        world_camera = CameraCfg(
            prim_path=_R1LITE_CAMERA_SPECS["world"]["prim_path"],
            update_period=0.0,
            width=_R1LITE_CAMERA_SPECS["world"]["width"],
            height=_R1LITE_CAMERA_SPECS["world"]["height"],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 5.0),
            ),
        )

    scene = InteractiveScene(_SceneCfg(num_envs=num_envs, env_spacing=spec.env_spacing))
    sim.reset()
    _configure_world_camera(scene, spec, sim)
    controller = RobotController(sim, scene, spec)
    controller.ee_marker.set_visibility(False)
    controller.goal_marker.set_visibility(False)
    controller.reset()
    scene.reset()
    cameras = {
        "head": scene["head_camera"],
        "left_hand": scene["left_hand_camera"],
        "right_hand": scene["right_hand_camera"],
        "world": scene["world_camera"],
    }
    aliases = {
        name: {
            "prim_path": spec_cfg["prim_path"],
            "width": spec_cfg["width"],
            "height": spec_cfg["height"],
        }
        for name, spec_cfg in _R1LITE_CAMERA_SPECS.items()
    }
    return scene, controller, cameras, None, aliases


_RECORD_BUILDERS: dict[str, tuple[str, Callable[[sim_utils.SimulationContext, int, str], tuple[InteractiveScene, RobotController, dict[str, object], Callable[[], None] | None, dict[str, dict[str, object]]]]]] = {
    "kinova": ("Isaac-Kinova-Gen3-MouseTeleop-v0", _build_kinova_record_scene),
    "agibot": ("Isaac-Agibot-G1-MouseTeleop-v0", _build_agibot_record_scene),
    "r1lite": ("Isaac-R1Lite-MouseTeleop-v0", _build_r1lite_record_scene),
}


def run_mouse_teleop_record(simulation_app, robot_name: str, args: MouseTeleopRecordArgs) -> None:
    if args.num_envs != 1:
        raise ValueError("Mouse teleop recording only supports --num_envs 1.")

    spec = resolve_stack_spec(robot_name, args.arm_side)
    env_name, build_scene = _RECORD_BUILDERS[robot_name]

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
    sim.set_camera_view(spec.camera_eye, spec.camera_target)

    scene, controller, cameras, sync_cameras, camera_aliases = build_scene(sim, args.num_envs, args.arm_side)
    ui = MouseRecordTeleopUI(
        f"{spec.window_title} Mouse Record",
        lin_step=args.lin_step,
        ang_step=args.ang_step,
        allow_arm_switch=spec.arm_switch_supported,
    )
    writer = MouseTeleopEpisodeWriter(
        args.dataset_file,
        args.capture_hz,
        args.append,
        env_name,
        camera_aliases,
        initial_arm_side=spec.arm_side,
        arm_switch_supported=spec.arm_switch_supported,
    )

    if sync_cameras is not None:
        sync_cameras()

    print(f"[INFO] {robot_name} mouse teleop recording ready.")
    print(f"[INFO] Recording to: {os.path.abspath(writer.dataset_file)}")
    print(f"[INFO] Capture rate: {args.capture_hz:.2f} Hz")
    print(f"[INFO] Active arm: {controller.active_arm_side}")

    sim_time = 0.0
    while simulation_app.is_running():
        if ui.consume_switch_arm_request():
            next_side = controller.switch_arm_side()
            ui.set_title(f"{controller.current_window_title()} Mouse Record")
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

        if ui.consume_save_request():
            writer.save_episode()
            controller.reset()
            scene.reset()
            writer.reset_episode()
            if sync_cameras is not None:
                sync_cameras()

        if ui.consume_discard_request():
            writer.discard_episode()
            controller.reset()
            scene.reset()
            if sync_cameras is not None:
                sync_cameras()

    if writer.frame_count > 0:
        print(
            f"[INFO] Exiting with an unsaved in-memory episode containing {writer.frame_count} frames. "
            "Use the Save Episode button before closing if you want to keep it."
        )
    writer.close()
