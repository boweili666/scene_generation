"""Omniverse UI window for mouse-driven scene data collection.

`MouseCommandCollectUI` builds a small Kit window with pose-step buttons
(±X/Y/Z/R/P/Yaw), gripper toggle, robot reset, optional arm switch, and
episode controls (start / stop+save / stop+discard). The `consume_*`
methods are polled from the main loop each step to translate clicks
into actions.

Originally bundled inside `scene_mouse_collect.py`; pulled out so the
mouse-collect orchestrator stays focused on scene assembly + episode
loop. `scene_mouse_collect.py` re-exports `MouseCommandCollectUI` for
backward compatibility.
"""

from __future__ import annotations

from typing import Callable

import torch


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
