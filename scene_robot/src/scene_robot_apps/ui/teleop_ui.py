"""Stack-cube-style teleop UI primitives.

`MouseClickTeleopUI` and `KeyboardTeleop` are the small Omniverse UIs
the stack-cube demo (and `RobotController.step_click_teleop` /
`step_keyboard_teleop`) rely on for ad-hoc operator control. They were
originally bundled inside `stack_cube.py`; pulling them under `ui/`
makes their role explicit and lets `stack_cube.py` shrink to the
actual stack-cube task code.
"""

from __future__ import annotations

import carb
import torch


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
