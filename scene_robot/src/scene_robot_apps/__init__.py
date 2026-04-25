from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "MouseTeleopRecordArgs",
    "run_mouse_teleop_record",
    "STACK_SPECS",
    "build_stack_scene",
    "run_stack_cube_demo",
]


def __getattr__(name: str) -> Any:
    # Lazy resolution into the new subdir layout so existing
    # `from scene_robot_apps import MouseTeleopRecordArgs` (etc.) still
    # works after the move into `pipelines/` and `control/`.
    if name in {"MouseTeleopRecordArgs", "run_mouse_teleop_record"}:
        module = import_module(".pipelines.mouse_teleop_record", __name__)
        return getattr(module, name)
    if name in {"STACK_SPECS", "build_stack_scene", "run_stack_cube_demo"}:
        module = import_module(".control.stack_cube", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
