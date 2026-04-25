"""Facade for backward-compat: re-exports from .pipelines.scene_mouse_collect.

The actual module lives at `scene_robot_apps/pipelines/scene_mouse_collect.py`.
External callers that did `from scene_robot_apps.scene_mouse_collect import ...`
or `from .scene_mouse_collect import ...` continue to work via this facade.
"""

from .pipelines.scene_mouse_collect import *  # noqa: F401, F403
from .pipelines.scene_mouse_collect import __all__  # noqa: F401
