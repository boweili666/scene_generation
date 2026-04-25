"""Stack-cube demo task: build the canonical scene + run the demo loop.

Everything that this file *used* to also host — robot specs
(`RobotStackSpec`, `STACK_SPECS`, ...), scene-cfg builders
(`build_single_robot_scene_cfg`, `build_merged_scene_cfg`), arm-side
helpers (`normalize_arm_side`, `_arm_config_fields`), and the teleop
UIs (`MouseClickTeleopUI`, `KeyboardTeleop`) — has moved out into
focused modules:

* `control/robot_spec.py` — the dataclasses + per-robot specs.
* `control/scene_cfg.py` — the InteractiveSceneCfg factories.
* `ui/teleop_ui.py` — the Omniverse teleop UI primitives.

What stays here is the actual stack-cube *task*: build a single-robot
scene, instantiate the controller, and run its `step_stack` state
machine in a loop. The agibot/kinova/r1lite stack-cube scripts under
`scene_robot/scripts/` go through this module via
`scene_robot_apps.run_stack_cube_demo`.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from .robot_spec import resolve_stack_spec
from .scene_cfg import build_single_robot_scene_cfg


# RobotController lives in robot_controller.py. Importing it at the top of
# this module would create a circular import (robot_controller imports
# RobotStackSpec / Phase / etc. from .robot_spec, and that subgraph is
# fine, but a top-level RobotController import here would still trigger
# the controller's own module-load chain at this file's import time). We
# import lazily inside build_stack_scene below. With
# `from __future__ import annotations` enabled, the type-hint reference
# in build_stack_scene's signature stays as a string and doesn't need a
# real symbol at module load time.


def build_stack_scene(
    sim: sim_utils.SimulationContext,
    robot_name: str,
    num_envs: int = 1,
    arm_side: str = "left",
) -> "tuple[InteractiveScene, RobotController]":
    from .robot_controller import RobotController

    spec = resolve_stack_spec(robot_name, arm_side)
    scene_cfg = build_single_robot_scene_cfg(spec)(num_envs=num_envs, env_spacing=spec.env_spacing)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    controller = RobotController(sim, scene, spec)
    controller.reset()
    scene.reset()
    return scene, controller


def run_stack_cube_demo(
    simulation_app,
    robot_name: str,
    device: str,
    num_envs: int = 1,
    arm_side: str = "left",
) -> None:
    spec = resolve_stack_spec(robot_name, arm_side)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device))
    sim.set_camera_view(spec.camera_eye, spec.camera_target)

    scene, controller = build_stack_scene(sim, robot_name, num_envs=num_envs, arm_side=arm_side)
    print(f"[INFO] {robot_name} stack-cube demo ready.")

    while simulation_app.is_running():
        controller.step_stack()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
