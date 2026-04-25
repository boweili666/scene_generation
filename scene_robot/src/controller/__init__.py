"""IsaacLab controller integration.

This package was originally larger — it shipped per-robot PID
controllers (`AgibotTeleopPIDController`, `KinovaBenchmarkPIDController`,
etc.) plus their `BasePIDPolicy` / `TeleopPIDPolicy` / `BenchmarkPIDPolicy`
/ `TrajTrackingPolicy` infrastructure. None of those were exercised by
the Isaac Sim pipelines in `scene_robot_apps/` (which all go through
`IsaacLabPoseController`), so the PID stack and the Dynamic2 configs /
unused agibot kinematics that only fed it have been removed.

What remains:

* `IsaacLabPoseController` / `IsaacLabBimanualPoseController` —
  diff-IK pose controllers used by every Isaac Sim pipeline.
* `controller.utils` — small geometry / viz-color helpers used by the
  per-robot config files.
* `controller.robot.*` — robot-specific kinematics + config classes
  that the Isaac controllers consume (Gen3SingleKinematics,
  AgibotG1ArmDynamic1Config, R1LiteFlatKinematics, etc.).
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "AgibotG1ArmDynamic1Config": "controller.robot.agibot.config.agibot_g1_arm_dynamic_1_config",
    "IsaacLabBimanualPoseController": "controller.isaaclab_pose_controller",
    "IsaacLabPoseController": "controller.isaaclab_pose_controller",
    "RobotConfig": "controller.robot.base.base_robot_config",
    "RobotKinematics": "controller.robot.base.base_robot_kinematics",
    "R1LiteUpperDynamic1Config": "controller.robot.r1lite.config.r1lite_upper_dynamic_1_config",
    "R1LiteFlatKinematics": "controller.robot.r1lite.kinematics.r1lite_flat_kinematics",
    "Gen3SingleDynamic1Config": "controller.robot.kinova.config.gen3_single_dynamic_1_config",
    "Gen3SingleKinematics": "controller.robot.kinova.kinematics.gen3_single_kinematics",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'controller' has no attribute '{name}'")
