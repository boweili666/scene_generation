from importlib import import_module

from .agibot import AgibotBenchmarkPIDController, AgibotTeleopPIDController
from .kinova import KinovaBenchmarkPIDController, KinovaTeleopPIDController
from .r1lite import R1LiteBenchmarkPIDController, R1LiteTeleopPIDController

_LAZY_EXPORTS = {
    "AgibotG1ArmDynamic1Config": "controller.robot.agibot.config.agibot_g1_arm_dynamic_1_config",
    "AgibotG1ArmKinematics": "controller.robot.agibot.kinematics.agibot_g1_arm_kinematics",
    "IsaacLabBimanualPoseController": "controller.isaaclab_pose_controller",
    "IsaacLabPoseController": "controller.isaaclab_pose_controller",
    "RobotConfig": "controller.robot.base.base_robot_config",
    "RobotKinematics": "controller.robot.base.base_robot_kinematics",
    "R1LiteUpperDynamic1Config": "controller.robot.r1lite.config.r1lite_upper_dynamic_1_config",
    "R1LiteUpperDynamic2Config": "controller.robot.r1lite.config.r1lite_upper_dynamic_2_config",
    "R1LiteFlatKinematics": "controller.robot.r1lite.kinematics.r1lite_flat_kinematics",
    "Gen3SingleDynamic1Config": "controller.robot.kinova.config.gen3_single_dynamic_1_config",
    "Gen3SingleDynamic2Config": "controller.robot.kinova.config.gen3_single_dynamic_2_config",
    "Gen3SingleKinematics": "controller.robot.kinova.kinematics.gen3_single_kinematics",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'controller' has no attribute '{name}'")
