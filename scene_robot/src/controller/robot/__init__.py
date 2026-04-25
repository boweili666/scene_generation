import os
from importlib import import_module

CONTROLLER_ROBOT_ROOT = os.path.dirname(os.path.realpath(__file__))
CONTROLLER_RESOURCE_DIR = os.path.join(os.path.dirname(CONTROLLER_ROBOT_ROOT), "resources")

from .base.base_robot_config import RobotConfig
from .base.base_robot_kinematics import RobotKinematics
from .agibot.config.agibot_g1_arm_dynamic_1_config import AgibotG1ArmDynamic1Config
from .kinova.config.gen3_single_dynamic_1_config import Gen3SingleDynamic1Config
from .r1lite.config.r1lite_upper_dynamic_1_config import R1LiteUpperDynamic1Config

_LAZY_EXPORTS = {
    "Gen3SingleKinematics": "controller.robot.kinova.kinematics.gen3_single_kinematics",
    "R1LiteFlatKinematics": "controller.robot.r1lite.kinematics.r1lite_flat_kinematics",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'controller.robot' has no attribute '{name}'")
