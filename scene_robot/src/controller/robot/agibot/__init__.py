from importlib import import_module

from .config import AgibotG1ArmDynamic1Config

_LAZY_EXPORTS = {
    "AgibotG1ArmKinematics": "controller.robot.agibot.kinematics.agibot_g1_arm_kinematics",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'controller.robot.agibot' has no attribute '{name}'")
