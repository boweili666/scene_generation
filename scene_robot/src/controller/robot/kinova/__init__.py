from importlib import import_module

from .config.gen3_single_dynamic_1_config import Gen3SingleDynamic1Config
from .config.gen3_single_dynamic_2_config import Gen3SingleDynamic2Config

_LAZY_EXPORTS = {
    "Gen3SingleKinematics": "controller.robot.kinova.kinematics.gen3_single_kinematics",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'controller.robot.kinova' has no attribute '{name}'")
