from importlib import import_module

from .config.r1lite_upper_dynamic_1_config import R1LiteUpperDynamic1Config
from .config.r1lite_upper_dynamic_2_config import R1LiteUpperDynamic2Config

_LAZY_EXPORTS = {
    "R1LiteFlatKinematics": "controller.robot.r1lite.kinematics.r1lite_flat_kinematics",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'controller.robot.r1lite' has no attribute '{name}'")

