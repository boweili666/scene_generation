# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for local Kinova Gen3 + Robotiq 2F-85 combined USD.

The following configuration parameters are available:

* :obj:`GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_CFG`: Local Gen3 (7-DoF) arm with Robotiq 2F-85 gripper.
* :obj:`GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_HIGH_PD_CFG`: Same robot with higher effort/PD gains.
* :obj:`GEN3_7DOF_VISION_ROBOTIQ_2F85_STACK_CUBE_DIFF_IK_CFG`: Stack-cube Diff-IK tuned variant.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_GEN3_ROBOTIQ_2F85_USD_PATH = os.path.join(
    os.path.dirname(__file__),
    "GEN3-7DOF-VISION_ROBOTIQ-2F85_COMBINED",
    "GEN3-7DOF-VISION_ROBOTIQ-2F85_COMBINED.usd",
)

GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_GEN3_ROBOTIQ_2F85_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.65,
            "joint_3": 0.0,
            "joint_4": 1.89,
            "joint_5": 0.0,
            "joint_6": 0.6,
            "joint_7": -1.57,
            "finger_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "left_inner_knuckle_joint": 0.0,
            "right_inner_knuckle_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]"],
            effort_limit={
                "joint_[1-4]": 39.0,
                "joint_[5-7]": 9.0,
            },
            stiffness={
                "joint_[1-4]": 40.0,
                "joint_[5-7]": 15.0,
            },
            damping={
                "joint_[1-4]": 1.0,
                "joint_[5-7]": 0.5,
            },
        ),
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=11.25,
            damping=0.1,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""Configuration of local Gen3 (7-DoF) arm with Robotiq 2F-85 gripper."""


GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_HIGH_PD_CFG = GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_CFG.copy()
GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = False
GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_HIGH_PD_CFG.actuators["arm"].effort_limit = {
    "joint_[1-4]": 10000000000.0,
    "joint_[5-7]": 10000000.0,
}
GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_HIGH_PD_CFG.actuators["arm"].effort_limit_sim = {
    "joint_[1-4]": 10000000000.0,
    "joint_[5-7]": 10000000.0,
}
GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_HIGH_PD_CFG.actuators["arm"].stiffness = {
    "joint_[1-4]": 10000000000000.0,
    "joint_[5-7]": 1000000000000.0,
}
GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_HIGH_PD_CFG.actuators["arm"].damping = {
    "joint_[1-4]": 100000000.0,
    "joint_[5-7]": 100000000.0,
}
GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_HIGH_PD_CFG.actuators["gripper_drive"].effort_limit_sim = 4.0
GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_HIGH_PD_CFG.actuators["gripper_drive"].stiffness = 8.0
GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_HIGH_PD_CFG.actuators["gripper_drive"].damping = 1.2
"""Configuration of local Gen3 (7-DoF) with higher effort/PD gains for IK tracking."""


GEN3_7DOF_VISION_ROBOTIQ_2F85_STACK_CUBE_DIFF_IK_CFG = GEN3_7DOF_VISION_ROBOTIQ_2F85_COMBINED_HIGH_PD_CFG.copy()
GEN3_7DOF_VISION_ROBOTIQ_2F85_STACK_CUBE_DIFF_IK_CFG.spawn.articulation_props.fix_root_link = True
GEN3_7DOF_VISION_ROBOTIQ_2F85_STACK_CUBE_DIFF_IK_CFG.spawn.rigid_props.max_depenetration_velocity = 1.0
"""Configuration of local Gen3 tuned for stack-cube Differential IK demos."""
