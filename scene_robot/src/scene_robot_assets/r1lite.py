# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for local R1Lite robot asset.

The following configuration parameters are available:

* :obj:`R1LITE_ROBOT_CFG`: Full articulated config (torso + dual-arm + grippers).
* :obj:`R1LITE_HIGH_PD_CFG`: Full articulated config with stiffer PD gains for task-space tracking.
* :obj:`R1LITE_BIMANUAL_CFG`: Bimanual manipulation config (torso + dual-arm + grippers).
* :obj:`R1LITE_STACK_CUBE_DIFF_IK_CFG`: Stack-cube Diff-IK tuned variant.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_R1LITE_USD_PATH = os.path.join(os.path.dirname(__file__), "r1lite", "robot", "robot.usd")

R1LITE_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_R1LITE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "torso_joint[1-3]": 0.0,
            "left_arm_joint[1-6]": 0.0,
            "right_arm_joint[1-6]": 0.0,
            "left_gripper_finger_joint[1-2]": 0.0,
            "right_gripper_finger_joint[1-2]": 0.0,
        },
    ),
    actuators={
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint[1-3]"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_arm_joint[1-6]"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_arm_joint[1-6]"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_gripper_finger_joint[1-2]"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["right_gripper_finger_joint[1-2]"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
    },
)
"""Configuration of local R1Lite robot using implicit actuator models."""


R1LITE_HIGH_PD_CFG = R1LITE_ROBOT_CFG.copy()
R1LITE_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
R1LITE_HIGH_PD_CFG.spawn.articulation_props.fix_root_link = True
R1LITE_HIGH_PD_CFG.actuators["torso"].effort_limit_sim = 900.0
R1LITE_HIGH_PD_CFG.actuators["torso"].velocity_limit_sim = 4.0
R1LITE_HIGH_PD_CFG.actuators["torso"].stiffness = 900.0
R1LITE_HIGH_PD_CFG.actuators["torso"].damping = 320.0
R1LITE_HIGH_PD_CFG.actuators["left_arm"].effort_limit_sim = 2500.0
R1LITE_HIGH_PD_CFG.actuators["left_arm"].velocity_limit_sim = 7.0
R1LITE_HIGH_PD_CFG.actuators["left_arm"].stiffness = 4500.0
R1LITE_HIGH_PD_CFG.actuators["left_arm"].damping = 320.0
R1LITE_HIGH_PD_CFG.actuators["right_arm"].effort_limit_sim = 2500.0
R1LITE_HIGH_PD_CFG.actuators["right_arm"].velocity_limit_sim = 7.0
R1LITE_HIGH_PD_CFG.actuators["right_arm"].stiffness = 4500.0
R1LITE_HIGH_PD_CFG.actuators["right_arm"].damping = 320.0
R1LITE_HIGH_PD_CFG.actuators["left_gripper"].effort_limit_sim = 700.0
R1LITE_HIGH_PD_CFG.actuators["left_gripper"].velocity_limit_sim = 6.0
R1LITE_HIGH_PD_CFG.actuators["left_gripper"].stiffness = 6000.0
R1LITE_HIGH_PD_CFG.actuators["left_gripper"].damping = 250.0
R1LITE_HIGH_PD_CFG.actuators["right_gripper"].effort_limit_sim = 700.0
R1LITE_HIGH_PD_CFG.actuators["right_gripper"].velocity_limit_sim = 6.0
R1LITE_HIGH_PD_CFG.actuators["right_gripper"].stiffness = 6000.0
R1LITE_HIGH_PD_CFG.actuators["right_gripper"].damping = 250.0
"""Configuration of local R1Lite with stiffer PD control."""


R1LITE_BIMANUAL_CFG = R1LITE_ROBOT_CFG.copy()
R1LITE_BIMANUAL_CFG.spawn.rigid_props.disable_gravity = True
R1LITE_BIMANUAL_CFG.spawn.articulation_props.fix_root_link = True
R1LITE_BIMANUAL_CFG.actuators = {
    "torso": ImplicitActuatorCfg(
        joint_names_expr=["torso_joint[1-3]"],
        effort_limit_sim=900.0,
        velocity_limit_sim=4.0,
        stiffness=90000000.0,
        damping=320000.0,
    ),
    "left_arm": ImplicitActuatorCfg(
        joint_names_expr=["left_arm_joint[1-6]"],
        effort_limit_sim=4500.0,
        velocity_limit_sim=9.0,
        stiffness=8000.0,
        damping=520.0,
    ),
    "right_arm": ImplicitActuatorCfg(
        joint_names_expr=["right_arm_joint[1-6]"],
        effort_limit_sim=4500.0,
        velocity_limit_sim=9.0,
        stiffness=8000.0,
        damping=520.0,
    ),
    "left_gripper": ImplicitActuatorCfg(
        joint_names_expr=["left_gripper_finger_joint[1-2]"],
        effort_limit_sim=1200.0,
        velocity_limit_sim=8.0,
        stiffness=9000.0,
        damping=450.0,
    ),
    "right_gripper": ImplicitActuatorCfg(
        joint_names_expr=["right_gripper_finger_joint[1-2]"],
        effort_limit_sim=1200.0,
        velocity_limit_sim=8.0,
        stiffness=9000.0,
        damping=450.0,
    ),
}
"""Configuration of local R1Lite for bimanual manipulation."""


R1LITE_STACK_CUBE_DIFF_IK_CFG = R1LITE_BIMANUAL_CFG.copy()
R1LITE_STACK_CUBE_DIFF_IK_CFG.actuators["left_arm"].stiffness = 150000.0
R1LITE_STACK_CUBE_DIFF_IK_CFG.actuators["left_arm"].damping = 22000.0
R1LITE_STACK_CUBE_DIFF_IK_CFG.actuators["left_gripper"].effort_limit_sim = 700.0
R1LITE_STACK_CUBE_DIFF_IK_CFG.actuators["left_gripper"].stiffness = 6000.0
R1LITE_STACK_CUBE_DIFF_IK_CFG.actuators["left_gripper"].damping = 250.0
R1LITE_STACK_CUBE_DIFF_IK_CFG.actuators["right_gripper"].effort_limit_sim = 700.0
R1LITE_STACK_CUBE_DIFF_IK_CFG.actuators["right_gripper"].stiffness = 6000.0
R1LITE_STACK_CUBE_DIFF_IK_CFG.actuators["right_gripper"].damping = 250.0
"""Configuration of local R1Lite tuned for stack-cube Differential IK demos."""
