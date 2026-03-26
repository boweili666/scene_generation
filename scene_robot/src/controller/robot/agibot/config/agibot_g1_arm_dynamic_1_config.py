from enum import IntEnum

import numpy as np

from controller.robot.base.base_robot_config import RobotConfig
from controller.utils import Geometry


class AgibotG1ArmDynamic1Config(RobotConfig):
    kinematics_class_name = "AgibotG1ArmKinematics"
    collision_spheres_json_path = None
    urdf_relpath_candidates = (
        "genie_sim/source/geniesim/app/robot_cfg/G1_omnipicker/G1_omnipicker.urdf",
        "genie_sim/source/geniesim/robot/isaac_sim/robot_urdf/G1_omnipicker.urdf",
    )

    def __init__(self, arm_side: str = "left") -> None:
        self.arm_side = arm_side.lower()
        super().__init__()

    def _post_init(self):
        if self.arm_side not in {"left", "right"}:
            raise ValueError(f"Unsupported Agibot arm side: {self.arm_side}")

        self.controlled_joint_names = tuple(
            f"idx2{i}_arm_l_joint{i}" if self.arm_side == "left" else f"idx6{i}_arm_r_joint{i}"
            for i in range(1, 8)
        )
        self.ee_frame_name = "arm_l_end_link" if self.arm_side == "left" else "arm_r_end_link"

        default_joint_values = {
            "idx01_body_joint1": 0.0,
            "idx02_body_joint2": 0.4363,
            "idx11_head_joint1": 0.8727,
            "idx12_head_joint2": 0.42,
            "idx21_arm_l_joint1": -1.0751,
            "idx22_arm_l_joint2": 0.6109,
            "idx23_arm_l_joint3": 0.2793,
            "idx24_arm_l_joint4": -1.2846,
            "idx25_arm_l_joint5": 0.7295,
            "idx26_arm_l_joint6": 1.4957,
            "idx27_arm_l_joint7": -0.1868,
            "idx61_arm_r_joint1": 1.0734,
            "idx62_arm_r_joint2": -0.6109,
            "idx63_arm_r_joint3": -0.2793,
            "idx64_arm_r_joint4": 1.2846,
            "idx65_arm_r_joint5": -0.7313,
            "idx66_arm_r_joint6": -1.4957,
            "idx67_arm_r_joint7": 0.1868,
            "idx41_gripper_l_outer_joint1": 0.0,
            "idx81_gripper_r_outer_joint1": 0.0,
        }
        self.locked_joint_positions = default_joint_values
        self.joint_to_lock = tuple(name for name in default_joint_values if name not in self.controlled_joint_names)

    NumTotalMotors = 7

    class RealMotors(IntEnum):
        Joint1 = 0
        Joint2 = 1
        Joint3 = 2
        Joint4 = 3
        Joint5 = 4
        Joint6 = 5
        Joint7 = 6

    RealMotorPosLimit = {
        RealMotors.Joint1: (-3.14, 3.14),
        RealMotors.Joint2: (-2.09, 1.48),
        RealMotors.Joint3: (-3.10, 3.10),
        RealMotors.Joint4: (-1.48, 1.48),
        RealMotors.Joint5: (-3.10, 3.10),
        RealMotors.Joint6: (-1.74, 1.74),
        RealMotors.Joint7: (-3.10, 3.10),
    }

    NormalMotor = [
        RealMotors.Joint1,
        RealMotors.Joint2,
        RealMotors.Joint3,
        RealMotors.Joint4,
        RealMotors.Joint5,
        RealMotors.Joint6,
        RealMotors.Joint7,
    ]
    WeakMotor = []
    DelicateMotor = []

    class DoFs(IntEnum):
        Joint1 = 0
        Joint2 = 1
        Joint3 = 2
        Joint4 = 3
        Joint5 = 4
        Joint6 = 5
        Joint7 = 6

    DefaultDoFVal = {
        DoFs.Joint1: 0.0,
        DoFs.Joint2: 0.0,
        DoFs.Joint3: 0.0,
        DoFs.Joint4: 0.0,
        DoFs.Joint5: 0.0,
        DoFs.Joint6: 0.0,
        DoFs.Joint7: 0.0,
    }

    class Control(IntEnum):
        vJoint1 = 0
        vJoint2 = 1
        vJoint3 = 2
        vJoint4 = 3
        vJoint5 = 4
        vJoint6 = 5
        vJoint7 = 6

    ControlLimit = {
        Control.vJoint1: 3.14,
        Control.vJoint2: 3.14,
        Control.vJoint3: 3.14,
        Control.vJoint4: 3.14,
        Control.vJoint5: 3.14,
        Control.vJoint6: 3.14,
        Control.vJoint7: 3.14,
    }

    NormalControl = [
        Control.vJoint1,
        Control.vJoint2,
        Control.vJoint3,
        Control.vJoint4,
        Control.vJoint5,
        Control.vJoint6,
        Control.vJoint7,
    ]
    WeakControl = []
    DelicateControl = []

    @property
    def num_state(self):
        return int(len(self.DoFs))

    def compose_state_from_dof(self, dof_pos, dof_vel):
        return np.asarray(dof_pos).reshape(-1)

    def decompose_state_to_dof_pos(self, state):
        return np.asarray(state).reshape(-1)

    def decompose_state_to_dof_vel(self, state):
        return np.zeros(self.num_dof)

    def dynamics_f(self, state):
        return np.zeros((self.num_state, 1))

    def dynamics_g(self, state):
        return np.eye(self.num_state)

    class MujocoDoFs(IntEnum):
        Joint1 = 0
        Joint2 = 1
        Joint3 = 2
        Joint4 = 3
        Joint5 = 4
        Joint6 = 5
        Joint7 = 6

    class MujocoMotors(IntEnum):
        Joint1 = 0
        Joint2 = 1
        Joint3 = 2
        Joint4 = 3
        Joint5 = 4
        Joint6 = 5
        Joint7 = 6

    MujocoDoF_to_DoF = {
        MujocoDoFs.Joint1: DoFs.Joint1,
        MujocoDoFs.Joint2: DoFs.Joint2,
        MujocoDoFs.Joint3: DoFs.Joint3,
        MujocoDoFs.Joint4: DoFs.Joint4,
        MujocoDoFs.Joint5: DoFs.Joint5,
        MujocoDoFs.Joint6: DoFs.Joint6,
        MujocoDoFs.Joint7: DoFs.Joint7,
    }
    DoF_to_MujocoDoF = {
        DoFs.Joint1: MujocoDoFs.Joint1,
        DoFs.Joint2: MujocoDoFs.Joint2,
        DoFs.Joint3: MujocoDoFs.Joint3,
        DoFs.Joint4: MujocoDoFs.Joint4,
        DoFs.Joint5: MujocoDoFs.Joint5,
        DoFs.Joint6: MujocoDoFs.Joint6,
        DoFs.Joint7: MujocoDoFs.Joint7,
    }
    MujocoMotor_to_Control = {
        MujocoMotors.Joint1: Control.vJoint1,
        MujocoMotors.Joint2: Control.vJoint2,
        MujocoMotors.Joint3: Control.vJoint3,
        MujocoMotors.Joint4: Control.vJoint4,
        MujocoMotors.Joint5: Control.vJoint5,
        MujocoMotors.Joint6: Control.vJoint6,
        MujocoMotors.Joint7: Control.vJoint7,
    }

    class RealDoFs(IntEnum):
        Joint1 = 0
        Joint2 = 1
        Joint3 = 2
        Joint4 = 3
        Joint5 = 4
        Joint6 = 5
        Joint7 = 6

    RealDoF_to_DoF = {
        RealDoFs.Joint1: DoFs.Joint1,
        RealDoFs.Joint2: DoFs.Joint2,
        RealDoFs.Joint3: DoFs.Joint3,
        RealDoFs.Joint4: DoFs.Joint4,
        RealDoFs.Joint5: DoFs.Joint5,
        RealDoFs.Joint6: DoFs.Joint6,
        RealDoFs.Joint7: DoFs.Joint7,
    }
    DoF_to_RealDoF = {
        DoFs.Joint1: RealDoFs.Joint1,
        DoFs.Joint2: RealDoFs.Joint2,
        DoFs.Joint3: RealDoFs.Joint3,
        DoFs.Joint4: RealDoFs.Joint4,
        DoFs.Joint5: RealDoFs.Joint5,
        DoFs.Joint6: RealDoFs.Joint6,
        DoFs.Joint7: RealDoFs.Joint7,
    }
    RealMotor_to_Control = {
        RealMotors.Joint1: Control.vJoint1,
        RealMotors.Joint2: Control.vJoint2,
        RealMotors.Joint3: Control.vJoint3,
        RealMotors.Joint4: Control.vJoint4,
        RealMotors.Joint5: Control.vJoint5,
        RealMotors.Joint6: Control.vJoint6,
        RealMotors.Joint7: Control.vJoint7,
    }

    class Frames(IntEnum):
        EE = 0

    CollisionVol = {
        Frames.EE: Geometry(type="sphere", radius=0.05),
    }

    AdjacentCollisionVolPairs = []
    SelfCollisionVolIgnored = []
    EnvCollisionVolIgnored = []
