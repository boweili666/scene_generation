from enum import IntEnum
from controller.robot.base.base_robot_config import RobotConfig
from controller.utils import Geometry, VizColor
import numpy as np

class R1LiteUpperDynamic2Config(RobotConfig):
    
    # ---------------------------------------------------------------------------- #
    #                                      Kinematics                              #
    # ---------------------------------------------------------------------------- #
    
    kinematics_class_name = "R1LiteFlatKinematics"
    mujoco_model_path = "r1lite/r1lite_upper_body_scene.xml"
    collision_spheres_json_path = "r1lite/config/r1lite_collision_spheres.json"
    joint_to_lock = [
        "left_gripper_finger_joint1",
        "left_gripper_finger_joint2",
        "right_gripper_finger_joint1",
        "right_gripper_finger_joint2"
    ]
    
    # ---------------------------------------------------------------------------- #
    #                                   hardware                                   #
    # ---------------------------------------------------------------------------- #
    
    NumTotalMotors = 25
    
    class RealMotors(IntEnum):

        TorsoJoint1 = 0
        TorsoJoint2 = 1
        TorsoJoint3 = 2

        LeftJoint1  = 3
        LeftJoint2  = 4
        LeftJoint3  = 5
        LeftJoint4  = 6
        LeftJoint5  = 7
        LeftJoint6  = 8

        RightJoint1 = 9
        RightJoint2 = 10
        RightJoint3 = 11
        RightJoint4 = 12
        RightJoint5 = 13
        RightJoint6 = 14

    # Based on https://support.unitree.com/home/en/G1_developer/about_G1
    RealMotorPosLimit = {
        }
    
    NormalMotor = [
        RealMotors.TorsoJoint1,
        RealMotors.TorsoJoint2,
        RealMotors.TorsoJoint3,

        RealMotors.LeftJoint1,
        RealMotors.LeftJoint2,
        RealMotors.LeftJoint3,
        RealMotors.LeftJoint4,
        RealMotors.LeftJoint5,
        RealMotors.LeftJoint6,

        RealMotors.RightJoint1,
        RealMotors.RightJoint2,
        RealMotors.RightJoint3,
        RealMotors.RightJoint4,
        RealMotors.RightJoint5,
        RealMotors.RightJoint6,

    ]
    
    WeakMotor = []
    
    DelicateMotor = []
    
    # ---------------------------------------------------------------------------- #
    #                                      DoF                                     #
    # ---------------------------------------------------------------------------- #

    class DoFs (IntEnum):

        TorsoJoint1 = 0
        TorsoJoint2 = 1
        TorsoJoint3 = 2

        LeftJoint1  = 3
        LeftJoint2  = 4
        LeftJoint3  = 5
        LeftJoint4  = 6
        LeftJoint5  = 7
        LeftJoint6  = 8

        RightJoint1 = 9
        RightJoint2 = 10
        RightJoint3 = 11
        RightJoint4 = 12
        RightJoint5 = 13
        RightJoint6 = 14


    DefaultDoFVal = {
        DoFs.TorsoJoint1 : 0.0,
        DoFs.TorsoJoint2 : 0.0,
        DoFs.TorsoJoint3 : 0.0,

        DoFs.LeftJoint1  : 0.0,
        DoFs.LeftJoint2  : 0.0,
        DoFs.LeftJoint3  : 0.0,
        DoFs.LeftJoint4  : 0.0,
        DoFs.LeftJoint5  : 0.0,
        DoFs.LeftJoint6  : 0.0,

        DoFs.RightJoint1 : 0.0,
        DoFs.RightJoint2 : 0.0,
        DoFs.RightJoint3 : 0.0,
        DoFs.RightJoint4 : 0.0,
        DoFs.RightJoint5 : 0.0,
        DoFs.RightJoint6 : 0.0,

    }

    # ---------------------------------------------------------------------------- #
    #                                   Dynamics                                   #
    # ---------------------------------------------------------------------------- #

    class Control (IntEnum):
        # Acceleration control

        aTorsoJoint1  = 0
        aTorsoJoint2  = 1
        aTorsoJoint3  = 2

        aLeftJoint1   = 3
        aLeftJoint2   = 4
        aLeftJoint3   = 5
        aLeftJoint4   = 6
        aLeftJoint5   = 7
        aLeftJoint6   = 8

        aRightJoint1  = 9
        aRightJoint2  = 10
        aRightJoint3  = 11
        aRightJoint4  = 12
        aRightJoint5  = 13
        aRightJoint6  = 14



    ControlLimit = {
        Control.aTorsoJoint1 : 10,
        Control.aTorsoJoint2 : 10,
        Control.aTorsoJoint3 : 10,

        Control.aLeftJoint1  : 10,
        Control.aLeftJoint2  : 10,
        Control.aLeftJoint3  : 10,
        Control.aLeftJoint4  : 10,
        Control.aLeftJoint5  : 10,
        Control.aLeftJoint6  : 10,

        Control.aRightJoint1 : 10,
        Control.aRightJoint2 : 10,
        Control.aRightJoint3 : 10,
        Control.aRightJoint4 : 10,
        Control.aRightJoint5 : 10,
        Control.aRightJoint6 : 10,

    }

    NormalControl = [
        Control.aTorsoJoint1,
        Control.aTorsoJoint2,
        Control.aTorsoJoint3,

        Control.aLeftJoint1,
        Control.aLeftJoint2,
        Control.aLeftJoint3,
        Control.aLeftJoint4,
        Control.aLeftJoint5,
        Control.aLeftJoint6,

        Control.aRightJoint1,
        Control.aRightJoint2,
        Control.aRightJoint3,
        Control.aRightJoint4,
        Control.aRightJoint5,
        Control.aRightJoint6,

    ]
    
    WeakControl = []
    
    DelicateControl = []

    '''
        x_dot = f(x) + g(x) * control

        For acceleration control, 
        
        state = [position; velocity]
    '''

    @property
    def num_state(self):
        return int(len(self.DoFs) * 2) # pos, vel for double integrator dynamics.

    def compose_state_from_dof(self, dof_pos, dof_vel):
        '''
            dof_pos: [num_dof,]
        '''
        state = np.concatenate((dof_pos.reshape(-1), dof_vel.reshape(-1)), axis=0)
        return state

    def decompose_state_to_dof_pos(self, state):
        '''
            state: [num_state,]
            return: [num_dof,]
        '''
        dof_pos = state.reshape(-1)[:self.num_dof] # Take only the first half entries.
        return dof_pos
    
    def decompose_state_to_dof_vel(self, state):
        '''
            state: [num_state,]
            return: [num_dof,]
        '''
        dof_vel = state.reshape(-1)[self.num_dof:] # Take only the second half entries.
        return dof_vel

    def dynamics_f(self, state):
        '''
            state: [num_state, 1]
            return: [num_state, 1]
        '''
        positions = state[:self.num_dof]
        velocities = state[self.num_dof:]
  
        f_x = np.zeros_like(state)
        f_x[:self.num_dof] = velocities
        return f_x

    def dynamics_g(self, state):
        '''
            state: [num_state, 1]
            return: [num_state, num_control]
        '''
        g_x = np.zeros((2 * self.num_dof, self.num_dof))
        g_x[self.num_dof:] = np.eye(self.num_dof)

        return g_x

    # ---------------------------------------------------------------------------- #
    #                                    MuJoCo                                    #
    # ---------------------------------------------------------------------------- #
    class MujocoDoFs(IntEnum):

        TorsoJoint1 = 0
        TorsoJoint2 = 1
        TorsoJoint3 = 2

        LeftJoint1  = 3
        LeftJoint2  = 4
        LeftJoint3  = 5
        LeftJoint4  = 6
        LeftJoint5  = 7
        LeftJoint6  = 8

        RightJoint1 = 11
        RightJoint2 = 12
        RightJoint3 = 13
        RightJoint4 = 14
        RightJoint5 = 15
        RightJoint6 = 16


    class MujocoMotors(IntEnum):

        TorsoJoint1 = 0
        TorsoJoint2 = 1
        TorsoJoint3 = 2

        LeftJoint1  = 3
        LeftJoint2  = 4
        LeftJoint3  = 5
        LeftJoint4  = 6
        LeftJoint5  = 7
        LeftJoint6  = 8

        RightJoint1 = 11
        RightJoint2 = 12
        RightJoint3 = 13
        RightJoint4 = 14
        RightJoint5 = 15
        RightJoint6 = 16


    # ---------------------------------------------------------------------------- #
    #                                   Mujoco PID Parameters                      #
    # ---------------------------------------------------------------------------- #
    
    # Kp parameters for Mujoco Motors
    MujocoMotorKps = {
        MujocoMotors.TorsoJoint1 : 3000.0,
        MujocoMotors.TorsoJoint2 : 3000.0,
        MujocoMotors.TorsoJoint3 : 3000.0,

        MujocoMotors.RightJoint1 : 1000.0,
        MujocoMotors.RightJoint2 : 1000.0,
        MujocoMotors.RightJoint3 : 1000.0,
        MujocoMotors.RightJoint4 : 1000.0,
        MujocoMotors.RightJoint5 : 1000.0,
        MujocoMotors.RightJoint6 : 1000.0,

        MujocoMotors.LeftJoint1  : 1000.0,
        MujocoMotors.LeftJoint2  : 1000.0,
        MujocoMotors.LeftJoint3  : 1000.0,
        MujocoMotors.LeftJoint4  : 1000.0,
        MujocoMotors.LeftJoint5  : 1000.0,
        MujocoMotors.LeftJoint6  : 1000.0,

    }
    
    # Kd parameters for Mujoco Motors
    MujocoMotorKds = {
        MujocoMotors.TorsoJoint1 : 0.1,
        MujocoMotors.TorsoJoint2 : 0.1,
        MujocoMotors.TorsoJoint3 : 0.1,

        MujocoMotors.RightJoint1 : 0.1,
        MujocoMotors.RightJoint2 : 0.1,
        MujocoMotors.RightJoint3 : 0.1,
        MujocoMotors.RightJoint4 : 0.1,
        MujocoMotors.RightJoint5 : 0.1,
        MujocoMotors.RightJoint6 : 0.1,

        MujocoMotors.LeftJoint1  : 0.1,
        MujocoMotors.LeftJoint2  : 0.1,
        MujocoMotors.LeftJoint3  : 0.1,
        MujocoMotors.LeftJoint4  : 0.1,
        MujocoMotors.LeftJoint5  : 0.1,
        MujocoMotors.LeftJoint6  : 0.1,

    }

    # ---------------------------------------------------------------------------- #
    #                                   Mujoco Mappings                            #
    # ---------------------------------------------------------------------------- #

    # Mapping from Mujoco DoFs to DoFs
    MujocoDoF_to_DoF = {
        MujocoDoFs.TorsoJoint1 : DoFs.TorsoJoint1,
        MujocoDoFs.TorsoJoint2 : DoFs.TorsoJoint2,
        MujocoDoFs.TorsoJoint3 : DoFs.TorsoJoint3,

        MujocoDoFs.RightJoint1 : DoFs.RightJoint1,
        MujocoDoFs.RightJoint2 : DoFs.RightJoint2,
        MujocoDoFs.RightJoint3 : DoFs.RightJoint3,
        MujocoDoFs.RightJoint4 : DoFs.RightJoint4,
        MujocoDoFs.RightJoint5 : DoFs.RightJoint5,
        MujocoDoFs.RightJoint6 : DoFs.RightJoint6,

        MujocoDoFs.LeftJoint1  : DoFs.LeftJoint1,
        MujocoDoFs.LeftJoint2  : DoFs.LeftJoint2,
        MujocoDoFs.LeftJoint3  : DoFs.LeftJoint3,
        MujocoDoFs.LeftJoint4  : DoFs.LeftJoint4,
        MujocoDoFs.LeftJoint5  : DoFs.LeftJoint5,
        MujocoDoFs.LeftJoint6  : DoFs.LeftJoint6,

    }

    # Mapping from DoFs to Mujoco DoFs
    DoF_to_MujocoDoF = {
        DoFs.TorsoJoint1 : MujocoDoFs.TorsoJoint1,
        DoFs.TorsoJoint2 : MujocoDoFs.TorsoJoint2,
        DoFs.TorsoJoint3 : MujocoDoFs.TorsoJoint3,

        DoFs.RightJoint1 : MujocoDoFs.RightJoint1,
        DoFs.RightJoint2 : MujocoDoFs.RightJoint2,
        DoFs.RightJoint3 : MujocoDoFs.RightJoint3,
        DoFs.RightJoint4 : MujocoDoFs.RightJoint4,
        DoFs.RightJoint5 : MujocoDoFs.RightJoint5,
        DoFs.RightJoint6 : MujocoDoFs.RightJoint6,

        DoFs.LeftJoint1  : MujocoDoFs.LeftJoint1,
        DoFs.LeftJoint2  : MujocoDoFs.LeftJoint2,
        DoFs.LeftJoint3  : MujocoDoFs.LeftJoint3,
        DoFs.LeftJoint4  : MujocoDoFs.LeftJoint4,
        DoFs.LeftJoint5  : MujocoDoFs.LeftJoint5,
        DoFs.LeftJoint6  : MujocoDoFs.LeftJoint6,

    }

    # Mapping from Mujoco Motors to Control
    MujocoMotor_to_Control = {
        MujocoMotors.TorsoJoint1 : Control.aTorsoJoint1,
        MujocoMotors.TorsoJoint2 : Control.aTorsoJoint2,
        MujocoMotors.TorsoJoint3 : Control.aTorsoJoint3,

        MujocoMotors.RightJoint1 : Control.aRightJoint1,
        MujocoMotors.RightJoint2 : Control.aRightJoint2,
        MujocoMotors.RightJoint3 : Control.aRightJoint3,
        MujocoMotors.RightJoint4 : Control.aRightJoint4,
        MujocoMotors.RightJoint5 : Control.aRightJoint5,
        MujocoMotors.RightJoint6 : Control.aRightJoint6,

        MujocoMotors.LeftJoint1  : Control.aLeftJoint1,
        MujocoMotors.LeftJoint2  : Control.aLeftJoint2,
        MujocoMotors.LeftJoint3  : Control.aLeftJoint3,
        MujocoMotors.LeftJoint4  : Control.aLeftJoint4,
        MujocoMotors.LeftJoint5  : Control.aLeftJoint5,
        MujocoMotors.LeftJoint6  : Control.aLeftJoint6,

    }
    
    # ---------------------------------------------------------------------------- #
    #                                    Real                                    #
    # ---------------------------------------------------------------------------- #
    class RealDoFs(IntEnum):
        

        TorsoJoint1 = 0
        TorsoJoint2 = 1
        TorsoJoint3 = 2

        LeftJoint1  = 3
        LeftJoint2  = 4
        LeftJoint3  = 5
        LeftJoint4  = 6
        LeftJoint5  = 7
        LeftJoint6  = 8

        RightJoint1 = 9
        RightJoint2 = 10
        RightJoint3 = 11
        RightJoint4 = 12
        RightJoint5 = 13
        RightJoint6 = 14
    
    # ---------------------------------------------------------------------------- #
    #                                   Real Mappings                                   #
    # ---------------------------------------------------------------------------- #
    
    
    # Mapping from Real DoFs to DoFs
    RealDoF_to_DoF = {
        RealDoFs.TorsoJoint1 : DoFs.TorsoJoint1,
        RealDoFs.TorsoJoint2 : DoFs.TorsoJoint2,
        RealDoFs.TorsoJoint3 : DoFs.TorsoJoint3,

        RealDoFs.RightJoint1 : DoFs.RightJoint1,
        RealDoFs.RightJoint2 : DoFs.RightJoint2,
        RealDoFs.RightJoint3 : DoFs.RightJoint3,
        RealDoFs.RightJoint4 : DoFs.RightJoint4,
        RealDoFs.RightJoint5 : DoFs.RightJoint5,
        RealDoFs.RightJoint6 : DoFs.RightJoint6,

        RealDoFs.LeftJoint1  : DoFs.LeftJoint1,
        RealDoFs.LeftJoint2  : DoFs.LeftJoint2,
        RealDoFs.LeftJoint3  : DoFs.LeftJoint3,
        RealDoFs.LeftJoint4  : DoFs.LeftJoint4,
        RealDoFs.LeftJoint5  : DoFs.LeftJoint5,
        RealDoFs.LeftJoint6  : DoFs.LeftJoint6,

    }

    # Mapping from DoFs to Real DoFs
    DoF_to_RealDoF = {
        DoFs.TorsoJoint1 : RealDoFs.TorsoJoint1,
        DoFs.TorsoJoint2 : RealDoFs.TorsoJoint2,
        DoFs.TorsoJoint3 : RealDoFs.TorsoJoint3,

        DoFs.RightJoint1 : RealDoFs.RightJoint1,
        DoFs.RightJoint2 : RealDoFs.RightJoint2,
        DoFs.RightJoint3 : RealDoFs.RightJoint3,
        DoFs.RightJoint4 : RealDoFs.RightJoint4,
        DoFs.RightJoint5 : RealDoFs.RightJoint5,
        DoFs.RightJoint6 : RealDoFs.RightJoint6,

        DoFs.LeftJoint1  : RealDoFs.LeftJoint1,
        DoFs.LeftJoint2  : RealDoFs.LeftJoint2,
        DoFs.LeftJoint3  : RealDoFs.LeftJoint3,
        DoFs.LeftJoint4  : RealDoFs.LeftJoint4,
        DoFs.LeftJoint5  : RealDoFs.LeftJoint5,
        DoFs.LeftJoint6  : RealDoFs.LeftJoint6,

    }

    # Mapping from real motors to Control
    RealMotor_to_Control = {
        RealMotors.TorsoJoint1 : Control.aTorsoJoint1,
        RealMotors.TorsoJoint2 : Control.aTorsoJoint2,
        RealMotors.TorsoJoint3 : Control.aTorsoJoint3,

        RealMotors.RightJoint1 : Control.aRightJoint1,
        RealMotors.RightJoint2 : Control.aRightJoint2,
        RealMotors.RightJoint3 : Control.aRightJoint3,
        RealMotors.RightJoint4 : Control.aRightJoint4,
        RealMotors.RightJoint5 : Control.aRightJoint5,
        RealMotors.RightJoint6 : Control.aRightJoint6,

        RealMotors.LeftJoint1  : Control.aLeftJoint1,
        RealMotors.LeftJoint2  : Control.aLeftJoint2,
        RealMotors.LeftJoint3  : Control.aLeftJoint3,
        RealMotors.LeftJoint4  : Control.aLeftJoint4,
        RealMotors.LeftJoint5  : Control.aLeftJoint5,
        RealMotors.LeftJoint6  : Control.aLeftJoint6,

    }

    # ---------------------------------------------------------------------------- #
    #                                   Cartesian                                  #
    # ---------------------------------------------------------------------------- #
    
    class Frames(IntEnum):
        
        L_ee = 0
        R_ee = 1
    
    # ---------------------------------------------------------------------------- #
    #                                   Collision                                  #
    # ---------------------------------------------------------------------------- #
    
    CollisionVol = {  
        Frames.L_ee: Geometry(type='sphere', radius=0.05),
        Frames.R_ee: Geometry(type='sphere', radius=0.05),
    }

    # Pairs of adjacent joints to be ignored in collision checking
    AdjacentCollisionVolPairs = []

    SelfCollisionVolIgnored = []
    
    EnvCollisionVolIgnored = []

    VisualizeSafeZone = [
        Frames.L_ee,
        Frames.R_ee,
    ]

    VisualizePhiTraj = [
        Frames.L_ee,
        Frames.R_ee,
    ]
