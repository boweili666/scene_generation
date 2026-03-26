import numpy as np


class BasePIDPolicy:
    def __init__(self, robot_cfg, robot_kinematics) -> None:
        self.robot_cfg = robot_cfg
        self.robot_kinematics = robot_kinematics
        self.num_dof = len(self.robot_cfg.DoFs)
        self.num_control = len(self.robot_cfg.Control)

    def tracking_pos_with_vel(
        self,
        dof_pos_target,
        dof_vel_target,
        dof_pos_current,
        dof_vel_current,
        k_p,
        k_d,
    ):
        dof_vel_nominal = k_p * (dof_pos_target - dof_pos_current) + k_d * (
            dof_vel_target - dof_vel_current
        )
        nominal_control = (
            np.linalg.pinv(self.robot_cfg.dynamics_g(dof_pos_current)) @ dof_vel_nominal
        )
        return nominal_control

    def tracking_pos_with_acc(
        self,
        dof_pos_target,
        dof_pos_current,
        dof_vel_current,
        k_p_vel,
        k_d_vel,
        k_p_acc,
        k_d_acc,
    ):
        nominal_dof_vel = k_p_vel * (dof_pos_target - dof_pos_current) - k_d_vel * dof_vel_current
        nominal_dof_acc = k_p_acc * (dof_pos_target - dof_pos_current) - k_d_acc * dof_vel_current
        nominal_control = np.linalg.pinv(
            self.robot_cfg.dynamics_g(np.concatenate((dof_pos_current, dof_vel_current)))
        ) @ np.concatenate((nominal_dof_vel, nominal_dof_acc))
        return nominal_control

    def tracking_pos_with_acc_zero_target(
        self,
        dof_pos_target,
        dof_pos_current,
        dof_vel_current,
        k_p,
        k_d,
    ):
        nominal_dof_vel = k_p * (dof_pos_target - dof_pos_current) + k_d * dof_vel_current
        nominal_dof_acc = nominal_dof_vel - dof_vel_current
        nominal_control = np.linalg.pinv(
            self.robot_cfg.dynamics_g(np.concatenate((dof_pos_current, dof_vel_current)))
        ) @ np.concatenate((nominal_dof_vel, nominal_dof_acc))
        return nominal_control

    def solve_ik_targets(self, goals, dof_pos_seed):
        dof_pos_target, _ = self.robot_kinematics.inverse_kinematics(goals, dof_pos_seed)
        return dof_pos_target

    def clamp_control(self, control):
        for control_id in self.robot_cfg.Control:
            control[control_id] = np.clip(
                control[control_id],
                -self.robot_cfg.ControlLimit[control_id],
                self.robot_cfg.ControlLimit[control_id],
            )
        return control
