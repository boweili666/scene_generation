import numpy as np

from controller.base_pid_policy import BasePIDPolicy


class TeleopPIDPolicy(BasePIDPolicy):
    def act(self, agent_feedback: dict, task_info: dict):
        info = {}
        dof_pos_cmd = agent_feedback["dof_pos_cmd"]
        dof_vel_cmd = agent_feedback["dof_vel_cmd"]
        dof_pos_fbk = agent_feedback["dof_pos_fbk"]
        dof_vel_fbk = agent_feedback["dof_vel_fbk"]
        robot_base_frame = agent_feedback["robot_base_frame"]
        goal_teleop = task_info["goal_teleop"]
        try:
            goals = []
            if "left" in goal_teleop:
                goals.append(np.linalg.inv(robot_base_frame) @ goal_teleop["left"][0])
            goals.append(np.linalg.inv(robot_base_frame) @ goal_teleop["right"][0])
            dof_pos_target = self.solve_ik_targets(goals, dof_pos_fbk)

            if "Dynamic1" in self.robot_cfg.__class__.__name__:
                control = self.tracking_pos_with_vel(
                    dof_pos_target,
                    np.zeros_like(dof_vel_fbk),
                    dof_pos_fbk,
                    dof_vel_fbk,
                    1.0 * np.ones(len(self.robot_cfg.Control)),
                    0.0 * np.ones(len(self.robot_cfg.Control)),
                )
            elif "Dynamic2" in self.robot_cfg.__class__.__name__:
                control = self.tracking_pos_with_acc_zero_target(
                    dof_pos_target,
                    dof_pos_cmd,
                    dof_vel_cmd,
                    1.0 * np.ones(len(self.robot_cfg.Control)),
                    0.0 * np.ones(len(self.robot_cfg.Control)),
                )
            else:
                raise ValueError(f"Unsupported robot config {self.robot_cfg.__class__.__name__}")

            info["ik_success"] = True
        except Exception as e:
            print("inverse_kinematics error", e)
            control = np.zeros_like(dof_pos_fbk)
            info["ik_success"] = False

        control = self.clamp_control(control)

        if "left_gripper_goal" in goal_teleop:
            info["left_gripper_goal"] = goal_teleop["left_gripper_goal"]
        if "right_gripper_goal" in goal_teleop:
            info["right_gripper_goal"] = goal_teleop["right_gripper_goal"]

        return control, info
