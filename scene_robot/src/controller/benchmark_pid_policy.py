import numpy as np
from scipy.spatial.transform import Rotation as R

from controller.base_pid_policy import BasePIDPolicy


class BenchmarkPIDPolicy(BasePIDPolicy):
    def act(self, agent_feedback: dict, task_info: dict):
        info = {}

        dof_pos_cmd = agent_feedback["dof_pos_cmd"]
        dof_vel_fbk = agent_feedback["dof_vel_fbk"]
        dof_pos_fbk = agent_feedback["dof_pos_fbk"]
        goal_teleop = task_info["goal_teleop"]
        robot_base_frame = agent_feedback["robot_base_frame"]

        dof_pos_current = dof_pos_cmd.copy()
        for dof in self.robot_cfg.DoFs:
            if dof.name in ["LinearX", "LinearY", "RotYaw"]:
                dof_pos_current[dof] = dof_pos_fbk[dof]

        try:
            dof_pos_target = dof_pos_fbk.copy()
            if task_info["arm_goal_enable"]:
                goals = []
                if "left" in goal_teleop:
                    goals.append(np.linalg.inv(robot_base_frame) @ goal_teleop["left"][0])
                goals.append(np.linalg.inv(robot_base_frame) @ goal_teleop["right"][0])
                dof_pos_target = self.solve_ik_targets(goals, dof_pos_fbk)

            if task_info["base_goal_enable"]:
                rot = R.from_matrix(goal_teleop["base"][:3, :3])
                euler = rot.as_euler("xyz")
                dof_pos_target[self.robot_cfg.DoFs.LinearX] = goal_teleop["base"][0, 3]
                dof_pos_target[self.robot_cfg.DoFs.LinearY] = goal_teleop["base"][1, 3]
                dof_pos_target[self.robot_cfg.DoFs.RotYaw] = euler[2]

            if "Dynamic1" in self.robot_cfg.__class__.__name__:
                control = self.tracking_pos_with_vel(
                    dof_pos_target,
                    np.zeros_like(dof_vel_fbk),
                    dof_pos_fbk,
                    dof_vel_fbk,
                    1.0 * np.ones(len(self.robot_cfg.Control)),
                    0.1 * np.ones(len(self.robot_cfg.Control)),
                )
            elif "Dynamic2" in self.robot_cfg.__class__.__name__:
                control = self.tracking_pos_with_acc(
                    dof_pos_target,
                    dof_pos_fbk,
                    dof_vel_fbk,
                    1.0 * np.ones(len(self.robot_cfg.DoFs)),
                    0.4 * np.ones(len(self.robot_cfg.DoFs)),
                    10.0 * np.ones(len(self.robot_cfg.DoFs)),
                    5.0 * np.ones(len(self.robot_cfg.DoFs)),
                )
            else:
                raise ValueError(f"Unsupported robot config {self.robot_cfg.__class__.__name__}")

            info["ik_success"] = True
        except Exception as e:
            print("inverse_kinematics error", e)
            control = np.zeros_like(dof_pos_cmd)
            info["ik_success"] = False
            raise e

        return self.clamp_control(control), info
