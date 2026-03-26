import numpy as np

from controller.base_pid_policy import BasePIDPolicy


class TrajTrackingPolicy(BasePIDPolicy):
    def __init__(self, robot_cfg, robot_kinematics) -> None:
        super().__init__(robot_cfg, robot_kinematics)
        self.target_traj = None
        self.traj_cnt = 0
        self.dof_pos_target_traj = []

    def plan(self, target_traj_cur, agent_feedback: dict, task_info: dict):
        dof_pos_fbk = agent_feedback["dof_pos_fbk"]
        goal_teleop = task_info["goal_teleop"]
        traj_len = len(goal_teleop["right"])
        if self.target_traj_changed(target_traj_cur):
            self.dof_pos_target_traj = []
            dof_pos_tmp = dof_pos_fbk.copy()
            for t in range(traj_len):
                goals = []
                if "left" in goal_teleop:
                    goals.append(target_traj_cur["left"][t])
                goals.append(target_traj_cur["right"][t])
                dof_pos_target = self.solve_ik_targets(goals, dof_pos_tmp)
                dof_pos_tmp = dof_pos_target
                self.dof_pos_target_traj.append(dof_pos_target)
            self.traj_cnt = 0
        else:
            if np.linalg.norm(dof_pos_fbk - self.dof_pos_target_traj[self.traj_cnt]) < 0.01:
                self.traj_cnt += 1
            if self.traj_cnt >= traj_len:
                self.traj_cnt = 0
        return self.dof_pos_target_traj[self.traj_cnt]

    def target_traj_changed(self, target_traj_cur):
        if self.target_traj is None:
            self.target_traj = target_traj_cur
            return True
        if len(self.target_traj["right"]) != len(target_traj_cur["right"]):
            self.target_traj = target_traj_cur
            return True
        for t in range(len(target_traj_cur["right"])):
            if not np.allclose(self.target_traj["right"][t], target_traj_cur["right"][t]):
                self.target_traj = target_traj_cur
                return True
            if "left" in target_traj_cur:
                if not np.allclose(self.target_traj["left"][t], target_traj_cur["left"][t]):
                    self.target_traj = target_traj_cur
                    return True
        return False

    def act(self, agent_feedback: dict, task_info: dict):
        info = {}
        dof_pos_cmd = agent_feedback["dof_pos_cmd"]
        dof_vel_cmd = agent_feedback["dof_vel_cmd"]
        dof_pos_fbk = agent_feedback["dof_pos_fbk"]
        dof_vel_fbk = agent_feedback["dof_vel_fbk"]

        robot_base_frame = agent_feedback["robot_base_frame"]
        target_traj_cur = {"right": np.linalg.inv(robot_base_frame) @ task_info["goal_teleop"]["right"]}
        if "left" in task_info["goal_teleop"]:
            target_traj_cur["left"] = np.linalg.inv(robot_base_frame) @ task_info["goal_teleop"]["left"]

        try:
            dof_pos_target = self.plan(target_traj_cur, agent_feedback, task_info)

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

        if "left_gripper_goal" in task_info["goal_teleop"]:
            info["left_gripper_goal"] = task_info["goal_teleop"]["left_gripper_goal"]
        if "right_gripper_goal" in task_info["goal_teleop"]:
            info["right_gripper_goal"] = task_info["goal_teleop"]["right_gripper_goal"]

        return control, info
