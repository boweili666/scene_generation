from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def _normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    if quat_wxyz.shape != (4,):
        quat_wxyz = quat_wxyz.reshape(4)
    norm = np.linalg.norm(quat_wxyz)
    if not np.isfinite(norm) or norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat_wxyz / norm


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = _normalize_quat_wxyz(quat_wxyz)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float)
    return R.from_quat(quat_xyzw).as_matrix()


def _pose_to_transform(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _quat_wxyz_to_matrix(quat_wxyz)
    transform[:3, 3] = pos
    return transform


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _matrix_to_quat_wxyz(rot_mat: np.ndarray) -> np.ndarray:
    return _quat_xyzw_to_wxyz(R.from_matrix(rot_mat).as_quat())


def _apply_delta_pose_local(ee_pos: torch.Tensor, ee_quat: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos_des = ee_pos + delta[:, :3]
    quat_des = torch.empty_like(ee_quat)
    for env_id in range(delta.shape[0]):
        current_quat = ee_quat[env_id].detach().cpu().numpy()
        delta_rot = R.from_rotvec(delta[env_id, 3:].detach().cpu().numpy())
        current_rot = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
        target_quat = _quat_xyzw_to_wxyz((current_rot * delta_rot).as_quat())
        quat_des[env_id] = torch.as_tensor(target_quat, dtype=ee_quat.dtype, device=ee_quat.device)
    return pos_des, quat_des


@dataclass
class _SolverBundle:
    solve: callable


class _KinovaSingleArmSolver:
    def __init__(self):
        from controller.robot.kinova.config.gen3_single_dynamic_1_config import Gen3SingleDynamic1Config
        from controller.robot.kinova.kinematics.gen3_single_kinematics import Gen3SingleKinematics

        self.robot_cfg = Gen3SingleDynamic1Config()
        self.robot_kinematics = Gen3SingleKinematics(robot_cfg=self.robot_cfg)

    def solve(self, target_pos: np.ndarray, target_quat: np.ndarray, current_q: np.ndarray) -> np.ndarray:
        goal = _pose_to_transform(target_pos, target_quat)
        q_des, _ = self.robot_kinematics.inverse_kinematics([goal], current_q)
        return np.asarray(q_des[: len(current_q)], dtype=float)


class _AgibotSingleArmSolver:
    _URDF_RELPATH_CANDIDATES = (
        "genie_sim/source/geniesim/app/robot_cfg/G1_omnipicker/G1_omnipicker.urdf",
        "genie_sim/source/geniesim/robot/isaac_sim/robot_urdf/G1_omnipicker.urdf",
    )
    _DEFAULT_JOINT_VALUES = {
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
    _TORSO_JOINTS = ("idx01_body_joint1", "idx02_body_joint2")
    _LEFT_JOINTS = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
    _RIGHT_JOINTS = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
    _LEFT_EE_FRAME = "arm_l_end_link"
    _RIGHT_EE_FRAME = "arm_r_end_link"

    def __init__(self, robot, arm_side: str):
        from controller.robot.agibot.config.agibot_g1_arm_dynamic_1_config import AgibotG1ArmDynamic1Config
        import casadi
        import pinocchio as pin
        from pinocchio import casadi as cpin

        self.robot_cfg = AgibotG1ArmDynamic1Config(arm_side=arm_side)
        self.arm_side = arm_side
        self.robot = robot
        self.casadi = casadi
        self.pin = pin

        urdf_path = self._resolve_urdf_path()
        full_model = pin.buildModelFromUrdf(urdf_path)
        q0_full = pin.neutral(full_model)
        for joint_name, joint_value in self._DEFAULT_JOINT_VALUES.items():
            joint_id = full_model.getJointId(joint_name)
            if joint_id == 0:
                continue
            q0_full[full_model.joints[joint_id].idx_q] = joint_value

        # Keep only body + dual-arm DoFs in the controller model.
        controlled_joint_names = set(self._TORSO_JOINTS) | set(self._LEFT_JOINTS) | set(self._RIGHT_JOINTS)
        joint_ids_to_lock = []
        for joint_id in range(1, full_model.njoints):
            joint_name = full_model.names[joint_id]
            if joint_name not in controlled_joint_names:
                joint_ids_to_lock.append(joint_id)

        self.model = pin.buildReducedModel(full_model, joint_ids_to_lock, q0_full)
        self.data = self.model.createData()
        self.nq = self.model.nq

        self.left_frame_id = self.model.getFrameId(self._LEFT_EE_FRAME)
        self.right_frame_id = self.model.getFrameId(self._RIGHT_EE_FRAME)

        self._sim_joint_id_map = {}
        for name in self._TORSO_JOINTS + self._LEFT_JOINTS + self._RIGHT_JOINTS:
            joint_ids, _ = self.robot.find_joints([name])
            if len(joint_ids) == 1:
                self._sim_joint_id_map[name] = int(joint_ids[0])

        self._q_idx_map = {}
        for name in self._TORSO_JOINTS + self._LEFT_JOINTS + self._RIGHT_JOINTS:
            joint_id = self.model.getJointId(name)
            if joint_id != 0:
                self._q_idx_map[name] = int(self.model.joints[joint_id].idx_q)

        self._arm_joint_names = self._LEFT_JOINTS if arm_side == "left" else self._RIGHT_JOINTS
        self._arm_q_indices = [self._q_idx_map[name] for name in self._arm_joint_names]

        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        left_pos_err = self.cdata.oMf[self.left_frame_id].translation - self.cTf_l[:3, 3]
        right_pos_err = self.cdata.oMf[self.right_frame_id].translation - self.cTf_r[:3, 3]
        left_rot_err = self.cdata.oMf[self.left_frame_id].rotation - self.cTf_l[:3, :3]
        right_rot_err = self.cdata.oMf[self.right_frame_id].rotation - self.cTf_r[:3, :3]
        self.left_pos_err_fn = casadi.Function("agibot_left_pos_err", [self.cq, self.cTf_l], [left_pos_err])
        self.right_pos_err_fn = casadi.Function("agibot_right_pos_err", [self.cq, self.cTf_r], [right_pos_err])
        self.left_rot_err_fn = casadi.Function("agibot_left_rot_err", [self.cq, self.cTf_l], [left_rot_err])
        self.right_rot_err_fn = casadi.Function("agibot_right_rot_err", [self.cq, self.cTf_r], [right_rot_err])

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.nq)
        self.param_q_ref = self.opti.parameter(self.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.opti.subject_to(self.opti.bounded(self.model.lowerPositionLimit, self.var_q, self.model.upperPositionLimit))

        translational_cost = casadi.sumsqr(self.left_pos_err_fn(self.var_q, self.param_tf_l)) + casadi.sumsqr(
            self.right_pos_err_fn(self.var_q, self.param_tf_r)
        )
        rotational_cost = casadi.sumsqr(self.left_rot_err_fn(self.var_q, self.param_tf_l)) + casadi.sumsqr(
            self.right_rot_err_fn(self.var_q, self.param_tf_r)
        )
        regularization_cost = casadi.sumsqr(self.var_q - self.param_q_ref)
        self.opti.minimize(80.0 * translational_cost + 2.0 * rotational_cost + 0.2 * regularization_cost)
        self.opti.solver(
            "ipopt",
            {
                "ipopt": {"print_level": 0, "max_iter": 80, "tol": 1e-4},
                "print_time": False,
                "calc_lam_p": False,
            },
        )

        self.init_data = pin.neutral(self.model)

    @classmethod
    def _resolve_urdf_path(cls) -> str:
        current = Path(__file__).resolve()
        for parent in current.parents:
            for relpath in cls._URDF_RELPATH_CANDIDATES:
                candidate = parent / relpath
                if candidate.exists():
                    return str(candidate)
        raise FileNotFoundError("Unable to resolve Agibot G1 URDF from candidates.")

    def _full_q_from_robot(self, env_id: int) -> np.ndarray:
        q = self.init_data.copy()
        joint_pos = self.robot.data.joint_pos[env_id]
        for name, sim_joint_id in self._sim_joint_id_map.items():
            q_idx = self._q_idx_map.get(name)
            if q_idx is None:
                continue
            q[q_idx] = float(joint_pos[sim_joint_id].detach().cpu().item())
        return q

    def solve(self, target_pos: np.ndarray, target_quat: np.ndarray, env_id: int) -> np.ndarray:
        goal = _pose_to_transform(target_pos, target_quat)
        q_ref = self._full_q_from_robot(env_id)

        self.pin.forwardKinematics(self.model, self.data, q_ref)
        self.pin.updateFramePlacements(self.model, self.data)
        left_goal = self.data.oMf[self.left_frame_id].homogeneous.copy()
        right_goal = self.data.oMf[self.right_frame_id].homogeneous.copy()
        if self.arm_side == "left":
            left_goal = goal
        else:
            right_goal = goal

        self.opti.set_value(self.param_q_ref, q_ref)
        self.opti.set_value(self.param_tf_l, left_goal)
        self.opti.set_value(self.param_tf_r, right_goal)
        self.opti.set_initial(self.var_q, q_ref)

        try:
            sol = self.opti.solve()
            q_des = np.asarray(sol.value(self.var_q)).reshape(-1)
        except Exception:
            q_des = np.asarray(self.opti.debug.value(self.var_q)).reshape(-1)

        if q_des.shape != q_ref.shape or not np.all(np.isfinite(q_des)):
            q_des = q_ref.copy()

        self.init_data = q_des
        arm_q = np.asarray([q_des[idx] for idx in self._arm_q_indices], dtype=float)
        return arm_q


class _AgibotBimanualSolver:
    _URDF_RELPATH_CANDIDATES = _AgibotSingleArmSolver._URDF_RELPATH_CANDIDATES
    _DEFAULT_JOINT_VALUES = _AgibotSingleArmSolver._DEFAULT_JOINT_VALUES
    _TORSO_JOINTS = _AgibotSingleArmSolver._TORSO_JOINTS
    _LEFT_JOINTS = _AgibotSingleArmSolver._LEFT_JOINTS
    _RIGHT_JOINTS = _AgibotSingleArmSolver._RIGHT_JOINTS
    _LEFT_EE_FRAME = _AgibotSingleArmSolver._LEFT_EE_FRAME
    _RIGHT_EE_FRAME = _AgibotSingleArmSolver._RIGHT_EE_FRAME

    def __init__(self, robot):
        import casadi
        import pinocchio as pin
        from pinocchio import casadi as cpin

        self.robot = robot
        self.pin = pin
        urdf_path = _AgibotSingleArmSolver._resolve_urdf_path()
        full_model = pin.buildModelFromUrdf(urdf_path)
        q0_full = pin.neutral(full_model)
        for joint_name, joint_value in self._DEFAULT_JOINT_VALUES.items():
            joint_id = full_model.getJointId(joint_name)
            if joint_id == 0:
                continue
            q0_full[full_model.joints[joint_id].idx_q] = joint_value

        controlled_joint_names = set(self._TORSO_JOINTS) | set(self._LEFT_JOINTS) | set(self._RIGHT_JOINTS)
        joint_ids_to_lock = []
        for joint_id in range(1, full_model.njoints):
            joint_name = full_model.names[joint_id]
            if joint_name not in controlled_joint_names:
                joint_ids_to_lock.append(joint_id)

        self.model = pin.buildReducedModel(full_model, joint_ids_to_lock, q0_full)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.left_frame_id = self.model.getFrameId(self._LEFT_EE_FRAME)
        self.right_frame_id = self.model.getFrameId(self._RIGHT_EE_FRAME)

        self._sim_joint_id_map = {}
        for name in self._TORSO_JOINTS + self._LEFT_JOINTS + self._RIGHT_JOINTS:
            joint_ids, _ = self.robot.find_joints([name])
            if len(joint_ids) == 1:
                self._sim_joint_id_map[name] = int(joint_ids[0])

        self._q_idx_map = {}
        for name in self._TORSO_JOINTS + self._LEFT_JOINTS + self._RIGHT_JOINTS:
            joint_id = self.model.getJointId(name)
            if joint_id != 0:
                self._q_idx_map[name] = int(self.model.joints[joint_id].idx_q)

        self._ordered_joint_names = self._TORSO_JOINTS + self._LEFT_JOINTS + self._RIGHT_JOINTS
        self._ordered_q_indices = [self._q_idx_map[name] for name in self._ordered_joint_names]

        cmodel = cpin.Model(self.model)
        cdata = cmodel.createData()
        cq = casadi.SX.sym("q", self.nq, 1)
        cTf_l = casadi.SX.sym("tf_l", 4, 4)
        cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(cmodel, cdata, cq)
        left_pos_err = cdata.oMf[self.left_frame_id].translation - cTf_l[:3, 3]
        right_pos_err = cdata.oMf[self.right_frame_id].translation - cTf_r[:3, 3]
        left_rot_err = cdata.oMf[self.left_frame_id].rotation - cTf_l[:3, :3]
        right_rot_err = cdata.oMf[self.right_frame_id].rotation - cTf_r[:3, :3]
        self.left_pos_err_fn = casadi.Function("agibot_bi_left_pos_err", [cq, cTf_l], [left_pos_err])
        self.right_pos_err_fn = casadi.Function("agibot_bi_right_pos_err", [cq, cTf_r], [right_pos_err])
        self.left_rot_err_fn = casadi.Function("agibot_bi_left_rot_err", [cq, cTf_l], [left_rot_err])
        self.right_rot_err_fn = casadi.Function("agibot_bi_right_rot_err", [cq, cTf_r], [right_rot_err])

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.nq)
        self.param_q_ref = self.opti.parameter(self.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.opti.subject_to(self.opti.bounded(self.model.lowerPositionLimit, self.var_q, self.model.upperPositionLimit))
        translational_cost = casadi.sumsqr(self.left_pos_err_fn(self.var_q, self.param_tf_l)) + casadi.sumsqr(
            self.right_pos_err_fn(self.var_q, self.param_tf_r)
        )
        rotational_cost = casadi.sumsqr(self.left_rot_err_fn(self.var_q, self.param_tf_l)) + casadi.sumsqr(
            self.right_rot_err_fn(self.var_q, self.param_tf_r)
        )
        regularization_cost = casadi.sumsqr(self.var_q - self.param_q_ref)
        self.opti.minimize(80.0 * translational_cost + 2.0 * rotational_cost + 0.2 * regularization_cost)
        self.opti.solver(
            "ipopt",
            {
                "ipopt": {"print_level": 0, "max_iter": 80, "tol": 1e-4},
                "print_time": False,
                "calc_lam_p": False,
            },
        )
        self.init_data = pin.neutral(self.model)

    def _full_q_from_robot(self, env_id: int) -> np.ndarray:
        q = self.init_data.copy()
        joint_pos = self.robot.data.joint_pos[env_id]
        for name, sim_joint_id in self._sim_joint_id_map.items():
            q_idx = self._q_idx_map.get(name)
            if q_idx is None:
                continue
            q[q_idx] = float(joint_pos[sim_joint_id].detach().cpu().item())
        return q

    def solve(
        self,
        left_target_pos: np.ndarray,
        left_target_quat: np.ndarray,
        right_target_pos: np.ndarray,
        right_target_quat: np.ndarray,
        env_id: int,
    ) -> np.ndarray:
        left_goal = _pose_to_transform(left_target_pos, left_target_quat)
        right_goal = _pose_to_transform(right_target_pos, right_target_quat)
        q_ref = self._full_q_from_robot(env_id)
        self.opti.set_value(self.param_q_ref, q_ref)
        self.opti.set_value(self.param_tf_l, left_goal)
        self.opti.set_value(self.param_tf_r, right_goal)
        self.opti.set_initial(self.var_q, q_ref)
        try:
            sol = self.opti.solve()
            q_des = np.asarray(sol.value(self.var_q)).reshape(-1)
        except Exception:
            q_des = np.asarray(self.opti.debug.value(self.var_q)).reshape(-1)
        if q_des.shape != q_ref.shape or not np.all(np.isfinite(q_des)):
            q_des = q_ref.copy()
        self.init_data = q_des
        return np.asarray([q_des[idx] for idx in self._ordered_q_indices], dtype=float)

    def current_ee_poses(self, env_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = self._full_q_from_robot(env_id)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        left_pose = self.data.oMf[self.left_frame_id].homogeneous
        right_pose = self.data.oMf[self.right_frame_id].homogeneous
        left_pos = np.asarray(left_pose[:3, 3], dtype=float)
        left_quat = _matrix_to_quat_wxyz(left_pose[:3, :3])
        right_pos = np.asarray(right_pose[:3, 3], dtype=float)
        right_quat = _matrix_to_quat_wxyz(right_pose[:3, :3])
        return left_pos, left_quat, right_pos, right_quat


class _R1LiteSingleArmSolver:
    TORSO_JOINTS = ("torso_joint1", "torso_joint2", "torso_joint3")
    RIGHT_JOINTS = tuple(f"right_arm_joint{i}" for i in range(1, 7))
    LEFT_JOINTS = tuple(f"left_arm_joint{i}" for i in range(1, 7))

    def __init__(self, robot, arm_side: str):
        from controller.robot.r1lite.config.r1lite_upper_dynamic_1_config import R1LiteUpperDynamic1Config
        from controller.robot.r1lite.kinematics.r1lite_flat_kinematics import R1LiteFlatKinematics

        self.robot = robot
        self.arm_side = arm_side
        self.robot_cfg = R1LiteUpperDynamic1Config()
        self.robot_kinematics = R1LiteFlatKinematics(robot_cfg=self.robot_cfg)
        device = robot.data.joint_pos.device
        self.torso_joint_ids = torch.tensor(robot.find_joints(list(self.TORSO_JOINTS))[0], dtype=torch.long, device=device)
        self.right_joint_ids = torch.tensor(robot.find_joints(list(self.RIGHT_JOINTS))[0], dtype=torch.long, device=device)
        self.left_joint_ids = torch.tensor(robot.find_joints(list(self.LEFT_JOINTS))[0], dtype=torch.long, device=device)

    def _current_full_q(self, env_id: int) -> np.ndarray:
        joint_pos = self.robot.data.joint_pos[env_id]
        torso = joint_pos[self.torso_joint_ids].detach().cpu().numpy()
        left = joint_pos[self.left_joint_ids].detach().cpu().numpy()
        right = joint_pos[self.right_joint_ids].detach().cpu().numpy()
        return np.concatenate([torso, left, right], axis=0)

    def solve(self, target_pos: np.ndarray, target_quat: np.ndarray, env_id: int) -> np.ndarray:
        full_q = self._current_full_q(env_id)
        fk_frames = self.robot_kinematics.forward_kinematics(full_q)
        left_goal = fk_frames[self.robot_cfg.Frames.L_ee]
        right_goal = fk_frames[self.robot_cfg.Frames.R_ee]
        target_goal = _pose_to_transform(target_pos, target_quat)
        if self.arm_side == "left":
            left_goal = target_goal
        else:
            right_goal = target_goal
        q_des, _ = self.robot_kinematics.inverse_kinematics([left_goal, right_goal], full_q)
        q_des = np.asarray(q_des, dtype=float)
        return q_des[3:9] if self.arm_side == "left" else q_des[9:15]


class _R1LiteBimanualSolver:
    TORSO_JOINTS = ("torso_joint1", "torso_joint2", "torso_joint3")
    RIGHT_JOINTS = tuple(f"right_arm_joint{i}" for i in range(1, 7))
    LEFT_JOINTS = tuple(f"left_arm_joint{i}" for i in range(1, 7))

    def __init__(self, robot):
        from controller.robot.r1lite.config.r1lite_upper_dynamic_1_config import R1LiteUpperDynamic1Config
        from controller.robot.r1lite.kinematics.r1lite_flat_kinematics import R1LiteFlatKinematics

        self.robot = robot
        self.robot_cfg = R1LiteUpperDynamic1Config()
        self.robot_kinematics = R1LiteFlatKinematics(robot_cfg=self.robot_cfg)
        device = robot.data.joint_pos.device
        self.torso_joint_ids = torch.tensor(robot.find_joints(list(self.TORSO_JOINTS))[0], dtype=torch.long, device=device)
        self.right_joint_ids = torch.tensor(robot.find_joints(list(self.RIGHT_JOINTS))[0], dtype=torch.long, device=device)
        self.left_joint_ids = torch.tensor(robot.find_joints(list(self.LEFT_JOINTS))[0], dtype=torch.long, device=device)

    def _current_full_q(self, env_id: int) -> np.ndarray:
        joint_pos = self.robot.data.joint_pos[env_id]
        torso = joint_pos[self.torso_joint_ids].detach().cpu().numpy()
        left = joint_pos[self.left_joint_ids].detach().cpu().numpy()
        right = joint_pos[self.right_joint_ids].detach().cpu().numpy()
        return np.concatenate([torso, left, right], axis=0)

    def solve(
        self,
        left_target_pos: np.ndarray,
        left_target_quat: np.ndarray,
        right_target_pos: np.ndarray,
        right_target_quat: np.ndarray,
        env_id: int,
    ) -> np.ndarray:
        full_q = self._current_full_q(env_id)
        left_goal = _pose_to_transform(left_target_pos, left_target_quat)
        right_goal = _pose_to_transform(right_target_pos, right_target_quat)
        q_des, _ = self.robot_kinematics.inverse_kinematics([left_goal, right_goal], full_q)
        return np.asarray(q_des, dtype=float)

    def current_ee_poses(self, env_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        full_q = self._current_full_q(env_id)
        fk_frames = self.robot_kinematics.forward_kinematics(full_q)
        right_pose = fk_frames[self.robot_cfg.Frames.R_ee]
        left_pose = fk_frames[self.robot_cfg.Frames.L_ee]
        left_pos = np.asarray(left_pose[:3, 3], dtype=float)
        left_quat = _matrix_to_quat_wxyz(left_pose[:3, :3])
        right_pos = np.asarray(right_pose[:3, 3], dtype=float)
        right_quat = _matrix_to_quat_wxyz(right_pose[:3, :3])
        return left_pos, left_quat, right_pos, right_quat


class IsaacLabPoseController:
    def __init__(
        self,
        robot_name: str,
        num_envs: int,
        device: str,
        robot=None,
        arm_side: str = "left",
        use_relative_mode: bool = False,
    ):
        self.robot_name = robot_name
        self.num_envs = num_envs
        self.device = device
        self.robot = robot
        self.arm_side = arm_side
        self.use_relative_mode = use_relative_mode
        self.ee_pos_des = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.ee_quat_des = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
        self.ee_quat_des[:, 0] = 1.0
        self._solvers = [self._build_solver() for _ in range(num_envs)]

    def _build_solver(self):
        if self.robot_name == "kinova":
            return _KinovaSingleArmSolver()
        if self.robot_name == "agibot":
            if self.robot is None:
                raise ValueError("agibot pose controller requires robot handle")
            return _AgibotSingleArmSolver(self.robot, self.arm_side)
        if self.robot_name == "r1lite":
            if self.robot is None:
                raise ValueError("r1lite pose controller requires robot handle")
            return _R1LiteSingleArmSolver(self.robot, self.arm_side)
        raise ValueError(f"Unsupported robot for pose controller: {self.robot_name}")

    def set_command(self, command: torch.Tensor, ee_pos: torch.Tensor | None = None, ee_quat: torch.Tensor | None = None):
        if self.use_relative_mode:
            if ee_pos is None or ee_quat is None:
                raise ValueError("Relative pose controller requires current ee pose")
            self.ee_pos_des, self.ee_quat_des = _apply_delta_pose_local(ee_pos, ee_quat, command)
        else:
            self.ee_pos_des = command[:, :3].clone()
            self.ee_quat_des = command[:, 3:7].clone()

    def compute(self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        del ee_pos, ee_quat, jacobian
        joint_pos_des = joint_pos.clone()
        for env_id, solver in enumerate(self._solvers):
            target_pos = self.ee_pos_des[env_id].detach().cpu().numpy()
            target_quat = self.ee_quat_des[env_id].detach().cpu().numpy()
            if self.robot_name in {"r1lite", "agibot"}:
                q_des = solver.solve(target_pos, target_quat, env_id)
            else:
                current_q = joint_pos[env_id].detach().cpu().numpy()
                q_des = solver.solve(target_pos, target_quat, current_q)
            joint_pos_des[env_id] = torch.as_tensor(q_des, device=joint_pos.device, dtype=joint_pos.dtype)
        return joint_pos_des


class IsaacLabBimanualPoseController:
    def __init__(self, robot_name: str, num_envs: int, device: str, robot):
        self.robot_name = robot_name
        self.num_envs = num_envs
        self.device = device
        self.robot = robot
        self.left_ee_pos_des = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.left_ee_quat_des = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
        self.left_ee_quat_des[:, 0] = 1.0
        self.right_ee_pos_des = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.right_ee_quat_des = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
        self.right_ee_quat_des[:, 0] = 1.0
        if robot_name not in {"r1lite", "agibot"}:
            raise ValueError(f"Unsupported robot for bimanual pose controller: {robot_name}")
        if robot_name == "r1lite":
            self._solvers = [_R1LiteBimanualSolver(robot) for _ in range(num_envs)]
        else:
            self._solvers = [_AgibotBimanualSolver(robot) for _ in range(num_envs)]

    def set_command(
        self,
        left_command: torch.Tensor,
        right_command: torch.Tensor,
    ):
        self.left_ee_pos_des = left_command[:, :3].clone()
        self.left_ee_quat_des = left_command[:, 3:7].clone()
        self.right_ee_pos_des = right_command[:, :3].clone()
        self.right_ee_quat_des = right_command[:, 3:7].clone()

    def compute(self) -> torch.Tensor:
        q_des_list: list[torch.Tensor] = []
        for env_id, solver in enumerate(self._solvers):
            q_des = solver.solve(
                self.left_ee_pos_des[env_id].detach().cpu().numpy(),
                self.left_ee_quat_des[env_id].detach().cpu().numpy(),
                self.right_ee_pos_des[env_id].detach().cpu().numpy(),
                self.right_ee_quat_des[env_id].detach().cpu().numpy(),
                env_id,
            )
            q_des_list.append(torch.as_tensor(q_des, device=self.device, dtype=torch.float32))
        return torch.stack(q_des_list, dim=0)

    def current_ee_poses_base(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        left_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        right_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        right_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        for env_id, solver in enumerate(self._solvers):
            l_pos, l_quat, r_pos, r_quat = solver.current_ee_poses(env_id)
            left_pos[env_id] = torch.as_tensor(l_pos, device=self.device, dtype=torch.float32)
            left_quat[env_id] = torch.as_tensor(l_quat, device=self.device, dtype=torch.float32)
            right_pos[env_id] = torch.as_tensor(r_pos, device=self.device, dtype=torch.float32)
            right_quat[env_id] = torch.as_tensor(r_quat, device=self.device, dtype=torch.float32)
        return left_pos, left_quat, right_pos, right_quat
