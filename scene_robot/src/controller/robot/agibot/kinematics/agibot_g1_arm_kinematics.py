from pathlib import Path

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from controller.robot.base.base_robot_config import RobotConfig
from controller.robot.base.base_robot_kinematics import RobotKinematics


class AgibotG1ArmKinematics(RobotKinematics):
    def __init__(self, robot_cfg: RobotConfig, **kwargs) -> None:
        super().__init__(robot_cfg)
        self.robot_cfg = robot_cfg
        self.urdf_path = self._resolve_urdf_path()
        self._init_fixed_base_kinematics()
        self._init_whole_body_kinematics()

    def _resolve_urdf_path(self) -> str:
        current = Path(__file__).resolve()
        for parent in current.parents:
            for relpath in self.robot_cfg.urdf_relpath_candidates:
                candidate = parent / relpath
                if candidate.exists():
                    return str(candidate)
        raise FileNotFoundError(f"Unable to resolve Agibot G1 URDF from {current}")

    def _default_reference_configuration(self, model: pin.Model) -> np.ndarray:
        q0 = pin.neutral(model)
        for joint_name, joint_value in self.robot_cfg.locked_joint_positions.items():
            joint_id = model.getJointId(joint_name)
            if joint_id == 0:
                continue
            joint = model.joints[joint_id]
            q0[joint.idx_q] = joint_value
        return q0

    def _build_reduced_model(self):
        full_model = pin.buildModelFromUrdf(self.urdf_path)
        q0 = self._default_reference_configuration(full_model)
        joint_ids_to_lock = [full_model.getJointId(name) for name in self.robot_cfg.joint_to_lock]
        reduced_model = pin.buildReducedModel(full_model, joint_ids_to_lock, q0)
        return reduced_model

    def _init_fixed_base_kinematics(self):
        self.reduced_fixed_base_model = self._build_reduced_model()
        self.reduced_fixed_base_data = self.reduced_fixed_base_model.createData()
        self.cmodel = cpin.Model(self.reduced_fixed_base_model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_fixed_base_model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.ee_frame_id = self.reduced_fixed_base_model.getFrameId(self.robot_cfg.ee_frame_name)

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf],
            [casadi.vertcat(self.cdata.oMf[self.ee_frame_id].translation - self.cTf[:3, 3])],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf],
            [casadi.vertcat(cpin.log3(self.cdata.oMf[self.ee_frame_id].rotation @ self.cTf[:3, :3].T))],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_fixed_base_model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_fixed_base_model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        trans_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf))
        rot_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf))
        reg_cost = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_fixed_base_model.lowerPositionLimit,
                self.var_q,
                self.reduced_fixed_base_model.upperPositionLimit,
            )
        )
        self.opti.minimize(10.0 * trans_cost + 1.0 * rot_cost + 0.01 * reg_cost + 0.01 * smooth_cost)
        self.opti.solver(
            "ipopt",
            {
                "ipopt": {"print_level": 0, "max_iter": 30, "tol": 1e-4},
                "print_time": False,
                "calc_lam_p": False,
            },
        )
        self.init_data = np.zeros(self.reduced_fixed_base_model.nq)

    def _init_whole_body_kinematics(self):
        self.model = self._build_reduced_model()
        self.data = self.model.createData()
        self.pin_frame_dict = {self.robot_cfg.Frames.EE.name: self.model.getFrameId(self.robot_cfg.ee_frame_name)}

    def update_base_frame(self, trans_world2base, dof):
        return trans_world2base

    def forward_kinematics(self, dof):
        self.pre_computation(dof)
        return self.get_forward_kinematics()

    def inverse_kinematics(self, T, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        ee_target = T[0]
        if current_lr_arm_motor_q is not None:
            self.init_data = np.asarray(current_lr_arm_motor_q).copy()
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_tf, ee_target)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            self.opti.solve()
            sol_q = np.asarray(self.opti.value(self.var_q)).reshape(-1)
            self.init_data = sol_q
            sol_tauff = pin.rnea(
                self.reduced_fixed_base_model,
                self.reduced_fixed_base_data,
                sol_q,
                np.zeros(self.reduced_fixed_base_model.nv),
                np.zeros(self.reduced_fixed_base_model.nv),
            )
            info = {"sol_tauff": sol_tauff, "success": True}
            return sol_q, info
        except Exception as exc:
            sol_q = np.asarray(self.opti.debug.value(self.var_q)).reshape(-1)
            self.init_data = sol_q
            raise RuntimeError("Agibot IK failed") from exc

    def pre_computation(self, dof_pos, dof_vel=None):
        q = np.asarray(dof_pos).reshape(-1)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data)
        pin.updateGlobalPlacements(self.model, self.data)
        if dof_vel is not None:
            dq = np.asarray(dof_vel).reshape(-1)
            pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)

    def get_jacobian(self, frame_name):
        return pin.getFrameJacobian(self.model, self.data, self.model.getFrameId(frame_name), pin.LOCAL_WORLD_ALIGNED)

    def get_jacobian_dot(self, frame_name):
        return pin.getFrameJacobianTimeVariation(
            self.model, self.data, self.model.getFrameId(frame_name), pin.LOCAL_WORLD_ALIGNED
        )

    def get_forward_kinematics(self):
        frames = np.zeros((len(self.robot_cfg.Frames), 4, 4))
        frames[self.robot_cfg.Frames.EE] = self.data.oMf[self.ee_frame_id].homogeneous
        return frames
