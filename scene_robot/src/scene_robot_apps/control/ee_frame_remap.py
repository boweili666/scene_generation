from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


EE_FRAME_REMAP_ROTATIONS = {
    "none": np.eye(3, dtype=float),
    "rot_y_pos_90": R.from_euler("y", 90.0, degrees=True).as_matrix(),
    "rot_y_neg_90": R.from_euler("y", -90.0, degrees=True).as_matrix(),
    "rot_z_pos_90": R.from_euler("z", 90.0, degrees=True).as_matrix(),
    "rot_z_neg_90": R.from_euler("z", -90.0, degrees=True).as_matrix(),
    "rot_x_pos_90_then_z_neg_90": (
        R.from_euler("z", -90.0, degrees=True).as_matrix() @ R.from_euler("x", 90.0, degrees=True).as_matrix()
    ),
    # x_new = z_old, y_new = -x_old, z_new = -y_old
    "x_forward_z_up": np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    ),
}


def quat_wxyz_to_matrix(quat_wxyz: Any) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
    return R.from_quat(quat_xyzw).as_matrix()


def matrix_to_quat_wxyz(rotation_matrix: Any) -> tuple[float, float, float, float]:
    quat_xyzw = R.from_matrix(np.asarray(rotation_matrix, dtype=float).reshape(3, 3)).as_quat()
    return (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )


def apply_local_ee_frame_remap_to_world_quat(
    quat_wxyz: Any,
    remap_name: str | None,
) -> tuple[float, float, float, float]:
    resolved_name = str(remap_name or "none").strip().lower()
    if resolved_name not in EE_FRAME_REMAP_ROTATIONS:
        raise ValueError(f"Unsupported ee frame remap '{remap_name}'. Choose from: {sorted(EE_FRAME_REMAP_ROTATIONS)}")
    rotation_offset = EE_FRAME_REMAP_ROTATIONS[resolved_name]
    base_rotation = quat_wxyz_to_matrix(quat_wxyz)
    return matrix_to_quat_wxyz(base_rotation @ rotation_offset)


def apply_inverse_local_ee_frame_remap_to_world_quat(
    quat_wxyz: Any,
    remap_name: str | None,
) -> tuple[float, float, float, float]:
    resolved_name = str(remap_name or "none").strip().lower()
    if resolved_name not in EE_FRAME_REMAP_ROTATIONS:
        raise ValueError(f"Unsupported ee frame remap '{remap_name}'. Choose from: {sorted(EE_FRAME_REMAP_ROTATIONS)}")
    rotation_offset = EE_FRAME_REMAP_ROTATIONS[resolved_name]
    base_rotation = quat_wxyz_to_matrix(quat_wxyz)
    return matrix_to_quat_wxyz(base_rotation @ rotation_offset.T)


def apply_local_translation_to_world_pos(
    pos_world: Any,
    quat_wxyz: Any,
    translation_local: Any,
) -> tuple[float, float, float]:
    base_pos = np.asarray(pos_world, dtype=float).reshape(3)
    local_translation = np.asarray(translation_local, dtype=float).reshape(3)
    rotation_world = quat_wxyz_to_matrix(quat_wxyz)
    translated = base_pos + (rotation_world @ local_translation)
    return (
        float(translated[0]),
        float(translated[1]),
        float(translated[2]),
    )
