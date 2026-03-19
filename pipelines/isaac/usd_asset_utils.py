from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)
Z_UP_TO_Y_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def normalize_up_axis_token(token: Any, *, default: str = "Y") -> str:
    text = str(token or default).strip().upper()
    if text.startswith("Z"):
        return "Z"
    return "Y"


def up_axis_correction_matrix(source_up_axis: Any, target_up_axis: Any) -> np.ndarray:
    source = normalize_up_axis_token(source_up_axis)
    target = normalize_up_axis_token(target_up_axis)
    if source == target:
        return np.eye(4, dtype=float)
    if source == "Y" and target == "Z":
        return Y_UP_TO_Z_UP.copy()
    if source == "Z" and target == "Y":
        return Z_UP_TO_Y_UP.copy()
    raise ValueError(f"Unsupported up-axis conversion: {source} -> {target}")


def compute_stage_local_to_scene_matrix(
    stage,
    *,
    target_up_axis: str = "Z",
    target_meters_per_unit: float = 1.0,
) -> np.ndarray:
    from pxr import UsdGeom

    if target_meters_per_unit <= 0.0:
        raise ValueError("target_meters_per_unit must be positive")

    source_up_axis = normalize_up_axis_token(UsdGeom.GetStageUpAxis(stage))
    source_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))

    correction = up_axis_correction_matrix(source_up_axis, target_up_axis)
    correction = correction.copy()
    correction[:3, :3] *= source_meters_per_unit / float(target_meters_per_unit)
    return correction


def compute_asset_local_to_scene_matrix(
    usd_path: Path | str,
    *,
    target_up_axis: str = "Z",
    target_meters_per_unit: float = 1.0,
) -> np.ndarray:
    from pxr import Usd

    stage = Usd.Stage.Open(str(Path(usd_path).expanduser().resolve()))
    if stage is None:
        raise ValueError(f"Failed to open USD stage: {usd_path}")
    return compute_stage_local_to_scene_matrix(
        stage,
        target_up_axis=target_up_axis,
        target_meters_per_unit=target_meters_per_unit,
    )


def is_identity_matrix(matrix: Any, *, atol: float = 1e-8) -> bool:
    return bool(np.allclose(np.asarray(matrix, dtype=float), np.eye(4, dtype=float), atol=atol))


def column_transform_to_row_major(matrix: Any) -> np.ndarray:
    return np.asarray(matrix, dtype=float).T.copy()


def transform_aligned_bbox(info: dict[str, Any], matrix: Any) -> dict[str, tuple[float, float, float]]:
    matrix_np = np.asarray(matrix, dtype=float)
    if is_identity_matrix(matrix_np):
        return {
            "size": tuple(float(v) for v in info["size"]),
            "center": tuple(float(v) for v in info["center"]),
        }

    size = np.asarray(info["size"], dtype=float)
    center = np.asarray(info["center"], dtype=float)
    half = size * 0.5

    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(center + np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=float))

    corners_h = np.concatenate([np.asarray(corners, dtype=float), np.ones((8, 1), dtype=float)], axis=1)
    transformed = (matrix_np @ corners_h.T).T[:, :3]
    mins = transformed.min(axis=0)
    maxs = transformed.max(axis=0)
    return {
        "size": tuple(float(v) for v in (maxs - mins)),
        "center": tuple(float(v) for v in ((mins + maxs) * 0.5)),
    }
