from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from .robot_workspaces import default_robot_workspace_specs


DUAL_ARM_ROBOTS = {"agibot", "r1lite"}


@dataclass(frozen=True)
class GraspExecutionPose:
    object_prim: str
    primitive_type: str
    candidate_id: str
    score: float
    position_world: tuple[float, float, float]
    quat_wxyz_world: tuple[float, float, float, float]
    approach_axis_world: tuple[float, float, float]
    closing_axis_world: tuple[float, float, float]
    width: float | None
    source_branch: str | None
    source_primitive_index: int

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FilteredGraspExecution:
    grasp: GraspExecutionPose
    arm_side: str
    pre_grasp_pos_world: tuple[float, float, float]
    pre_grasp_quat_world: tuple[float, float, float, float]
    lift_pos_world: tuple[float, float, float]
    lift_quat_world: tuple[float, float, float, float]
    retreat_pos_world: tuple[float, float, float]
    retreat_quat_world: tuple[float, float, float, float]
    base_frame_xy: tuple[float, float]
    pre_grasp_base_frame_xy: tuple[float, float]
    workspace_margin_xy: tuple[float, float]
    support_clearance: float
    score: float
    start_pose_position_error: float | None = None
    start_pose_rotation_error_deg: float | None = None
    ranking_score: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(-1)
    if array.shape[0] != 3:
        raise ValueError(f"Expected 3D vector, got shape {array.shape}")
    norm = float(np.linalg.norm(array))
    if norm < 1e-8:
        return np.zeros(3, dtype=float)
    return array / norm


def _axes_to_rotation_matrix(approach_axis: Any, closing_axis: Any) -> np.ndarray:
    x_axis = _normalize(approach_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    y_axis = np.asarray(closing_axis, dtype=float).reshape(3)
    y_axis = y_axis - x_axis * float(np.dot(y_axis, x_axis))
    y_axis = _normalize(y_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(ref, x_axis))) > 0.95:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
        y_axis = _normalize(np.cross(ref, x_axis))
    z_axis = _normalize(np.cross(x_axis, y_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _quat_wxyz_from_matrix(rotation_matrix: Any) -> tuple[float, float, float, float]:
    quat_xyzw = R.from_matrix(np.asarray(rotation_matrix, dtype=float).reshape(3, 3)).as_quat()
    return (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )


def _matrix_from_quat_wxyz(quat_wxyz: Any) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
    return R.from_quat(quat_xyzw).as_matrix()


def _rotate_xy(x: float, y: float, yaw_deg: float) -> tuple[float, float]:
    yaw_rad = math.radians(float(yaw_deg))
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    return (
        (cos_yaw * float(x)) - (sin_yaw * float(y)),
        (sin_yaw * float(x)) + (cos_yaw * float(y)),
    )


def _world_to_base_xy(base_pose: tuple[float, float, float, float], point_world: tuple[float, float, float]) -> tuple[float, float]:
    base_x, base_y, _base_z, base_yaw_deg = (float(value) for value in base_pose)
    dx = float(point_world[0]) - base_x
    dy = float(point_world[1]) - base_y
    return _rotate_xy(dx, dy, -base_yaw_deg)


def _support_local_xy(
    support_center_xy: tuple[float, float],
    support_yaw_deg: float,
    point_world: tuple[float, float, float],
) -> tuple[float, float]:
    dx = float(point_world[0]) - float(support_center_xy[0])
    dy = float(point_world[1]) - float(support_center_xy[1])
    return _rotate_xy(dx, dy, -float(support_yaw_deg))


def _mean_range_value(value: Any) -> float | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0] + value[1]) * 0.5
    return None


def _sample_axis_slide_values(slide_range_world: Any, sample_count: int) -> list[float]:
    if not isinstance(slide_range_world, (list, tuple)) or len(slide_range_world) != 2:
        return [0.0]
    lo = float(slide_range_world[0])
    hi = float(slide_range_world[1])
    count = max(1, int(sample_count))
    if count == 1 or abs(hi - lo) < 1.0e-8:
        return [float((lo + hi) * 0.5)]
    return [float(value) for value in np.linspace(lo, hi, count)]


def _orthonormal_perp_basis(axis_world: Any) -> tuple[np.ndarray, np.ndarray]:
    axis = _normalize(axis_world)
    ref = np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(np.dot(axis, ref))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    e1 = _normalize(np.cross(axis, ref))
    e2 = _normalize(np.cross(axis, e1))
    return e1, e2


def _make_ring_candidates(
    *,
    object_prim: str,
    primitive_index: int,
    primitive: dict[str, Any],
    center_point: np.ndarray,
    axis_world: np.ndarray,
    radius: float,
    sample_count: int,
    slide_index: int = 0,
) -> list[GraspExecutionPose]:
    poses: list[GraspExecutionPose] = []
    e1, e2 = _orthonormal_perp_basis(axis_world)
    count = max(4, int(sample_count))
    source_branch = primitive.get("source_branch")
    primitive_score = float(primitive.get("score", 1.0))
    for index in range(count):
        theta = (2.0 * math.pi * float(index)) / float(count)
        radial = _normalize((math.cos(theta) * e1) + (math.sin(theta) * e2))
        tangential = _normalize(np.cross(axis_world, radial))
        if np.linalg.norm(tangential) < 1e-8:
            tangential = e1
        position = center_point + radial * float(radius)
        approach_axis = -radial
        base_score = float(primitive_score)
        for orientation_index, closing_axis in enumerate((tangential, -tangential)):
            rotation = _axes_to_rotation_matrix(approach_axis, closing_axis)
            poses.append(
                GraspExecutionPose(
                    object_prim=object_prim,
                    primitive_type="axis_band",
                    candidate_id=f"{object_prim}:axis_ring:{primitive_index}:{int(slide_index)}:{index}:{orientation_index}",
                    score=base_score,
                    position_world=(float(position[0]), float(position[1]), float(position[2])),
                    quat_wxyz_world=_quat_wxyz_from_matrix(rotation),
                    approach_axis_world=(float(approach_axis[0]), float(approach_axis[1]), float(approach_axis[2])),
                    closing_axis_world=(float(closing_axis[0]), float(closing_axis[1]), float(closing_axis[2])),
                    width=float(max(0.0, radius * 2.0)),
                    source_branch=source_branch,
                    source_primitive_index=int(primitive_index),
                )
            )
    return poses


def expand_grasp_candidates(
    scene_grasp_payload: dict[str, Any],
    *,
    target_prim: str | None = None,
    axis_band_slide_samples: int = 3,
    axis_band_ring_samples: int = 12,
) -> list[GraspExecutionPose]:
    objects = scene_grasp_payload.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError("scene_grasp_payload['objects'] must be a dict")

    selected_prim_paths = [str(target_prim)] if target_prim is not None else sorted(objects)
    candidates: list[GraspExecutionPose] = []

    for prim_path in selected_prim_paths:
        object_payload = objects.get(prim_path)
        if not isinstance(object_payload, dict):
            continue
        primitives = object_payload.get("grasp_primitives_world", [])
        if not isinstance(primitives, list):
            continue
        for primitive_index, primitive in enumerate(primitives):
            if not isinstance(primitive, dict):
                continue
            primitive_type = str(primitive.get("type") or "").strip()
            source_branch = primitive.get("source_branch")
            primitive_score = float(primitive.get("score", 1.0))
            if primitive_type == "point_grasp":
                point_world = primitive.get("point_world", [0.0, 0.0, 0.0])
                approach_dirs = primitive.get("approach_dirs_world", [])
                closing_dirs = primitive.get("closing_dirs_world", [])
                if not approach_dirs or not closing_dirs:
                    continue
                width = _mean_range_value(primitive.get("width_range"))
                for approach_idx, approach in enumerate(approach_dirs):
                    for closing_idx, closing in enumerate(closing_dirs):
                        rotation = _axes_to_rotation_matrix(approach, closing)
                        candidates.append(
                            GraspExecutionPose(
                                object_prim=str(prim_path),
                                primitive_type=primitive_type,
                                candidate_id=f"{prim_path}:point:{primitive_index}:{approach_idx}:{closing_idx}",
                                score=primitive_score,
                                position_world=(float(point_world[0]), float(point_world[1]), float(point_world[2])),
                                quat_wxyz_world=_quat_wxyz_from_matrix(rotation),
                                approach_axis_world=tuple(float(value) for value in _normalize(approach)),
                                closing_axis_world=tuple(float(value) for value in _normalize(closing)),
                                width=None if width is None else float(width),
                                source_branch=source_branch,
                                source_primitive_index=int(primitive_index),
                            )
                        )
                continue

            if primitive_type == "axis_band":
                point_world = np.asarray(primitive.get("point_world", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
                axis_world = _normalize(primitive.get("axis_world", [1.0, 0.0, 0.0]))
                slide_values = _sample_axis_slide_values(primitive.get("slide_range_world"), axis_band_slide_samples)
                approach_dirs = primitive.get("approach_dirs_world", [])
                closing_dirs = primitive.get("closing_dirs_world", [])
                width = _mean_range_value(primitive.get("width_range"))
                if approach_dirs and closing_dirs:
                    for slide_idx, slide in enumerate(slide_values):
                        sampled_point = point_world + (axis_world * float(slide))
                        for approach_idx, approach in enumerate(approach_dirs):
                            for closing_idx, closing in enumerate(closing_dirs):
                                rotation = _axes_to_rotation_matrix(approach, closing)
                                candidates.append(
                                    GraspExecutionPose(
                                        object_prim=str(prim_path),
                                        primitive_type=primitive_type,
                                        candidate_id=f"{prim_path}:axis:{primitive_index}:{slide_idx}:{approach_idx}:{closing_idx}",
                                        score=primitive_score,
                                        position_world=(
                                            float(sampled_point[0]),
                                            float(sampled_point[1]),
                                            float(sampled_point[2]),
                                        ),
                                        quat_wxyz_world=_quat_wxyz_from_matrix(rotation),
                                        approach_axis_world=tuple(float(value) for value in _normalize(approach)),
                                        closing_axis_world=tuple(float(value) for value in _normalize(closing)),
                                        width=None if width is None else float(width),
                                        source_branch=source_branch,
                                        source_primitive_index=int(primitive_index),
                                    )
                                )
                elif str(primitive.get("radial_symmetry") or "").strip().lower() == "full":
                    radius = max(0.005, float(width or 0.06) * 0.5)
                    for slide_idx, slide in enumerate(slide_values):
                        sampled_point = point_world + (axis_world * float(slide))
                        candidates.extend(
                            _make_ring_candidates(
                                object_prim=str(prim_path),
                                primitive_index=primitive_index,
                                primitive=primitive,
                                center_point=sampled_point,
                                axis_world=axis_world,
                                radius=radius,
                                sample_count=axis_band_ring_samples,
                                slide_index=slide_idx,
                            )
                        )
                continue

            if primitive_type == "pose_set":
                poses_world = primitive.get("poses_world", [])
                if not isinstance(poses_world, list):
                    continue
                for pose_index, pose in enumerate(poses_world):
                    if not isinstance(pose, dict):
                        continue
                    rotation = np.asarray(pose.get("rotation_matrix"), dtype=float)
                    translation = pose.get("translation", [0.0, 0.0, 0.0])
                    if rotation.shape != (3, 3):
                        continue
                    candidates.append(
                        GraspExecutionPose(
                            object_prim=str(prim_path),
                            primitive_type=primitive_type,
                            candidate_id=f"{prim_path}:pose_set:{primitive_index}:{pose_index}",
                            score=float(pose.get("score", primitive_score)),
                            position_world=(float(translation[0]), float(translation[1]), float(translation[2])),
                            quat_wxyz_world=_quat_wxyz_from_matrix(rotation),
                            approach_axis_world=tuple(float(value) for value in rotation[:, 0]),
                            closing_axis_world=tuple(float(value) for value in rotation[:, 1]),
                            width=float(pose["width"]) if pose.get("width") is not None else None,
                            source_branch=source_branch,
                            source_primitive_index=int(primitive_index),
                        )
                    )

    candidates.sort(key=lambda item: (float(item.score), item.candidate_id), reverse=True)
    return candidates
def infer_arm_side(
    robot: str,
    base_pose: tuple[float, float, float, float],
    position_world: tuple[float, float, float],
    *,
    preferred_arm_side: str | None = None,
    center_deadband: float = 0.06,
) -> str:
    if robot not in DUAL_ARM_ROBOTS:
        return "left"
    preferred = str(preferred_arm_side or "").strip().lower()
    if preferred in {"left", "right"}:
        return preferred
    _x_base, y_base = _world_to_base_xy(base_pose, position_world)
    if y_base > float(center_deadband):
        return "left"
    if y_base < -float(center_deadband):
        return "right"
    return "left"


def _point_in_support_xy(
    support_center_xy: tuple[float, float],
    support_half_extents_xy: tuple[float, float],
    support_yaw_deg: float,
    point_world: tuple[float, float, float],
    *,
    margin: float = 0.0,
) -> bool:
    local_x, local_y = _support_local_xy(support_center_xy, support_yaw_deg, point_world)
    return (
        abs(local_x) <= float(support_half_extents_xy[0]) + float(margin)
        and abs(local_y) <= float(support_half_extents_xy[1]) + float(margin)
    )


def _point_in_bbox(
    point_world: tuple[float, float, float],
    bbox_world: dict[str, Any] | None,
    *,
    margin: float = 0.0,
) -> bool:
    if not isinstance(bbox_world, dict):
        return False
    mins = bbox_world.get("min")
    maxs = bbox_world.get("max")
    if not isinstance(mins, (list, tuple)) or not isinstance(maxs, (list, tuple)) or len(mins) < 3 or len(maxs) < 3:
        return False
    return all(
        float(mins[idx]) - float(margin) <= float(point_world[idx]) <= float(maxs[idx]) + float(margin)
        for idx in range(3)
    )


def filter_grasp_candidates_geometry(
    candidates: list[GraspExecutionPose],
    *,
    robot: str,
    base_pose: tuple[float, float, float, float],
    support_center_xy: tuple[float, float],
    support_half_extents_xy: tuple[float, float],
    support_yaw_deg: float,
    support_top_z: float,
    target_bbox_world: dict[str, Any] | None,
    preferred_arm_side: str | None = None,
    workspace_margin: float = 0.02,
    body_clearance_margin: float = 0.08,
    pre_grasp_distance: float = 0.10,
    lift_height: float = 0.10,
    retreat_distance: float = 0.08,
    approach_clearance: float = 0.006,
    path_samples: int = 7,
) -> list[FilteredGraspExecution]:
    workspace_specs = default_robot_workspace_specs()
    if robot not in workspace_specs:
        raise ValueError(f"Unsupported robot '{robot}'. Choose from: {sorted(workspace_specs)}")
    workspace_spec = workspace_specs[robot]
    working_area = workspace_spec.working_area
    workspace_bounds = working_area.bounds_xy()
    body_half_x = float(workspace_spec.body_half_extents_xy[0]) + float(body_clearance_margin)
    body_half_y = float(workspace_spec.body_half_extents_xy[1]) + float(body_clearance_margin)

    filtered: list[FilteredGraspExecution] = []
    for candidate in candidates:
        arm_side = infer_arm_side(
            robot,
            base_pose,
            candidate.position_world,
            preferred_arm_side=preferred_arm_side,
        )
        grasp_xy_base = _world_to_base_xy(base_pose, candidate.position_world)
        if not (
            workspace_bounds[0] + float(workspace_margin) <= grasp_xy_base[0] <= workspace_bounds[1] - float(workspace_margin)
            and workspace_bounds[2] + float(workspace_margin) <= grasp_xy_base[1] <= workspace_bounds[3] - float(workspace_margin)
        ):
            continue

        if grasp_xy_base[0] <= body_half_x and abs(grasp_xy_base[1]) <= body_half_y:
            continue

        if robot in DUAL_ARM_ROBOTS and preferred_arm_side in {"left", "right"}:
            if arm_side != preferred_arm_side:
                continue
        elif robot in DUAL_ARM_ROBOTS:
            if arm_side == "left" and grasp_xy_base[1] < -0.02:
                continue
            if arm_side == "right" and grasp_xy_base[1] > 0.02:
                continue

        approach_axis = np.asarray(candidate.approach_axis_world, dtype=float)
        approach_axis = _normalize(approach_axis)
        if np.linalg.norm(approach_axis) < 1e-8:
            continue

        grasp_pos = np.asarray(candidate.position_world, dtype=float)
        pre_grasp_pos = grasp_pos - (approach_axis * float(pre_grasp_distance))
        pre_grasp_xy_base = _world_to_base_xy(
            base_pose,
            (float(pre_grasp_pos[0]), float(pre_grasp_pos[1]), float(pre_grasp_pos[2])),
        )
        if not (
            workspace_bounds[0] + float(workspace_margin) <= pre_grasp_xy_base[0] <= workspace_bounds[1] - float(workspace_margin)
            and workspace_bounds[2] + float(workspace_margin) <= pre_grasp_xy_base[1] <= workspace_bounds[3] - float(workspace_margin)
        ):
            continue
        if pre_grasp_xy_base[0] <= body_half_x and abs(pre_grasp_xy_base[1]) <= body_half_y:
            continue

        grasp_world = tuple(float(value) for value in grasp_pos.tolist())
        support_clearance = float(grasp_world[2]) - float(support_top_z)
        if _point_in_support_xy(
            support_center_xy,
            support_half_extents_xy,
            support_yaw_deg,
            grasp_world,
            margin=0.02,
        ) and support_clearance < -0.01:
            continue

        path_ok = True
        sample_count = max(2, int(path_samples))
        for alpha in np.linspace(0.0, 1.0, sample_count, endpoint=True):
            sample_pos = ((1.0 - alpha) * pre_grasp_pos) + (alpha * grasp_pos)
            sample_world = (
                float(sample_pos[0]),
                float(sample_pos[1]),
                float(sample_pos[2]),
            )
            if _point_in_bbox(sample_world, target_bbox_world, margin=0.02):
                continue
            if _point_in_support_xy(
                support_center_xy,
                support_half_extents_xy,
                support_yaw_deg,
                sample_world,
                margin=0.01,
            ) and float(sample_world[2]) < float(support_top_z) + float(approach_clearance):
                path_ok = False
                break
        if not path_ok:
            continue

        lift_pos = grasp_pos + np.array([0.0, 0.0, float(lift_height)], dtype=float)
        retreat_pos = lift_pos - (approach_axis * float(retreat_distance))
        filtered.append(
            FilteredGraspExecution(
                grasp=candidate,
                arm_side=arm_side,
                pre_grasp_pos_world=(float(pre_grasp_pos[0]), float(pre_grasp_pos[1]), float(pre_grasp_pos[2])),
                pre_grasp_quat_world=candidate.quat_wxyz_world,
                lift_pos_world=(float(lift_pos[0]), float(lift_pos[1]), float(lift_pos[2])),
                lift_quat_world=candidate.quat_wxyz_world,
                retreat_pos_world=(float(retreat_pos[0]), float(retreat_pos[1]), float(retreat_pos[2])),
                retreat_quat_world=candidate.quat_wxyz_world,
                base_frame_xy=(float(grasp_xy_base[0]), float(grasp_xy_base[1])),
                pre_grasp_base_frame_xy=(float(pre_grasp_xy_base[0]), float(pre_grasp_xy_base[1])),
                workspace_margin_xy=(float(workspace_margin), float(workspace_margin)),
                support_clearance=support_clearance,
                score=float(candidate.score),
            )
        )

    filtered.sort(
        key=lambda item: (
            float(item.score),
            float(item.support_clearance),
            float(item.base_frame_xy[0]),
        ),
        reverse=True,
    )
    return filtered


def pose_error_metrics(
    current_pos: Any,
    current_quat_wxyz: Any,
    target_pos: Any,
    target_quat_wxyz: Any,
) -> tuple[float, float]:
    current_pos_array = np.asarray(current_pos, dtype=float).reshape(3)
    target_pos_array = np.asarray(target_pos, dtype=float).reshape(3)
    position_error = float(np.linalg.norm(current_pos_array - target_pos_array))

    current_rot = _matrix_from_quat_wxyz(current_quat_wxyz)
    target_rot = _matrix_from_quat_wxyz(target_quat_wxyz)
    relative = R.from_matrix(target_rot @ current_rot.T)
    rotation_error_deg = float(np.degrees(relative.magnitude()))
    return position_error, rotation_error_deg


def rank_filtered_grasp_candidates_by_start_pose(
    candidates: list[FilteredGraspExecution],
    *,
    current_pos_world: tuple[float, float, float],
    current_quat_wxyz: tuple[float, float, float, float],
    position_weight: float = 0.30,
    rotation_weight: float = 0.10,
    use_grasp_orientation: bool = False,
) -> list[FilteredGraspExecution]:
    ranked: list[FilteredGraspExecution] = []
    for candidate in candidates:
        target_quat_wxyz = candidate.grasp.quat_wxyz_world if use_grasp_orientation else candidate.pre_grasp_quat_world
        pos_error, rot_error_deg = pose_error_metrics(
            current_pos_world,
            current_quat_wxyz,
            candidate.pre_grasp_pos_world,
            target_quat_wxyz,
        )
        ranking_score = (
            float(candidate.score)
            - (float(position_weight) * float(pos_error))
            - (float(rotation_weight) * (float(rot_error_deg) / 180.0))
        )
        ranked.append(
            replace(
                candidate,
                start_pose_position_error=float(pos_error),
                start_pose_rotation_error_deg=float(rot_error_deg),
                ranking_score=float(ranking_score),
            )
        )

    ranked.sort(
        key=lambda item: (
            float(item.ranking_score if item.ranking_score is not None else item.score),
            float(item.score),
            float(item.support_clearance),
            float(item.base_frame_xy[0]),
        ),
        reverse=True,
    )
    return ranked
