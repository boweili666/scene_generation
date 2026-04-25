from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from .execution import FilteredGraspExecution


def _import_pxr():
    from pxr import Gf, UsdGeom

    return Gf, UsdGeom


def _gf_matrix_to_numpy(matrix) -> np.ndarray:
    return np.array([[float(matrix[i][j]) for j in range(4)] for i in range(4)], dtype=float)


def _rotation_only_row(linear_3x3: np.ndarray) -> np.ndarray:
    # Row-vector: p_out = p_in @ linear. Drop any non-unit scale via SVD.
    col_linear = np.asarray(linear_3x3, dtype=float).T
    u, _s, vh = np.linalg.svd(col_linear)
    rotation_col = u @ vh
    if np.linalg.det(rotation_col) < 0.0:
        u[:, -1] *= -1.0
        rotation_col = u @ vh
    return rotation_col.T


def _resolve_parent_prim(stage, parent_prim_path: str):
    # Accept both the literal scene-graph path (e.g. `/World/bolt_2`) and the
    # live env-clone (e.g. `/World/envs/env_0/GeneratedScene/bolt_2`) that
    # isaaclab's InteractiveScene produces when loading the scene USD as a
    # UsdFileCfg reference. Mirrors `grasp_scene_adapter._resolve_stage_prim`'s
    # preference order so the visualization runs in both entry-points.
    from pathlib import PurePosixPath

    literal = stage.GetPrimAtPath(parent_prim_path)
    prim_name = PurePosixPath(parent_prim_path).name
    suffix = f"/{prim_name}"

    best_prim = None
    best_rank: tuple[int, int, str] | None = None
    for candidate in stage.Traverse():
        if not candidate.IsValid():
            continue
        path_str = str(candidate.GetPath())
        if not path_str.endswith(suffix):
            continue
        if "/envs/" in path_str and f"/GeneratedScene/{prim_name}" in path_str:
            rank = (0, path_str.count("/"), path_str)
        elif f"/GeneratedScene/{prim_name}" in path_str:
            rank = (1, path_str.count("/"), path_str)
        elif path_str == parent_prim_path:
            rank = (2, path_str.count("/"), path_str)
        elif "/envs/" in path_str:
            rank = (3, path_str.count("/"), path_str)
        else:
            rank = (4, path_str.count("/"), path_str)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_prim = candidate

    if best_prim is not None:
        return best_prim
    return literal if literal.IsValid() else None


def _compute_parent_world_transform(stage, parent_prim_path: str | None):
    # Return the parent prim plus (world_row_inv, rot_row_inv) used to convert
    # world-space pose tuples into the parent's local frame. Return None if
    # `parent_prim_path` is falsy or the prim can't be resolved / isn't
    # Xformable.
    if not parent_prim_path:
        return None
    from pxr import Usd, UsdGeom

    parent_prim = _resolve_parent_prim(stage, parent_prim_path)
    if parent_prim is None or not parent_prim.IsValid() or not parent_prim.IsA(UsdGeom.Xformable):
        return None
    world_row = _gf_matrix_to_numpy(
        UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    )
    try:
        world_row_inv = np.linalg.inv(world_row)
    except np.linalg.LinAlgError:
        return None
    rot_row = _rotation_only_row(world_row[:3, :3])
    rot_row_inv = rot_row.T
    return parent_prim, world_row_inv, rot_row_inv


def _world_pose_to_parent_local(
    position_world: tuple[float, float, float],
    quat_wxyz_world: tuple[float, float, float, float],
    parent_world_row_inv: np.ndarray,
    parent_rot_row_inv: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    pw = np.array([float(position_world[0]), float(position_world[1]), float(position_world[2]), 1.0], dtype=float)
    p_local = pw @ parent_world_row_inv
    if abs(float(p_local[3])) > 1e-12:
        p_local = p_local / float(p_local[3])
    quat_xyzw_world = np.array(
        [float(quat_wxyz_world[1]), float(quat_wxyz_world[2]), float(quat_wxyz_world[3]), float(quat_wxyz_world[0])],
        dtype=float,
    )
    rot_world_col = R.from_quat(quat_xyzw_world).as_matrix()
    # Convert column-vector rotation to row-vector convention, compose with parent
    # row-rotation inverse, then convert back to column-vector for scipy.
    rot_world_row = rot_world_col.T
    rot_local_row = rot_world_row @ parent_rot_row_inv
    rot_local_col = rot_local_row.T
    quat_xyzw_local = R.from_matrix(rot_local_col).as_quat()
    return (
        (float(p_local[0]), float(p_local[1]), float(p_local[2])),
        (
            float(quat_xyzw_local[3]),
            float(quat_xyzw_local[0]),
            float(quat_xyzw_local[1]),
            float(quat_xyzw_local[2]),
        ),
    )


def _gf_quatd_from_wxyz(quat_wxyz: tuple[float, float, float, float]):
    Gf, _UsdGeom = _import_pxr()
    return Gf.Quatf(
        float(quat_wxyz[0]),
        Gf.Vec3f(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
    )


def _set_visual_prim_appearance(geom, color: tuple[float, float, float], opacity: float) -> None:
    _Gf, UsdGeom = _import_pxr()
    geom.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([tuple(float(value) for value in color)])
    geom.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([float(opacity)])
    imageable = UsdGeom.Imageable(geom.GetPrim())
    imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.render)
    imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)


def _add_cube(
    stage,
    prim_path: str,
    *,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    color: tuple[float, float, float],
    opacity: float,
) -> str:
    Gf, UsdGeom = _import_pxr()
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in translate]))
    xform.AddScaleOp().Set(Gf.Vec3f(*[max(1.0e-4, float(value)) for value in scale]))
    _set_visual_prim_appearance(cube, color, opacity)
    return prim_path


def _add_sphere(
    stage,
    prim_path: str,
    *,
    center: tuple[float, float, float],
    radius: float,
    color: tuple[float, float, float],
    opacity: float,
) -> str:
    Gf, UsdGeom = _import_pxr()
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.CreateRadiusAttr(max(1.0e-4, float(radius)))
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in center]))
    _set_visual_prim_appearance(sphere, color, opacity)
    return prim_path


def _add_pose_frame(
    stage,
    prim_path: str,
    *,
    position_world: tuple[float, float, float],
    quat_wxyz_world: tuple[float, float, float, float],
    axis_length: float,
    axis_thickness: float,
    opacity: float,
) -> list[str]:
    Gf, UsdGeom = _import_pxr()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root_xform = UsdGeom.Xformable(root.GetPrim())
    root_xform.ClearXformOpOrder()
    root_xform.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in position_world]))
    root_xform.AddOrientOp().Set(_gf_quatd_from_wxyz(quat_wxyz_world))
    root_imageable = UsdGeom.Imageable(root.GetPrim())
    root_imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.render)
    root_imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)

    created = [
        _add_cube(
            stage,
            f"{prim_path}/AxisX",
            translate=(float(axis_length) * 0.5, 0.0, 0.0),
            scale=(float(axis_length), float(axis_thickness), float(axis_thickness)),
            color=(0.95, 0.25, 0.25),
            opacity=opacity,
        ),
        _add_cube(
            stage,
            f"{prim_path}/AxisY",
            translate=(0.0, float(axis_length) * 0.5, 0.0),
            scale=(float(axis_thickness), float(axis_length), float(axis_thickness)),
            color=(0.25, 0.95, 0.25),
            opacity=opacity,
        ),
        _add_cube(
            stage,
            f"{prim_path}/AxisZ",
            translate=(0.0, 0.0, float(axis_length) * 0.5),
            scale=(float(axis_thickness), float(axis_thickness), float(axis_length)),
            color=(0.25, 0.45, 0.95),
            opacity=opacity,
        ),
        _add_cube(
            stage,
            f"{prim_path}/Center",
            translate=(0.0, 0.0, 0.0),
            scale=(float(axis_thickness) * 1.6, float(axis_thickness) * 1.6, float(axis_thickness) * 1.6),
            color=(0.96, 0.96, 0.96),
            opacity=min(1.0, float(opacity) + 0.15),
        ),
    ]
    return created


def _segment_quat_wxyz(start: tuple[float, float, float], end: tuple[float, float, float]) -> tuple[float, float, float, float]:
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(delta))
    if length < 1.0e-8:
        return (1.0, 0.0, 0.0, 0.0)
    x_axis = delta / length
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(x_axis, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    y_axis = ref - (x_axis * float(np.dot(ref, x_axis)))
    y_axis /= max(1.0e-8, float(np.linalg.norm(y_axis)))
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(1.0e-8, float(np.linalg.norm(z_axis)))
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(1.0e-8, float(np.linalg.norm(y_axis)))
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat_xyzw = R.from_matrix(rotation).as_quat()
    return (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )


def _add_segment(
    stage,
    prim_path: str,
    *,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    thickness: float,
    color: tuple[float, float, float],
    opacity: float,
) -> str | None:
    midpoint = tuple(float((float(a) + float(b)) * 0.5) for a, b in zip(start, end))
    length = float(np.linalg.norm(np.asarray(end, dtype=float) - np.asarray(start, dtype=float)))
    if length < 1.0e-6:
        return None
    Gf, UsdGeom = _import_pxr()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root_xform = UsdGeom.Xformable(root.GetPrim())
    root_xform.ClearXformOpOrder()
    root_xform.AddTranslateOp().Set(Gf.Vec3d(*midpoint))
    root_xform.AddOrientOp().Set(_gf_quatd_from_wxyz(_segment_quat_wxyz(start, end)))
    root_imageable = UsdGeom.Imageable(root.GetPrim())
    root_imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.render)
    root_imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    return _add_cube(
        stage,
        f"{prim_path}/Body",
        translate=(float(length) * 0.5, 0.0, 0.0),
        scale=(float(length), float(thickness), float(thickness)),
        color=color,
        opacity=opacity,
    )


def add_grasp_candidates_visuals_to_stage(
    stage,
    *,
    root_prim_path: str,
    ranked_candidates: list[FilteredGraspExecution],
    selected_candidate: FilteredGraspExecution | None = None,
    max_candidates: int = 0,
    selected_axis_length: float = 0.18,
    candidate_axis_length: float = 0.09,
    axis_thickness: float = 0.012,
    parent_prim_path: str | None = None,
) -> dict[str, Any]:
    # When `parent_prim_path` is provided, the visuals are created as children of
    # that prim with the grasp poses expressed in the parent's local frame. This
    # makes the markers share the exact same xformOp chain as the visual mesh, so
    # they stay glued to it regardless of runtime physics / Fabric-USD sync quirks.
    _Gf, UsdGeom = _import_pxr()
    parent_state = _compute_parent_world_transform(stage, parent_prim_path)
    if parent_state is not None:
        parent_prim, parent_world_row_inv, parent_rot_row_inv = parent_state
        effective_root_path = f"{parent_prim.GetPath()}{root_prim_path}" if root_prim_path.startswith("/") else f"{parent_prim.GetPath()}/{root_prim_path}"
    else:
        parent_world_row_inv = None
        parent_rot_row_inv = None
        effective_root_path = root_prim_path

    stage.RemovePrim(effective_root_path)
    root = UsdGeom.Xform.Define(stage, effective_root_path).GetPrim()
    root_imageable = UsdGeom.Imageable(root)
    root_imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    root_imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.render)

    created_paths: list[str] = []

    def _pose_for(candidate):
        if parent_world_row_inv is None:
            return (
                tuple(float(v) for v in candidate.grasp.position_world),
                tuple(float(v) for v in candidate.grasp.quat_wxyz_world),
            )
        return _world_pose_to_parent_local(
            candidate.grasp.position_world,
            candidate.grasp.quat_wxyz_world,
            parent_world_row_inv,
            parent_rot_row_inv,
        )

    if selected_candidate is not None:
        selected_root = f"{effective_root_path}/Selected"
        pos, quat = _pose_for(selected_candidate)
        created_paths.extend(
            _add_pose_frame(
                stage,
                f"{selected_root}/GraspPose",
                position_world=pos,
                quat_wxyz_world=quat,
                axis_length=float(selected_axis_length),
                axis_thickness=float(axis_thickness),
                opacity=0.88,
            )
        )

    candidate_count = max(0, int(max_candidates))
    if candidate_count > 0:
        for index, candidate in enumerate(ranked_candidates[:candidate_count]):
            candidate_root = f"{effective_root_path}/Candidates/Candidate_{index:02d}"
            pos, quat = _pose_for(candidate)
            created_paths.extend(
                _add_pose_frame(
                    stage,
                    f"{candidate_root}/GraspPose",
                    position_world=pos,
                    quat_wxyz_world=quat,
                    axis_length=float(candidate_axis_length),
                    axis_thickness=float(axis_thickness) * 0.75,
                    opacity=0.42,
                )
            )

    return {
        "visual_root_path": effective_root_path,
        "candidate_count": min(len(ranked_candidates), candidate_count),
        "selected_candidate_id": None if selected_candidate is None else selected_candidate.grasp.candidate_id,
        "selected_grasp_path": None if selected_candidate is None else f"{effective_root_path}/Selected/GraspPose",
        "created_paths": created_paths,
    }


def add_pose_frames_to_stage(
    stage,
    *,
    root_prim_path: str,
    pose_frames: list[dict[str, Any]],
    axis_length: float = 0.075,
    axis_thickness: float = 0.006,
) -> dict[str, Any]:
    # Each entry may set its own "parent_prim_path": when present, that pose frame
    # is re-parented under that prim with its pose converted to the prim's local
    # frame, so it tracks that prim via the USD hierarchy. Entries without a
    # parent are drawn in absolute world coordinates under `root_prim_path`.
    _Gf, UsdGeom = _import_pxr()

    stage.RemovePrim(root_prim_path)
    root = UsdGeom.Xform.Define(stage, root_prim_path).GetPrim()
    root_imageable = UsdGeom.Imageable(root)
    root_imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    root_imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.render)

    created_paths: list[str] = []
    parent_state_cache: dict[str, Any] = {}

    for index, entry in enumerate(pose_frames):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or f"Pose_{index:02d}").replace(" ", "_")
        position_world = entry.get("position_world")
        quat_wxyz_world = entry.get("quat_wxyz_world")
        if not isinstance(position_world, (list, tuple)) or len(position_world) != 3:
            continue
        if not isinstance(quat_wxyz_world, (list, tuple)) or len(quat_wxyz_world) != 4:
            continue

        entry_parent_path = entry.get("parent_prim_path")
        if entry_parent_path:
            if entry_parent_path not in parent_state_cache:
                parent_state_cache[entry_parent_path] = _compute_parent_world_transform(
                    stage, entry_parent_path
                )
            parent_state = parent_state_cache[entry_parent_path]
        else:
            parent_state = None

        if parent_state is not None:
            parent_prim, parent_world_inv, parent_rot_inv = parent_state
            target_root = f"{parent_prim.GetPath()}{root_prim_path}/{name}" if root_prim_path.startswith("/") else f"{parent_prim.GetPath()}/{root_prim_path}/{name}"
            stage.RemovePrim(target_root)
            parent_xf_root = UsdGeom.Xform.Define(stage, target_root).GetPrim()
            UsdGeom.Imageable(parent_xf_root).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            UsdGeom.Imageable(parent_xf_root).CreatePurposeAttr().Set(UsdGeom.Tokens.render)
            pos, quat = _world_pose_to_parent_local(
                tuple(float(v) for v in position_world),
                tuple(float(v) for v in quat_wxyz_world),
                parent_world_inv,
                parent_rot_inv,
            )
        else:
            target_root = f"{root_prim_path}/{name}"
            pos = tuple(float(v) for v in position_world)
            quat = tuple(float(v) for v in quat_wxyz_world)

        created_paths.extend(
            _add_pose_frame(
                stage,
                target_root,
                position_world=pos,
                quat_wxyz_world=quat,
                axis_length=float(entry.get("axis_length", axis_length)),
                axis_thickness=float(entry.get("axis_thickness", axis_thickness)),
                opacity=float(entry.get("opacity", 0.88)),
            )
        )

    return {
        "visual_root_path": root_prim_path,
        "pose_count": len(created_paths),
        "created_paths": created_paths,
    }
