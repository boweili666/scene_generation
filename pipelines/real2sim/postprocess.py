from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=float)
X_AXIS = np.array([1.0, 0.0, 0.0], dtype=float)
Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)
OUTPUT_NAME_RE = re.compile(r"(\d+)$")
PRE_SCENE_FILENAME = "scene_merged_pre.glb"
POST_SCENE_FILENAME = "scene_merged_post.glb"
PRE_POSES_FILENAME = "poses_pre.json"
POST_POSES_FILENAME = "poses_post.json"


@dataclass
class PlacementState:
    name: str
    mesh: trimesh.Trimesh
    rotation: np.ndarray
    scale: np.ndarray
    translation: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self.rotation @ np.diag(self.scale)
        matrix[:3, 3] = self.translation
        return matrix

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        transformed = trimesh.transform_points(self.mesh.vertices, self.matrix)
        return transformed.min(axis=0), transformed.max(axis=0)


def copy_placement(placement: PlacementState) -> PlacementState:
    return PlacementState(
        name=placement.name,
        mesh=placement.mesh,
        rotation=placement.rotation.copy(),
        scale=placement.scale.copy(),
        translation=placement.translation.copy(),
    )


def _as_float_vector(value: Any, *, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 0 and length == 1:
        array = np.array([float(array)], dtype=float)
    array = array.reshape(-1)
    if array.size == 1 and length > 1:
        array = np.repeat(array, length)
    if array.size != length:
        raise ValueError(f"Expected vector of length {length}, got shape {array.shape}")
    return array.astype(float, copy=False)


def _as_float_matrix4(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform matrix, got shape {matrix.shape}")
    return matrix.astype(float, copy=False)


def pose_translation_to_glb(translation: Any) -> np.ndarray:
    return _as_float_vector(translation, length=3)


def pose_rotation_to_glb(rotation_wxyz: Any) -> np.ndarray:
    quat_wxyz = _as_float_vector(rotation_wxyz, length=4)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def upright_rotation_from_current(rotation: np.ndarray) -> np.ndarray:
    world_x = rotation @ X_AXIS
    world_z = rotation @ Z_AXIS
    horizontal_x = world_x.copy()
    horizontal_x[1] = 0.0
    horizontal_z = world_z.copy()
    horizontal_z[1] = 0.0
    norm_x = float(np.linalg.norm(horizontal_x))
    norm_z = float(np.linalg.norm(horizontal_z))

    if norm_x >= norm_z and norm_x > 1e-8:
        x_axis = horizontal_x / norm_x
        z_axis = np.cross(x_axis, Y_AXIS)
    elif norm_z > 1e-8:
        z_axis = horizontal_z / norm_z
        x_axis = np.cross(Y_AXIS, z_axis)
    else:
        x_axis = X_AXIS.copy()
        z_axis = Z_AXIS.copy()

    x_axis /= np.linalg.norm(x_axis)
    z_axis /= np.linalg.norm(z_axis)
    upright = np.column_stack([x_axis, Y_AXIS, z_axis])
    if np.linalg.det(upright) < 0.0:
        upright[:, 2] *= -1.0
    return upright


def is_nearly_y_up(rotation: np.ndarray, *, cos_threshold: float = 0.97) -> bool:
    up = rotation @ Y_AXIS
    return float(np.dot(up, Y_AXIS)) >= cos_threshold


def extract_scene_object_paths(scene_graph_data: dict[str, Any]) -> list[str]:
    ordered_paths: list[str] = []
    seen: set[str] = set()

    objects = scene_graph_data.get("objects")
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            path = obj.get("path") or obj.get("usd_path")
            if isinstance(path, str) and path and path not in seen:
                ordered_paths.append(path)
                seen.add(path)

    obj_map = scene_graph_data.get("obj")
    if isinstance(obj_map, dict):
        for path in obj_map.keys():
            if isinstance(path, str) and path and path not in seen:
                ordered_paths.append(path)
                seen.add(path)

    return ordered_paths


def _support_pair(edge: dict[str, Any]) -> tuple[str, str] | None:
    relation = str(edge.get("relation", "")).strip().lower()
    source = edge.get("source")
    target = edge.get("target")
    if not isinstance(source, str) or not isinstance(target, str):
        return None
    if "supported by" in relation:
        return source, target
    if "supports" in relation:
        return target, source
    return None


def extract_support_pairs(scene_graph_data: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    edges = scene_graph_data.get("edges", {})
    obj_edges = edges.get("obj-obj") if isinstance(edges, dict) else None
    if not isinstance(obj_edges, list):
        return pairs

    for edge in obj_edges:
        if not isinstance(edge, dict):
            continue
        pair = _support_pair(edge)
        if pair is None or pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def output_name_sort_key(name: str) -> tuple[int, str]:
    match = OUTPUT_NAME_RE.search(name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def resolve_support_penetration(
    placements: dict[str, PlacementState],
    support_pairs: list[tuple[str, str]],
    *,
    clearance: float = 1e-4,
) -> int:
    total_adjustments = 0
    if not support_pairs:
        return total_adjustments

    max_passes = max(1, len(support_pairs))
    for _ in range(max_passes):
        moved_this_pass = False
        for top_name, base_name in support_pairs:
            top = placements.get(top_name)
            base = placements.get(base_name)
            if top is None or base is None:
                continue

            top_min, top_max = top.bounds
            base_min, base_max = base.bounds
            overlap_x = min(top_max[0], base_max[0]) - max(top_min[0], base_min[0])
            overlap_z = min(top_max[2], base_max[2]) - max(top_min[2], base_min[2])
            if overlap_x < -clearance or overlap_z < -clearance:
                continue

            target_bottom = base_max[1] + clearance
            if top_min[1] >= target_bottom:
                continue

            top.translation[1] += target_bottom - top_min[1]
            total_adjustments += 1
            moved_this_pass = True
        if not moved_this_pass:
            break

    return total_adjustments


def preserve_relative_support_transforms(
    original_placements: dict[str, PlacementState],
    placements: dict[str, PlacementState],
    support_pairs: list[tuple[str, str]],
) -> int:
    updated_count = 0
    if not support_pairs:
        return updated_count

    max_passes = max(1, len(support_pairs))
    for _ in range(max_passes):
        changed = False
        for top_name, base_name in support_pairs:
            top_original = original_placements.get(top_name)
            base_original = original_placements.get(base_name)
            top_current = placements.get(top_name)
            base_current = placements.get(base_name)
            if top_original is None or base_original is None or top_current is None or base_current is None:
                continue

            delta_rotation = base_current.rotation @ base_original.rotation.T
            delta_translation = base_current.translation - delta_rotation @ base_original.translation
            next_rotation = delta_rotation @ top_original.rotation
            next_translation = delta_rotation @ top_original.translation + delta_translation

            rotation_changed = not np.allclose(top_current.rotation, next_rotation)
            translation_changed = not np.allclose(top_current.translation, next_translation)
            if not rotation_changed and not translation_changed:
                continue

            top_current.rotation = next_rotation
            top_current.translation = next_translation
            updated_count += 1
            changed = True
        if not changed:
            break

    return updated_count


def ground_unsupported_roots(
    original_placements: dict[str, PlacementState],
    placements: dict[str, PlacementState],
    support_pairs: list[tuple[str, str]],
    root_objects: set[str],
    *,
    floor_y: float = 0.0,
    epsilon: float = 1e-6,
) -> int:
    grounded_count = 0
    for object_name in sorted(root_objects, key=output_name_sort_key):
        placement = placements.get(object_name)
        if placement is None:
            continue
        bounds_min, _ = placement.bounds
        delta_y = floor_y - float(bounds_min[1])
        if abs(delta_y) <= epsilon:
            continue
        placement.translation[1] += delta_y
        grounded_count += 1

    if grounded_count > 0:
        preserve_relative_support_transforms(original_placements, placements, support_pairs)

    return grounded_count


def _pose_to_placement(name: str, mesh: trimesh.Trimesh, pose_entry: dict[str, Any]) -> PlacementState:
    scene_transform = pose_entry.get("scene_transform")
    if scene_transform is not None:
        matrix = _as_float_matrix4(scene_transform)
        rotation, scale = _rotation_and_scale_from_linear(matrix[:3, :3])
        translation = np.asarray(matrix[:3, 3], dtype=float)
        return PlacementState(
            name=name,
            mesh=mesh,
            rotation=rotation,
            scale=scale,
            translation=translation,
        )

    rotation = pose_rotation_to_glb(pose_entry.get("rotation", [1.0, 0.0, 0.0, 0.0]))
    scale = _as_float_vector(pose_entry.get("scale", [1.0, 1.0, 1.0]), length=3)
    translation = pose_translation_to_glb(pose_entry.get("translation", [0.0, 0.0, 0.0]))
    return PlacementState(
        name=name,
        mesh=mesh,
        rotation=rotation,
        scale=scale,
        translation=translation,
    )


def _rotation_and_scale_from_linear(linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    linear = np.asarray(linear, dtype=float)
    scale = np.linalg.norm(linear, axis=0)
    safe_scale = np.where(scale > 1e-8, scale, 1.0)
    rotation = linear / safe_scale
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation, safe_scale


def _update_pose_entry(pose_entry: dict[str, Any], placement: PlacementState) -> dict[str, Any]:
    updated = dict(pose_entry)
    quat_xyzw = Rotation.from_matrix(placement.rotation).as_quat()
    quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]
    updated["rotation"] = [quat_wxyz]
    updated["translation"] = [[float(v) for v in placement.translation.tolist()]]
    updated["scale"] = [[float(v) for v in placement.scale.tolist()]]
    updated["scene_transform"] = [[float(v) for v in row] for row in placement.matrix.tolist()]
    updated.setdefault("scene_transform_convention", "gltf_y_up")
    return updated


def _copy_artifact(source: Path, destination: Path, *, overwrite: bool) -> None:
    if not source.exists():
        return
    if destination.exists() and not overwrite:
        return
    shutil.copy2(source, destination)


def postprocess_real2sim_outputs(
    output_dir: str | Path,
    scene_graph_path: str | Path | None = None,
    *,
    upright_cos_threshold: float = 0.97,
    support_clearance: float = 1e-4,
) -> dict[str, int]:
    output_root = Path(output_dir)
    objects_dir = output_root / "objects"
    poses_path = output_root / "poses.json"
    poses_partial_path = output_root / "poses_partial.json"
    scene_path = output_root / "scene_merged.glb"
    pre_scene_path = output_root / PRE_SCENE_FILENAME
    post_scene_path = output_root / POST_SCENE_FILENAME
    pre_poses_path = output_root / PRE_POSES_FILENAME
    post_poses_path = output_root / POST_POSES_FILENAME

    if not objects_dir.exists():
        raise FileNotFoundError(f"Real2Sim objects directory not found: {objects_dir}")
    if not poses_path.exists():
        raise FileNotFoundError(f"Real2Sim poses file not found: {poses_path}")

    _copy_artifact(scene_path, pre_scene_path, overwrite=False)
    _copy_artifact(poses_path, pre_poses_path, overwrite=False)

    source_poses_path = poses_partial_path if poses_partial_path.exists() else poses_path
    poses = json.loads(source_poses_path.read_text(encoding="utf-8"))
    if not isinstance(poses, dict):
        raise ValueError(f"Expected {source_poses_path} to contain an object map")

    object_names = sorted(
        {p.stem for p in objects_dir.glob("*.glb")} & {str(k) for k in poses.keys()},
        key=output_name_sort_key,
    )
    if not object_names:
        raise ValueError(f"No matching object GLBs / poses found under {output_root}")

    scene_graph_data: dict[str, Any] = {}
    if scene_graph_path:
        graph_path = Path(scene_graph_path)
        if graph_path.exists():
            loaded = json.loads(graph_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                scene_graph_data = loaded

    ordered_scene_paths = extract_scene_object_paths(scene_graph_data)
    scene_path_to_output = {
        scene_path_key: output_name
        for scene_path_key, output_name in zip(ordered_scene_paths, object_names)
    }
    support_pairs = [
        (scene_path_to_output[top], scene_path_to_output[base])
        for top, base in extract_support_pairs(scene_graph_data)
        if top in scene_path_to_output and base in scene_path_to_output
    ]
    supported_objects = {top for top, _ in support_pairs}
    unsupported_root_objects = {object_name for object_name in object_names if object_name not in supported_objects}

    original_placements: dict[str, PlacementState] = {}
    placements: dict[str, PlacementState] = {}
    forced_upright_count = 0
    snapped_upright_count = 0

    for object_name in object_names:
        object_path = objects_dir / f"{object_name}.glb"
        mesh_scene = trimesh.load(object_path, force="scene")
        mesh = mesh_scene.to_geometry()
        original_placement = _pose_to_placement(object_name, mesh, poses[object_name])
        original_placements[object_name] = original_placement
        placements[object_name] = copy_placement(original_placement)

    for object_name in object_names:
        placement = placements[object_name]
        if object_name not in supported_objects:
            upright_rotation = upright_rotation_from_current(placement.rotation)
            if not np.allclose(placement.rotation, upright_rotation):
                if is_nearly_y_up(placement.rotation, cos_threshold=upright_cos_threshold):
                    snapped_upright_count += 1
            placement.rotation = upright_rotation
            forced_upright_count += 1

    preserve_relative_support_transforms(original_placements, placements, support_pairs)
    grounded_roots = ground_unsupported_roots(
        original_placements,
        placements,
        support_pairs,
        unsupported_root_objects,
    )
    penetration_adjustments = resolve_support_penetration(
        placements,
        support_pairs,
        clearance=support_clearance,
    )

    merged_scene = trimesh.Scene()
    for object_name in object_names:
        placement = placements[object_name]
        merged_scene.add_geometry(
            placement.mesh,
            node_name=object_name,
            geom_name=object_name,
            transform=placement.matrix,
        )

    scene_path.write_bytes(merged_scene.export(file_type="glb"))
    updated_poses = {
        object_name: _update_pose_entry(poses[object_name], placements[object_name])
        for object_name in object_names
    }
    poses_path.write_text(
        json.dumps(updated_poses, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _copy_artifact(scene_path, post_scene_path, overwrite=True)
    _copy_artifact(poses_path, post_poses_path, overwrite=True)

    return {
        "objects": len(object_names),
        "support_pairs": len(support_pairs),
        "forced_upright": forced_upright_count,
        "snapped_upright": snapped_upright_count,
        "grounded_roots": grounded_roots,
        "penetration_adjustments": penetration_adjustments,
    }
