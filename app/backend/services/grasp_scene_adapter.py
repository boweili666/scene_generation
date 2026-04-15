from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .grasp_asset_cache import (
    ASSET_GRASP_CACHE_SCHEMA_VERSION,
    ManifestObjectRecord,
    cache_entry_path,
    default_grasp_annotation_root,
    load_cached_grasp_annotation,
    load_manifest_object_records,
)


SCENE_GRASP_PROPOSALS_SCHEMA_VERSION = "scene_grasp_proposals_v1"
USD_ROW_VECTOR_TRANSFORM_CONVENTION = "USD row-vector homogeneous 4x4 from pxr Gf.Matrix4d"


def _import_pxr():
    from pxr import Usd, UsdGeom

    return Usd, UsdGeom


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def default_scene_grasp_proposals_path(scene_usd_path: str | Path) -> Path:
    scene_path = Path(scene_usd_path).resolve()
    suffix = scene_path.suffix or ".usd"
    return scene_path.with_suffix(f"{suffix}.grasp_proposals.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _relative_to(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _as_numpy_matrix(matrix: Any) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.shape == (16,):
        array = array.reshape(4, 4)
    if array.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform matrix, got shape {array.shape}")
    return array.astype(float, copy=False)


def _row_transform_point(point_xyz: list[float] | tuple[float, ...] | np.ndarray, matrix: np.ndarray) -> np.ndarray:
    point4 = np.array([float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0], dtype=float)
    transformed = point4 @ matrix
    if abs(float(transformed[3])) < 1e-12:
        return transformed[:3]
    return transformed[:3] / float(transformed[3])


def _rotation_scale_from_row_linear(linear: np.ndarray) -> tuple[np.ndarray, float]:
    col_linear = np.asarray(linear, dtype=float).T
    u, singular_values, vh = np.linalg.svd(col_linear)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    scale_hint = float(np.mean(singular_values)) if singular_values.size else 1.0
    return rotation, scale_hint


def _transform_direction(direction_xyz: list[float] | tuple[float, ...] | np.ndarray, rotation_col: np.ndarray) -> list[float]:
    direction = np.asarray(direction_xyz, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return [0.0, 0.0, 0.0]
    world = rotation_col @ (direction / norm)
    world_norm = float(np.linalg.norm(world))
    if world_norm < 1e-8:
        return [0.0, 0.0, 0.0]
    return (world / world_norm).tolist()


def _scale_range(range_value: Any, scale_hint: float) -> list[float] | None:
    if not isinstance(range_value, (list, tuple)) or len(range_value) != 2:
        return None
    return [float(range_value[0]) * float(scale_hint), float(range_value[1]) * float(scale_hint)]


def _transform_pose_set_poses(poses_local: list[dict[str, Any]], transform: np.ndarray) -> list[dict[str, Any]]:
    rotation_col, scale_hint = _rotation_scale_from_row_linear(transform[:3, :3])
    poses_world: list[dict[str, Any]] = []
    for pose in poses_local:
        if not isinstance(pose, dict):
            continue
        entry = dict(pose)
        translation = pose.get("translation")
        if isinstance(translation, (list, tuple)) and len(translation) == 3:
            entry["translation"] = _row_transform_point(translation, transform).tolist()
        rotation_matrix = pose.get("rotation_matrix")
        if isinstance(rotation_matrix, list):
            local_rotation = np.asarray(rotation_matrix, dtype=float)
            if local_rotation.shape == (3, 3):
                entry["rotation_matrix"] = (rotation_col @ local_rotation).tolist()
        for size_key in ("width", "height", "depth"):
            if size_key in pose:
                entry[size_key] = float(pose[size_key]) * float(scale_hint)
        poses_world.append(entry)
    return poses_world


def _transform_grasp_primitive_to_world(primitive: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
    primitive_type = str(primitive.get("type") or "").strip()
    rotation_col, scale_hint = _rotation_scale_from_row_linear(transform[:3, :3])
    transformed = {
        "type": primitive_type,
        "coordinate_frame": "world",
        "score": primitive.get("score"),
        "source_branch": primitive.get("source_branch"),
        "source_primitive": primitive,
    }

    if primitive_type == "point_grasp":
        transformed["point_world"] = _row_transform_point(primitive.get("point_local", [0.0, 0.0, 0.0]), transform).tolist()
        transformed["approach_dirs_world"] = [
            _transform_direction(direction, rotation_col)
            for direction in primitive.get("approach_dirs_local", [])
        ]
        transformed["closing_dirs_world"] = [
            _transform_direction(direction, rotation_col)
            for direction in primitive.get("closing_dirs_local", [])
        ]
        width_range = _scale_range(primitive.get("width_range"), scale_hint)
        depth_range = _scale_range(primitive.get("depth_range"), scale_hint)
        if width_range is not None:
            transformed["width_range"] = width_range
        if depth_range is not None:
            transformed["depth_range"] = depth_range
        return transformed

    if primitive_type == "axis_band":
        transformed["point_world"] = _row_transform_point(primitive.get("point_local", [0.0, 0.0, 0.0]), transform).tolist()
        transformed["axis_world"] = _transform_direction(primitive.get("axis_local", [1.0, 0.0, 0.0]), rotation_col)
        transformed["approach_dirs_world"] = [
            _transform_direction(direction, rotation_col)
            for direction in primitive.get("approach_dirs_local", [])
        ]
        transformed["closing_dirs_world"] = [
            _transform_direction(direction, rotation_col)
            for direction in primitive.get("closing_dirs_local", [])
        ]
        slide_range_world = _scale_range(primitive.get("slide_range"), scale_hint)
        width_range = _scale_range(primitive.get("width_range"), scale_hint)
        depth_range = _scale_range(primitive.get("depth_range"), scale_hint)
        if slide_range_world is not None:
            transformed["slide_range_world"] = slide_range_world
        if width_range is not None:
            transformed["width_range"] = width_range
        if depth_range is not None:
            transformed["depth_range"] = depth_range
        radial_symmetry = primitive.get("radial_symmetry")
        if radial_symmetry is not None:
            transformed["radial_symmetry"] = radial_symmetry
        return transformed

    if primitive_type == "pose_set":
        poses_local = primitive.get("poses_local", [])
        transformed["poses_world"] = _transform_pose_set_poses(
            poses_local if isinstance(poses_local, list) else [],
            transform,
        )
        return transformed

    return transformed


def _annotation_payload_from_cache(cache_payload: dict[str, Any]) -> dict[str, Any]:
    schema_version = str(cache_payload.get("schema_version") or "").strip()
    if schema_version == ASSET_GRASP_CACHE_SCHEMA_VERSION:
        return cache_payload
    if schema_version == "grasp_primitives_v1":
        return cache_payload
    raise ValueError(f"Unsupported grasp annotation schema '{schema_version}'")


def _manifest_fallback_row_transform(manifest_payload: dict[str, Any]) -> np.ndarray | None:
    matrix = manifest_payload.get("usd_transform")
    if matrix is None:
        return None
    transform = _as_numpy_matrix(matrix)
    convention = str(manifest_payload.get("usd_transform_convention") or "").lower()
    if "column-vector" in convention:
        return transform.T
    return transform


def _resolve_stage_prim(stage, prim_path: str):
    Usd, _UsdGeom = _import_pxr()

    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        return prim

    prim_name = Path(prim_path).name
    suffix = f"/{prim_name}"
    best_prim = None
    best_rank: tuple[int, int, str] | None = None

    for candidate in stage.Traverse():
        if not candidate.IsValid():
            continue
        path_str = str(candidate.GetPath())
        if not path_str.endswith(suffix):
            continue
        if f"/GeneratedScene/{prim_name}" in path_str:
            rank = (0, path_str.count("/"), path_str)
        elif "/GeneratedScene/" in path_str:
            rank = (1, path_str.count("/"), path_str)
        elif "/envs/" in path_str:
            rank = (2, path_str.count("/"), path_str)
        else:
            rank = (3, path_str.count("/"), path_str)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_prim = candidate
    return best_prim


def _row_rotation_from_row_linear(linear: np.ndarray) -> np.ndarray:
    rotation_col, _scale_hint = _rotation_scale_from_row_linear(linear)
    return rotation_col.T


def _asset_ref_local_rotation_only(stage_prim) -> np.ndarray | None:
    # Grasp annotations are produced by the grasp_annotator, which reads GLB vertices
    # via trimesh — i.e., it works in the GLB's native glTF meter frame. When the GLB
    # is converted to an asset USD, the converter typically scales vertex numbers to
    # centimeters and sets `metersPerUnit=0.01`. At scene assembly time, `AssetRef`
    # gets an `assetNormalization` xformOp that bakes both the Y-up→Z-up axis flip
    # AND the cm→m unit correction (see pipelines/isaac/usd_asset_utils.py).
    #
    # Because the annotation is already in physical meters (GLB meters), we must NOT
    # apply the cm→m scale from `assetNormalization` a second time. We only want its
    # axis-flip rotation. The root prim's transform (containing placement and scale
    # randomization) is applied via `world_matrix` in `_prim_world_transform`.
    Usd, UsdGeom = _import_pxr()

    asset_ref = stage_prim.GetChild("AssetRef")
    if not asset_ref.IsValid() or not asset_ref.IsA(UsdGeom.Xformable):
        return None

    local = UsdGeom.Xformable(asset_ref).GetLocalTransformation()
    local_matrix = _as_numpy_matrix(local[0] if isinstance(local, tuple) else local)
    if np.allclose(local_matrix, np.eye(4), atol=1e-8):
        return None
    rotation_only = np.eye(4, dtype=float)
    rotation_only[:3, :3] = _row_rotation_from_row_linear(local_matrix[:3, :3])
    return rotation_only


def _prim_world_transform(stage, prim_path: str) -> np.ndarray | None:
    Usd, UsdGeom = _import_pxr()

    prim = _resolve_stage_prim(stage, prim_path)
    if prim is None or not prim.IsValid():
        return None
    world_matrix = _as_numpy_matrix(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    asset_ref_rotation = _asset_ref_local_rotation_only(prim)
    if asset_ref_rotation is None:
        return world_matrix
    return asset_ref_rotation @ world_matrix


def _annotation_cache_path_for_record(annotation_root: Path, record: ManifestObjectRecord) -> Path:
    return cache_entry_path(annotation_root, record.asset_id)


def _build_scene_grasp_proposals_from_stage(
    stage,
    manifest_path: Path,
    *,
    annotation_root: Path,
    scene_usd_path: str | None = None,
    output_path: str | Path | None = None,
    target_prim_paths: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    resolved_annotation_root = Path(annotation_root).resolve()
    manifest_payload = _load_json(manifest)
    object_records = load_manifest_object_records(manifest)
    target_filter = {str(value).strip() for value in (target_prim_paths or []) if str(value).strip()}

    objects_payload: dict[str, Any] = {}
    missing_annotation_paths: list[str] = []
    missing_scene_prims: list[str] = []

    manifest_object_payload = manifest_payload.get("objects", {})
    if not isinstance(manifest_object_payload, dict):
        manifest_object_payload = {}

    for record in object_records:
        if target_filter and record.prim_path not in target_filter:
            continue
        annotation_path = _annotation_cache_path_for_record(resolved_annotation_root, record)
        if not annotation_path.exists():
            missing_annotation_paths.append(str(annotation_path))
            continue

        raw_annotation = load_cached_grasp_annotation(annotation_path)
        annotation_payload = _annotation_payload_from_cache(raw_annotation)
        world_transform = _prim_world_transform(stage, record.prim_path)
        if world_transform is None:
            fallback = manifest_object_payload.get(record.prim_path)
            if isinstance(fallback, dict):
                world_transform = _manifest_fallback_row_transform(fallback)
        if world_transform is None:
            missing_scene_prims.append(record.prim_path)
            continue

        primitives = annotation_payload.get("grasp_primitives", [])
        world_primitives = [
            _transform_grasp_primitive_to_world(primitive, world_transform)
            for primitive in primitives
            if isinstance(primitive, dict)
        ]
        objects_payload[record.prim_path] = {
            "asset_id": record.asset_id,
            "class": record.class_name,
            "caption": record.caption,
            "source": record.source,
            "source_glb": str(record.glb_path),
            "source_usd": str(record.usd_path) if record.usd_path else None,
            "annotation_path": str(annotation_path),
            "annotation_schema_version": annotation_payload.get("schema_version"),
            "category": annotation_payload.get("category"),
            "world_transform": world_transform.tolist(),
            "world_transform_convention": USD_ROW_VECTOR_TRANSFORM_CONVENTION,
            "grasp_primitives_world": world_primitives,
        }

    payload = {
        "schema_version": SCENE_GRASP_PROPOSALS_SCHEMA_VERSION,
        "scene_usd_path": None if scene_usd_path is None else str(scene_usd_path),
        "manifest_path": str(manifest),
        "annotation_root": str(resolved_annotation_root),
        "objects": objects_payload,
        "summary": {
            "object_count": len(objects_payload),
            "missing_annotation_count": len(missing_annotation_paths),
            "missing_scene_prim_count": len(missing_scene_prims),
        },
        "missing_annotation_paths": missing_annotation_paths,
        "missing_scene_prims": missing_scene_prims,
    }
    if output_path is not None:
        _write_json(Path(output_path).resolve(), payload)
    return payload


def build_scene_grasp_proposals(
    scene_usd_path: str | Path,
    manifest_path: str | Path,
    *,
    annotation_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    scene_path = Path(scene_usd_path).resolve()
    manifest = Path(manifest_path).resolve()
    resolved_annotation_root = Path(annotation_root).resolve() if annotation_root is not None else default_grasp_annotation_root(manifest)

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene USD not found: {scene_path}")
    if not manifest.exists():
        raise FileNotFoundError(f"Real2Sim manifest not found: {manifest}")

    manifest_payload = _load_json(manifest)
    object_records = load_manifest_object_records(manifest)

    Usd, _ = _import_pxr()
    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise ValueError(f"Failed to open USD stage: {scene_path}")

    return _build_scene_grasp_proposals_from_stage(
        stage,
        manifest,
        annotation_root=resolved_annotation_root,
        scene_usd_path=str(scene_path),
        output_path=output_path,
    )


def build_stage_grasp_proposals(
    stage,
    manifest_path: str | Path,
    *,
    annotation_root: str | Path | None = None,
    output_path: str | Path | None = None,
    target_prim_paths: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    resolved_annotation_root = Path(annotation_root).resolve() if annotation_root is not None else default_grasp_annotation_root(manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Real2Sim manifest not found: {manifest}")
    return _build_scene_grasp_proposals_from_stage(
        stage,
        manifest,
        annotation_root=resolved_annotation_root,
        scene_usd_path=None,
        output_path=output_path,
        target_prim_paths=target_prim_paths,
    )
