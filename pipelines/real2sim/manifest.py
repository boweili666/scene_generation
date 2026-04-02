from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_FILENAME = "real2sim_asset_manifest.json"
USD_TRANSFORM_CONVENTION = "USD z-up, column-vector homogeneous 4x4"
# Real2Sim pose outputs are already expressed in meters. Unit normalization for the
# converted USD assets happens when those assets are referenced into a Z-up meter stage.
GLTF_TO_USD_UNIT_SCALE = 1.0
GLTF_Y_UP_TO_USD_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)
REAL2SIM_SOURCE = "real2sim"
OUTPUT_NAME_RE = re.compile(r"(\d+)$")
MASK_METADATA_FILENAME = "mask_metadata.json"
ASSIGNMENT_FILENAME = "assignment.json"


def _as_float_matrix4(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform matrix, got shape {matrix.shape}")
    return matrix.astype(float, copy=False)


def gltf_scene_transform_to_usd_transform(
    scene_transform: Any,
    *,
    unit_scale: float = GLTF_TO_USD_UNIT_SCALE,
) -> np.ndarray:
    matrix = _as_float_matrix4(scene_transform)
    basis = GLTF_Y_UP_TO_USD_Z_UP
    usd_matrix = basis @ matrix @ np.linalg.inv(basis)
    usd_matrix = usd_matrix.copy()
    usd_matrix[:3, 3] *= float(unit_scale)
    return usd_matrix


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _scene_objects_by_source(scene_graph_data: dict[str, Any], *, source: str) -> list[tuple[str, dict[str, Any]]]:
    ordered: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    objects = scene_graph_data.get("objects")
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            path = obj.get("path") or obj.get("usd_path")
            if not isinstance(path, str) or not path or path in seen:
                continue
            if obj.get("source") != source:
                continue
            ordered.append((path, obj))
            seen.add(path)

    obj_map = scene_graph_data.get("obj")
    if isinstance(obj_map, dict):
        for path, meta in obj_map.items():
            if not isinstance(path, str) or not path or path in seen or not isinstance(meta, dict):
                continue
            if meta.get("source") != source:
                continue
            ordered.append((path, meta))
            seen.add(path)

    return ordered


def _relative_to(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _discover_scene_usd(output_root: Path) -> Path | None:
    for name in ("scene_merged_post.usd", "scene_merged.usd"):
        candidate = output_root / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _discover_object_usd(output_root: Path, object_name: str) -> Path | None:
    for candidate in (
        output_root / "usd_objects" / f"{object_name}.usd",
        output_root / "objects" / f"{object_name}.usd",
    ):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _discover_mask_metadata(output_root: Path) -> Path | None:
    for candidate in (
        output_root / MASK_METADATA_FILENAME,
        output_root.parent / "masks" / MASK_METADATA_FILENAME,
    ):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _discover_assignment(output_root: Path) -> Path | None:
    candidate = output_root / ASSIGNMENT_FILENAME
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _output_name_sort_key(name: str) -> tuple[int, str]:
    match = OUTPUT_NAME_RE.search(name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def _normalize_class_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _build_output_prompt_lookup(mask_metadata: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for output_name, meta in mask_metadata.items():
        if not isinstance(output_name, str) or not isinstance(meta, dict):
            continue
        prompt = _normalize_class_name(meta.get("prompt"))
        if prompt:
            lookup[output_name] = prompt
    return lookup


def _build_assignment_lookup(assignment_payload: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in assignment_payload.get("assignments", []):
        if not isinstance(row, dict):
            continue
        scene_path = row.get("scene_path")
        output_name = row.get("output_name")
        if isinstance(scene_path, str) and scene_path and isinstance(output_name, str) and output_name:
            lookup[scene_path] = output_name
    return lookup


def _load_assignment_lookup(output_root: Path) -> dict[str, str]:
    assignment_path = _discover_assignment(output_root)
    if assignment_path is None:
        return {}
    return _build_assignment_lookup(_load_json(assignment_path))


def _coerce_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _bbox_area(value: Any) -> float:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return -1.0
    x_min = _coerce_finite_float(value[0])
    y_min = _coerce_finite_float(value[1])
    x_max = _coerce_finite_float(value[2])
    y_max = _coerce_finite_float(value[3])
    if x_min is None or y_min is None or x_max is None or y_max is None:
        return -1.0
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    return width * height


def _build_output_match_score_lookup(
    poses: dict[str, Any],
    mask_metadata: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    lookup: dict[str, tuple[float, float]] = {}
    output_names = {
        str(name)
        for name in set(poses.keys()) | set(mask_metadata.keys())
    }
    for output_name in output_names:
        pose_entry = poses.get(output_name)
        pose_iou = -1.0
        if isinstance(pose_entry, dict):
            coerced_iou = _coerce_finite_float(pose_entry.get("iou"))
            if coerced_iou is not None:
                pose_iou = coerced_iou
        meta = mask_metadata.get(output_name)
        bbox_area = _bbox_area(meta.get("bbox_xyxy")) if isinstance(meta, dict) else -1.0
        lookup[output_name] = (pose_iou, bbox_area)
    return lookup


def _take_first_matching_output(
    available_outputs: list[str],
    *,
    target_class: Any,
    output_prompt_lookup: dict[str, str],
    output_score_lookup: dict[str, tuple[float, float]] | None = None,
) -> str | None:
    normalized_target = _normalize_class_name(target_class)
    if not normalized_target:
        return None
    matching_outputs = [
        output_name
        for output_name in available_outputs
        if output_prompt_lookup.get(output_name) == normalized_target
    ]
    if not matching_outputs:
        return None

    if output_score_lookup is None:
        output_score_lookup = {}

    def sort_key(output_name: str) -> tuple[float, float, tuple[int, str]]:
        pose_iou, bbox_area = output_score_lookup.get(output_name, (-1.0, -1.0))
        return (-pose_iou, -bbox_area, _output_name_sort_key(output_name))

    return min(matching_outputs, key=sort_key)


def build_real2sim_asset_manifest(
    output_dir: str | Path,
    *,
    scene_graph_path: str | Path,
    poses_filename: str = "poses.json",
    unit_scale: float = GLTF_TO_USD_UNIT_SCALE,
) -> tuple[Path, dict[str, Any]]:
    output_root = Path(output_dir)
    scene_graph_file = Path(scene_graph_path)
    if not scene_graph_file.exists():
        raise FileNotFoundError(f"Scene graph not found: {scene_graph_file}")

    scene_graph = _load_json(scene_graph_file)
    poses = _load_json(output_root / poses_filename)
    objects_dir = output_root / "objects"
    mask_metadata_path = _discover_mask_metadata(output_root)
    mask_metadata = _load_json(mask_metadata_path) if mask_metadata_path is not None else {}
    output_prompt_lookup = _build_output_prompt_lookup(mask_metadata)
    output_score_lookup = _build_output_match_score_lookup(poses, mask_metadata)
    assignment_lookup = _load_assignment_lookup(output_root)

    object_names = sorted(
        {p.stem for p in objects_dir.glob("*.glb")} & {str(k) for k in poses.keys()},
        key=_output_name_sort_key,
    )
    ordered_scene_objects = _scene_objects_by_source(scene_graph, source=REAL2SIM_SOURCE)

    matched_entries: dict[str, dict[str, Any]] = {}
    scene_usd_path = _discover_scene_usd(output_root)
    remaining_outputs = list(object_names)
    unmatched_scene_paths: list[str] = []

    for prim_path, meta in ordered_scene_objects:
        object_name = assignment_lookup.get(prim_path)
        if object_name not in remaining_outputs:
            object_name = _take_first_matching_output(
                remaining_outputs,
                target_class=meta.get("class") or meta.get("class_name"),
                output_prompt_lookup=output_prompt_lookup,
                output_score_lookup=output_score_lookup,
            )
        if object_name is None and remaining_outputs:
            object_name = remaining_outputs[0]
        if object_name is None:
            unmatched_scene_paths.append(prim_path)
            continue
        pose_entry = poses.get(object_name)
        if not isinstance(pose_entry, dict):
            continue
        remaining_outputs.remove(object_name)
        scene_transform = pose_entry.get("scene_transform")
        if scene_transform is None:
            raise ValueError(f"Pose entry '{object_name}' is missing required scene_transform")
        usd_transform = gltf_scene_transform_to_usd_transform(scene_transform, unit_scale=unit_scale)
        object_usd_path = _discover_object_usd(output_root, object_name)
        matched_entries[prim_path] = {
            "prim_path": prim_path,
            "output_name": object_name,
            "class": meta.get("class") or meta.get("class_name"),
            "caption": meta.get("caption"),
            "source": meta.get("source"),
            "glb_path": _relative_to(output_root, objects_dir / f"{object_name}.glb"),
            "usd_path": _relative_to(output_root, object_usd_path) or _relative_to(output_root, scene_usd_path),
            "usd_prim_path": None if object_usd_path is not None else f"/World/{object_name}",
            "scene_transform": scene_transform,
            "scene_transform_convention": pose_entry.get(
                "scene_transform_convention", "glTF y-up, column-vector homogeneous 4x4"
            ),
            "usd_transform": [[float(v) for v in row] for row in usd_transform.tolist()],
            "usd_transform_convention": USD_TRANSFORM_CONVENTION,
            "usd_unit_scale": float(unit_scale),
        }

    unmatched_outputs = remaining_outputs
    manifest = {
        "version": 1,
        "source": REAL2SIM_SOURCE,
        "scene_graph_path": str(scene_graph_file.resolve()),
        "results_root": str(output_root.resolve()),
        "scene_glb": _relative_to(output_root, output_root / "scene_merged.glb"),
        "scene_post_glb": _relative_to(output_root, output_root / "scene_merged_post.glb"),
        "scene_usd": _relative_to(output_root, scene_usd_path),
        "gltf_to_usd_basis": [[float(v) for v in row] for row in GLTF_Y_UP_TO_USD_Z_UP.tolist()],
        "usd_unit_scale": float(unit_scale),
        "objects": matched_entries,
        "unmatched_scene_paths": unmatched_scene_paths,
        "unmatched_outputs": unmatched_outputs,
    }

    manifest_path = output_root / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path, manifest
