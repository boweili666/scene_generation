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


def _output_name_sort_key(name: str) -> tuple[int, str]:
    match = OUTPUT_NAME_RE.search(name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


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

    object_names = sorted(
        {p.stem for p in objects_dir.glob("*.glb")} & {str(k) for k in poses.keys()},
        key=_output_name_sort_key,
    )
    ordered_scene_objects = _scene_objects_by_source(scene_graph, source=REAL2SIM_SOURCE)
    ordered_scene_paths = [path for path, _ in ordered_scene_objects]

    matched_entries: dict[str, dict[str, Any]] = {}
    scene_usd_path = _discover_scene_usd(output_root)
    for (prim_path, meta), object_name in zip(ordered_scene_objects, object_names):
        pose_entry = poses.get(object_name)
        if not isinstance(pose_entry, dict):
            continue
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

    unmatched_scene_paths = ordered_scene_paths[len(object_names) :]
    unmatched_outputs = object_names[len(ordered_scene_paths) :]
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
