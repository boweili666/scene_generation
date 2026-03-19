from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


RESAMPLE_MODES = {"joint", "lock_real2sim"}


def validate_resample_mode(mode: str | None) -> str:
    normalized = str(mode or "joint").strip().lower()
    if normalized not in RESAMPLE_MODES:
        raise ValueError(
            f"Invalid resample_mode '{mode}'. Allowed values: {sorted(RESAMPLE_MODES)}"
        )
    return normalized


def support_parent_map(edges: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        relation = str(edge.get("relation", ""))
        parts = [part.strip() for part in relation.split(",") if part.strip()]
        if "supported by" in parts:
            mapping[source] = target
        elif "supports" in parts:
            mapping[target] = source
    return mapping


def _as_float_matrix4(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform matrix, got shape {matrix.shape}")
    return matrix.astype(float, copy=False)


def column_transform_to_row_major(matrix: Any) -> List[List[float]]:
    column_major = _as_float_matrix4(matrix)
    row_major = column_major.T
    return [[float(v) for v in row] for row in row_major.tolist()]


def row_transform_to_column_major(matrix: Any) -> np.ndarray:
    row_major = _as_float_matrix4(matrix)
    return row_major.T.copy()


def build_real2sim_usd_transform_lookup(manifest: Dict[str, Any] | None) -> Dict[str, np.ndarray]:
    if not isinstance(manifest, dict):
        return {}
    objects = manifest.get("objects")
    if not isinstance(objects, dict):
        return {}

    lookup: Dict[str, np.ndarray] = {}
    for prim, entry in objects.items():
        if not isinstance(entry, dict):
            continue
        usd_transform = entry.get("usd_transform")
        if usd_transform is None:
            continue
        lookup[str(prim)] = _as_float_matrix4(usd_transform)
    return lookup


def _estimate_center(meta: Dict[str, Any]) -> np.ndarray:
    bbox = meta.get("world_3d_bbox") or meta.get("3d_bbox")
    if not bbox:
        return np.zeros(3, dtype=float)
    corners = np.asarray(bbox, dtype=float)
    if corners.ndim != 2 or corners.shape[1] != 3:
        return np.zeros(3, dtype=float)
    return corners.mean(axis=0)


def _build_center_lookup(
    data: Dict[str, Any],
    asset_bbox_lookup: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]],
) -> Dict[str, np.ndarray]:
    centers: Dict[str, np.ndarray] = {}
    for prim, meta in data.get("obj", {}).items():
        if asset_bbox_lookup and prim in asset_bbox_lookup:
            centers[prim] = np.asarray(asset_bbox_lookup[prim].get("center", (0.0, 0.0, 0.0)), dtype=float)
        else:
            centers[prim] = _estimate_center(meta if isinstance(meta, dict) else {})
    return centers


def _world_center_from_column_transform(
    transform: np.ndarray,
    local_center: np.ndarray,
) -> Tuple[float, float, float]:
    center_h = np.ones(4, dtype=float)
    center_h[:3] = np.asarray(local_center, dtype=float)
    world_h = np.asarray(transform, dtype=float) @ center_h
    return (float(world_h[0]), float(world_h[1]), float(world_h[2]))


def _root_real2sim_anchor(
    prim: str,
    real2sim_prims: set[str],
    parent_lookup: Dict[str, str],
) -> Optional[str]:
    current = prim
    visited: set[str] = set()
    while True:
        if current in visited:
            return None
        visited.add(current)
        parent = parent_lookup.get(current)
        if parent is None:
            return current
        if parent not in real2sim_prims:
            return None
        current = parent


def apply_lock_real2sim_relative_transforms(
    data: Dict[str, Any],
    object_entries: List[Dict[str, Any]],
    placements: Dict[str, Tuple[float, float, float]],
    *,
    asset_bbox_lookup: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None,
    real2sim_manifest: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[float, float, float]], Dict[str, Any]]:
    entries = [
        {
            **entry,
            "transform": [[float(v) for v in row] for row in entry.get("transform", [])],
        }
        for entry in object_entries
    ]
    placements_out = dict(placements)

    manifest_transforms = build_real2sim_usd_transform_lookup(real2sim_manifest)
    if not manifest_transforms:
        return entries, placements_out, {
            "mode_applied": False,
            "reason": "missing_real2sim_manifest_transforms",
            "real2sim_roots": [],
            "locked_real2sim_children": [],
            "skipped_real2sim_prims": [],
            "missing_manifest_transforms": [],
        }

    entry_lookup = {entry.get("prim"): entry for entry in entries if entry.get("prim")}
    parent_lookup = support_parent_map(data.get("edges", {}).get("obj-obj", []))
    centers = _build_center_lookup(data, asset_bbox_lookup)
    real2sim_prims = {
        prim
        for prim, meta in data.get("obj", {}).items()
        if isinstance(meta, dict) and meta.get("source") == "real2sim"
    }

    root_by_prim: Dict[str, Optional[str]] = {}
    roots: set[str] = set()
    skipped: set[str] = set()
    for prim in sorted(real2sim_prims):
        root = _root_real2sim_anchor(prim, real2sim_prims, parent_lookup)
        root_by_prim[prim] = root
        if root is None:
            skipped.add(prim)
        else:
            roots.add(root)

    locked_children: List[str] = []
    missing_transforms: set[str] = set()

    for root in sorted(roots):
        root_entry = entry_lookup.get(root)
        root_initial = manifest_transforms.get(root)
        if root_entry is None or root_initial is None:
            missing_transforms.add(root)
            continue

        sampled_root = row_transform_to_column_major(root_entry["transform"])
        delta = sampled_root @ np.linalg.inv(root_initial)

        for prim, anchor in root_by_prim.items():
            if prim == root or anchor != root:
                continue
            child_entry = entry_lookup.get(prim)
            child_initial = manifest_transforms.get(prim)
            if child_entry is None or child_initial is None:
                missing_transforms.add(prim)
                continue

            child_world = delta @ child_initial
            child_entry["transform"] = column_transform_to_row_major(child_world)
            placements_out[prim] = _world_center_from_column_transform(
                child_world,
                centers.get(prim, np.zeros(3, dtype=float)),
            )
            locked_children.append(prim)

    return entries, placements_out, {
        "mode_applied": bool(locked_children),
        "reason": None,
        "real2sim_roots": sorted(roots),
        "locked_real2sim_children": sorted(locked_children),
        "skipped_real2sim_prims": sorted(skipped),
        "missing_manifest_transforms": sorted(missing_transforms),
    }
