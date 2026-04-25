from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np


DEFAULT_MIN_SCALE = 0.85
DEFAULT_MAX_SCALE = 1.15


@dataclass(frozen=True)
class SceneObjectSpec:
    prim_path: str
    label: str
    caption: str | None
    support_parent: str | None
    asset_ref_prim_path: str


def _import_pxr():
    from pxr import Gf, Usd, UsdGeom

    return Gf, Usd, UsdGeom


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_adjacent_path(scene_usd_path: Path, relative_parts: tuple[str, ...]) -> Path | None:
    for parent in (scene_usd_path.parent, *scene_usd_path.parents):
        candidate = parent.joinpath(*relative_parts)
        if candidate.exists():
            return candidate.resolve()
    return None


def _auto_scene_graph_path(scene_usd_path: Path) -> Path | None:
    return _resolve_adjacent_path(scene_usd_path, ("scene_graph", "current_scene_graph.json"))


def _auto_manifest_path(scene_usd_path: Path) -> Path | None:
    return _resolve_adjacent_path(scene_usd_path, ("real2sim", "scene_results", "real2sim_asset_manifest.json"))


def _as_numpy_matrix(matrix) -> np.ndarray:
    return np.array([[float(matrix[row][col]) for col in range(4)] for row in range(4)], dtype=float)


def _as_gf_matrix(matrix: np.ndarray, Gf) -> Any:
    rows = tuple(tuple(float(matrix[row, col]) for col in range(4)) for row in range(4))
    return Gf.Matrix4d(rows)


def _transform_point(point_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    point4 = np.array([float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0], dtype=float)
    transformed = point4 @ matrix
    if abs(transformed[3]) < 1e-12:
        return transformed[:3]
    return transformed[:3] / transformed[3]


def _bbox_payload_from_range(rng) -> dict[str, list[float]]:
    bmin = rng.GetMin()
    bmax = rng.GetMax()
    min_xyz = [float(bmin[0]), float(bmin[1]), float(bmin[2])]
    max_xyz = [float(bmax[0]), float(bmax[1]), float(bmax[2])]
    return {
        "min": min_xyz,
        "max": max_xyz,
        "center": [
            float((min_xyz[0] + max_xyz[0]) * 0.5),
            float((min_xyz[1] + max_xyz[1]) * 0.5),
            float((min_xyz[2] + max_xyz[2]) * 0.5),
        ],
        "size": [
            float(max_xyz[0] - min_xyz[0]),
            float(max_xyz[1] - min_xyz[1]),
            float(max_xyz[2] - min_xyz[2]),
        ],
    }


def _compute_world_bbox(stage, prim_path: str) -> dict[str, list[float]]:
    _, Usd, UsdGeom = _import_pxr()

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim does not exist: {prim_path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "guide"], useExtentsHint=True)
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        raise ValueError(f"Failed to compute world bbox for: {prim_path}")
    return _bbox_payload_from_range(rng)


def _compute_world_matrix(stage, prim_path: str) -> np.ndarray:
    _, Usd, UsdGeom = _import_pxr()

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim does not exist: {prim_path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return _as_numpy_matrix(matrix)


def _get_transform_op(prim, *, op_suffix: str | None = None):
    _, _, UsdGeom = _import_pxr()

    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            return op
    if xformable.GetOrderedXformOps():
        raise ValueError(
            f"Prim {prim.GetPath()} uses non-matrix xform ops; this utility expects transform ops from scene_service USDs."
        )
    return xformable.AddTransformOp(opSuffix=(op_suffix or ""))


def _get_local_matrix(prim) -> np.ndarray:
    _, _, UsdGeom = _import_pxr()

    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    if ops:
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                return _as_numpy_matrix(op.Get())
        raise ValueError(
            f"Prim {prim.GetPath()} uses non-matrix xform ops; this utility expects transform ops from scene_service USDs."
        )
    return _as_numpy_matrix(xformable.GetLocalTransformation()[0])


def _set_local_matrix(prim, matrix: np.ndarray, *, op_suffix: str | None = None) -> None:
    Gf, _, _ = _import_pxr()

    op = _get_transform_op(prim, op_suffix=op_suffix)
    op.Set(_as_gf_matrix(matrix, Gf))


def _translate_prim(prim, delta_xyz: np.ndarray) -> np.ndarray:
    matrix = _get_local_matrix(prim)
    matrix[3, 0] += float(delta_xyz[0])
    matrix[3, 1] += float(delta_xyz[1])
    matrix[3, 2] += float(delta_xyz[2])
    _set_local_matrix(prim, matrix)
    return matrix


def _find_asset_ref_prim(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim does not exist: {prim_path}")
    asset_ref = stage.GetPrimAtPath(f"{prim_path}/AssetRef")
    if asset_ref.IsValid():
        return asset_ref
    for child in prim.GetChildren():
        if child.GetTypeName() == "Xform":
            return child
    raise ValueError(f"Prim has no Xform child usable as asset reference: {prim_path}")


def _support_parent_lookup(scene_graph_data: dict[str, Any] | None) -> dict[str, str]:
    if not scene_graph_data:
        return {}
    lookup: dict[str, str] = {}
    for edge in scene_graph_data.get("edges", {}).get("obj-obj", []):
        if str(edge.get("relation")).strip().lower() != "supported by":
            continue
        child = str(edge.get("source") or "").strip()
        parent = str(edge.get("target") or "").strip()
        if child and parent:
            lookup[child] = parent
    return lookup


def _discover_real2sim_objects(stage, scene_graph_data: dict[str, Any] | None, manifest_data: dict[str, Any] | None) -> list[SceneObjectSpec]:
    support_lookup = _support_parent_lookup(scene_graph_data)
    object_specs: dict[str, SceneObjectSpec] = {}

    if scene_graph_data:
        for prim_path, meta in scene_graph_data.get("obj", {}).items():
            if str(meta.get("source")).strip().lower() != "real2sim":
                continue
            prim = stage.GetPrimAtPath(str(prim_path))
            if not prim.IsValid():
                continue
            asset_ref_prim = _find_asset_ref_prim(stage, str(prim_path))
            object_specs[str(prim_path)] = SceneObjectSpec(
                prim_path=str(prim_path),
                label=str(meta.get("class") or Path(str(prim_path)).name),
                caption=meta.get("caption"),
                support_parent=support_lookup.get(str(prim_path)),
                asset_ref_prim_path=str(asset_ref_prim.GetPath()),
            )

    if not object_specs and manifest_data:
        for prim_path, meta in manifest_data.get("objects", {}).items():
            if str(meta.get("source")).strip().lower() != "real2sim":
                continue
            prim = stage.GetPrimAtPath(str(prim_path))
            if not prim.IsValid():
                continue
            asset_ref_prim = _find_asset_ref_prim(stage, str(prim_path))
            object_specs[str(prim_path)] = SceneObjectSpec(
                prim_path=str(prim_path),
                label=str(meta.get("class") or Path(str(prim_path)).name),
                caption=meta.get("caption"),
                support_parent=None,
                asset_ref_prim_path=str(asset_ref_prim.GetPath()),
            )

    return [object_specs[path] for path in sorted(object_specs)]


def _topological_order(objects: list[SceneObjectSpec]) -> list[str]:
    selected = {item.prim_path for item in objects}
    children_by_parent: dict[str, list[str]] = {}
    roots: list[str] = []
    for item in objects:
        parent = item.support_parent
        if parent and parent in selected:
            children_by_parent.setdefault(parent, []).append(item.prim_path)
        else:
            roots.append(item.prim_path)

    ordered: list[str] = []
    visited: set[str] = set()

    def visit(path: str) -> None:
        if path in visited:
            return
        visited.add(path)
        ordered.append(path)
        for child_path in sorted(children_by_parent.get(path, [])):
            visit(child_path)

    for root_path in sorted(roots):
        visit(root_path)
    for item in objects:
        visit(item.prim_path)
    return ordered


def _validate_scale(scale: float, *, prim_path: str) -> float:
    value = float(scale)
    if value <= 0.0:
        raise ValueError(f"Scale for {prim_path} must be positive, got {value}.")
    return value


def _serialize_vector(values: np.ndarray) -> list[float]:
    return [float(values[0]), float(values[1]), float(values[2])]


def randomize_real2sim_asset_scales(
    scene_usd_path: str | Path,
    *,
    output_usd_path: str | Path,
    output_metadata_path: str | Path | None = None,
    scene_graph_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    min_scale: float = DEFAULT_MIN_SCALE,
    max_scale: float = DEFAULT_MAX_SCALE,
    seed: int | None = None,
    shared_scale: bool = False,
    global_scale: float | None = None,
    scale_overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    _, Usd, _ = _import_pxr()

    source_path = Path(scene_usd_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Scene USD does not exist: {source_path}")
    if min_scale <= 0.0 or max_scale <= 0.0:
        raise ValueError("min_scale and max_scale must be positive.")
    if min_scale > max_scale:
        raise ValueError(f"min_scale ({min_scale}) must be <= max_scale ({max_scale}).")
    if global_scale is not None and float(global_scale) <= 0.0:
        raise ValueError(f"global_scale must be positive, got {global_scale}.")

    resolved_scene_graph = Path(scene_graph_path).resolve() if scene_graph_path else _auto_scene_graph_path(source_path)
    resolved_manifest = Path(manifest_path).resolve() if manifest_path else _auto_manifest_path(source_path)
    scene_graph_data = _load_json(resolved_scene_graph)
    manifest_data = _load_json(resolved_manifest)

    stage = Usd.Stage.Open(str(source_path))
    if stage is None:
        raise ValueError(f"Failed to open USD: {source_path}")

    objects = _discover_real2sim_objects(stage, scene_graph_data, manifest_data)
    if not objects:
        raise ValueError(f"No real2sim objects found in stage: {source_path}")

    object_by_path = {item.prim_path: item for item in objects}
    support_paths = {
        item.support_parent
        for item in objects
        if item.support_parent and stage.GetPrimAtPath(item.support_parent).IsValid()
    }
    scale_rng = random.Random(seed)

    sampled_scales: dict[str, float] = {}
    overrides = {str(path): float(scale) for path, scale in (scale_overrides or {}).items()}
    if overrides and (shared_scale or global_scale is not None):
        raise ValueError("scale_overrides cannot be combined with shared_scale/global_scale because all objects must share one scale.")

    scale_mode = "per_object"
    sampled_shared_scale: float | None = None
    if global_scale is not None:
        scale_mode = "global_fixed"
        sampled_shared_scale = float(_validate_scale(float(global_scale), prim_path="/World"))
    elif shared_scale:
        scale_mode = "shared_sampled"
        sampled_shared_scale = float(
            _validate_scale(
                scale_rng.uniform(float(min_scale), float(max_scale)),
                prim_path="/World",
            )
        )

    for item in objects:
        if sampled_shared_scale is not None:
            sampled_scales[item.prim_path] = float(sampled_shared_scale)
        elif item.prim_path in overrides:
            sampled_scales[item.prim_path] = _validate_scale(overrides[item.prim_path], prim_path=item.prim_path)
        else:
            sampled_scales[item.prim_path] = _validate_scale(
                scale_rng.uniform(float(min_scale), float(max_scale)),
                prim_path=item.prim_path,
            )

    original_bboxes: dict[str, dict[str, list[float]]] = {}
    original_world_matrices: dict[str, np.ndarray] = {}
    original_local_translations: dict[str, list[float]] = {}
    for prim_path in sorted({item.prim_path for item in objects} | set(support_paths)):
        original_world_matrices[prim_path] = _compute_world_matrix(stage, prim_path)
        original_bboxes[prim_path] = _compute_world_bbox(stage, prim_path)
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            original_local_translations[prim_path] = _serialize_vector(_get_local_matrix(prim)[3, :3])

    for item in objects:
        object_prim = stage.GetPrimAtPath(item.prim_path)
        object_matrix = _get_local_matrix(object_prim)
        scaled_object_matrix = object_matrix.copy()
        scaled_object_matrix[:3, :3] *= float(sampled_scales[item.prim_path])
        _set_local_matrix(object_prim, scaled_object_matrix)

    stage_world = stage.GetPrimAtPath("/World")
    if not stage_world.IsValid():
        raise ValueError("Stage does not contain /World")
    world_matrix = _compute_world_matrix(stage, "/World")

    free_root_paths = []
    for item in objects:
        parent = item.support_parent
        if not parent or not stage.GetPrimAtPath(parent).IsValid():
            free_root_paths.append(item.prim_path)
    if free_root_paths:
        root_centroid = np.mean(
            [np.array(original_bboxes[path]["center"], dtype=float) for path in free_root_paths],
            axis=0,
        )
        root_layout_scale = float(np.mean([sampled_scales[path] for path in free_root_paths]))
    else:
        root_centroid = _transform_point(np.zeros(3, dtype=float), world_matrix)
        root_layout_scale = 1.0

    ordered_paths = _topological_order(objects)
    for prim_path in ordered_paths:
        item = object_by_path[prim_path]
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim disappeared while editing stage: {prim_path}")

        current_bbox = _compute_world_bbox(stage, prim_path)
        current_center = np.array(current_bbox["center"], dtype=float)

        if item.support_parent and stage.GetPrimAtPath(item.support_parent).IsValid():
            parent_world_new = _compute_world_matrix(stage, item.support_parent)
            anchor_local = _transform_point(
                np.array(original_bboxes[prim_path]["center"], dtype=float),
                np.linalg.inv(original_world_matrices[item.support_parent]),
            )
            target_center = _transform_point(anchor_local, parent_world_new)
            _translate_prim(prim, target_center - current_center)

            child_bbox = _compute_world_bbox(stage, prim_path)
            parent_bbox = _compute_world_bbox(stage, item.support_parent)
            original_gap = float(original_bboxes[prim_path]["min"][2]) - float(original_bboxes[item.support_parent]["max"][2])
            child_bottom = float(child_bbox["min"][2])
            target_bottom = float(parent_bbox["max"][2]) + original_gap
            _translate_prim(prim, np.array([0.0, 0.0, target_bottom - child_bottom], dtype=float))
            continue

        original_center = np.array(original_bboxes[prim_path]["center"], dtype=float)
        target_center = root_centroid + (original_center - root_centroid) * root_layout_scale
        _translate_prim(prim, target_center - current_center)

        shifted_bbox = _compute_world_bbox(stage, prim_path)
        original_bottom = float(original_bboxes[prim_path]["min"][2])
        shifted_bottom = float(shifted_bbox["min"][2])
        _translate_prim(prim, np.array([0.0, 0.0, original_bottom - shifted_bottom], dtype=float))

    output_path = Path(output_usd_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output_path))

    metadata_payload: dict[str, Any] = {
        "scene_input_usd": str(source_path),
        "scene_output_usd": str(output_path),
        "scene_graph_path": str(resolved_scene_graph) if resolved_scene_graph else None,
        "manifest_path": str(resolved_manifest) if resolved_manifest else None,
        "seed": seed,
        "scale_range": [float(min_scale), float(max_scale)],
        "scale_mode": scale_mode,
        "shared_scale": float(sampled_shared_scale) if sampled_shared_scale is not None else None,
        "root_layout_mode": "mean_root_scale_about_root_centroid",
        "root_layout_scale": float(root_layout_scale),
        "objects": {},
    }

    for item in objects:
        prim = stage.GetPrimAtPath(item.prim_path)
        local_matrix = _get_local_matrix(prim)
        metadata_payload["objects"][item.prim_path] = {
            "label": item.label,
            "caption": item.caption,
            "support_parent": item.support_parent,
            "asset_ref_prim_path": item.asset_ref_prim_path,
            "scale": float(sampled_scales[item.prim_path]),
            "translation_before": original_local_translations[item.prim_path],
            "translation_after": _serialize_vector(local_matrix[3, :3]),
            "bbox_before": original_bboxes[item.prim_path],
            "bbox_after": _compute_world_bbox(stage, item.prim_path),
        }

    metadata_path = Path(output_metadata_path).resolve() if output_metadata_path else output_path.with_suffix(".randomization.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "scene_input_usd": str(source_path),
        "scene_output_usd": str(output_path),
        "metadata_path": str(metadata_path),
        "scene_graph_path": str(resolved_scene_graph) if resolved_scene_graph else None,
        "manifest_path": str(resolved_manifest) if resolved_manifest else None,
        "object_count": len(objects),
        "scale_mode": scale_mode,
        "shared_scale": float(sampled_shared_scale) if sampled_shared_scale is not None else None,
        "root_layout_scale": float(root_layout_scale),
        "scales": {path: float(value) for path, value in sampled_scales.items()},
    }


def collect_real2sim_layout_snapshot(
    scene_usd_path: str | Path,
    *,
    scene_graph_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    _, Usd, _ = _import_pxr()

    usd_path = Path(scene_usd_path).resolve()
    resolved_scene_graph = Path(scene_graph_path).resolve() if scene_graph_path else _auto_scene_graph_path(usd_path)
    resolved_manifest = Path(manifest_path).resolve() if manifest_path else _auto_manifest_path(usd_path)
    scene_graph_data = _load_json(resolved_scene_graph)
    manifest_data = _load_json(resolved_manifest)

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Failed to open USD: {usd_path}")

    objects = _discover_real2sim_objects(stage, scene_graph_data, manifest_data)
    snapshot: dict[str, Any] = {
        "scene_usd_path": str(usd_path),
        "scene_graph_path": str(resolved_scene_graph) if resolved_scene_graph else None,
        "manifest_path": str(resolved_manifest) if resolved_manifest else None,
        "objects": {},
    }
    for item in objects:
        snapshot["objects"][item.prim_path] = {
            "label": item.label,
            "caption": item.caption,
            "support_parent": item.support_parent,
            "bbox": _compute_world_bbox(stage, item.prim_path),
        }
    return snapshot


def visualize_real2sim_scale_randomization(
    before_scene_usd_path: str | Path,
    after_scene_usd_path: str | Path,
    *,
    output_image_path: str | Path,
    scene_graph_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    dpi: int = 180,
) -> dict[str, Any]:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    from matplotlib import patches
    import matplotlib.pyplot as plt

    before = collect_real2sim_layout_snapshot(
        before_scene_usd_path,
        scene_graph_path=scene_graph_path,
        manifest_path=manifest_path,
    )
    after = collect_real2sim_layout_snapshot(
        after_scene_usd_path,
        scene_graph_path=scene_graph_path,
        manifest_path=manifest_path,
    )
    metadata = _load_json(Path(metadata_path).resolve()) if metadata_path else None

    object_paths = sorted(set(before["objects"]).intersection(after["objects"]))
    if not object_paths:
        raise ValueError("No overlapping real2sim objects found between the two USDs.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    top_ax, side_ax = axes
    cmap = plt.get_cmap("tab10")

    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = []

    for index, prim_path in enumerate(object_paths):
        color = cmap(index % 10)
        before_bbox = before["objects"][prim_path]["bbox"]
        after_bbox = after["objects"][prim_path]["bbox"]
        label = before["objects"][prim_path]["label"]
        scale_text = ""
        if metadata:
            object_meta = metadata.get("objects", {}).get(prim_path, {})
            if "scale" in object_meta:
                scale_text = f" x{float(object_meta['scale']):.3f}"

        top_ax.add_patch(
            patches.Rectangle(
                (before_bbox["min"][0], before_bbox["min"][1]),
                before_bbox["size"][0],
                before_bbox["size"][1],
                linewidth=1.6,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
                alpha=0.65,
            )
        )
        top_ax.add_patch(
            patches.Rectangle(
                (after_bbox["min"][0], after_bbox["min"][1]),
                after_bbox["size"][0],
                after_bbox["size"][1],
                linewidth=2.0,
                edgecolor=color,
                facecolor="none",
                linestyle="-",
            )
        )
        top_ax.plot(before_bbox["center"][0], before_bbox["center"][1], marker="o", color=color, alpha=0.55)
        top_ax.plot(after_bbox["center"][0], after_bbox["center"][1], marker="x", color=color)
        top_ax.annotate(
            f"{label}{scale_text}",
            xy=(after_bbox["center"][0], after_bbox["center"][1]),
            xytext=(6, 6),
            textcoords="offset points",
            color=color,
            fontsize=9,
        )

        side_ax.add_patch(
            patches.Rectangle(
                (before_bbox["min"][0], before_bbox["min"][2]),
                before_bbox["size"][0],
                before_bbox["size"][2],
                linewidth=1.6,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
                alpha=0.65,
            )
        )
        side_ax.add_patch(
            patches.Rectangle(
                (after_bbox["min"][0], after_bbox["min"][2]),
                after_bbox["size"][0],
                after_bbox["size"][2],
                linewidth=2.0,
                edgecolor=color,
                facecolor="none",
                linestyle="-",
            )
        )
        side_ax.plot(before_bbox["center"][0], before_bbox["center"][2], marker="o", color=color, alpha=0.55)
        side_ax.plot(after_bbox["center"][0], after_bbox["center"][2], marker="x", color=color)

        for bbox in (before_bbox, after_bbox):
            all_x.extend([float(bbox["min"][0]), float(bbox["max"][0])])
            all_y.extend([float(bbox["min"][1]), float(bbox["max"][1])])
            all_z.extend([float(bbox["min"][2]), float(bbox["max"][2])])

    top_ax.set_title("Top View (X-Y)")
    top_ax.set_xlabel("X (m)")
    top_ax.set_ylabel("Y (m)")
    top_ax.grid(True, alpha=0.25)
    top_ax.set_aspect("equal", adjustable="box")

    side_ax.set_title("Side View (X-Z)")
    side_ax.set_xlabel("X (m)")
    side_ax.set_ylabel("Z (m)")
    side_ax.grid(True, alpha=0.25)
    side_ax.set_aspect("equal", adjustable="box")

    if all_x and all_y and all_z:
        x_margin = max((max(all_x) - min(all_x)) * 0.08, 0.05)
        y_margin = max((max(all_y) - min(all_y)) * 0.08, 0.05)
        z_margin = max((max(all_z) - min(all_z)) * 0.08, 0.05)
        top_ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        top_ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        side_ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        side_ax.set_ylim(min(all_z) - z_margin, max(all_z) + z_margin)

    output_path = Path(output_image_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    return {
        "before_scene_usd": str(Path(before_scene_usd_path).resolve()),
        "after_scene_usd": str(Path(after_scene_usd_path).resolve()),
        "output_image_path": str(output_path),
        "object_count": len(object_paths),
    }
