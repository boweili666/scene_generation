"""Loading and lazy-building scene grasp proposals + manifest resolution.

Scene auto grasp collection needs a JSON payload of grasp candidates
keyed by target prim. That payload is produced by the grasp annotator
pipeline (`tools/build_grasp_asset_cache.py` + the per-asset cache);
this module is a thin loader on top of that:

* `_load_or_build_scene_grasp_payload` is the entry point used by the
  pipeline. It locates the proposals JSON next to the scene USD,
  resolves the manifest + annotation root, optionally builds a missing
  per-target cache on demand, and returns the resolved payload.
* `_maybe_lazy_build_target_annotation` triggers a single-asset cache
  rebuild via `ensure_asset_grasp_cache_for_prim` if the target is the
  first one we need and the cache is missing.
* `_world_bbox_payload` snapshots a target prim's world-space AABB —
  used downstream to filter candidates that wouldn't fit through the
  bounding box.

The module also bundles small path-resolution helpers
(`_auto_manifest_path`, `_resolve_adjacent_path`, `_resolve_manifest_and_annotation_root`)
so the manifest discovery logic stays alongside the I/O it powers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.backend.services.grasp_asset_cache import (
    cache_entry_path_for_prim,
    default_grasp_annotation_root,
    ensure_asset_grasp_cache_for_prim,
)
from app.backend.services.grasp_scene_adapter import (
    build_stage_grasp_proposals,
    default_scene_grasp_proposals_path,
)


def _resolve_adjacent_path(scene_usd_path: Path, relative_parts: tuple[str, ...]) -> Path | None:
    for parent in (scene_usd_path.parent, *scene_usd_path.parents):
        candidate = parent.joinpath(*relative_parts)
        if candidate.exists():
            return candidate.resolve()
    return None


def _auto_manifest_path(scene_usd_path: str | Path) -> Path | None:
    return _resolve_adjacent_path(
        Path(scene_usd_path).resolve(),
        ("real2sim", "scene_results", "real2sim_asset_manifest.json"),
    )


def _resolve_manifest_and_annotation_root(args) -> tuple[Path, Path]:
    scene_usd_path = Path(args.scene_usd_path).resolve()
    manifest_path = (
        Path(args.manifest_path).resolve() if args.manifest_path else _auto_manifest_path(scene_usd_path)
    )
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(
            "No scene grasp proposals found and no Real2Sim manifest could be resolved. "
            "Run build_grasp_asset_cache.py / export_scene_grasp_proposals.py first or pass --manifest-path."
        )
    annotation_root = (
        Path(args.annotation_root).resolve()
        if args.annotation_root
        else default_grasp_annotation_root(manifest_path)
    )
    return manifest_path, annotation_root


def _maybe_lazy_build_target_annotation(
    args,
    manifest_path: Path,
    annotation_root: Path,
    *,
    target_prim: str | None,
) -> bool:
    if not args.lazy_build_target_annotation or not target_prim:
        return False
    target_cache_path = cache_entry_path_for_prim(manifest_path, target_prim, output_root=annotation_root)
    if target_cache_path is None:
        raise ValueError(f"Target prim '{target_prim}' was not found in manifest {manifest_path}.")
    if target_cache_path.exists():
        return False
    print(f"[INFO] Missing grasp annotation cache for target {target_prim}; building it now.")
    ensure_asset_grasp_cache_for_prim(
        manifest_path,
        target_prim,
        output_root=annotation_root,
        resume=True,
    )
    return True


def _load_or_build_scene_grasp_payload(
    stage,
    args,
    *,
    target_prim: str | None = None,
) -> tuple[dict[str, Any], Path]:
    scene_usd_path = Path(args.scene_usd_path).resolve()
    proposals_path = (
        Path(args.scene_grasp_proposals_path).resolve()
        if args.scene_grasp_proposals_path
        else default_scene_grasp_proposals_path(scene_usd_path)
    )
    manifest_path, annotation_root = _resolve_manifest_and_annotation_root(args)
    _maybe_lazy_build_target_annotation(
        args,
        manifest_path,
        annotation_root,
        target_prim=target_prim,
    )
    payload = build_stage_grasp_proposals(
        stage,
        manifest_path,
        annotation_root=annotation_root,
        output_path=proposals_path,
        target_prim_paths=None if target_prim is None else [target_prim],
    )
    objects = payload.get("objects", {})
    if target_prim is not None and (not isinstance(objects, dict) or target_prim not in objects):
        target_cache_path = cache_entry_path_for_prim(manifest_path, target_prim, output_root=annotation_root)
        missing_cache_note = ""
        if target_cache_path is not None and not target_cache_path.exists():
            missing_cache_note = f" Missing target cache: {target_cache_path}."
        raise RuntimeError(
            f"Failed to resolve scene grasp proposals for target '{target_prim}'."
            f"{missing_cache_note}"
        )
    return payload, proposals_path


def _world_bbox_payload(stage, prim_path: str) -> dict[str, list[float]] | None:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "guide", "render"], useExtentsHint=True)
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if aligned.IsEmpty():
        return None
    bmin = aligned.GetMin()
    bmax = aligned.GetMax()
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
