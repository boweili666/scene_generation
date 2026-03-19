from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ResolvedAsset:
    asset_path: Path
    source: str
    reference_prim_path: Optional[str] = None


def find_asset_matches_by_name(name: str, usd_paths: Iterable[Path]) -> List[Path]:
    needle = name.lower()
    return [p for p in usd_paths if needle in p.name.lower()]


def first_asset_match(name: str, usd_paths: Iterable[Path]) -> Optional[Path]:
    matches = find_asset_matches_by_name(name, usd_paths)
    return matches[0] if matches else None


def object_label(meta: Dict[str, Any], prim: str) -> str:
    return str(meta.get("class") or meta.get("class_name") or Path(prim).name)


def load_real2sim_manifest(manifest_path: Path | None) -> Dict[str, Any]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in manifest: {manifest_path}")
    return data


def build_real2sim_uniform_scale_lookup(manifest: Dict[str, Any] | None) -> Dict[str, float]:
    if not isinstance(manifest, dict):
        return {}
    objects = manifest.get("objects")
    if not isinstance(objects, dict):
        return {}

    lookup: Dict[str, float] = {}
    for prim, entry in objects.items():
        if not isinstance(entry, dict):
            continue
        usd_transform = entry.get("usd_transform")
        if not (
            isinstance(usd_transform, list)
            and len(usd_transform) == 4
            and all(isinstance(row, list) and len(row) == 4 for row in usd_transform)
        ):
            continue

        column_norms = []
        for col in range(3):
            sq = 0.0
            for row in range(3):
                sq += float(usd_transform[row][col]) ** 2
            column_norms.append(math.sqrt(sq))

        scale = sum(column_norms) / 3.0
        if scale > 1e-6:
            lookup[str(prim)] = float(scale)
    return lookup


def _resolve_manifest_entry(manifest: Dict[str, Any], prim: str) -> Optional[ResolvedAsset]:
    objects = manifest.get("objects")
    if not isinstance(objects, dict):
        return None
    entry = objects.get(prim)
    if not isinstance(entry, dict):
        return None

    results_root_raw = manifest.get("results_root")
    if not isinstance(results_root_raw, str) or not results_root_raw:
        return None
    results_root = Path(results_root_raw).resolve()

    usd_path_raw = entry.get("usd_path")
    scene_usd_raw = manifest.get("scene_usd")
    reference_prim_path = entry.get("usd_prim_path")

    asset_path: Optional[Path] = None
    if isinstance(usd_path_raw, str) and usd_path_raw:
        asset_path = (results_root / usd_path_raw).resolve()
    elif isinstance(scene_usd_raw, str) and scene_usd_raw:
        asset_path = (results_root / scene_usd_raw).resolve()

    if asset_path is None or not asset_path.exists():
        return None

    return ResolvedAsset(
        asset_path=asset_path,
        source="real2sim",
        reference_prim_path=reference_prim_path if isinstance(reference_prim_path, str) and reference_prim_path else None,
    )


def build_asset_match_lookup(
    data: Dict[str, Any],
    fallback_usd_paths: List[Path],
    *,
    retrieval_usd_paths: Optional[List[Path]] = None,
    real2sim_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, ResolvedAsset]:
    retrieval_pool = retrieval_usd_paths if retrieval_usd_paths is not None else fallback_usd_paths
    manifest = real2sim_manifest or {}
    asset_match_lookup: Dict[str, ResolvedAsset] = {}

    for prim, meta in data.get("obj", {}).items():
        if not isinstance(meta, dict):
            continue

        source = meta.get("source")
        label = object_label(meta, prim)
        fallback_match = first_asset_match(label, fallback_usd_paths) or first_asset_match(Path(prim).name, fallback_usd_paths)

        if source == "retrieval":
            retrieval_match = first_asset_match(label, retrieval_pool) or first_asset_match(Path(prim).name, retrieval_pool)
            if retrieval_match:
                asset_match_lookup[prim] = ResolvedAsset(asset_path=retrieval_match, source="retrieval")
            continue

        if source == "real2sim":
            resolved = _resolve_manifest_entry(manifest, prim)
            if resolved is not None:
                asset_match_lookup[prim] = resolved
            continue

        if fallback_match:
            asset_match_lookup[prim] = ResolvedAsset(asset_path=fallback_match, source="fallback")

    return asset_match_lookup
