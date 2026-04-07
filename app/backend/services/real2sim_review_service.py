from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pipelines.real2sim.manifest import MANIFEST_FILENAME, MASK_METADATA_FILENAME, _load_json
from pipelines.real2sim.vlm_assignment import (
    ASSIGNMENT_FILENAME,
    NUMBERED_MASK_OVERLAY_FILENAME,
    build_mask_label_index,
    build_scene_assignment_context,
    render_numbered_masks,
)


LOW_CONFIDENCE_THRESHOLD = 0.75


def _relative_to(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), root.resolve())
    except ValueError:
        return str(path.resolve())


def _clamp_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _resolve_mask_metadata_path(results_root: Path, masks_root: Path) -> Path | None:
    for candidate in (results_root / MASK_METADATA_FILENAME, masks_root / MASK_METADATA_FILENAME):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _collect_mask_paths(masks_root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in masks_root.glob("*.png")
            if path.is_file() and path.name.lower() != "image.png"
        ],
        key=lambda path: (0, int(path.stem), path.name) if path.stem.isdigit() else (1, path.stem, path.name),
    )


def _resolve_review_image_path(masks_root: Path, latest_input_image: Path | None) -> Path | None:
    segmented_image = masks_root / "image.png"
    if segmented_image.exists() and segmented_image.is_file():
        return segmented_image
    if latest_input_image is not None and latest_input_image.exists() and latest_input_image.is_file():
        return latest_input_image
    return None


def _build_review_context(
    *,
    scene_graph_path: str | Path,
    masks_dir: str | Path,
    results_dir: str | Path,
    latest_input_image: str | Path | None,
) -> dict[str, Any]:
    scene_graph_file = Path(scene_graph_path)
    if not scene_graph_file.exists():
        raise FileNotFoundError(f"Scene graph not found: {scene_graph_file}")

    masks_root = Path(masks_dir)
    if not masks_root.exists():
        raise FileNotFoundError(f"Real2Sim masks directory not found: {masks_root}")

    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    scene_graph_data = _load_json(scene_graph_file)
    scene_context = build_scene_assignment_context(scene_graph_data)
    scene_objects = sorted(
        [
            {
                "path": str(obj.get("path") or ""),
                "class": obj.get("class"),
                "caption": obj.get("caption"),
                "source": obj.get("source"),
            }
            for obj in scene_context.get("objects", [])
            if isinstance(obj, dict) and isinstance(obj.get("path"), str) and obj.get("path")
        ],
        key=lambda obj: obj["path"],
    )
    if not scene_objects:
        raise ValueError("No real2sim scene objects are available for manual review.")

    mask_metadata_path = _resolve_mask_metadata_path(results_root, masks_root)
    mask_paths = _collect_mask_paths(masks_root)
    if not mask_paths:
        raise ValueError("No Real2Sim masks are available for manual review.")

    mask_index = build_mask_label_index(mask_paths, mask_metadata_path=mask_metadata_path)
    if not mask_index:
        raise ValueError("Failed to build the mask index for manual review.")

    latest_image_path = Path(latest_input_image) if latest_input_image else None
    review_image_path = _resolve_review_image_path(masks_root, latest_image_path)
    overlay_path = results_root / NUMBERED_MASK_OVERLAY_FILENAME
    if review_image_path is not None and not overlay_path.exists():
        render_numbered_masks(review_image_path, mask_index, overlay_path)

    assignment_path = results_root / ASSIGNMENT_FILENAME
    manifest_path = results_root / MANIFEST_FILENAME
    return {
        "scene_graph_file": scene_graph_file,
        "scene_graph_data": scene_graph_data,
        "scene_context": scene_context,
        "scene_objects": scene_objects,
        "masks_root": masks_root,
        "results_root": results_root,
        "mask_index": mask_index,
        "mask_metadata_path": mask_metadata_path,
        "review_image_path": review_image_path,
        "overlay_path": overlay_path if overlay_path.exists() else None,
        "assignment_path": assignment_path,
        "manifest_path": manifest_path,
    }


def _normalize_assignments(
    raw_assignments: list[dict[str, Any]],
    *,
    scene_objects: list[dict[str, Any]],
    mask_index: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[int], list[dict[str, Any]]]:
    valid_scene_paths = {str(obj["path"]) for obj in scene_objects}
    label_lookup = {
        int(entry["mask_label"]): entry
        for entry in mask_index
        if isinstance(entry, dict) and isinstance(entry.get("mask_label"), int)
    }
    object_lookup = {str(obj["path"]): obj for obj in scene_objects}

    assignments: list[dict[str, Any]] = []
    low_confidence: list[dict[str, Any]] = []
    seen_scene_paths: set[str] = set()
    seen_mask_labels: set[int] = set()

    for row in raw_assignments:
        if not isinstance(row, dict):
            continue
        scene_path = row.get("scene_path")
        mask_label = row.get("mask_label")
        if not isinstance(scene_path, str) or scene_path not in valid_scene_paths:
            continue
        if not isinstance(mask_label, int) or mask_label not in label_lookup:
            continue
        if scene_path in seen_scene_paths or mask_label in seen_mask_labels:
            continue

        confidence = _clamp_confidence(row.get("confidence", 0.0))
        entry = label_lookup[mask_label]
        assignment = {
            "scene_path": scene_path,
            "mask_label": mask_label,
            "output_name": entry["output_name"],
            "confidence": confidence,
            "reason": row.get("reason"),
            "scene_class": object_lookup[scene_path].get("class"),
            "scene_caption": object_lookup[scene_path].get("caption"),
        }
        assignments.append(assignment)
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            low_confidence.append(assignment)
        seen_scene_paths.add(scene_path)
        seen_mask_labels.add(mask_label)

    unmatched_scene_paths = sorted(valid_scene_paths - seen_scene_paths)
    unmatched_mask_labels = sorted(set(label_lookup.keys()) - seen_mask_labels)
    return assignments, unmatched_scene_paths, unmatched_mask_labels, low_confidence


def _load_manifest_summary(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists() or not manifest_path.is_file():
        return {
            "exists": False,
            "matched_objects": 0,
            "unmatched_scene_paths": [],
            "unmatched_outputs": [],
        }

    payload = _load_json(manifest_path)
    objects = payload.get("objects")
    object_count = len(objects) if isinstance(objects, dict) else 0
    return {
        "exists": True,
        "matched_objects": object_count,
        "unmatched_scene_paths": [
            str(path) for path in payload.get("unmatched_scene_paths", []) if isinstance(path, str) and path
        ],
        "unmatched_outputs": [
            str(name) for name in payload.get("unmatched_outputs", []) if isinstance(name, str) and name
        ],
    }


def _build_review_payload(
    context: dict[str, Any],
    *,
    raw_assignment_payload: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    results_root = context["results_root"]
    scene_graph_file = context["scene_graph_file"]
    review_image_path = context["review_image_path"]
    overlay_path = context["overlay_path"]
    scene_objects = context["scene_objects"]
    mask_index = context["mask_index"]
    assignment_payload = raw_assignment_payload if isinstance(raw_assignment_payload, dict) else {}

    assignments, unmatched_scene_paths, unmatched_mask_labels, low_confidence = _normalize_assignments(
        assignment_payload.get("assignments", []),
        scene_objects=scene_objects,
        mask_index=mask_index,
    )
    manifest_summary = _load_manifest_summary(context["manifest_path"])

    payload = {
        "version": 1,
        "source": source,
        "scene_graph_path": str(scene_graph_file.resolve()),
        "image_path": _relative_to(results_root, review_image_path),
        "overlay_image_path": _relative_to(results_root, overlay_path),
        "mask_labels": [
            {
                "mask_label": int(entry["mask_label"]),
                "output_name": entry["output_name"],
                "mask_path": str(Path(entry["mask_path"]).resolve()),
                "mask_path_relative": _relative_to(results_root, Path(entry["mask_path"])),
                "prompt": entry.get("prompt"),
                "prompt_key": entry.get("prompt_key"),
                "bbox_xyxy": entry.get("bbox_xyxy"),
            }
            for entry in mask_index
        ],
        "scene_objects": scene_objects,
        "assignments": assignments,
        "unmatched_scene_paths": unmatched_scene_paths,
        "unmatched_mask_labels": unmatched_mask_labels,
        "low_confidence_assignments": low_confidence,
        "summary": {
            "matched_assignments": len(assignments),
            "scene_objects": len(scene_objects),
            "mask_labels": len(mask_index),
            "unmatched_scene_paths": len(unmatched_scene_paths),
            "unmatched_mask_labels": len(unmatched_mask_labels),
            "low_confidence_assignments": len(low_confidence),
        },
        "manifest": manifest_summary,
        "needs_attention": bool(
            unmatched_scene_paths
            or unmatched_mask_labels
            or low_confidence
            or manifest_summary["unmatched_scene_paths"]
            or manifest_summary["unmatched_outputs"]
        ),
        "assignment_path": str(context["assignment_path"].resolve()),
        "manifest_path": str(context["manifest_path"].resolve()),
        "review_image_abs_path": str(review_image_path.resolve()) if review_image_path is not None else None,
        "overlay_image_abs_path": str(overlay_path.resolve()) if overlay_path is not None else None,
    }
    return payload


def load_assignment_review(
    *,
    scene_graph_path: str | Path,
    masks_dir: str | Path,
    results_dir: str | Path,
    latest_input_image: str | Path | None = None,
) -> dict[str, Any]:
    context = _build_review_context(
        scene_graph_path=scene_graph_path,
        masks_dir=masks_dir,
        results_dir=results_dir,
        latest_input_image=latest_input_image,
    )
    raw_assignment = _load_json(context["assignment_path"]) if context["assignment_path"].exists() else {}
    source = str(raw_assignment.get("source") or "manual_review")
    return _build_review_payload(context, raw_assignment_payload=raw_assignment, source=source)


def save_assignment_review(
    *,
    assignments: list[dict[str, Any]],
    scene_graph_path: str | Path,
    masks_dir: str | Path,
    results_dir: str | Path,
    latest_input_image: str | Path | None = None,
) -> dict[str, Any]:
    context = _build_review_context(
        scene_graph_path=scene_graph_path,
        masks_dir=masks_dir,
        results_dir=results_dir,
        latest_input_image=latest_input_image,
    )
    results_root = context["results_root"]
    assignment_path = context["assignment_path"]

    valid_scene_paths = {str(obj["path"]) for obj in context["scene_objects"]}
    valid_mask_labels = {
        int(entry["mask_label"]): entry
        for entry in context["mask_index"]
        if isinstance(entry, dict) and isinstance(entry.get("mask_label"), int)
    }
    used_scene_paths: set[str] = set()
    used_mask_labels: set[int] = set()
    normalized_rows: list[dict[str, Any]] = []

    for row in assignments:
        if not isinstance(row, dict):
            continue
        scene_path = str(row.get("scene_path") or "").strip()
        if not scene_path:
            continue
        if scene_path not in valid_scene_paths:
            raise ValueError(f"Unknown scene_path in manual assignment: {scene_path}")

        mask_label_raw = row.get("mask_label")
        if not isinstance(mask_label_raw, int):
            raise ValueError("mask_label must be an integer in manual assignment review.")
        if mask_label_raw not in valid_mask_labels:
            raise ValueError(f"Unknown mask_label in manual assignment: {mask_label_raw}")
        if scene_path in used_scene_paths:
            raise ValueError(f"Scene object was assigned more than once: {scene_path}")
        if mask_label_raw in used_mask_labels:
            raise ValueError(f"Mask label was assigned more than once: {mask_label_raw}")

        mask_entry = valid_mask_labels[mask_label_raw]
        normalized_rows.append(
            {
                "scene_path": scene_path,
                "mask_label": mask_label_raw,
                "output_name": mask_entry["output_name"],
                "confidence": _clamp_confidence(row.get("confidence", 1.0)) or 1.0,
                "reason": str(row.get("reason") or "Confirmed via manual review.").strip(),
            }
        )
        used_scene_paths.add(scene_path)
        used_mask_labels.add(mask_label_raw)

    payload = _build_review_payload(
        context,
        raw_assignment_payload={
            "assignments": normalized_rows,
        },
        source="manual_review",
    )
    assignment_to_save = {
        "version": payload["version"],
        "source": "manual_review",
        "scene_graph_path": payload["scene_graph_path"],
        "image_path": payload["image_path"],
        "overlay_image_path": payload["overlay_image_path"],
        "mask_labels": payload["mask_labels"],
        "assignments": [
            {
                "scene_path": row["scene_path"],
                "mask_label": row["mask_label"],
                "output_name": row["output_name"],
                "confidence": row["confidence"],
                "reason": row.get("reason"),
            }
            for row in payload["assignments"]
        ],
        "unmatched_scene_paths": payload["unmatched_scene_paths"],
        "unmatched_mask_labels": payload["unmatched_mask_labels"],
    }
    assignment_path.write_text(json.dumps(assignment_to_save, indent=2, ensure_ascii=False), encoding="utf-8")

    poses_path = results_root / "poses.json"
    if poses_path.exists() and poses_path.is_file():
        from pipelines.real2sim.manifest import build_real2sim_asset_manifest

        build_real2sim_asset_manifest(results_root, scene_graph_path=context["scene_graph_file"])

    return load_assignment_review(
        scene_graph_path=scene_graph_path,
        masks_dir=masks_dir,
        results_dir=results_dir,
        latest_input_image=latest_input_image,
    )
