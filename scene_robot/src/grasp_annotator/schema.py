from __future__ import annotations

import os
from pathlib import Path


def _relpath(path_value: str | None, object_dir: Path) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    try:
        return os.path.relpath(path, object_dir)
    except ValueError:
        return str(path)


def _collect_artifacts(stage: dict, keys: list[str], object_dir: Path) -> dict:
    artifacts = {}
    for key in keys:
        value = stage.get(key)
        rel = _relpath(value, object_dir) if isinstance(value, str) else None
        if rel:
            artifacts[key] = rel
    return artifacts


def _extract_grasp_primitives(stage: dict | None) -> list[dict]:
    if not isinstance(stage, dict):
        return []
    primitives = stage.get("grasp_primitives")
    if isinstance(primitives, list):
        return [primitive for primitive in primitives if isinstance(primitive, dict)]
    primitive = stage.get("grasp_primitive")
    if isinstance(primitive, dict):
        return [primitive]
    return []


def build_annotation(result: dict, glb_path: Path, object_dir: Path) -> dict:
    category = result.get("classification", {}).get("category")
    annotation = {
        "schema_version": "grasp_primitives_v1",
        "object_name": result.get("object_name", glb_path.stem),
        "source_glb": str(glb_path.resolve()),
        "category": category,
        "grasp_primitives": [],
        "artifacts": {
            "pipeline_result": "pipeline_result.json",
        },
    }

    classification_json = result.get("classification_json")
    rel_classification = _relpath(classification_json, object_dir) if isinstance(classification_json, str) else None
    if rel_classification:
        annotation["artifacts"]["classification_json"] = rel_classification

    if category == "handle_tool_object":
        stage = result.get("handle_tool_stage", {})
        annotation["grasp_primitives"] = _extract_grasp_primitives(stage)
        annotation["artifacts"].update(
            _collect_artifacts(
                stage,
                [
                    "skeleton_view_image",
                    "candidate_overlay",
                    "candidate_json",
                    "final_effect_image",
                    "open3d_effect_image",
                ],
                object_dir,
            )
        )
    elif category == "axis_object":
        stage = result.get("axis_object_stage", {})
        annotation["grasp_primitives"] = _extract_grasp_primitives(stage)
        annotation["artifacts"].update(
            _collect_artifacts(
                stage,
                [
                    "axis_samples_overlay",
                    "axis_samples_json",
                    "final_effect_image",
                    "open3d_effect_image",
                ],
                object_dir,
            )
        )
    elif category == "graspnet_object":
        stage = result.get("graspnet_object_stage", {})
        annotation["grasp_primitives"] = _extract_grasp_primitives(stage)
        annotation["artifacts"].update(_collect_artifacts(stage, ["final_effect_image"], object_dir))

    return annotation
