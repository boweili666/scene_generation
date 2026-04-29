from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

if __package__:
    from .manifest import REAL2SIM_SOURCE, _load_json, _scene_objects_by_source
else:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from pipelines.real2sim.manifest import REAL2SIM_SOURCE, _load_json, _scene_objects_by_source


ASSIGNMENT_FILENAME = "assignment.json"
NUMBERED_MASK_OVERLAY_FILENAME = "numbered_masks_overlay.png"
_PALETTE = [
    (230, 57, 70),
    (29, 78, 216),
    (46, 125, 50),
    (255, 140, 0),
    (111, 66, 193),
    (0, 121, 107),
    (214, 51, 132),
    (120, 144, 156),
]


def _mask_sort_key(mask_path: Path) -> tuple[int, int | str, str]:
    stem = mask_path.stem
    if stem.isdigit():
        return (0, int(stem), mask_path.name)
    return (1, stem, mask_path.name)


def _coerce_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x_min, y_min, x_max, y_max = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    if x_max < x_min or y_max < y_min:
        return None
    return x_min, y_min, x_max, y_max


def _bbox_from_mask(mask_path: Path) -> tuple[int, int, int, int] | None:
    # PIL `Image.open` is lazy; the actual decode happens inside `convert`.
    # If the mask file is being written concurrently by the segmentation
    # step (race during early polling), the decode raises OSError or
    # struct.error. Treat any such unreadable-mask failure as "no bbox
    # available yet" so the review still loads with whatever masks are
    # already complete.
    try:
        mask_image = Image.open(mask_path).convert("RGBA")
    except Exception:  # noqa: BLE001 - any read/decode failure means skip
        return None
    alpha = mask_image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return left, top, max(left, right - 1), max(top, bottom - 1)


def _relative_string(root: Path, path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), root.resolve())
    except ValueError:
        return str(path.resolve())


def _load_font(image_size: tuple[int, int]) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    target_size = max(18, min(image_size) // 18)
    for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, target_size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_mask_label_index(
    mask_paths: Iterable[Path],
    *,
    mask_metadata_path: Path | None = None,
) -> list[dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if mask_metadata_path is not None and mask_metadata_path.exists():
        metadata = _load_json(mask_metadata_path)

    sorted_masks = sorted(
        (
            path
            for path in mask_paths
            if path.is_file() and path.suffix.lower() == ".png" and path.name.lower() != "image.png"
        ),
        key=_mask_sort_key,
    )

    mask_index: list[dict[str, Any]] = []
    for mask_label, mask_path in enumerate(sorted_masks, start=1):
        output_name = mask_path.stem
        meta = metadata.get(output_name, {}) if isinstance(metadata.get(output_name), dict) else {}
        bbox = _coerce_bbox(meta.get("bbox_xyxy")) or _bbox_from_mask(mask_path)
        mask_index.append(
            {
                "mask_label": mask_label,
                "output_name": output_name,
                "mask_path": str(mask_path.resolve()),
                "prompt": meta.get("prompt"),
                "prompt_key": meta.get("prompt_key"),
                "bbox_xyxy": list(bbox) if bbox is not None else None,
            }
        )

    return mask_index


def render_numbered_masks(
    image_path: str | Path,
    mask_index: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    image_file = Path(image_path)
    output_file = Path(output_path)
    base = Image.open(image_file).convert("RGBA")
    overlay = base.copy()
    font = _load_font(base.size)

    for entry in mask_index:
        mask_path = Path(str(entry.get("mask_path") or ""))
        if not mask_path.exists():
            continue
        color = _PALETTE[(int(entry["mask_label"]) - 1) % len(_PALETTE)]
        mask_image = Image.open(mask_path).convert("RGBA")
        alpha = mask_image.getchannel("A")
        alpha = alpha.point(lambda value: int(min(150, value * 0.45)))
        tint = Image.new("RGBA", overlay.size, (*color, 0))
        tint.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, tint)

    draw = ImageDraw.Draw(overlay)
    for entry in mask_index:
        bbox = _coerce_bbox(entry.get("bbox_xyxy"))
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        color = _PALETTE[(int(entry["mask_label"]) - 1) % len(_PALETTE)]
        draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=max(3, min(base.size) // 400))

        label_text = str(entry["mask_label"])
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        box_width = text_width + 20
        box_height = text_height + 14
        box_left = max(0, min(x_min, base.size[0] - box_width))
        box_top = max(0, min(y_min - box_height - 6, base.size[1] - box_height))
        if box_top < y_min - box_height - 6:
            box_top = max(0, min(y_min + 6, base.size[1] - box_height))
        draw.rounded_rectangle(
            (box_left, box_top, box_left + box_width, box_top + box_height),
            radius=10,
            fill=(*color, 235),
            outline=(255, 255, 255, 255),
            width=2,
        )
        draw.text((box_left + 10, box_top + 7), label_text, fill=(255, 255, 255, 255), font=font)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_file)
    return output_file


def build_scene_assignment_context(scene_graph_data: dict[str, Any]) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    valid_paths: set[str] = set()
    for path, meta in _scene_objects_by_source(scene_graph_data, source=REAL2SIM_SOURCE):
        row = {
            "path": path,
            "class": meta.get("class") or meta.get("class_name"),
            "caption": meta.get("caption"),
            "source": meta.get("source"),
        }
        objects.append(row)
        valid_paths.add(path)

    edges = scene_graph_data.get("edges")
    obj_obj_edges: list[dict[str, Any]] = []
    obj_wall_edges: list[dict[str, Any]] = []
    if isinstance(edges, dict):
        for edge in edges.get("obj-obj", []):
            if not isinstance(edge, dict):
                continue
            if edge.get("source") in valid_paths and edge.get("target") in valid_paths:
                obj_obj_edges.append(edge)
        for edge in edges.get("obj-wall", []):
            if not isinstance(edge, dict):
                continue
            if edge.get("source") in valid_paths:
                obj_wall_edges.append(edge)

    return {
        "objects": objects,
        "edges": {
            "obj-obj": obj_obj_edges,
            "obj-wall": obj_wall_edges,
        },
    }


def _normalize_assignment_result(
    raw_result: dict[str, Any],
    *,
    scene_context: dict[str, Any],
    mask_index: list[dict[str, Any]],
    output_root: Path,
    scene_graph_path: Path,
    image_path: Path,
    overlay_path: Path,
) -> dict[str, Any]:
    valid_scene_paths = {
        str(obj.get("path"))
        for obj in scene_context.get("objects", [])
        if isinstance(obj, dict) and isinstance(obj.get("path"), str)
    }
    label_lookup = {
        int(entry["mask_label"]): entry
        for entry in mask_index
        if isinstance(entry, dict) and isinstance(entry.get("mask_label"), int)
    }

    assignments: list[dict[str, Any]] = []
    seen_scene_paths: set[str] = set()
    seen_mask_labels: set[int] = set()
    for row in raw_result.get("assignments", []):
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
        try:
            confidence = max(0.0, min(1.0, float(row.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        entry = label_lookup[mask_label]
        assignments.append(
            {
                "scene_path": scene_path,
                "mask_label": mask_label,
                "output_name": entry["output_name"],
                "confidence": confidence,
                "reason": row.get("reason"),
            }
        )
        seen_scene_paths.add(scene_path)
        seen_mask_labels.add(mask_label)

    unmatched_scene_paths = sorted(valid_scene_paths - seen_scene_paths)
    unmatched_mask_labels = sorted(set(label_lookup.keys()) - seen_mask_labels)

    return {
        "version": 1,
        "source": "vlm_mask_assignment",
        "scene_graph_path": str(scene_graph_path.resolve()),
        "image_path": _relative_string(output_root, image_path),
        "overlay_image_path": _relative_string(output_root, overlay_path),
        "mask_labels": [
            {
                "mask_label": int(entry["mask_label"]),
                "output_name": entry["output_name"],
                "mask_path": _relative_string(output_root, Path(entry["mask_path"])),
                "prompt": entry.get("prompt"),
                "prompt_key": entry.get("prompt_key"),
                "bbox_xyxy": entry.get("bbox_xyxy"),
            }
            for entry in mask_index
        ],
        "assignments": assignments,
        "unmatched_scene_paths": unmatched_scene_paths,
        "unmatched_mask_labels": unmatched_mask_labels,
    }


def generate_vlm_mask_assignment(
    image_path: str | Path,
    mask_paths: Iterable[Path],
    scene_graph_path: str | Path,
    output_dir: str | Path,
    *,
    mask_metadata_path: str | Path | None = None,
    model: str | None = None,
) -> Path | None:
    scene_graph_file = Path(scene_graph_path)
    if not scene_graph_file.exists():
        return None

    output_root = Path(output_dir)
    metadata_file = Path(mask_metadata_path) if mask_metadata_path is not None else None
    mask_index = build_mask_label_index(mask_paths, mask_metadata_path=metadata_file)
    if not mask_index:
        return None

    overlay_path = output_root / NUMBERED_MASK_OVERLAY_FILENAME
    render_numbered_masks(image_path, mask_index, overlay_path)

    scene_graph_data = _load_json(scene_graph_file)
    scene_context = build_scene_assignment_context(scene_graph_data)
    if not scene_context.get("objects"):
        return None

    try:
        from app.backend.services.openai_service import assign_real2sim_masks_with_images, encode_image_b64
    except Exception as exc:  # pragma: no cover - environment-dependent import failure
        raise RuntimeError(f"OpenAI service is unavailable for VLM mask assignment: {exc}") from exc

    raw_result = assign_real2sim_masks_with_images(
        scene_context,
        mask_index,
        original_image_b64=encode_image_b64(Path(image_path)),
        overlay_image_b64=encode_image_b64(overlay_path),
        model=model,
    )
    assignment = _normalize_assignment_result(
        raw_result,
        scene_context=scene_context,
        mask_index=mask_index,
        output_root=output_root,
        scene_graph_path=scene_graph_file,
        image_path=Path(image_path),
        overlay_path=overlay_path,
    )
    assignment_path = output_root / ASSIGNMENT_FILENAME
    assignment_path.write_text(json.dumps(assignment, indent=2, ensure_ascii=False), encoding="utf-8")
    return assignment_path
