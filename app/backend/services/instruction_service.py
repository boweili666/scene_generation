from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List

from ..config import (
    DEFAULT_PLACEMENTS_PATH,
    DEFAULT_RENDER_PATH,
    LATEST_INPUT_IMAGE,
    SCENE_GRAPH_PATH,
)
from .instruction_router import (
    prune_placements_to_scene_graph,
    reconcile_placements_after_graph_edit,
    validate_route_decision,
)
from .openai_service import (
    SCHEMA,
    edit_placements_with_instruction,
    edit_scene_graph_with_instruction,
    normalize_scene_graph_payload,
    parse_scene_graph_from_image,
    parse_scene_graph_from_text,
    read_json_file,
    route_scene_instruction,
    write_json_file,
)

ALLOWED_OBJ_OBJ_RELATIONS = set(
    SCHEMA["properties"]["edges"]["properties"]["obj-obj"]["items"]["properties"]["relation"]["enum"]
)
ALLOWED_OBJ_WALL_RELATIONS = set(
    SCHEMA["properties"]["edges"]["properties"]["obj-wall"]["items"]["properties"]["relation"]["enum"]
)


def _encode_image_bytes(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def _load_scene_graph(path: str | Path = SCENE_GRAPH_PATH) -> Dict[str, Any] | None:
    graph_path = Path(path)
    if not graph_path.exists():
        return None
    try:
        payload = read_json_file(graph_path)
    except json.JSONDecodeError:
        return None
    return normalize_scene_graph_payload(payload)


def _normalize_placements_payload(payload: Dict[str, Any]) -> Dict[str, List[float]]:
    normalized: Dict[str, List[float]] = {}
    if not isinstance(payload, dict):
        return normalized
    for path, value in payload.items():
        if isinstance(value, (list, tuple)) and len(value) in (3, 4):
            normalized[str(path)] = [float(part) for part in value]
            continue
        if isinstance(value, dict) and all(key in value for key in ("x", "y", "z")):
            row = [float(value["x"]), float(value["y"]), float(value["z"])]
            if value.get("yaw") is not None:
                row.append(float(value["yaw"]))
            normalized[str(path)] = row
    return normalized


def _load_placements(path: str | Path = DEFAULT_PLACEMENTS_PATH) -> Dict[str, List[float]]:
    placements_path = Path(path)
    if not placements_path.exists():
        return {}
    try:
        payload = read_json_file(placements_path)
    except json.JSONDecodeError:
        return {}
    return _normalize_placements_payload(payload)


def _write_placements(path: str | Path, placements: Dict[str, List[float]]) -> None:
    serializable = {key: list(value) for key, value in sorted(placements.items())}
    write_json_file(Path(path), serializable)


def _save_latest_input_image(image_bytes: bytes) -> None:
    LATEST_INPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    LATEST_INPUT_IMAGE.write_bytes(image_bytes)


def _current_render_image_b64() -> str | None:
    render_path = Path(DEFAULT_RENDER_PATH)
    if not render_path.exists():
        return None
    raw = render_path.read_bytes()
    return _encode_image_bytes(raw, "image/png")


def _generate_scene_graph(
    instruction: str,
    class_names_raw: str,
    image_bytes: bytes | None,
) -> Dict[str, Any]:
    if image_bytes is not None:
        return parse_scene_graph_from_image(image_bytes, class_names_raw, text=instruction)
    if not instruction.strip():
        raise ValueError("A text instruction or reference image is required.")
    return parse_scene_graph_from_text(instruction, class_names_raw)


def _merge_placement_updates(
    scene_graph: Dict[str, Any],
    placements: Dict[str, List[float]],
    updates_payload: Dict[str, Any],
) -> tuple[Dict[str, List[float]], List[str]]:
    next_placements = prune_placements_to_scene_graph(scene_graph, placements)
    valid_paths = set((scene_graph.get("obj") or {}).keys())
    updated_paths: List[str] = []
    for item in updates_payload.get("updates", []):
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        if path not in valid_paths:
            raise ValueError(f"Placement update referenced unknown object path: {path}")
        row = [
            float(item["x"]),
            float(item["y"]),
            float(item["z"]),
        ]
        yaw_deg = item.get("yaw_deg")
        if yaw_deg is not None:
            row.append(float(yaw_deg))
        next_placements[path] = row
        updated_paths.append(path)
    return next_placements, sorted(dict.fromkeys(updated_paths))


def _empty_scene_graph() -> Dict[str, Any]:
    return {
        "scene": {
            "room_type": "empty room",
            "dimensions": {"length": 5.0, "width": 5.0, "height": 3.0, "unit": "m"},
            "materials": {"floor": "concrete", "walls": "paint"},
        },
        "obj": {},
        "edges": {"obj-obj": [], "obj-wall": []},
    }


def _dedupe_edge_payload(edges: List[Dict[str, Any]], *, wall_mode: bool = False) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "").strip()
        relation = str(edge.get("relation") or "").strip()
        target = str(edge.get("target") or "").strip()
        signature = (source, relation) if wall_mode else (source, target, relation)
        if signature in seen:
            continue
        seen.add(signature)
        payload = {"source": source, "relation": relation}
        if not wall_mode:
            payload["target"] = target
        elif target:
            payload["target"] = target
        deduped.append(payload)
    return deduped


def _validate_saved_scene_graph(scene_graph: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_scene_graph_payload(scene_graph)
    if not isinstance(normalized.get("scene"), dict):
        raise ValueError("Scene graph must contain a 'scene' object.")

    obj_map = normalized.get("obj") or {}
    obj_paths = set(obj_map.keys())
    obj_obj_edges = _dedupe_edge_payload((normalized.get("edges") or {}).get("obj-obj") or [])
    obj_wall_edges = _dedupe_edge_payload((normalized.get("edges") or {}).get("obj-wall") or [], wall_mode=True)

    for path, meta in obj_map.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Object '{path}' must be a JSON object.")
        if meta.get("source") not in {"real2sim", "retrieval"}:
            raise ValueError(f"Object '{path}' must have source real2sim or retrieval.")

    for edge in obj_obj_edges:
        if edge["source"] not in obj_paths or edge["target"] not in obj_paths:
            raise ValueError("Each obj-obj edge must reference existing object paths.")
        if edge["source"] == edge["target"]:
            raise ValueError("Scene graph edges cannot be self-relations.")
        if edge["relation"] not in ALLOWED_OBJ_OBJ_RELATIONS:
            raise ValueError(
                f"Unsupported obj-obj relation '{edge['relation']}'. "
                f"Allowed: {sorted(ALLOWED_OBJ_OBJ_RELATIONS)}"
            )

    for edge in obj_wall_edges:
        if edge["source"] not in obj_paths:
            raise ValueError("Each obj-wall edge must reference an existing object path.")
        if edge["relation"] not in ALLOWED_OBJ_WALL_RELATIONS:
            raise ValueError(
                f"Unsupported obj-wall relation '{edge['relation']}'. "
                f"Allowed: {sorted(ALLOWED_OBJ_WALL_RELATIONS)}"
            )

    normalized["edges"] = {
        "obj-obj": obj_obj_edges,
        "obj-wall": obj_wall_edges,
    }
    return normalized


def save_scene_graph_state(scene_graph: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _validate_saved_scene_graph(scene_graph)
    current_placements = _load_placements()
    next_placements = prune_placements_to_scene_graph(normalized, current_placements)
    invalidated_paths = sorted(set(current_placements.keys()) - set(next_placements.keys()))

    write_json_file(Path(SCENE_GRAPH_PATH), normalized)
    _write_placements(DEFAULT_PLACEMENTS_PATH, next_placements)

    warnings: List[str] = []
    if invalidated_paths:
        warnings.append("Removed object placements were dropped and may need resampling.")

    return {
        "status": "ok",
        "scene_graph": normalized,
        "placements": next_placements,
        "invalidated_placements": invalidated_paths,
        "warnings": warnings,
    }


def apply_instruction(
    instruction: str,
    *,
    class_names_raw: str = "",
    image_bytes: bytes | None = None,
) -> Dict[str, Any]:
    text = (instruction or "").strip()
    if image_bytes is not None:
        _save_latest_input_image(image_bytes)

    current_graph = _load_scene_graph()
    current_placements = _load_placements()
    graph_image_b64 = _encode_image_bytes(image_bytes) if image_bytes is not None else None
    placement_image_b64 = graph_image_b64 or _current_render_image_b64()

    if current_graph is None:
        next_graph = _generate_scene_graph(text, class_names_raw, image_bytes)
        next_placements: Dict[str, List[float]] = {}
        write_json_file(Path(SCENE_GRAPH_PATH), next_graph)
        _write_placements(DEFAULT_PLACEMENTS_PATH, next_placements)
        route = {
            "mode": "graph",
            "confidence": 1.0,
            "signals": ["bootstrap"],
            "objects": [],
            "reason": "No current scene graph existed; generated a new scene graph.",
            "needs_resample": False,
            "requires_existing_placements": False,
            "apply_placement_editor": False,
        }
        return {
            "status": "ok",
            "route": route,
            "llm_route": None,
            "scene_graph": next_graph,
            "placements": next_placements,
            "invalidated_placements": [],
            "updated_paths": [],
            "warnings": [],
        }

    if not text and image_bytes is not None:
        raise ValueError("A text instruction is required when editing an existing scene.")
    if not text:
        raise ValueError("instruction is required")

    llm_route = route_scene_instruction(text, current_graph, current_placements)
    route = validate_route_decision(text, current_graph, current_placements, llm_route)

    warnings: List[str] = []
    invalidated_paths: List[str] = []
    updated_paths: List[str] = []

    if route["mode"] == "reset":
        next_graph = _generate_scene_graph(text, class_names_raw, image_bytes) if (text or image_bytes) else _empty_scene_graph()
        next_placements = {}
        warnings.append("Reset cleared existing placements.")
    elif route["mode"] == "graph":
        next_graph = edit_scene_graph_with_instruction(current_graph, text, image_b64=graph_image_b64)
        next_placements, invalidated_paths = reconcile_placements_after_graph_edit(
            current_graph,
            next_graph,
            current_placements,
            route,
        )
        if invalidated_paths:
            warnings.append("Changed graph paths were removed from placements and will need resampling.")
    elif route["mode"] == "placement":
        if route["requires_existing_placements"]:
            raise ValueError("Placement edits require existing sampled placements. Run Edit Scene or Resample once first.")
        next_graph = current_graph
        next_placements, updated_paths = _merge_placement_updates(
            current_graph,
            current_placements,
            edit_placements_with_instruction(
                current_graph,
                current_placements,
                text,
                image_b64=placement_image_b64,
            ),
        )
    else:
        next_graph = edit_scene_graph_with_instruction(current_graph, text, image_b64=graph_image_b64)
        next_placements, invalidated_paths = reconcile_placements_after_graph_edit(
            current_graph,
            next_graph,
            current_placements,
            route,
        )
        if route["apply_placement_editor"] and next_placements:
            next_placements, updated_paths = _merge_placement_updates(
                next_graph,
                next_placements,
                edit_placements_with_instruction(
                    next_graph,
                    next_placements,
                    text,
                    image_b64=placement_image_b64,
                ),
            )
        if route["needs_resample"]:
            warnings.append("Semantic relation changes may require Edit Scene or Resample to refresh geometry.")

    next_placements = prune_placements_to_scene_graph(next_graph, next_placements)
    write_json_file(Path(SCENE_GRAPH_PATH), next_graph)
    _write_placements(DEFAULT_PLACEMENTS_PATH, next_placements)

    return {
        "status": "ok",
        "route": route,
        "llm_route": llm_route,
        "scene_graph": next_graph,
        "placements": next_placements,
        "invalidated_placements": invalidated_paths,
        "updated_paths": updated_paths,
        "warnings": warnings,
    }
