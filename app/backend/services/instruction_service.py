from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..config import (
    DEFAULT_PLACEMENTS_PATH,
    DEFAULT_RENDER_PATH,
    LATEST_INPUT_IMAGE,
    SCENE_GRAPH_PATH,
)
from .instruction_router import (
    normalize_instruction_text,
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
TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9_./-]+")


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


def _load_placements_payload_raw(path: str | Path = DEFAULT_PLACEMENTS_PATH) -> Dict[str, Any]:
    placements_path = Path(path)
    if not placements_path.exists():
        return {}
    try:
        payload = read_json_file(placements_path)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_placements(path: str | Path = DEFAULT_PLACEMENTS_PATH) -> Dict[str, List[float]]:
    return _normalize_placements_payload(_load_placements_payload_raw(path))


def _translate_aabb(aabb: Dict[str, Any], dx: float, dy: float, dz: float) -> Dict[str, Any] | None:
    if not isinstance(aabb, dict):
        return None

    def _translate_vec3(value: Any) -> List[float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        return [
            float(value[0]) + dx,
            float(value[1]) + dy,
            float(value[2]) + dz,
        ]

    translated = {
        "min": _translate_vec3(aabb.get("min")),
        "max": _translate_vec3(aabb.get("max")),
        "center": _translate_vec3(aabb.get("center")),
    }
    size = aabb.get("size")
    if isinstance(size, (list, tuple)) and len(size) == 3:
        translated["size"] = [float(size[0]), float(size[1]), float(size[2])]
    else:
        translated["size"] = None

    if not translated["min"] or not translated["max"] or not translated["center"] or not translated["size"]:
        return None
    return translated


def _write_placements(path: str | Path, placements: Dict[str, List[float]]) -> None:
    target_path = Path(path)
    previous_payload = _load_placements_payload_raw(target_path)
    serializable: Dict[str, Dict[str, Any]] = {}
    for key, value in sorted(placements.items()):
        entry: Dict[str, Any] = {
            "x": float(value[0]),
            "y": float(value[1]),
            "z": float(value[2]),
        }
        if len(value) >= 4:
            entry["yaw"] = float(value[3])

        previous = previous_payload.get(key)
        if isinstance(previous, dict) and all(axis in previous for axis in ("x", "y", "z")):
            translated_aabb = _translate_aabb(
                previous.get("aabb"),
                float(value[0]) - float(previous["x"]),
                float(value[1]) - float(previous["y"]),
                float(value[2]) - float(previous["z"]),
            )
            if translated_aabb is not None:
                entry["aabb"] = translated_aabb
            if "yaw" not in entry and previous.get("yaw") is not None:
                entry["yaw"] = float(previous["yaw"])

        serializable[key] = entry

    write_json_file(target_path, serializable)


def _save_latest_input_image(image_bytes: bytes) -> None:
    LATEST_INPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    LATEST_INPUT_IMAGE.write_bytes(image_bytes)


def _current_render_image_b64() -> str | None:
    render_path = Path(DEFAULT_RENDER_PATH)
    if not render_path.exists():
        return None
    raw = render_path.read_bytes()
    return _encode_image_bytes(raw, "image/png")


def _object_reference_tokens(path: str, meta: Dict[str, Any]) -> List[str]:
    tokens = {path.lower(), path.split("/")[-1].lower()}
    class_name = str(meta.get("class") or meta.get("class_name") or "").strip().lower()
    caption = str(meta.get("caption") or "").strip().lower()
    if class_name:
        tokens.add(class_name)
    if caption:
        tokens.add(caption)
        for token in TOKEN_SPLIT_RE.split(caption):
            if len(token) >= 2:
                tokens.add(token)
    return sorted(token for token in tokens if token)


def _ordered_mentioned_paths(text: str, scene_graph: Dict[str, Any], paths: List[str]) -> List[str]:
    normalized = normalize_instruction_text(text)
    ordered: List[Tuple[int, str]] = []
    for path in paths:
        meta = (scene_graph.get("obj") or {}).get(path) or {}
        indices = [
            normalized.find(token)
            for token in _object_reference_tokens(path, meta if isinstance(meta, dict) else {})
            if normalized.find(token) >= 0
        ]
        if indices:
            ordered.append((min(indices), path))
    ordered.sort()
    return [path for _, path in ordered]


def _placement_editor_hint(instruction: str, scene_graph: Dict[str, Any], route: Dict[str, Any]) -> str | None:
    ordered_paths = _ordered_mentioned_paths(
        instruction,
        scene_graph,
        [str(path) for path in (route.get("objects") or []) if str(path)],
    )
    signals = set(route.get("signals") or [])
    if not ordered_paths or not signals & {"move", "rotate"}:
        return None
    if len(ordered_paths) >= 2:
        return (
            f"Movable object: {ordered_paths[0]}\n"
            f"Reference object: {ordered_paths[1]}\n"
            "Update only the movable object unless the instruction explicitly asks to move both."
        )
    return f"Movable object: {ordered_paths[0]}\nUpdate only that object."


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
    *,
    allowed_paths: set[str] | None = None,
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
        if allowed_paths is not None and path not in allowed_paths:
            continue
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


def _allowed_placement_update_paths(
    route: Dict[str, Any],
    instruction: str,
    scene_graph: Dict[str, Any],
) -> set[str] | None:
    mentioned_paths = _ordered_mentioned_paths(
        instruction,
        scene_graph,
        [str(path) for path in (route.get("objects") or []) if str(path)],
    )
    if not mentioned_paths:
        return None
    signals = set(route.get("signals") or [])
    if len(mentioned_paths) == 2 and signals & {"move", "rotate"} and not signals & {"create", "delete", "replace"}:
        return {mentioned_paths[0]}
    if signals & {"relation", "support", "create", "delete", "replace"}:
        return None
    return set(mentioned_paths)


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


def _fallback_assistant_message(
    route: Dict[str, Any],
    updated_paths: List[str],
    invalidated_paths: List[str],
    warnings: List[str],
) -> str:
    mode = str(route.get("mode") or "")
    if mode == "reset":
        return "Scene reset completed."
    if mode == "placement":
        if updated_paths:
            label = ", ".join(updated_paths[:3])
            suffix = " ..." if len(updated_paths) > 3 else ""
            return f"Updated placement for {len(updated_paths)} object(s): {label}{suffix}."
        return "Placement edit completed."
    if mode == "graph":
        if invalidated_paths:
            return "Scene graph updated. Some placements were invalidated and may need resampling."
        return "Scene graph updated."
    if warnings:
        return warnings[0]
    return str(route.get("reason") or "Instruction applied.")


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
    current_placements_raw = _load_placements_payload_raw()
    graph_image_b64 = _encode_image_bytes(image_bytes) if image_bytes is not None else None
    # Placement edits are text-only on purpose; images bias the model toward screen-space directions.
    placement_image_b64 = None

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
            "assistant_message": "Created a new scene graph from the current input.",
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
    placement_editor_hint = _placement_editor_hint(text, current_graph, route)

    warnings: List[str] = []
    invalidated_paths: List[str] = []
    updated_paths: List[str] = []
    assistant_message: str | None = None

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
        allowed_paths = _allowed_placement_update_paths(route, text, current_graph)
        placement_updates = edit_placements_with_instruction(
            current_graph,
            current_placements,
            text,
            image_b64=placement_image_b64,
            placement_context=current_placements_raw,
            editing_hint=placement_editor_hint,
        )
        assistant_message = str(placement_updates.get("message") or "").strip() or None
        next_placements, updated_paths = _merge_placement_updates(
            current_graph,
            current_placements,
            placement_updates,
            allowed_paths=allowed_paths,
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
            allowed_paths = _allowed_placement_update_paths(route, text, next_graph)
            placement_updates = edit_placements_with_instruction(
                next_graph,
                next_placements,
                text,
                image_b64=placement_image_b64,
                placement_context=current_placements_raw,
                editing_hint=placement_editor_hint,
            )
            assistant_message = str(placement_updates.get("message") or "").strip() or None
            next_placements, updated_paths = _merge_placement_updates(
                next_graph,
                next_placements,
                placement_updates,
                allowed_paths=allowed_paths,
            )
        if route["needs_resample"]:
            warnings.append("Semantic relation changes may require Edit Scene or Resample to refresh geometry.")

    next_placements = prune_placements_to_scene_graph(next_graph, next_placements)
    write_json_file(Path(SCENE_GRAPH_PATH), next_graph)
    _write_placements(DEFAULT_PLACEMENTS_PATH, next_placements)
    assistant_message = assistant_message or _fallback_assistant_message(
        route,
        updated_paths,
        invalidated_paths,
        warnings,
    )

    return {
        "status": "ok",
        "route": route,
        "llm_route": llm_route,
        "assistant_message": assistant_message,
        "scene_graph": next_graph,
        "placements": next_placements,
        "invalidated_placements": invalidated_paths,
        "updated_paths": updated_paths,
        "warnings": warnings,
    }
