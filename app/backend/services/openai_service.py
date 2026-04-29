import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in local test envs
    OpenAI = None  # type: ignore[assignment]

from ..config import DEFAULT_MODEL
from .instruction_router import build_router_rulebook


SCENE_GRAPH_MODEL = os.getenv("SCENE_GRAPH_MODEL", "gpt-5.4-mini")
SCENE_GRAPH_ROUTER_MODEL = os.getenv("SCENE_GRAPH_ROUTER_MODEL", SCENE_GRAPH_MODEL)
SCENE_GRAPH_EDITOR_MODEL = os.getenv("SCENE_GRAPH_EDITOR_MODEL", SCENE_GRAPH_MODEL)
PLACEMENT_EDITOR_MODEL = os.getenv("PLACEMENT_EDITOR_MODEL", DEFAULT_MODEL)
client: Optional[Any] = None
ALLOWED_OBJECT_SOURCES = {"real2sim", "retrieval"}

SYSTEM_PROMPT = r"""
You are an expert in 3D Scene Graph Construction.

Your task is to convert a natural language scene description into a structured Scene Graph in strict JSON format.

--------------------------------
SCENE-LEVEL INFORMATION
--------------------------------
Extract global room information under "scene":
- room_type
- dimensions (length, width, height, unit)
- materials (floor, walls)

If the user explicitly specifies the room type, use it directly.
If the user does not explicitly specify room type, infer a plausible room_type from the mentioned objects,
and choose suitable dimensions and floor/wall materials that are typical for that inferred room type.
Room dimensions may be adjusted based on the scene object list (object count/size/layout complexity),
but must remain realistic for the selected room type.

Room, walls, floor, ceiling are NOT objects.
They must not appear in the object list.

--------------------------------
OBJECT RULES
--------------------------------
1. Extract ONLY explicitly mentioned physical objects.
2. Do NOT invent objects.
3. Exclude room, floor, ceiling, walls.
4. Assign unique integer IDs starting from 0.
5. Return objects in an `objects` array.
6. Each object must include:
   - `path`: "/World/<ClassName>_<ID>"
   - `id`: integer ID
   - `class`: lowercase class name
   - `caption`: short caption
   - `source`: one of `real2sim` or `retrieval`
7. Class names must be lowercase.
8. Caption max 6 words.
9. Choose `source` conservatively:
   - use `real2sim` for objects that should come from the current observed real scene / uploaded image
   - use `retrieval` for objects that should come from the asset library or are newly imagined from text
10. You may infer implicit object-object pairing relations from common priors when strongly plausible.
   Example: chair and table are typically "face to" and "adjacent" unless explicitly contradicted.
11. One physical object instance must map to exactly one node.
    - Do NOT create duplicate nodes for the same visible instance because of class ambiguity or synonyms.
    - If an object could plausibly be described by multiple classes (for example `jar` vs `glass`, `cup` vs `mug`), choose exactly one class.
    - Put secondary details such as lid/material/style into `caption`, not into an extra object node.

--------------------------------
RELATION TYPES
--------------------------------

OBJ-OBJ RELATIONS

Position:
- left
- right
- in front of
- behind

Orientation:
- face to
- face same as

Vertical Support:
- supported by
- supports

Alignment:
- center aligned

Proximity:
- adjacent

--------------------------------
OBJ-WALL RELATIONS

- against wall
- in corner

Definitions:
- "against wall" means object directly touches a wall surface.
- "in corner" means object is positioned at the intersection of two walls.

--------------------------------
RELATION CONSTRAINTS
--------------------------------

1. Horizontal relations must be bidirectional:
   If A left B -> B right A
   If A in front of B -> B behind A

2. Vertical support must appear as a pair:
   A supported by B
   B supports A

3. Orientation relations are directional and do not require reverse edges.

4. No:
   - Duplicate edges
   - Self-relations
   - Contradictory relations
   - Physically impossible relations
5. Implicit relations from priors must remain conservative and consistent with explicit text.
6. Edge endpoints must use object prim-path strings exactly.
   - `source` and `target` in `edges.obj-obj` MUST be values from `objects[*].path`.
   - `source` in `edges.obj-wall` MUST be a value from `objects[*].path`.
   - NEVER use numeric ids (e.g. "1", 1) as edge endpoints.

--------------------------------
STRICT OUTPUT FORMAT
--------------------------------

Return ONLY valid JSON.
No explanations.
No markdown.
No extra text.
""".strip()

SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {
            "type": "object",
            "properties": {
                "room_type": {"type": ["string", "null"]},
                "dimensions": {
                    "type": "object",
                    "properties": {
                        "length": {"type": ["number", "null"]},
                        "width": {"type": ["number", "null"]},
                        "height": {"type": ["number", "null"]},
                        "unit": {"type": ["string", "null"]},
                    },
                    "required": ["length", "width", "height", "unit"],
                    "additionalProperties": False,
                },
                "materials": {
                    "type": "object",
                    "properties": {
                        "floor": {"type": ["string", "null"]},
                        "walls": {"type": ["string", "null"]},
                    },
                    "required": ["floor", "walls"],
                    "additionalProperties": False,
                },
            },
            "required": ["room_type", "dimensions", "materials"],
            "additionalProperties": False,
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "id": {"type": "integer", "minimum": 0},
                    "class": {"type": "string"},
                    "caption": {"type": "string"},
                    "source": {"type": "string", "enum": ["real2sim", "retrieval"]},
                },
                "required": ["path", "id", "class", "caption", "source"],
                "additionalProperties": False,
            },
        },
        "edges": {
            "type": "object",
            "properties": {
                "obj-obj": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "relation": {
                                "type": "string",
                                "enum": [
                                    "left",
                                    "right",
                                    "in front of",
                                    "behind",
                                    "face to",
                                    "face same as",
                                    "supported by",
                                    "supports",
                                    "center aligned",
                                    "adjacent",
                                ],
                            },
                        },
                        "required": ["source", "target", "relation"],
                        "additionalProperties": False,
                    },
                },
                "obj-wall": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "relation": {
                                "type": "string",
                                "enum": ["against wall", "in corner"],
                            },
                        },
                        "required": ["source", "relation"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["obj-obj", "obj-wall"],
            "additionalProperties": False,
        },
    },
    "required": ["scene", "objects", "edges"],
    "additionalProperties": False,
}

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["graph", "placement", "both", "reset"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["mode", "confidence", "reason"],
    "additionalProperties": False,
}

PLACEMENT_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": ["string", "null"]},
        "updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "yaw_deg": {"type": ["number", "null"]},
                },
                "required": ["path", "x", "y", "z", "yaw_deg"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["message", "updates"],
    "additionalProperties": False,
}

MASK_ASSIGNMENT_MODEL = os.getenv("REAL2SIM_MASK_ASSIGNMENT_MODEL", SCENE_GRAPH_MODEL)
MASK_ASSIGNMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene_path": {"type": "string"},
                    "mask_label": {"type": "integer", "minimum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": ["string", "null"]},
                },
                "required": ["scene_path", "mask_label", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
        "unmatched_scene_paths": {
            "type": "array",
            "items": {"type": "string"},
        },
        "unmatched_mask_labels": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        },
    },
    "required": ["assignments", "unmatched_scene_paths", "unmatched_mask_labels"],
    "additionalProperties": False,
}

SCENE_GRAPH_EDITOR_PROMPT = """
You edit an existing scene graph in response to an instruction.

Editing rules:
- Preserve unaffected scene metadata, objects, and relations.
- Only make the minimum graph changes required by the instruction.
- If the instruction adds an object, assign a unique id and path.
- If the instruction removes an object, remove all incident edges too.
- If the instruction changes semantic relations like left/right/on/against wall, update edges accordingly.
- Do not output placements or numeric coordinates.
- Return the complete updated scene graph in the provided schema.
""".strip()

PLACEMENT_EDITOR_PROMPT = """
You edit object placements for an existing scene.

Editing rules:
- Update only placements for existing objects already present in the scene graph.
- Use object paths exactly as provided.
- Return only the placement updates needed for the instruction.
- Also return a short `message` in the user's language that summarizes what you changed.
- If the instruction explicitly mentions one object, return an update only for that object.
- If the instruction moves object X relative to object Y, infer the new placement for X from the current placements and AABBs, and do not move Y.
- Do not reposition nearby or supporting objects unless the instruction explicitly asks for them too.
- Use the provided AABB geometry when deciding relative positions or support-surface positions.
- Use world coordinates, not camera/view coordinates, when interpreting directions.
- Direction convention: +y = right, -y = left, +x = front, -x = back.
- If yaw is not changed, keep yaw_deg as null.
- Do not add or remove scene-graph objects or relations.
""".strip()

MASK_ASSIGNMENT_PROMPT = """
You match numbered segmentation masks in an image to existing real-scene objects from a scene graph.

Matching rules:
- Use the numbered mask overlay image and the original image together.
- Use each mask label at most once.
- Only assign masks that clearly correspond to a listed scene object.
- Treat each mask candidate's `prompt` as a strong prior, but use image evidence, caption details, color, material, support, and relative position to break ties.
- When multiple scene objects share the same class, use caption details and scene-graph relations to create the best one-to-one assignment.
- If exact identity is uncertain, still choose the most plausible one-to-one assignment and lower confidence.
- Never invent scene paths or mask labels that are not provided.
- Return JSON only.
""".strip()

def read_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def encode_image_b64(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime = path.suffix.lstrip(".").lower() or "png"
    return f"data:image/{mime};base64,{b64}"


def _build_class_hint(class_names_raw: str) -> str:
    if not class_names_raw:
        return ""
    try:
        names = json.loads(class_names_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"class_names must be valid JSON list: {exc}") from exc
    unique_names = sorted({n for n in names if n})
    if not unique_names:
        return ""
    return "You must only use the following classes (do not invent others): " + ", ".join(
        unique_names
    )


def _scene_graph_schema_format() -> Dict[str, Any]:
    return _json_schema_format("scene_graph", SCHEMA)


def _json_schema_format(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": schema,
        }
    }


def _get_openai_client() -> Any:
    global client
    if client is None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed in the current Python environment.")
        client = OpenAI()
    return client


def _parse_scene_graph_response(resp: Any) -> Dict[str, Any]:
    try:
        parsed = json.loads(resp.output_text)
    except Exception as exc:
        raise ValueError(f"Model did not return valid scene graph JSON: {exc}") from exc
    return _normalize_scene_graph_payload(parsed)


def _validate_normalized_scene_graph(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Scene graph payload must be a JSON object.")

    obj_map = payload.get("obj")
    if not isinstance(obj_map, dict):
        raise ValueError("Scene graph payload must contain 'obj' as a dict keyed by prim path.")

    edges = payload.get("edges")
    if not isinstance(edges, dict):
        raise ValueError("Scene graph payload must contain 'edges' as an object.")

    obj_obj_edges = edges.get("obj-obj")
    obj_wall_edges = edges.get("obj-wall")
    if not isinstance(obj_obj_edges, list) or not isinstance(obj_wall_edges, list):
        raise ValueError("Scene graph edges must include 'obj-obj' and 'obj-wall' arrays.")

    obj_paths = set(obj_map.keys())
    for path, meta in obj_map.items():
        if not isinstance(path, str) or not path:
            raise ValueError("Object paths must be non-empty strings.")
        if not isinstance(meta, dict):
            raise ValueError("Each object entry must be a JSON object.")
        source = meta.get("source")
        if source is None:
            raise ValueError(
                f"Object '{path}' is missing required source. Allowed: {sorted(ALLOWED_OBJECT_SOURCES)}"
            )
        if source not in ALLOWED_OBJECT_SOURCES:
            raise ValueError(
                f"Object '{path}' has invalid source '{source}'. Allowed: {sorted(ALLOWED_OBJECT_SOURCES)}"
            )

    for edge in obj_obj_edges:
        if not isinstance(edge, dict):
            raise ValueError("Each edge in 'obj-obj' must be a JSON object.")
        source = edge.get("source")
        target = edge.get("target")
        if source not in obj_paths or target not in obj_paths:
            raise ValueError("Each 'obj-obj' edge must reference object paths present in 'obj'.")

    for edge in obj_wall_edges:
        if not isinstance(edge, dict):
            raise ValueError("Each edge in 'obj-wall' must be a JSON object.")
        source = edge.get("source")
        if source not in obj_paths:
            raise ValueError("Each 'obj-wall' edge must reference an object path present in 'obj'.")

    return payload


def _normalize_scene_graph_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Scene graph payload must be a JSON object.")

    if "objects" in payload:
        objects = payload.get("objects")
        if not isinstance(objects, list):
            raise ValueError("Scene graph payload field 'objects' must be an array.")

        obj_map: Dict[str, Dict[str, Any]] = {}
        for item in objects:
            if not isinstance(item, dict):
                raise ValueError("Each scene graph object must be a JSON object.")
            path = item.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError("Each scene graph object must include a non-empty 'path'.")
            if path in obj_map:
                raise ValueError(f"Duplicate object path in scene graph payload: {path}")
            obj_map[path] = {
                "id": item.get("id"),
                "class": item.get("class"),
                "caption": item.get("caption"),
                "source": item.get("source"),
            }

        normalized = {
            "scene": payload.get("scene"),
            "obj": obj_map,
            "edges": payload.get("edges"),
        }
        return _validate_normalized_scene_graph(normalized)

    return _validate_normalized_scene_graph(payload)


def normalize_scene_graph_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_scene_graph_payload(payload)


def _parse_output_json(resp: Any, label: str) -> Dict[str, Any]:
    try:
        return json.loads(resp.output_text)
    except Exception as exc:
        raise ValueError(f"Model did not return valid {label} JSON: {exc}") from exc


def _scene_graph_prompt_payload(scene_graph: Dict[str, Any]) -> Dict[str, Any]:
    objects = []
    for path, meta in sorted((scene_graph.get("obj") or {}).items()):
        if not isinstance(meta, dict):
            continue
        objects.append(
            {
                "path": path,
                "id": meta.get("id"),
                "class": meta.get("class") or meta.get("class_name"),
                "caption": meta.get("caption"),
                "source": meta.get("source"),
            }
        )
    return {
        "scene": scene_graph.get("scene"),
        "objects": objects,
        "edges": scene_graph.get("edges", {"obj-obj": [], "obj-wall": []}),
    }


def _placements_prompt_payload(placements: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for path, payload in sorted((placements or {}).items()):
        if isinstance(payload, (list, tuple)) and len(payload) >= 3:
            rows.append(
                {
                    "path": path,
                    "x": payload[0],
                    "y": payload[1],
                    "z": payload[2],
                    "yaw_deg": payload[3] if len(payload) >= 4 else None,
                }
            )
            continue
        if isinstance(payload, dict) and all(key in payload for key in ("x", "y", "z")):
            row = {
                "path": path,
                "x": payload["x"],
                "y": payload["y"],
                "z": payload["z"],
                "yaw_deg": payload.get("yaw"),
            }
            if isinstance(payload.get("aabb"), dict):
                row["aabb"] = payload["aabb"]
            rows.append(row)
    return {"placements": rows}


def route_scene_instruction(
    instruction: str,
    scene_graph: Dict[str, Any],
    placements: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = "\n\n".join(
        [
            build_router_rulebook(),
            "Current scene graph objects:",
            json.dumps(_scene_graph_prompt_payload(scene_graph).get("objects", []), ensure_ascii=False, indent=2),
            f"Placements available: {bool(placements)}",
            f'User instruction: "{instruction.strip()}"',
            "Return JSON only.",
        ]
    )
    response = _get_openai_client().responses.create(
        model=SCENE_GRAPH_ROUTER_MODEL,
        input=prompt,
        text=_json_schema_format("instruction_route", ROUTER_SCHEMA),
    )
    return _parse_output_json(response, "routing")


def assign_real2sim_masks_with_images(
    scene_context: Dict[str, Any],
    mask_candidates: list[Dict[str, Any]],
    *,
    original_image_b64: str,
    overlay_image_b64: str,
    model: str | None = None,
) -> Dict[str, Any]:
    prompt = "\n\n".join(
        [
            "Match the numbered masks to the scene graph objects.",
            "Scene graph objects and relations:",
            json.dumps(scene_context, ensure_ascii=False, indent=2),
            "Mask candidates:",
            json.dumps(mask_candidates, ensure_ascii=False, indent=2),
            (
                "The numbered overlay uses `mask_label` values. "
                "The `output_name` field is an internal ID and may help bookkeeping, "
                "but the visible number on the image is `mask_label`."
            ),
        ]
    )
    response = _get_openai_client().responses.create(
        model=model or MASK_ASSIGNMENT_MODEL,
        instructions=MASK_ASSIGNMENT_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": original_image_b64},
                    {"type": "input_image", "image_url": overlay_image_b64},
                ],
            }
        ],
        text=_json_schema_format("mask_assignments", MASK_ASSIGNMENT_SCHEMA),
    )
    return _parse_output_json(response, "mask assignment")


def edit_scene_graph_with_instruction(
    scene_graph: Dict[str, Any],
    instruction: str,
    *,
    image_b64: str | None = None,
) -> Dict[str, Any]:
    content = [
        {
            "type": "input_text",
            "text": (
                f"Instruction:\n{instruction.strip()}\n\n"
                "Current scene graph:\n"
                f"{json.dumps(_scene_graph_prompt_payload(scene_graph), ensure_ascii=False, indent=2)}"
            ),
        }
    ]
    if image_b64:
        content.append({"type": "input_image", "image_url": image_b64})
    response = _get_openai_client().responses.create(
        model=SCENE_GRAPH_EDITOR_MODEL,
        instructions=f"{SYSTEM_PROMPT}\n\n{SCENE_GRAPH_EDITOR_PROMPT}",
        input=[{"role": "user", "content": content}],
        text=_scene_graph_schema_format(),
    )
    return _parse_scene_graph_response(response)


def edit_placements_with_instruction(
    scene_graph: Dict[str, Any],
    placements: Dict[str, Any],
    instruction: str,
    *,
    image_b64: str | None = None,
    placement_context: Dict[str, Any] | None = None,
    editing_hint: str | None = None,
) -> Dict[str, Any]:
    hint_block = f"\n\nEditing hint:\n{editing_hint.strip()}" if editing_hint else ""
    content = [
        {
            "type": "input_text",
            "text": (
                f"Instruction:\n{instruction.strip()}\n\n"
                "Scene graph objects:\n"
                f"{json.dumps(_scene_graph_prompt_payload(scene_graph).get('objects', []), ensure_ascii=False, indent=2)}\n\n"
                "Current placements:\n"
                f"{json.dumps(_placements_prompt_payload(placement_context or placements), ensure_ascii=False, indent=2)}"
                f"{hint_block}"
            ),
        }
    ]
    if image_b64:
        content.append({"type": "input_image", "image_url": image_b64})
    response = _get_openai_client().responses.create(
        model=PLACEMENT_EDITOR_MODEL,
        instructions=PLACEMENT_EDITOR_PROMPT,
        input=[{"role": "user", "content": content}],
        text=_json_schema_format("placement_updates", PLACEMENT_UPDATE_SCHEMA),
    )
    return _parse_output_json(response, "placement update")


def parse_scene_graph_from_text(text: str, class_names_raw: str = "") -> Dict[str, Any]:
    class_hint = _build_class_hint(class_names_raw)
    user_prompt = text if not class_hint else f"{text}\n\n{class_hint}"
    response = _get_openai_client().responses.create(
        model=SCENE_GRAPH_MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        text=_scene_graph_schema_format(),
    )
    return _parse_scene_graph_response(response)


def _build_image_scene_graph_prompt(text: str = "", class_names_raw: str = "") -> str:
    class_hint = _build_class_hint(class_names_raw)
    prompt = (
        "Analyze this image and produce a scene graph that follows the provided schema."
        + (f" {class_hint}" if class_hint else "")
        + " Infer room type/dimensions/materials from visible context when unspecified."
    )
    text = text.strip()
    if text:
        prompt += (
            " The user also provided this text instruction: "
            f"{text} "
            "TEXT GOVERNS OBJECT MEMBERSHIP — IMAGE GROUNDS PROPERTIES. "
            "If the text enumerates which objects the user wants (e.g. 'a red bowl and a box on the table', "
            "'I want X, Y, Z'), the scene graph MUST contain ONLY those enumerated objects, plus the minimum "
            "supporting parent surface they are placed on if the text references it (e.g. 'on the table' -> "
            "include the table). Do NOT add any other object visible in the image (cups, drills, tools, "
            "clutter, decorations, etc.) just because it appears in the photo. The image is for grounding "
            "visual properties (color, material, geometry, relative position) of the requested objects only. "
            "If the text instead asks an open-ended question (e.g. 'describe this scene', 'reconstruct what "
            "you see', no specific object list), THEN you may include all clearly visible objects from the "
            "image. "
            "Source assignment for the included objects: use source=real2sim if the requested object is "
            "clearly present in the uploaded image; use source=retrieval if the user requested it in text "
            "but it is not visible in the image."
        )
    return prompt


def parse_scene_graph_from_image(
    image_bytes: bytes,
    class_names_raw: str = "",
    *,
    text: str = "",
) -> Dict[str, Any]:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt_content = [
        {
            "type": "input_text",
            "text": _build_image_scene_graph_prompt(text=text, class_names_raw=class_names_raw),
        },
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{image_b64}",
        },
    ]
    response = _get_openai_client().responses.create(
        model=SCENE_GRAPH_MODEL,
        instructions=SYSTEM_PROMPT,
        input=[
            {"role": "user", "content": prompt_content},
        ],
        text=_scene_graph_schema_format(),
    )
    return _parse_scene_graph_response(response)


def call_gpt_json_editor_with_image(
    model: str, data: Dict[str, Any], instruction: str, image_b64: str
) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are given: (1) an image, (2) a JSON object with numeric parameters. "
                "Update the JSON to satisfy the instruction while preserving structure and keys. "
                "Use the y-axis to reason about left/right and the x-axis to reason about front/behind. "
                "Reply with JSON only, no commentary."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Instruction: {instruction}\nCurrent JSON:\n"
                        f"{json.dumps(data, ensure_ascii=False)}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_b64}},
            ],
        },
    ]

    resp = _get_openai_client().chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc}") from exc
