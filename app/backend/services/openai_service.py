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


SCENE_GRAPH_MODEL = os.getenv("SCENE_GRAPH_MODEL", "gpt-5")
client: Optional[Any] = None

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
7. Class names must be lowercase.
8. Caption max 6 words.
9. You may infer implicit object-object pairing relations from common priors when strongly plausible.
   Example: chair and table are typically "face to" and "adjacent" unless explicitly contradicted.

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
                },
                "required": ["path", "id", "class", "caption"],
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
    return {
        "format": {
            "type": "json_schema",
            "name": "scene_graph",
            "strict": True,
            "schema": SCHEMA,
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
            }

        normalized = {
            "scene": payload.get("scene"),
            "obj": obj_map,
            "edges": payload.get("edges"),
        }
        return _validate_normalized_scene_graph(normalized)

    return _validate_normalized_scene_graph(payload)


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


def parse_scene_graph_from_image(image_bytes: bytes, class_names_raw: str = "") -> Dict[str, Any]:
    class_hint = _build_class_hint(class_names_raw)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt_content = [
        {
            "type": "input_text",
            "text": (
                "Analyze this image and produce a scene graph that follows "
                "the provided schema."
            )
            + (f" {class_hint}" if class_hint else "")
            + " Infer room type/dimensions/materials from visible context when unspecified.",
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
