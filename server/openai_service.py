import base64
import json
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

try:
    from .config import DEFAULT_MODEL
    from .prompts import SYSTEM_PROMPT, VISION_SYSTEM_PROMPT
    from .schemas import SceneGraph
except ImportError:
    from config import DEFAULT_MODEL
    from prompts import SYSTEM_PROMPT, VISION_SYSTEM_PROMPT
    from schemas import SceneGraph


client = OpenAI()


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


def parse_scene_graph_from_text(text: str, class_names_raw: str = "") -> Dict[str, Any]:
    class_hint = _build_class_hint(class_names_raw)
    user_prompt = text if not class_hint else f"{text}\n{class_hint}"
    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        text_format=SceneGraph,
    )
    return response.output_parsed.model_dump()


def parse_scene_graph_from_image(image_bytes: bytes, class_names_raw: str = "") -> Dict[str, Any]:
    class_hint = _build_class_hint(class_names_raw)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt_content = [
        {
            "type": "input_text",
            "text": (
                "Analyze this image and produce a scene graph that follows"
                " the provided schema."
            )
            + (f" {class_hint}" if class_hint else ""),
        },
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{image_b64}",
        },
    ]
    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_content},
        ],
        text_format=SceneGraph,
    )
    return response.output_parsed.model_dump()


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

    resp = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc}") from exc
