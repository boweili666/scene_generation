from __future__ import annotations

import base64
import json
import os
from pathlib import Path


CLASSIFICATION_PROMPT = """
You are an Object Primitive Annotation Agent in a robotics perception system.

Given three orthographic renderings (front, side, top) of one GLB object with world axis overlays, classify the object into exactly one category:
- axis_object
- handle_tool_object
- graspnet_object

Category definitions:
- axis_object: rotational symmetry around a central axis (bottle, pen, spray can).
- handle_tool_object: object with a distinctive interaction skeleton, including preferred human grasp poses and specific interaction points (teapot, hanger, hammer).
- graspnet_object: all remaining objects that are not axis_object or handle_tool_object, including container, planar, and irregular shapes; use graspnet-style grasp candidates.

Disambiguation rules:
- If the object has a clear task-oriented grasp region and a distinct interaction region, prefer handle_tool_object.
- Teapots, hangers, and hammers must be handle_tool_object.
- Use graspnet_object when the object is not axis_object or handle_tool_object.

Geometry conventions:
- world up = +Y
- world front = -Z
- front_view camera looks along -Z
- side_view camera looks along -X
- top_view camera looks along -Y

Return JSON only with this schema:
{
  "object_name": "string",
  "category": "axis_object|handle_tool_object|graspnet_object",
  "views": ["front_view", "side_view", "top_view"],
  "primitives": {
    "axis_object": {
      "object_axis": "X|Y|Z",
      "center_point": [0.0, 0.0, 0.0]
    },
    "handle_tool_object": {
      "handle_center": [0.0, 0.0, 0.0],
      "handle_axis": "X|Y|Z",
      "skeleton_view": "front_view|side_view|top_view",
      "skeleton_graph": "brief text description"
    },
    "graspnet_object": {
      "candidate_grasp_points": [[0.0, 0.0, 0.0]],
      "surface_normals": ["X|Y|Z"]
    }
  }
}

Rules:
- Step 1: classify into exactly one category.
- Step 2: set views exactly to ["front_view", "side_view", "top_view"].
- Step 3: in primitives, include exactly one key matching the predicted category.
- Do not include primitive keys from other categories.
- For handle_tool_object, skeleton_view is required and must be one of front_view, side_view, top_view.
- Choose skeleton_view where the grasp region and interaction region are both most visible with minimal occlusion.
- Prefer geometry-based reasoning from silhouettes and axis cues.
- Do not output any extra text outside JSON.
""".strip()


KP_PICKER_PROMPT = """
You are a robotic grasp planner for handle_tool_object.

Input includes:
- One orthographic image with grasp candidate IDs overlaid.
- Candidate list with id, center_xy, gripper_axis_xy, width_px, score.
- Skeleton view name and skeleton graph text.

Task:
- Pick exactly one candidate id for a stable grasp on the primary grasp region.
- Prefer a candidate on the handle/body grasp zone, avoiding tips/extreme ends if possible.
- Prefer high score candidates if geometry is plausible.

Return JSON only:
{
  "selected_candidate_id": 0,
  "grasp_intent": "short string",
  "confidence": 0.0
}
""".strip()


AXIS_PICKER_PROMPT = """
You are a robotic grasp planner for axis_object.

Input includes:
- One orthographic image with indexed points sampled along the object's main axis.
- A list of axis sample points with point_id and world xyz.

Task:
- Pick exactly one point_id where humans most commonly grasp this axis object.
- Prefer stable mid-body grasp regions, avoiding very top and very bottom unless clearly better.

Return JSON only:
{
  "selected_point_id": 0,
  "grasp_intent": "short string",
  "confidence": 0.0
}
""".strip()


HANDLE_ORIENTATION_PROMPT = """
You are selecting a grasp closing direction for a handle_tool_object at a chosen grasp keypoint.

Choose exactly one direction label from:
- X
- Y
- Z
- X+Y
- X-Y
- X+Z
- X-Z
- Y+Z
- Y-Z

Interpretation:
- X/Y/Z means world-axis-parallel closing direction.
- A+B or A-B means a 45-degree direction between two world axes.

Strict grasp-direction rules:
- The jaw closing direction must go ACROSS the local handle cross-section, not ALONG the handle length.
- So the closing direction should be approximately perpendicular to local handle tangent near the selected keypoint.
- Reject directions that are mostly parallel to the local handle tangent.
- If uncertain, prefer the label that is most perpendicular to the local handle tangent and also not parallel to the approach axis.

Return JSON only:
{
  "direction_label": "X|Y|Z|X+Y|X-Y|X+Z|X-Z|Y+Z|Y-Z",
  "reason": "short string",
  "confidence": 0.0
}
""".strip()


def image_to_data_url(image_path: Path) -> str:
    mime = "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def call_openai_json(model: str, system_prompt: str, user_content: list[dict]) -> dict:
    from openai import BadRequestError, OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    payload = {
        "model": model,
        "temperature": 0,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ],
    }

    try:
        resp = client.responses.create(**payload)
    except BadRequestError as exc:
        msg = str(exc)
        if "Unsupported parameter" not in msg or "temperature" not in msg:
            raise
        payload.pop("temperature", None)
        resp = client.responses.create(**payload)

    text = resp.output_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model output is not valid JSON. Raw output:\n{text}") from exc


def classify_object(model: str, object_name: str, rendered: dict) -> dict:
    user_text = (
        f"Object name: {object_name}. "
        "Classify into one category and return JSON only. "
        "Use axis overlays to determine the axis-oriented grasp/specific view fields."
    )
    user_content = [
        {"type": "input_text", "text": user_text},
        {"type": "input_text", "text": "front_view (with world axis overlay):"},
        {"type": "input_image", "image_url": image_to_data_url(Path(rendered["front_view"]["axes_image_path"]))},
        {"type": "input_text", "text": "side_view (with world axis overlay):"},
        {"type": "input_image", "image_url": image_to_data_url(Path(rendered["side_view"]["axes_image_path"]))},
        {"type": "input_text", "text": "top_view (with world axis overlay):"},
        {"type": "input_image", "image_url": image_to_data_url(Path(rendered["top_view"]["axes_image_path"]))},
    ]
    return call_openai_json(model, CLASSIFICATION_PROMPT, user_content)


def extract_skeleton_view(classification: dict) -> str:
    primitives = classification.get("primitives", {})
    handle_primitive = primitives.get("handle_tool_object", {}) if isinstance(primitives, dict) else {}
    view_name = handle_primitive.get("skeleton_view")
    if view_name in {"front_view", "side_view", "top_view"}:
        return view_name
    return "front_view"


def pick_handle_candidate(
    model: str,
    object_name: str,
    skeleton_view: str,
    skeleton_graph: str,
    candidates: list[dict],
    candidate_overlay_path: Path,
) -> dict:
    candidates_compact = [
        {
            "id": c["id"],
            "center_xy": c["center_xy"],
            "gripper_axis_xy": c["gripper_axis_xy"],
            "width_px": c["width_px"],
            "score": c["score"],
            "branch_id": c["branch_id"],
        }
        for c in candidates
    ]
    user_content = [
        {"type": "input_text", "text": f"Object: {object_name}. Skeleton view: {skeleton_view}. Pick exactly one best candidate ID for grasp."},
        {"type": "input_text", "text": f"Skeleton graph: {skeleton_graph}"},
        {"type": "input_text", "text": "Candidates JSON:"},
        {"type": "input_text", "text": json.dumps(candidates_compact, ensure_ascii=False)},
        {"type": "input_text", "text": "Candidate overlay image with IDs:"},
        {"type": "input_image", "image_url": image_to_data_url(candidate_overlay_path)},
    ]
    return call_openai_json(model, KP_PICKER_PROMPT, user_content)


def pick_handle_orientation(
    model: str,
    object_name: str,
    skeleton_view: str,
    selected_xy: tuple[int, int],
    axes_image_path: Path,
    candidate_overlay_path: Path,
) -> dict:
    user_content = [
        {
            "type": "input_text",
            "text": (
                f"Object: {object_name}. View: {skeleton_view}. "
                f"Selected keypoint pixel: ({int(selected_xy[0])}, {int(selected_xy[1])}). "
                "Choose best grasp closing direction label."
            ),
        },
        {"type": "input_text", "text": "View image with world-axis overlay:"},
        {"type": "input_image", "image_url": image_to_data_url(axes_image_path)},
        {"type": "input_text", "text": "Candidate overlay image (selected ID already chosen):"},
        {"type": "input_image", "image_url": image_to_data_url(candidate_overlay_path)},
    ]
    return call_openai_json(model, HANDLE_ORIENTATION_PROMPT, user_content)


def pick_axis_sample(
    model: str,
    object_name: str,
    axis_label: str,
    projected_samples: list[dict],
    sample_overlay_path: Path,
) -> dict:
    user_content = [
        {"type": "input_text", "text": f"Object: {object_name}. Axis label: {axis_label}. Pick one point_id."},
        {"type": "input_text", "text": "Axis sampled points JSON:"},
        {"type": "input_text", "text": json.dumps(projected_samples, ensure_ascii=False)},
        {"type": "input_text", "text": "Axis sampled points image with IDs:"},
        {"type": "input_image", "image_url": image_to_data_url(sample_overlay_path)},
    ]
    return call_openai_json(model, AXIS_PICKER_PROMPT, user_content)
