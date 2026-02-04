# server/server.py
from flask import Flask, send_from_directory, jsonify, request
import subprocess
import os
import json
import base64
from pathlib import Path
import base64

from flask import send_file
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from datetime import datetime

# ==================================================
# 路径配置
# ==================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB_DIR = os.path.join(BASE_DIR, "web")

ISAAC_PYTHON = os.path.expanduser(
    "~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh"
)
ISAAC_SCRIPT = os.path.expanduser(
    "~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/bowei/save_figure.py"
)
ASSET_CONVERTER_SCRIPT = os.path.expanduser(
    "~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/standalone_examples/api/omni.kit.asset_converter/convert.py"
)
SAM3_PYTHON = os.path.expanduser(
    "/home/lbw/miniconda3/envs/sam3/bin/python"
)
SAM3_MESH_GEN = os.path.expanduser(
    "/home/lbw/3dgen-project/sam3/test_mesh_gen.py"
)
SAM3_MESH_OUTPUT = os.path.expanduser(
    "/home/lbw/3dgen-project/sam3/outputs/meshes"
)
GENMESH_ROOT = os.path.expanduser(
    "/home/lbw/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/genmesh"
)

SCENE_GRAPH_PATH = os.path.expanduser(
    "~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/bowei/my_viewer/full_scene_graph.json"
)

# ==================================================
# Flask
# ==================================================
app = Flask(
    __name__,
    static_folder=os.path.join(WEB_DIR, "assets"),
    static_url_path="/assets",
)

# ==================================================
# OpenAI Client
# ==================================================
client = OpenAI()
LOG_PATH = os.path.join(BASE_DIR, "real2sim.log")

# ==================================================
# Scene Graph Schema（parse-safe）
# ==================================================
class SceneObject(BaseModel):
    path: str
    class_name: str
    id: int


class SceneEdge(BaseModel):
    source: str
    source_name: str
    target: str
    target_name: str
    relation: str


class SceneGraph(BaseModel):
    objects: List[SceneObject]
    edges: List[SceneEdge]


SYSTEM_PROMPT = """
You are a specialist in 3D Scene Reconstruction.
Your task is to transform natural language descriptions into a structured Scene Graph (JSON).

Rules:
- Extract objects (exclude floor/ground)
- Assign unique integer IDs starting from 0
- Use /World/[ClassName]_[ID] as object path
- Supported relations: supported by, supports, left, right, front, behind, near
- Relations must be logically consistent
- For support relations, put the supporting object first; the supported object must not be listed first
Vertical Support (supported by / supports):

    If Object A is resting on Object B's top surface, define the relation as A supported by B (and B supports A).

Horizontal Directions:

    Assign left, right, front, and behind based on the relative 3D positions of the objects from the camera's perspective.

    Consistency: If A is left of B, then B must be right of A.
json format:
{
  "obj": {
    "<USD path>": { "class": "<lowercase class name>", "id": <int>, "caption": "<short caption>" }
  },
  "edges": {
    "obj-obj": [
      { "source": "<path>", "source_name": "<class>", "target": "<path>", "target_name": "<class>", "relation": "<supported by|supports|left|right|front|behind|near>" }
    ],
  }
}
Return ONLY JSON, no explanations.
"""

# ==================================================
# Vision Prompt (Image → Scene Graph)
# ==================================================
VISION_SYSTEM_PROMPT = """
You are a specialist in 3D Scene Reconstruction. Convert the provided image
into a structured Scene Graph (JSON). Use only objects that are visually
present; do not invent objects. Floor/ground must not be an object.

Rules for relations:
- "supported by" / "supports" when one object rests on another.
- Positional: left, right, front, behind based on camera view.
- Proximity: near when objects are close but not touching.
- Ensure bidirectional consistency (if A left of B, then B right of A).

Output schema:
{
  "objects": [
    {"path": "/World/SM_<Class>_<ID>", "class_name": "string", "id": int}
  ],
  "edges": [
    {"source": "pathA", "source_name": "classA", "target": "pathB", "target_name": "classB", "relation": "left,near"}
  ]
}

IDs start at 0 and increment. Return ONLY the JSON code block.
"""

# ==================================================
# Vision Prompt (Image → Scene Graph)
# ==================================================
VISION_SYSTEM_PROMPT = """
You are a specialist in 3D Scene Reconstruction. Convert the provided image
into a structured Scene Graph (JSON). Use only objects that are visually
present; do not invent objects. Floor/ground must not be an object.

Rules for relations:
- "supported by" / "supports" when one object rests on another.
- Positional: left, right, front, behind based on camera view.
- Proximity: near when objects are close but not touching.
- Ensure bidirectional consistency (if A left of B, then B right of A).

Output schema:
{
  "objects": [
    {"path": "/World/SM_<Class>_<ID>", "class_name": "string", "id": int}
  ],
  "edges": [
    {"source": "pathA", "source_name": "classA", "target": "pathB", "target_name": "classB", "relation": "left,near"}
  ]
}

IDs start at 0 and increment. Return ONLY the JSON code block.
"""

# ==================================================
# 首页
# ==================================================
@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")

# ==================================================
# Scene Graph（从文件读）
# ==================================================
@app.route("/scene_graph")
def scene_graph():
    if not os.path.exists(SCENE_GRAPH_PATH):
        return jsonify({"error": "Scene graph file not found"}), 404

    return send_file(
        SCENE_GRAPH_PATH,
        mimetype="application/json"
    )

# ==================================================
# 自然语言 → Scene Graph
# ==================================================
@app.route("/nl_to_scene", methods=["POST"])
def nl_to_scene():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Empty input"}), 400

    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        text_format=SceneGraph,
    )

    scene_graph = response.output_parsed
    scene_graph_json = scene_graph.model_dump()

    # 👉 关键：写回文件，右侧 Scene Graph 自动刷新
    with open(SCENE_GRAPH_PATH, "w") as f:
        json.dump(scene_graph_json, f, indent=2)

    return jsonify(scene_graph_json)

# ==================================================
# 图片上传 → Scene Graph
# ==================================================
@app.route("/scene_from_input", methods=["POST"])
def scene_from_input():
    """
    Accepts multipart form data:
    - image: optional file; when present we generate from image only (vision prompt)
    - text: optional string; used when image not provided
    - class_names: optional JSON list of strings (e.g., from browser directory picker)
    """
    file = request.files.get("image")
    text = request.form.get("text", "").strip()
    class_names_raw = request.form.get("class_names", "")

    class_hint = ""
    names = []

    if class_names_raw:
        try:
            names.extend(json.loads(class_names_raw))
        except json.JSONDecodeError:
            return jsonify({"error": "class_names must be valid JSON list"}), 400

    if names:
        unique_names = sorted({n for n in names if n})
        class_hint = (
            "You must only use the following classes (do not invent others): "
            + ", ".join(unique_names)
        )

    if file:
        image_bytes = file.read()
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
    else:
        if not text:
            return jsonify({"error": "Empty input"}), 400

        user_prompt = text
        if class_hint:
            user_prompt = f"{text}\n{class_hint}"

        response = client.responses.parse(
            model="gpt-4o-2024-08-06",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            text_format=SceneGraph,
        )

    scene_graph = response.output_parsed
    scene_graph_json = scene_graph.model_dump()

    with open(SCENE_GRAPH_PATH, "w") as f:
        json.dump(scene_graph_json, f, indent=2)

    return jsonify(scene_graph_json)

# ==================================================
# 按钮触发 Isaac Sim
# ==================================================
@app.route("/generate")
def generate():
    try:
        result = subprocess.run(
            [ISAAC_PYTHON, ISAAC_SCRIPT],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )

        print("=== Isaac Sim stdout ===")
        print(result.stdout)

        return jsonify({"status": "ok"})

    except subprocess.TimeoutExpired:
        return jsonify({
            "status": "error",
            "msg": "Isaac Sim timed out"
        }), 500

    except subprocess.CalledProcessError as e:
        print("=== Isaac Sim stderr ===")
        print(e.stderr)
        return jsonify({
            "status": "error",
            "msg": "Isaac Sim failed",
            "detail": e.stderr
        }), 500

# ==================================================
# Real2Sim pipeline
# ==================================================
@app.route("/real2sim")
def real2sim():
    def run_step(cmd, timeout, label, env=None):
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env or os.environ.copy(),
        )
        ts = datetime.now().isoformat()
        log_text = (
            f"[{ts}] === {label} ===\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}\n"
        )
        print(log_text)
        with open(LOG_PATH, "a") as logf:
            logf.write(log_text)
        return result

    # Common env for Isaac steps: force warp to CPU to avoid CUDA driver mismatch
    isaac_env = os.environ.copy()
    isaac_env["WARP_DISABLE_CUDA"] = "1"

    try:
        # 1) Run SAM3 mesh generation
        run_step([SAM3_PYTHON, SAM3_MESH_GEN], timeout=600, label="sam3_mesh_gen")

        # 2) Convert meshes to USD using asset converter
        run_step(
            [
                ISAAC_PYTHON,
                ASSET_CONVERTER_SCRIPT,
                "--folders",
                SAM3_MESH_OUTPUT,
            ],
            timeout=600,
            label="asset_converter",
            env=isaac_env,
        )

        # 3) Render with save_figure.py using generated mesh root
        run_step(
            [
                ISAAC_PYTHON,
                ISAAC_SCRIPT,
                "--asset-root",
                GENMESH_ROOT,
            ],
            timeout=600,
            label="save_figure",
            env=isaac_env,
        )

        return jsonify({"status": "ok"})

    except subprocess.TimeoutExpired:
        return jsonify({
            "status": "error",
            "msg": "Real2Sim pipeline timed out"
        }), 500

    except subprocess.CalledProcessError as e:
        ts = datetime.now().isoformat()
        fail_text = f"[{ts}] === {e.cmd} failed ===\n{e.stderr}\n"
        print(fail_text)
        with open(LOG_PATH, "a") as logf:
            logf.write(fail_text)
        return jsonify({
            "status": "error",
            "msg": "Real2Sim pipeline failed",
            "detail": e.stderr
        }), 500

# ==================================================
# 启动
# ==================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
