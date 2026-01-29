# server/server.py
from flask import Flask, send_from_directory, jsonify, request
import subprocess
import os
import json

from flask import send_file
from openai import OpenAI
from pydantic import BaseModel
from typing import List

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
# 启动
# ==================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
