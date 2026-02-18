import os
from pathlib import Path


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
SAM3_PYTHON = os.path.expanduser("/home/lbw/miniconda3/envs/sam3/bin/python")
SAM3_MESH_GEN = os.path.expanduser("/home/lbw/3dgen-project/sam3/test_mesh_gen.py")
SAM3_MESH_OUTPUT = os.path.expanduser("/home/lbw/3dgen-project/sam3/outputs/meshes")
GENMESH_ROOT = os.path.expanduser(
    "/home/lbw/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/genmesh"
)

SCENE_GRAPH_PATH = os.path.expanduser(
    "~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/bowei/my_viewer/full_scene_graph.json"
)

DEFAULT_PLACEMENTS_PATH = Path(
    "/home/lbw/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/bowei/my_viewer/placements/placements_default.json"
)
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_RENDER_PATH = Path(WEB_DIR) / "assets" / "renders" / "render.png"
LOG_PATH = os.path.join(BASE_DIR, "real2sim.log")

# Real2Sim option2
TOPVIEW_SCRIPT = os.path.expanduser("/home/lbw/3dgen-project/gpttest/generate_top_view.py")
TOPVIEW_DEFAULT_INPUT = os.path.expanduser("/home/lbw/3dgen-project/gpttest/scene_60.jpeg")
TOPVIEW_DEFAULT_OUTPUT = os.path.expanduser(
    "/home/lbw/3dgen-project/scene60_mesh_rts/scene_60_top_view.png"
)
SAM3_RELATIVE_XY_SCRIPT = os.path.expanduser("/home/lbw/3dgen-project/sam3/sam3_relative_xy.py")
ARRANGE_FROM_CSV_SCRIPT = os.path.expanduser(
    "/home/lbw/3dgen-project/scene60_mesh_rts/arrange_from_csv_yaw_yup.py"
)
ARRANGE_INPUT_DIR = os.path.expanduser("/home/lbw/3dgen-project/scene60_mesh_rts")
OPTION2_DEFAULT_CSV = os.path.expanduser(
    "/home/lbw/3dgen-project/scene60_mesh_rts/sam3_bbox_relative.csv"
)
