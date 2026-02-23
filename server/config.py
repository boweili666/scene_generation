import os
from pathlib import Path


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB_DIR = os.path.join(BASE_DIR, "web")

ISAAC_PYTHON = "python"
ISAAC_SCRIPT = os.path.join(BASE_DIR, "isaac_local", "scripts", "save_figure.py")
ASSET_CONVERTER_SCRIPT = os.path.join(BASE_DIR, "isaac_local", "scripts", "asset_convert.py")
SAM3_PYTHON = "python"
SAM3_MESH_GEN = os.path.expanduser("/home/lbw/3dgen-project/sam3/test_mesh_gen.py")
SAM3_MESH_OUTPUT = os.path.expanduser("/home/lbw/3dgen-project/sam3/outputs/meshes")
GENMESH_ROOT = os.path.join(BASE_DIR, "genmesh")

SCENE_GRAPH_PATH = os.path.join(BASE_DIR, "isaac_local", "my_viewer", "full_scene_graph.json")

DEFAULT_PLACEMENTS_PATH = Path(
    os.path.join(BASE_DIR, "isaac_local", "my_viewer", "placements", "placements_default.json")
)
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_RENDER_PATH = Path(WEB_DIR) / "assets" / "renders" / "render.png"
LOG_PATH = os.path.join(BASE_DIR, "real2sim.log")

# Real2Sim option2
OPTION2_ROOT_DIR = BASE_DIR
OPTION2_PIPELINE_DIR = os.path.join(BASE_DIR, "option2_pipeline")
OPTION2_RUNTIME_DIR = os.path.join(OPTION2_PIPELINE_DIR, "runtime")
OPTION2_STEP1_SEGMENT_SCRIPT = os.path.join("option2_pipeline", "step1_segment_scene60_all_objects.py")
TOPVIEW_SCRIPT = os.path.join("option2_pipeline", "step2_generate_top_view.py")
TOPVIEW_DEFAULT_INPUT = os.path.join("..", "sam3", "scene_60.jpeg")
TOPVIEW_DEFAULT_OUTPUT = os.path.join("option2_pipeline", "runtime", "scene_60_top_view.png")
SAM3_RELATIVE_XY_SCRIPT = os.path.join("option2_pipeline", "step3_sam3_relative_xy.py")
OPTION2_STEP4_ARRANGE_WRAPPER = os.path.join("option2_pipeline", "step4_arrange_from_csv.py")
ARRANGE_INPUT_DIR = os.path.join("option2_pipeline", "runtime")
OPTION2_MASK_OUTPUT = os.path.join("option2_pipeline", "runtime", "masks")
OPTION2_DEFAULT_CSV = os.path.join("option2_pipeline", "runtime", "sam3_bbox_relative.csv")
OPTION2_MESH_OUTPUT_DIR = os.path.join("option2_pipeline", "runtime", "meshes")
OPTION2_REUSE_MESH_DIR = os.path.join("option2_pipeline", "runtime", "meshes")
OPTION2_DEFAULT_OUTPUT_GLB = os.path.join("option2_pipeline", "runtime", "scene_from_csv_yaw.glb")
OPTION2_DEFAULT_OUTPUT_JSON = os.path.join(
    "option2_pipeline", "runtime", "scene_from_csv_yaw_transforms.json"
)
OPTION2_SKIP_SAM3D_DEFAULT = True
