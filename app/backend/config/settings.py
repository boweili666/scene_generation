import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = PROJECT_ROOT / "app"
BACKEND_DIR = APP_DIR / "backend"
FRONTEND_DIR = APP_DIR / "frontend"
FRONTEND_ASSETS_DIR = FRONTEND_DIR / "assets"
LEGACY_WEB_DIR = PROJECT_ROOT / "web"

RUNTIME_DIR = Path(os.environ.get("SCENE_UI_RUNTIME_DIR", PROJECT_ROOT / "runtime")).resolve()
LOGS_DIR = Path(os.environ.get("SCENE_UI_LOGS_DIR", PROJECT_ROOT / "logs")).resolve()

REAL2SIM_RUNTIME_DIR = RUNTIME_DIR / "real2sim"
SCENE_SERVICE_RUNTIME_DIR = RUNTIME_DIR / "scene_service"
SCENE_SERVICE_USD_DIR = SCENE_SERVICE_RUNTIME_DIR / "usd"
SCENE_GRAPH_RUNTIME_DIR = RUNTIME_DIR / "scene_graph"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
RENDERS_DIR = RUNTIME_DIR / "renders"

BASE_DIR = str(PROJECT_ROOT)
WEB_DIR = str(FRONTEND_DIR)

ISAAC_PYTHON = os.environ.get("ISAAC_PYTHON", "python")
ISAAC_SCRIPT = str(PROJECT_ROOT / "isaac_local" / "scripts" / "save_figure.py")
ASSET_CONVERTER_SCRIPT = str(PROJECT_ROOT / "isaac_local" / "scripts" / "asset_convert.py")
SAM3_PYTHON = os.environ.get("SAM3_PYTHON", "python")
SAM3_MESH_GEN = os.path.expanduser(os.environ.get("SAM3_MESH_GEN", "/home/lbw/3dgen-project/sam3/test_mesh_gen.py"))
SAM3_MESH_OUTPUT = os.path.expanduser(os.environ.get("SAM3_MESH_OUTPUT", "/home/lbw/3dgen-project/sam3/outputs/meshes"))
GENMESH_ROOT = str(PROJECT_ROOT / "genmesh")

LEGACY_SCENE_GRAPH_PATH = PROJECT_ROOT / "isaac_local" / "my_viewer" / "full_scene_graph.json"
SCENE_GRAPH_PATH = str((SCENE_GRAPH_RUNTIME_DIR / "current_scene_graph.json").resolve())

DEFAULT_PLACEMENTS_PATH = (PROJECT_ROOT / "isaac_local" / "my_viewer" / "placements" / "placements_default.json").resolve()
DEFAULT_MODEL = os.environ.get("SCENE_GRAPH_DEFAULT_MODEL", "gpt-5-nano")
DEFAULT_RENDER_PATH = (RENDERS_DIR / "render.png").resolve()
LATEST_INPUT_IMAGE = (UPLOADS_DIR / "latest_input.jpg").resolve()
LOG_PATH = str((LOGS_DIR / "real2sim.log").resolve())
SCENE_SERVICE_LOG_PATH = str((LOGS_DIR / "scene_service.log").resolve())

# Real2Sim option2
OPTION2_ROOT_DIR = str(PROJECT_ROOT)
OPTION2_PIPELINE_DIR = str((PROJECT_ROOT / "option2_pipeline").resolve())
OPTION2_RUNTIME_DIR = str(REAL2SIM_RUNTIME_DIR.resolve())
OPTION2_STEP1_SEGMENT_SCRIPT = os.path.join("option2_pipeline", "step1_segment_scene60_all_objects.py")
TOPVIEW_SCRIPT = os.path.join("option2_pipeline", "step2_generate_top_view.py")
TOPVIEW_DEFAULT_INPUT = os.path.join("..", "sam3", "scene_60.jpeg")
TOPVIEW_DEFAULT_OUTPUT = str((REAL2SIM_RUNTIME_DIR / "scene_60_top_view.png").resolve())
SAM3_RELATIVE_XY_SCRIPT = os.path.join("option2_pipeline", "step3_sam3_relative_xy.py")
OPTION2_STEP4_ARRANGE_WRAPPER = os.path.join("option2_pipeline", "step4_arrange_from_csv.py")
ARRANGE_INPUT_DIR = str(REAL2SIM_RUNTIME_DIR.resolve())
OPTION2_MASK_OUTPUT = str((REAL2SIM_RUNTIME_DIR / "masks").resolve())
OPTION2_DEFAULT_CSV = str((REAL2SIM_RUNTIME_DIR / "sam3_bbox_relative.csv").resolve())
OPTION2_MESH_OUTPUT_DIR = str((REAL2SIM_RUNTIME_DIR / "meshes").resolve())
OPTION2_REUSE_MESH_DIR = str((REAL2SIM_RUNTIME_DIR / "meshes").resolve())
OPTION2_DEFAULT_OUTPUT_GLB = str((REAL2SIM_RUNTIME_DIR / "scene_from_csv_yaw.glb").resolve())
OPTION2_DEFAULT_OUTPUT_JSON = str((REAL2SIM_RUNTIME_DIR / "scene_from_csv_yaw_transforms.json").resolve())
OPTION2_SKIP_SAM3D_DEFAULT = True
OPTION2_SCENE_RESULTS_DIR = str((REAL2SIM_RUNTIME_DIR / "scene_results").resolve())
PREDICT_STREAM_SERVER = os.environ.get("PREDICT_STREAM_SERVER", "http://128.2.204.110:8000")
SEND_MASKS_TO_PREDICT_SCRIPT = os.path.join("scripts", "send_masks_to_predict.py")

DIRECTORIES_TO_CREATE = [
    FRONTEND_ASSETS_DIR,
    LOGS_DIR,
    RUNTIME_DIR,
    REAL2SIM_RUNTIME_DIR,
    SCENE_SERVICE_RUNTIME_DIR,
    SCENE_SERVICE_USD_DIR,
    SCENE_GRAPH_RUNTIME_DIR,
    UPLOADS_DIR,
    RENDERS_DIR,
    Path(OPTION2_MASK_OUTPUT),
    Path(OPTION2_MESH_OUTPUT_DIR),
    Path(OPTION2_SCENE_RESULTS_DIR),
]


def ensure_runtime_layout() -> None:
    for path in DIRECTORIES_TO_CREATE:
        path.mkdir(parents=True, exist_ok=True)

    # Seed runtime files from legacy locations on first run.
    legacy_render = LEGACY_WEB_DIR / "assets" / "renders" / "render.png"
    if not DEFAULT_RENDER_PATH.exists() and legacy_render.exists():
        DEFAULT_RENDER_PATH.write_bytes(legacy_render.read_bytes())

    legacy_upload = LEGACY_WEB_DIR / "assets" / "uploads" / "latest_input.jpg"
    if not LATEST_INPUT_IMAGE.exists() and legacy_upload.exists():
        LATEST_INPUT_IMAGE.write_bytes(legacy_upload.read_bytes())

    if not Path(SCENE_GRAPH_PATH).exists() and LEGACY_SCENE_GRAPH_PATH.exists():
        Path(SCENE_GRAPH_PATH).write_text(LEGACY_SCENE_GRAPH_PATH.read_text(encoding="utf-8"), encoding="utf-8")


ensure_runtime_layout()
