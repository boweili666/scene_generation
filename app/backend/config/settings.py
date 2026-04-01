import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = PROJECT_ROOT / "app"
BACKEND_DIR = APP_DIR / "backend"
FRONTEND_DIR = APP_DIR / "frontend"
FRONTEND_ASSETS_DIR = FRONTEND_DIR / "assets"

RUNTIME_DIR = Path(os.environ.get("SCENE_UI_RUNTIME_DIR", PROJECT_ROOT / "runtime")).resolve()
LOGS_DIR = Path(os.environ.get("SCENE_UI_LOGS_DIR", PROJECT_ROOT / "logs")).resolve()

REAL2SIM_RUNTIME_DIR = RUNTIME_DIR / "real2sim"
SCENE_SERVICE_RUNTIME_DIR = RUNTIME_DIR / "scene_service"
SCENE_SERVICE_USD_DIR = SCENE_SERVICE_RUNTIME_DIR / "usd"
SCENE_SERVICE_PLACEMENTS_DIR = SCENE_SERVICE_RUNTIME_DIR / "placements"
SCENE_GRAPH_RUNTIME_DIR = RUNTIME_DIR / "scene_graph"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
RENDERS_DIR = RUNTIME_DIR / "renders"

BASE_DIR = str(PROJECT_ROOT)
WEB_DIR = str(FRONTEND_DIR)

ISAAC_PYTHON = os.environ.get("ISAAC_PYTHON", "python")
ISAAC_SCRIPT = str(PROJECT_ROOT / "pipelines" / "isaac" / "scene_renderer.py")
ASSET_CONVERTER_SCRIPT = str(PROJECT_ROOT / "pipelines" / "isaac" / "mesh_to_usd_converter.py")
ISAAC_ASSET_ROOT = (PROJECT_ROOT / "pipelines" / "isaac" / "assets").resolve()
RETRIEVAL_ASSET_ROOT = Path(os.environ.get("RETRIEVAL_ASSET_ROOT", PROJECT_ROOT / "testusd")).resolve()
SAM3_PYTHON = os.environ.get("SAM3_PYTHON", "python")
SAM3_MESH_GEN = os.path.expanduser(os.environ.get("SAM3_MESH_GEN", "/home/lbw/3dgen-project/sam3/test_mesh_gen.py"))
SAM3_MESH_OUTPUT = os.path.expanduser(os.environ.get("SAM3_MESH_OUTPUT", "/home/lbw/3dgen-project/sam3/outputs/meshes"))
GENMESH_ROOT = str(PROJECT_ROOT / "genmesh")

SCENE_GRAPH_PATH = str((SCENE_GRAPH_RUNTIME_DIR / "current_scene_graph.json").resolve())

DEFAULT_PLACEMENTS_PATH = (SCENE_SERVICE_PLACEMENTS_DIR / "placements_default.json").resolve()
DEFAULT_MODEL = os.environ.get("SCENE_GRAPH_DEFAULT_MODEL", "gpt-5-nano")
DEFAULT_RENDER_PATH = (RENDERS_DIR / "render.png").resolve()
LATEST_INPUT_IMAGE = (UPLOADS_DIR / "latest_input.jpg").resolve()
LOG_PATH = str((LOGS_DIR / "real2sim.log").resolve())
SCENE_SERVICE_LOG_PATH = str((LOGS_DIR / "scene_service.log").resolve())

# Real2Sim
REAL2SIM_ROOT_DIR = PROJECT_ROOT.resolve()
REAL2SIM_SEGMENT_SCRIPT = os.path.join("pipelines", "real2sim", "object_segmentation_pipeline.py")
REAL2SIM_PREDICT_STREAM_CLIENT = os.path.join("pipelines", "real2sim", "streaming_generation_client.py")
REAL2SIM_PREDICT_STREAM_SERVER_SCRIPT = os.path.join("pipelines", "real2sim", "predict_stream_server.py")
REAL2SIM_MASK_OUTPUT_DIR = (REAL2SIM_RUNTIME_DIR / "masks").resolve()
REAL2SIM_MESH_OUTPUT_DIR = (REAL2SIM_RUNTIME_DIR / "meshes").resolve()
REAL2SIM_REUSE_MESH_DIR = (REAL2SIM_RUNTIME_DIR / "meshes").resolve()
REAL2SIM_SCENE_RESULTS_DIR = (REAL2SIM_RUNTIME_DIR / "scene_results").resolve()
REAL2SIM_MANIFEST_PATH = (REAL2SIM_SCENE_RESULTS_DIR / "real2sim_asset_manifest.json").resolve()
PREDICT_STREAM_SERVER = os.environ.get("PREDICT_STREAM_SERVER", "http://127.0.0.1:8002")
SAM3D_OBJECTS_ROOT = Path(
    os.environ.get("SAM3D_OBJECTS_ROOT", PROJECT_ROOT / "third_party" / "sam-3d-objects")
).resolve()
SAM3D_PIPELINE_CONFIG = Path(
    os.environ.get("SAM3D_PIPELINE_CONFIG", SAM3D_OBJECTS_ROOT / "checkpoints" / "hf" / "pipeline.yaml")
).resolve()

DIRECTORIES_TO_CREATE = [
    FRONTEND_ASSETS_DIR,
    LOGS_DIR,
    RUNTIME_DIR,
    RETRIEVAL_ASSET_ROOT,
    REAL2SIM_RUNTIME_DIR,
    SCENE_SERVICE_RUNTIME_DIR,
    SCENE_SERVICE_USD_DIR,
    SCENE_SERVICE_PLACEMENTS_DIR,
    SCENE_GRAPH_RUNTIME_DIR,
    UPLOADS_DIR,
    RENDERS_DIR,
    REAL2SIM_MASK_OUTPUT_DIR,
    REAL2SIM_MESH_OUTPUT_DIR,
    REAL2SIM_SCENE_RESULTS_DIR,
]


def ensure_runtime_layout() -> None:
    for path in DIRECTORIES_TO_CREATE:
        path.mkdir(parents=True, exist_ok=True)

    placements_path = Path(DEFAULT_PLACEMENTS_PATH)
    if not placements_path.exists():
        placements_path.write_text("{}\n", encoding="utf-8")


ensure_runtime_layout()
