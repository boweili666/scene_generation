import os
import subprocess
from datetime import datetime

try:
    from .config import (
        ASSET_CONVERTER_SCRIPT,
        ARRANGE_FROM_CSV_SCRIPT,
        ARRANGE_INPUT_DIR,
        GENMESH_ROOT,
        ISAAC_PYTHON,
        ISAAC_SCRIPT,
        LOG_PATH,
        OPTION2_DEFAULT_CSV,
        SAM3_MESH_GEN,
        SAM3_MESH_OUTPUT,
        SAM3_RELATIVE_XY_SCRIPT,
        SAM3_PYTHON,
        TOPVIEW_DEFAULT_INPUT,
        TOPVIEW_DEFAULT_OUTPUT,
        TOPVIEW_SCRIPT,
    )
except ImportError:
    from config import (
        ASSET_CONVERTER_SCRIPT,
        ARRANGE_FROM_CSV_SCRIPT,
        ARRANGE_INPUT_DIR,
        GENMESH_ROOT,
        ISAAC_PYTHON,
        ISAAC_SCRIPT,
        LOG_PATH,
        OPTION2_DEFAULT_CSV,
        SAM3_MESH_GEN,
        SAM3_MESH_OUTPUT,
        SAM3_RELATIVE_XY_SCRIPT,
        SAM3_PYTHON,
        TOPVIEW_DEFAULT_INPUT,
        TOPVIEW_DEFAULT_OUTPUT,
        TOPVIEW_SCRIPT,
    )


def run_generate() -> None:
    subprocess.run(
        [ISAAC_PYTHON, ISAAC_SCRIPT],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
    )


def _run_step(cmd, timeout, label, env=None):
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
    with open(LOG_PATH, "a", encoding="utf-8") as logf:
        logf.write(log_text)
    return result


def run_real2sim() -> None:
    isaac_env = os.environ.copy()
    isaac_env["WARP_DISABLE_CUDA"] = "1"

    _run_step([SAM3_PYTHON, SAM3_MESH_GEN], timeout=600, label="sam3_mesh_gen")
    _run_step(
        [ISAAC_PYTHON, ASSET_CONVERTER_SCRIPT, "--folders", SAM3_MESH_OUTPUT],
        timeout=600,
        label="asset_converter",
        env=isaac_env,
    )
    _run_step(
        [ISAAC_PYTHON, ISAAC_SCRIPT, "--asset-root", GENMESH_ROOT],
        timeout=600,
        label="save_figure",
        env=isaac_env,
    )


def run_real2sim_option2(payload: dict | None = None) -> dict:
    payload = payload or {}
    topview_input = str(payload.get("topview_input") or TOPVIEW_DEFAULT_INPUT)
    topview_output = str(payload.get("topview_output") or TOPVIEW_DEFAULT_OUTPUT)
    prompts = payload.get("prompts") or ["table"]
    if not isinstance(prompts, list) or not all(isinstance(p, str) and p for p in prompts):
        raise ValueError("prompts must be a non-empty list of strings")
    reference = str(payload.get("reference") or "table")
    csv_output = str(payload.get("csv_output") or OPTION2_DEFAULT_CSV)
    input_dir = str(payload.get("input_dir") or ARRANGE_INPUT_DIR)
    output_glb = str(payload.get("output_glb") or os.path.join(input_dir, "scene_from_csv_yaw.glb"))
    output_json = str(
        payload.get("output_json") or os.path.join(input_dir, "scene_from_csv_yaw_transforms.json")
    )
    topview_python = str(payload.get("topview_python") or SAM3_PYTHON)
    sam3_python = str(payload.get("sam3_python") or SAM3_PYTHON)
    arrange_python = str(payload.get("arrange_python") or SAM3_PYTHON)

    # step1
    _run_step([SAM3_PYTHON, SAM3_MESH_GEN], timeout=600, label="option2_sam3_mesh_gen")

    # step2
    _run_step(
        [
            topview_python,
            TOPVIEW_SCRIPT,
            "--input",
            topview_input,
            "--output",
            topview_output,
        ],
        timeout=600,
        label="option2_generate_top_view",
    )

    # step3
    _run_step(
        [
            sam3_python,
            SAM3_RELATIVE_XY_SCRIPT,
            "--image",
            topview_output,
            "--prompts",
            *prompts,
            "--reference",
            reference,
            "--output",
            csv_output,
        ],
        timeout=1200,
        label="option2_sam3_relative_xy",
    )

    # step4
    _run_step(
        [
            arrange_python,
            ARRANGE_FROM_CSV_SCRIPT,
            "--input-dir",
            input_dir,
            "--csv-path",
            csv_output,
            "--output-glb",
            output_glb,
            "--output-json",
            output_json,
        ],
        timeout=1800,
        label="option2_arrange_from_csv",
    )

    return {
        "topview_input": topview_input,
        "topview_output": topview_output,
        "csv_output": csv_output,
        "input_dir": input_dir,
        "output_glb": output_glb,
        "output_json": output_json,
        "prompts": prompts,
        "reference": reference,
    }


def log_pipeline_failure(error: subprocess.CalledProcessError) -> None:
    ts = datetime.now().isoformat()
    fail_text = f"[{ts}] === {error.cmd} failed ===\n{error.stderr}\n"
    print(fail_text)
    with open(LOG_PATH, "a", encoding="utf-8") as logf:
        logf.write(fail_text)
