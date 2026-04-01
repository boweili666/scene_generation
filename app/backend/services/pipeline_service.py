import os
import subprocess
import json
import selectors
import time
from pathlib import Path
from datetime import datetime

from ..config import (
    ASSET_CONVERTER_SCRIPT,
    ISAAC_PYTHON,
    LATEST_INPUT_IMAGE,
    LOG_PATH,
    PREDICT_STREAM_SERVER,
    REAL2SIM_MASK_OUTPUT_DIR,
    REAL2SIM_MESH_OUTPUT_DIR,
    REAL2SIM_ROOT_DIR,
    REAL2SIM_REUSE_MESH_DIR,
    REAL2SIM_SCENE_RESULTS_DIR,
    REAL2SIM_SEGMENT_SCRIPT,
    REAL2SIM_PREDICT_STREAM_CLIENT,
    SCENE_GRAPH_PATH,
    SAM3_PYTHON,
)


def run_generate() -> None:
    run_real2sim()


def _append_log_entry(entry: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as logf:
        logf.write(entry)


def log_real2sim_event(message: str, job_id: str | None = None, level: str = "INFO") -> None:
    prefix = f"[{datetime.now().isoformat()}] [{level}]"
    if job_id:
        prefix += f" [job={job_id}]"

    lines = message.splitlines() or [""]
    entries: list[str] = []
    for line in lines:
        entry = f"{prefix} {line}\n"
        print(entry, end="")
        entries.append(entry)
    _append_log_entry("".join(entries))


def read_real2sim_log(offset: int = 0, limit: int = 65536) -> dict:
    path = Path(LOG_PATH)
    if not path.exists():
        return {"content": "", "next_offset": 0, "size": 0, "truncated": False}

    file_size = path.stat().st_size
    safe_offset = max(0, min(int(offset), file_size))
    safe_limit = max(1024, min(int(limit), 262144))

    with path.open("rb") as f:
        f.seek(safe_offset)
        chunk = f.read(safe_limit)
        next_offset = f.tell()

    return {
        "content": chunk.decode("utf-8", errors="ignore"),
        "next_offset": next_offset,
        "size": file_size,
        "truncated": next_offset < file_size,
    }


def get_real2sim_log_size() -> int:
    path = Path(LOG_PATH)
    return path.stat().st_size if path.exists() else 0


def _run_step(cmd, timeout, label, env=None, cwd=None, job_id: str | None = None):
    ts = datetime.now().isoformat()
    header = (
        f"[{ts}]"
        + (f" [job={job_id}]" if job_id else "")
        + f" === {label} ===\n"
        f"CMD: {' '.join(cmd)}\n"
        f"CWD: {cwd or os.getcwd()}\n"
        f"--- stream start ---\n"
    )
    print(header, end="")
    _append_log_entry(header)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or os.environ.copy(),
        cwd=cwd,
        bufsize=1,
    )

    sel = selectors.DefaultSelector()
    if process.stdout is not None:
        sel.register(process.stdout, selectors.EVENT_READ, data="STDOUT")
    if process.stderr is not None:
        sel.register(process.stderr, selectors.EVENT_READ, data="STDERR")

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    start_time = time.monotonic()
    last_activity = start_time
    heartbeat_interval = 15.0

    def _write_stream_line(stream_name: str, line: str) -> None:
        ts_line = datetime.now().isoformat()
        entry = f"[{ts_line}] [{stream_name}]"
        if job_id:
            entry += f" [job={job_id}]"
        entry += f" {line}"
        print(entry, end="")
        _append_log_entry(entry)

    while True:
        now = time.monotonic()
        if timeout is not None and now - start_time > timeout:
            process.kill()
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        events = sel.select(timeout=0.5)
        if not events:
            if process.poll() is not None:
                break
            if now - last_activity >= heartbeat_interval:
                heartbeat = (
                    f"[{datetime.now().isoformat()}] [HEARTBEAT]"
                    + (f" [job={job_id}]" if job_id else "")
                    + f" {label} still running "
                    f"({int(now - start_time)}s)\n"
                )
                print(heartbeat, end="")
                _append_log_entry(heartbeat)
                last_activity = now
            continue

        for key, _ in events:
            stream_name = key.data
            stream = key.fileobj
            line = stream.readline()
            if line:
                last_activity = time.monotonic()
                _write_stream_line(stream_name, line)
                if stream_name == "STDOUT":
                    stdout_chunks.append(line)
                else:
                    stderr_chunks.append(line)
            else:
                try:
                    sel.unregister(stream)
                except Exception:
                    pass
                stream.close()

        if process.poll() is not None and not sel.get_map():
            break

    return_code = process.wait()
    footer = (
        f"[{datetime.now().isoformat()}]"
        + (f" [job={job_id}]" if job_id else "")
        + f" --- stream end --- exit={return_code} label={label}\n"
    )
    print(footer, end="")
    _append_log_entry(footer)

    if return_code != 0:
        raise subprocess.CalledProcessError(
            returncode=return_code,
            cmd=cmd,
            output="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )

    class _Result:
        def __init__(self, stdout: str, stderr: str):
            self.stdout = stdout
            self.stderr = stderr

    return _Result("".join(stdout_chunks), "".join(stderr_chunks))


def _extract_prompts_from_scene_graph(scene_graph_path: str) -> list[str]:
    path = Path(scene_graph_path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompts: list[str] = []
    objects = data.get("objects")
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if obj.get("source") != "real2sim":
                continue
            val = obj.get("class_name") or obj.get("class")
            if isinstance(val, str) and val.strip():
                prompts.append(val.strip().lower())

    obj_map = data.get("obj")
    if isinstance(obj_map, dict):
        for obj in obj_map.values():
            if not isinstance(obj, dict):
                continue
            if obj.get("source") != "real2sim":
                continue
            val = obj.get("class_name") or obj.get("class")
            if isinstance(val, str) and val.strip():
                prompts.append(val.strip().lower())

    deduped: list[str] = []
    seen: set[str] = set()
    for item in prompts:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def collect_scene_result_artifacts(real2sim_root: str, scene_results_dir: str) -> dict:
    root = Path(real2sim_root).resolve()
    results_root = (root / Path(scene_results_dir)).resolve()
    objects_dir = results_root / "objects"

    def _rel(path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            return str(resolved)

    object_glbs: list[str] = []
    if objects_dir.exists():
        object_glbs = sorted(
            _rel(p)
            for p in objects_dir.glob("*.glb")
            if p.is_file()
        )

    scene_glb = None
    if results_root.exists():
        candidates = sorted(
            p for p in results_root.glob("*.glb") if p.is_file() and p.name.lower().endswith(".glb")
        )
        if candidates:
            preferred = next((p for p in candidates if p.name == "scene_merged.glb"), None)
            chosen = preferred or candidates[0]
            scene_glb = _rel(chosen)

    poses_json = None
    poses_path = results_root / "poses.json"
    if poses_path.exists() and poses_path.is_file():
        poses_json = _rel(poses_path)

    scene_usd = None
    for scene_usd_name in ("scene_merged_post.usd", "scene_merged.usd"):
        scene_usd_path = results_root / scene_usd_name
        if scene_usd_path.exists() and scene_usd_path.is_file():
            scene_usd = _rel(scene_usd_path)
            break

    manifest_json = None
    manifest_path = results_root / "real2sim_asset_manifest.json"
    if manifest_path.exists() and manifest_path.is_file():
        manifest_json = _rel(manifest_path)

    return {
        "scene_results_dir": str(results_root),
        "object_glbs": object_glbs,
        "scene_glb": scene_glb,
        "poses_json": poses_json,
        "scene_usd": scene_usd,
        "manifest_json": manifest_json,
    }


def run_real2sim(payload: dict | None = None, job_id: str | None = None) -> dict:
    payload = payload or {}
    real2sim_root = Path(str(payload.get("real2sim_root_dir") or REAL2SIM_ROOT_DIR)).resolve()
    real2sim_root.mkdir(parents=True, exist_ok=True)

    image_path = str(payload.get("image_path") or LATEST_INPUT_IMAGE)
    scene_graph_path = str(payload.get("scene_graph_path") or SCENE_GRAPH_PATH)
    scene_graph_prompts = _extract_prompts_from_scene_graph(scene_graph_path)
    prompts = payload.get("prompts") or scene_graph_prompts
    log_real2sim_event(
        f"Starting Real2Sim with image={image_path} scene_graph={scene_graph_path}",
        job_id=job_id,
    )
    if not prompts:
        raise ValueError("No object prompts found in scene graph. Generate scene graph first.")
    if not isinstance(prompts, list) or not all(isinstance(p, str) and p for p in prompts):
        raise ValueError("prompts must be a non-empty list of strings")
    prompts = [p.strip().lower() for p in prompts if p and p.strip()]
    if not prompts:
        raise ValueError("prompts resolved to empty after normalization")
    log_real2sim_event(
        f"Resolved {len(prompts)} prompt(s): {', '.join(prompts)}",
        job_id=job_id,
    )

    if not Path(image_path).exists():
        raise ValueError(f"Input image not found: {image_path}. Upload image first via /scene_from_input.")
    if not Path(scene_graph_path).exists():
        raise ValueError(f"Scene graph not found: {scene_graph_path}. Generate scene graph first.")

    mask_output = str(payload.get("mask_output") or REAL2SIM_MASK_OUTPUT_DIR)
    mesh_output_dir = str(payload.get("mesh_output_dir") or REAL2SIM_MESH_OUTPUT_DIR)
    reuse_mesh_dir = str(payload.get("reuse_mesh_dir") or REAL2SIM_REUSE_MESH_DIR)
    sam3_python = str(payload.get("sam3_python") or SAM3_PYTHON)
    predict_stream_server = str(payload.get("predict_stream_server") or PREDICT_STREAM_SERVER)
    scene_results_dir = str(payload.get("scene_results_dir") or REAL2SIM_SCENE_RESULTS_DIR)

    (real2sim_root / Path(mask_output)).mkdir(parents=True, exist_ok=True)
    (real2sim_root / Path(mesh_output_dir)).mkdir(parents=True, exist_ok=True)
    (real2sim_root / Path(scene_results_dir)).mkdir(parents=True, exist_ok=True)

    step1_cmd = [
        sam3_python,
        "-u",
        REAL2SIM_SEGMENT_SCRIPT,
        "--image",
        image_path,
        "--scene-graph",
        scene_graph_path,
        "--output-root",
        mask_output,
        "--mesh-output-dir",
        mesh_output_dir,
        "--reuse-mesh-dir",
        reuse_mesh_dir,
        "--prompts",
        *prompts,
    ]
    _run_step(
        step1_cmd,
        timeout=1800,
        label="sam3_segment_objects_only",
        cwd=str(real2sim_root),
        job_id=job_id,
    )

    masks_dir_abs = (real2sim_root / Path(mask_output)).resolve()
    image_png_abs = (masks_dir_abs / "image.png").resolve()
    if not image_png_abs.exists():
        raise ValueError(f"Expected segmented image not found: {image_png_abs}")

    scene_results_abs = (real2sim_root / Path(scene_results_dir)).resolve()
    step2_cmd = [
        sam3_python,
        "-u",
        REAL2SIM_PREDICT_STREAM_CLIENT,
        "--server",
        predict_stream_server,
        "--image",
        str(image_png_abs),
        "--scene-graph",
        scene_graph_path,
        "--mask-dir",
        str(masks_dir_abs),
        "--output-dir",
        str(scene_results_abs),
        "--converter-python",
        str(ISAAC_PYTHON),
        "--asset-converter-script",
        str(ASSET_CONVERTER_SCRIPT),
    ]
    _run_step(
        step2_cmd,
        timeout=7200,
        label="predict_stream_generate_glbs",
        cwd=str(real2sim_root),
        job_id=job_id,
    )

    scene_artifacts = collect_scene_result_artifacts(str(real2sim_root), scene_results_dir)
    log_real2sim_event(
        "Real2Sim artifacts ready: "
        f"{len(scene_artifacts.get('object_glbs', []))} object GLB(s), "
        f"scene_glb={scene_artifacts.get('scene_glb')}, poses_json={scene_artifacts.get('poses_json')}",
        job_id=job_id,
    )

    return {
        "mode": "sam3-segmentation-plus-glb",
        "real2sim_root_dir": str(real2sim_root),
        "image_path": image_path,
        "scene_graph_path": scene_graph_path,
        "prompts": prompts,
        "mask_output": mask_output,
        "scene_results_dir": scene_results_dir,
        "predict_stream_server": predict_stream_server,
        **scene_artifacts,
    }


def log_pipeline_failure(error: subprocess.CalledProcessError) -> None:
    ts = datetime.now().isoformat()
    fail_text = f"[{ts}] === {error.cmd} failed ===\n{error.stderr}\n"
    print(fail_text)
    _append_log_entry(fail_text)
