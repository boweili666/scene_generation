import json
import os
import subprocess
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from flask import jsonify, request, send_file, send_from_directory

from ..config import (
    DEFAULT_MODEL,
    DEFAULT_PLACEMENTS_PATH,
    DEFAULT_RENDER_PATH,
    LATEST_INPUT_IMAGE,
    LOG_PATH,
    REAL2SIM_MASK_OUTPUT_DIR,
    REAL2SIM_ROOT_DIR,
    REAL2SIM_RUNTIME_DIR,
    REAL2SIM_SCENE_RESULTS_DIR,
    SCENE_GRAPH_PATH,
    WEB_DIR,
)
from ..services.openai_service import (
    call_gpt_json_editor_with_image,
    encode_image_b64,
    parse_scene_graph_from_image,
    parse_scene_graph_from_text,
    read_json_file,
    write_json_file,
)
from ..services.pipeline_service import (
    collect_scene_result_artifacts,
    get_real2sim_log_size,
    log_pipeline_failure,
    log_real2sim_event,
    read_real2sim_log,
    run_real2sim,
)


_REAL2SIM_JOBS: dict[str, dict] = {}
_REAL2SIM_JOBS_LOCK = threading.Lock()


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _to_runtime_file_url(abs_path: Path) -> str | None:
    runtime_root = REAL2SIM_RUNTIME_DIR.resolve()
    target = abs_path.resolve()
    if target != runtime_root and runtime_root not in target.parents:
        return None
    rel = target.relative_to(runtime_root).as_posix()
    ts = int(target.stat().st_mtime_ns) if target.exists() else 0
    return f"/runtime_file/{rel}?ts={ts}"


def _to_artifact_urls(artifacts: dict) -> dict:
    result = dict(artifacts)
    object_urls: list[str] = []
    for rel in artifacts.get("object_glbs", []):
        p = Path(artifacts["real2sim_root_dir"]) / rel
        url = _to_runtime_file_url(p)
        if url:
            object_urls.append(url)
    result["object_glb_urls"] = object_urls

    scene_glb = artifacts.get("scene_glb")
    if scene_glb:
        scene_url = _to_runtime_file_url(Path(artifacts["real2sim_root_dir"]) / scene_glb)
        result["scene_glb_url"] = scene_url
    else:
        result["scene_glb_url"] = None

    poses_json = artifacts.get("poses_json")
    if poses_json:
        poses_url = _to_runtime_file_url(Path(artifacts["real2sim_root_dir"]) / poses_json)
        result["poses_json_url"] = poses_url
    else:
        result["poses_json_url"] = None

    scene_usd = artifacts.get("scene_usd")
    if scene_usd:
        scene_usd_url = _to_runtime_file_url(Path(artifacts["real2sim_root_dir"]) / scene_usd)
        result["scene_usd_url"] = scene_usd_url
    else:
        result["scene_usd_url"] = None

    manifest_json = artifacts.get("manifest_json")
    if manifest_json:
        manifest_url = _to_runtime_file_url(Path(artifacts["real2sim_root_dir"]) / manifest_json)
        result["manifest_json_url"] = manifest_url
    else:
        result["manifest_json_url"] = None
    return result


def _start_real2sim_job(payload: dict) -> str:
    job_id = uuid.uuid4().hex
    with _REAL2SIM_JOBS_LOCK:
        _REAL2SIM_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": _utcnow_iso(),
            "updated_at": _utcnow_iso(),
            "error": None,
            "traceback": None,
            "artifacts": {},
            "log_path": LOG_PATH,
            "log_start_offset": get_real2sim_log_size(),
            "payload": payload,
        }

    def _runner():
        with _REAL2SIM_JOBS_LOCK:
            job = _REAL2SIM_JOBS.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["updated_at"] = _utcnow_iso()

        log_real2sim_event("Job queued -> running", job_id=job_id)
        try:
            artifacts = run_real2sim(payload, job_id=job_id)
            with _REAL2SIM_JOBS_LOCK:
                job = _REAL2SIM_JOBS.get(job_id)
                if not job:
                    return
                job["status"] = "succeeded"
                job["updated_at"] = _utcnow_iso()
                job["artifacts"] = artifacts
            log_real2sim_event("Job completed successfully", job_id=job_id)
        except Exception as e:
            tb = traceback.format_exc()
            log_real2sim_event(f"Job failed: {e}", job_id=job_id, level="ERROR")
            log_real2sim_event(tb.rstrip(), job_id=job_id, level="TRACE")
            with _REAL2SIM_JOBS_LOCK:
                job = _REAL2SIM_JOBS.get(job_id)
                if not job:
                    return
                job["status"] = "failed"
                job["updated_at"] = _utcnow_iso()
                job["error"] = str(e)
                job["traceback"] = tb

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    return job_id


def _get_real2sim_job_status(job_id: str) -> dict | None:
    with _REAL2SIM_JOBS_LOCK:
        base = _REAL2SIM_JOBS.get(job_id)
        if not base:
            return None
        job = dict(base)

    payload = job.get("payload") or {}
    real2sim_root = str(payload.get("real2sim_root_dir") or REAL2SIM_ROOT_DIR)
    scene_results_dir = str(payload.get("scene_results_dir") or REAL2SIM_SCENE_RESULTS_DIR)
    live_artifacts = collect_scene_result_artifacts(real2sim_root, scene_results_dir)

    merged_artifacts = dict(job.get("artifacts") or {})
    merged_artifacts.setdefault("real2sim_root_dir", real2sim_root)
    merged_artifacts.setdefault("scene_results_dir", scene_results_dir)
    merged_artifacts["object_glbs"] = live_artifacts.get("object_glbs", [])
    merged_artifacts["scene_glb"] = live_artifacts.get("scene_glb")
    merged_artifacts["poses_json"] = live_artifacts.get("poses_json")

    mask_output = str(payload.get("mask_output") or REAL2SIM_MASK_OUTPUT_DIR)
    masks_dir = (Path(real2sim_root).resolve() / Path(mask_output)).resolve()
    expected_objects = None
    if masks_dir.exists() and masks_dir.is_dir():
        expected_objects = len(
            [
                p
                for p in masks_dir.glob("*.png")
                if p.is_file() and p.name.lower() != "image.png"
            ]
        )

    generated_objects = len(merged_artifacts["object_glbs"])
    has_merged_scene = bool(merged_artifacts.get("scene_glb"))
    phase = "queued"
    percent = 0
    if job.get("status") == "queued":
        phase = "queued"
        percent = 0
    elif job.get("status") == "running":
        if expected_objects is None or expected_objects == 0:
            phase = "segmenting"
            percent = 15
        elif generated_objects < expected_objects:
            phase = "generating_glbs"
            ratio = generated_objects / max(expected_objects, 1)
            percent = min(94, 20 + int(ratio * 70))
        elif not has_merged_scene:
            phase = "merging_scene"
            percent = 95
        else:
            phase = "finalizing"
            percent = 98
    elif job.get("status") == "succeeded":
        phase = "completed"
        percent = 100
    elif job.get("status") == "failed":
        phase = "failed"
        percent = 100 if has_merged_scene else 0

    job["progress"] = {
        "phase": phase,
        "percent": percent,
        "expected_objects": expected_objects,
        "generated_objects": generated_objects,
        "has_merged_scene": has_merged_scene,
    }
    job["artifacts"] = _to_artifact_urls(merged_artifacts)
    job.pop("payload", None)
    return job


def _write_scene_graph(scene_graph_json):
    write_json_file(Path(SCENE_GRAPH_PATH), scene_graph_json)


def _handle_edit_json(payload):
    instruction = (payload.get("instruction") or "").strip()
    if not instruction:
        return None, jsonify({"error": "instruction is required"}), 400

    raw_input = payload.get("input")
    raw_output = payload.get("output")
    raw_image = payload.get("image")
    model = payload.get("model") or DEFAULT_MODEL

    src = Path(raw_input).expanduser() if raw_input else DEFAULT_PLACEMENTS_PATH
    if not src.exists():
        return None, jsonify({"error": f"Input JSON not found: {src}"}), 404

    image_path = Path(raw_image).expanduser() if raw_image else DEFAULT_RENDER_PATH
    if not image_path.exists():
        return None, jsonify({"error": f"Image not found: {image_path}"}), 404

    try:
        data = read_json_file(src)
    except Exception as e:
        return None, jsonify({"error": f"Failed to read input JSON: {e}"}), 500

    try:
        image_b64 = encode_image_b64(image_path)
    except Exception as e:
        return None, jsonify({"error": f"Failed to load image: {e}"}), 500

    try:
        updated = call_gpt_json_editor_with_image(model, data, instruction, image_b64)
    except Exception as e:
        return None, jsonify({"error": f"OpenAI call failed: {e}"}), 500

    dst = Path(raw_output).expanduser() if raw_output else src

    try:
        write_json_file(dst, updated)
    except Exception as e:
        return (
            None,
            jsonify({"error": f"Failed to write output JSON: {e}", "path": str(dst)}),
            500,
        )

    return {
        "status": "ok",
        "input": str(src),
        "image": str(image_path),
        "output": str(dst),
        "json": updated,
    }, None, None


def register_routes(app):
    @app.route("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.route("/render_image")
    def render_image():
        path = Path(DEFAULT_RENDER_PATH)
        if not path.exists():
            return jsonify({"error": "Render image not found"}), 404
        return send_file(path)

    @app.route("/latest_input_image")
    def latest_input_image():
        path = Path(LATEST_INPUT_IMAGE)
        if not path.exists():
            return jsonify({"error": "Latest input image not found"}), 404
        return send_file(path)

    @app.route("/runtime_file/<path:relpath>")
    def runtime_file(relpath):
        runtime_root = REAL2SIM_RUNTIME_DIR.resolve()
        target = (runtime_root / relpath).resolve()
        if target != runtime_root and runtime_root not in target.parents:
            return jsonify({"error": "Invalid runtime file path"}), 400
        if not target.exists() or not target.is_file():
            return jsonify({"error": "Runtime file not found"}), 404
        return send_file(target)

    @app.route("/scene_graph")
    def scene_graph():
        if not os.path.exists(SCENE_GRAPH_PATH):
            return jsonify({"error": "Scene graph file not found"}), 404
        return send_file(SCENE_GRAPH_PATH, mimetype="application/json")

    @app.route("/nl_to_scene", methods=["POST"])
    def nl_to_scene():
        payload = request.json or {}
        text = payload.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty input"}), 400

        scene_graph_json = parse_scene_graph_from_text(text)
        _write_scene_graph(scene_graph_json)
        return jsonify(scene_graph_json)

    @app.route("/scene_from_input", methods=["POST"])
    def scene_from_input():
        file = request.files.get("image")
        text = request.form.get("text", "").strip()
        class_names_raw = request.form.get("class_names", "")

        try:
            if file:
                image_bytes = file.read()
                LATEST_INPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
                LATEST_INPUT_IMAGE.write_bytes(image_bytes)
                scene_graph_json = parse_scene_graph_from_image(
                    image_bytes,
                    class_names_raw,
                    text=text,
                )
            else:
                if not text:
                    return jsonify({"error": "Empty input"}), 400
                scene_graph_json = parse_scene_graph_from_text(text, class_names_raw)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        _write_scene_graph(scene_graph_json)
        return jsonify(scene_graph_json)

    @app.route("/edit_json", methods=["POST"])
    def edit_json():
        payload = request.json or {}
        resp, err_json, status = _handle_edit_json(payload)
        if err_json:
            return err_json, status
        return jsonify(resp)

    @app.route("/real2sim/start", methods=["POST"])
    def real2sim_start():
        payload = request.json or {}
        job_id = _start_real2sim_job(payload)
        job = _get_real2sim_job_status(job_id)
        return jsonify(
            {
                "status": "ok",
                "job_id": job_id,
                "log_start_offset": job.get("log_start_offset", 0) if job else 0,
                "log_path": LOG_PATH,
            }
        )

    @app.route("/real2sim/status/<job_id>")
    def real2sim_status(job_id):
        job = _get_real2sim_job_status(job_id)
        if not job:
            return jsonify({"status": "error", "msg": "job not found"}), 404
        return jsonify({"status": "ok", "job": job})

    @app.route("/real2sim/log")
    def real2sim_log():
        try:
            offset = int(request.args.get("offset", 0))
        except (TypeError, ValueError):
            return jsonify({"status": "error", "msg": "offset must be an integer"}), 400

        try:
            limit = int(request.args.get("limit", 65536))
        except (TypeError, ValueError):
            return jsonify({"status": "error", "msg": "limit must be an integer"}), 400

        data = read_real2sim_log(offset=offset, limit=limit)
        return jsonify({"status": "ok", **data, "path": LOG_PATH})
