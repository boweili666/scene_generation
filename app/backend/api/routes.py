import json
from pathlib import Path

from flask import jsonify, request, send_file, send_from_directory

from ..config import (
    DEFAULT_MODEL,
    DEFAULT_PLACEMENTS_PATH,
    DEFAULT_RENDER_PATH,
    LATEST_INPUT_IMAGE,
    LOG_PATH,
    REAL2SIM_MESH_OUTPUT_DIR,
    REAL2SIM_MASK_OUTPUT_DIR,
    REAL2SIM_ROOT_DIR,
    REAL2SIM_REUSE_MESH_DIR,
    REAL2SIM_SCENE_RESULTS_DIR,
    RUNTIME_DIR,
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
from ..services.agent_service import get_agent_state_response, handle_agent_message, sync_real2sim_job_to_session
from ..services.instruction_service import apply_instruction as apply_scene_instruction
from ..services.instruction_service import save_scene_graph_state
from ..services.pipeline_service import (
    get_real2sim_job_status as get_real2sim_job_status_raw,
    read_real2sim_log,
    start_real2sim_job,
)
from ..services.real2sim_review_service import load_assignment_review, save_assignment_review
from ..services.runtime_context import (
    create_run,
    create_session,
    resolve_runtime_context,
)

def _request_value(name: str) -> str | None:
    query_value = request.args.get(name)
    if isinstance(query_value, str) and query_value.strip():
        return query_value.strip()

    form_value = request.form.get(name)
    if isinstance(form_value, str) and form_value.strip():
        return form_value.strip()

    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        json_value = payload.get(name)
        if isinstance(json_value, str) and json_value.strip():
            return json_value.strip()
    return None


def _resolve_request_runtime_context(*, create: bool = False):
    session_id = _request_value("session_id")
    run_id = _request_value("run_id")
    return resolve_runtime_context(session_id=session_id, run_id=run_id, create=create)


def _request_runtime_paths(*, create: bool = False) -> dict[str, object]:
    context = _resolve_request_runtime_context(create=create)
    if context is None:
        return {
            "context": None,
            "runtime_root": RUNTIME_DIR.resolve(),
            "latest_input_image": Path(LATEST_INPUT_IMAGE),
            "scene_graph_path": Path(SCENE_GRAPH_PATH),
            "render_path": Path(DEFAULT_RENDER_PATH),
            "placements_path": Path(DEFAULT_PLACEMENTS_PATH),
            "real2sim_root_dir": Path(REAL2SIM_ROOT_DIR),
            "real2sim_mask_output_dir": Path(REAL2SIM_MASK_OUTPUT_DIR),
            "real2sim_mesh_output_dir": Path(REAL2SIM_MESH_OUTPUT_DIR),
            "real2sim_reuse_mesh_dir": Path(REAL2SIM_REUSE_MESH_DIR),
            "real2sim_scene_results_dir": Path(REAL2SIM_SCENE_RESULTS_DIR),
            "real2sim_log_path": Path(LOG_PATH),
        }

    return {
        "context": context,
        "runtime_root": context.runtime_root.resolve(),
        "latest_input_image": context.latest_input_image,
        "scene_graph_path": context.scene_graph_path,
        "render_path": context.render_path,
        "placements_path": context.default_placements_path,
        "real2sim_root_dir": context.run_root,
        "real2sim_mask_output_dir": context.real2sim_masks_dir,
        "real2sim_mesh_output_dir": context.real2sim_meshes_dir,
        "real2sim_reuse_mesh_dir": context.real2sim_meshes_dir,
        "real2sim_scene_results_dir": context.real2sim_scene_results_dir,
        "real2sim_log_path": context.real2sim_log_path,
    }


def _to_runtime_file_url(abs_path: Path) -> str | None:
    runtime_root = RUNTIME_DIR.resolve()
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

    object_usd_urls: list[str] = []
    for rel in artifacts.get("object_usds", []):
        p = Path(artifacts["real2sim_root_dir"]) / rel
        url = _to_runtime_file_url(p)
        if url:
            object_usd_urls.append(url)
    result["object_usd_urls"] = object_usd_urls

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

    assignment_json = artifacts.get("assignment_json")
    if assignment_json:
        assignment_url = _to_runtime_file_url(Path(artifacts["real2sim_root_dir"]) / assignment_json)
        result["assignment_json_url"] = assignment_url
    else:
        result["assignment_json_url"] = None

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


def _annotate_assignment_review_urls(review: dict | None) -> dict | None:
    if not isinstance(review, dict):
        return None

    result = dict(review)
    overlay_abs = review.get("overlay_image_abs_path")
    review_abs = review.get("review_image_abs_path")
    manifest_path = review.get("manifest_path")
    result["overlay_image_url"] = _to_runtime_file_url(Path(overlay_abs)) if overlay_abs else None
    result["review_image_url"] = _to_runtime_file_url(Path(review_abs)) if review_abs else None
    result["manifest_url"] = _to_runtime_file_url(Path(manifest_path)) if manifest_path else None

    rows: list[dict] = []
    for row in review.get("mask_labels", []):
        if not isinstance(row, dict):
            continue
        next_row = dict(row)
        mask_path = row.get("mask_path")
        next_row["mask_url"] = _to_runtime_file_url(Path(mask_path)) if mask_path else None
        rows.append(next_row)
    result["mask_labels"] = rows
    return result


def _start_real2sim_job(payload: dict) -> dict[str, object]:
    return start_real2sim_job(payload)


def _get_real2sim_job_status(job_id: str) -> dict | None:
    raw_job = get_real2sim_job_status_raw(job_id)
    if raw_job is None:
        return None
    session_state = sync_real2sim_job_to_session(raw_job)
    job = dict(raw_job)
    job["artifacts"] = _to_artifact_urls(job.get("artifacts") or {})
    if session_state is not None:
        job["session_state"] = session_state
    return job


def _write_scene_graph(scene_graph_json, *, scene_graph_path: str | Path = SCENE_GRAPH_PATH):
    write_json_file(Path(scene_graph_path), scene_graph_json)


def _handle_edit_json(payload, *, default_input_path: str | Path, default_image_path: str | Path):
    instruction = (payload.get("instruction") or "").strip()
    if not instruction:
        return None, jsonify({"error": "instruction is required"}), 400

    raw_input = payload.get("input")
    raw_output = payload.get("output")
    raw_image = payload.get("image")
    model = payload.get("model") or DEFAULT_MODEL

    src = Path(raw_input).expanduser() if raw_input else Path(default_input_path)
    if not src.exists():
        return None, jsonify({"error": f"Input JSON not found: {src}"}), 404

    image_path = Path(raw_image).expanduser() if raw_image else Path(default_image_path)
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
    @app.route("/sessions", methods=["POST"])
    def session_create():
        payload = request.get_json(silent=True) or {}
        try:
            context = create_session(
                session_id=payload.get("session_id"),
                run_id=payload.get("run_id"),
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"status": "ok", "context": context.to_dict()})

    @app.route("/sessions/<session_id>/runs", methods=["POST"])
    def session_run_create(session_id):
        payload = request.get_json(silent=True) or {}
        try:
            context = create_run(session_id, run_id=payload.get("run_id"))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"status": "ok", "context": context.to_dict()})

    @app.route("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.route("/render_image")
    def render_image():
        paths = _request_runtime_paths()
        path = Path(paths["render_path"])
        if not path.exists():
            return jsonify({"error": "Render image not found"}), 404
        return send_file(path)

    @app.route("/latest_input_image")
    def latest_input_image():
        paths = _request_runtime_paths()
        path = Path(paths["latest_input_image"])
        if not path.exists():
            return jsonify({"error": "Latest input image not found"}), 404
        return send_file(path)

    @app.route("/runtime_file/<path:relpath>")
    def runtime_file(relpath):
        runtime_root = RUNTIME_DIR.resolve()
        target = (runtime_root / relpath).resolve()
        if target != runtime_root and runtime_root not in target.parents:
            return jsonify({"error": "Invalid runtime file path"}), 400
        if not target.exists() or not target.is_file():
            return jsonify({"error": "Runtime file not found"}), 404
        return send_file(target)

    @app.route("/scene_graph", methods=["GET", "POST"])
    def scene_graph():
        paths = _request_runtime_paths(create=request.method == "POST")
        if request.method == "POST":
            payload = request.json or {}
            scene_graph_json = payload.get("scene_graph") if "scene_graph" in payload else payload
            try:
                result = save_scene_graph_state(
                    scene_graph_json,
                    scene_graph_path=paths["scene_graph_path"],
                    placements_path=paths["placements_path"],
                )
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            context = paths.get("context")
            if context is not None:
                result["session_id"] = context.session_id
                result["run_id"] = context.run_id
            return jsonify(result)
        scene_graph_path = Path(paths["scene_graph_path"])
        if not scene_graph_path.exists():
            return jsonify({"error": "Scene graph file not found"}), 404
        return send_file(scene_graph_path, mimetype="application/json")

    @app.route("/nl_to_scene", methods=["POST"])
    def nl_to_scene():
        paths = _request_runtime_paths(create=True)
        payload = request.json or {}
        text = payload.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty input"}), 400

        scene_graph_json = parse_scene_graph_from_text(text)
        _write_scene_graph(scene_graph_json, scene_graph_path=paths["scene_graph_path"])
        response = dict(scene_graph_json)
        context = paths.get("context")
        if context is not None:
            response["_session"] = {"session_id": context.session_id, "run_id": context.run_id}
        return jsonify(response)

    @app.route("/scene_from_input", methods=["POST"])
    def scene_from_input():
        paths = _request_runtime_paths(create=True)
        file = request.files.get("image")
        text = request.form.get("text", "").strip()
        class_names_raw = request.form.get("class_names", "")

        try:
            if file:
                image_bytes = file.read()
                latest_input_path = Path(paths["latest_input_image"])
                latest_input_path.parent.mkdir(parents=True, exist_ok=True)
                latest_input_path.write_bytes(image_bytes)
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

        _write_scene_graph(scene_graph_json, scene_graph_path=paths["scene_graph_path"])
        response = dict(scene_graph_json)
        context = paths.get("context")
        if context is not None:
            response["_session"] = {"session_id": context.session_id, "run_id": context.run_id}
        return jsonify(response)

    @app.route("/apply_instruction", methods=["POST"])
    def apply_instruction():
        paths = _request_runtime_paths(create=True)
        image_bytes = None
        class_names_raw = ""
        text = ""

        if request.content_type and request.content_type.startswith("multipart/form-data"):
            file = request.files.get("image")
            text = request.form.get("text", "").strip()
            class_names_raw = request.form.get("class_names", "")
            if file:
                image_bytes = file.read()
        else:
            payload = request.json or {}
            text = (payload.get("text") or payload.get("instruction") or "").strip()
            class_names_raw = payload.get("class_names", "")

        try:
            result = apply_scene_instruction(
                text,
                class_names_raw=class_names_raw,
                image_bytes=image_bytes,
                scene_graph_path=paths["scene_graph_path"],
                placements_path=paths["placements_path"],
                latest_input_image_path=paths["latest_input_image"],
                render_path=paths["render_path"],
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500

        context = paths.get("context")
        if context is not None:
            result["session_id"] = context.session_id
            result["run_id"] = context.run_id
        return jsonify(result)

    @app.route("/agent/message", methods=["POST"])
    def agent_message():
        payload = request.get_json(silent=True) if not (request.content_type or "").startswith("multipart/form-data") else None
        text = ""
        class_names_raw = ""
        image_bytes = None
        action = None
        resample_mode = None
        scene_endpoint = None

        if payload is not None:
            text = (payload.get("text") or payload.get("instruction") or "").strip()
            class_names_raw = payload.get("class_names", "") or ""
            action = payload.get("action")
            resample_mode = payload.get("resample_mode")
            scene_endpoint = payload.get("scene_endpoint")
        else:
            text = (request.form.get("text") or request.form.get("instruction") or "").strip()
            class_names_raw = request.form.get("class_names", "") or ""
            action = request.form.get("action")
            resample_mode = request.form.get("resample_mode")
            scene_endpoint = request.form.get("scene_endpoint")
            file = request.files.get("image")
            if file:
                image_bytes = file.read()

        try:
            context = _resolve_request_runtime_context(create=True)
            result = handle_agent_message(
                session_id=context.session_id if context is not None else None,
                run_id=context.run_id if context is not None else None,
                text=text,
                image_bytes=image_bytes,
                class_names_raw=class_names_raw,
                action=action,
                resample_mode=resample_mode,
                scene_endpoint=scene_endpoint,
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 502

        return jsonify(result)

    @app.route("/agent/state", methods=["GET"])
    def agent_state():
        try:
            context = _resolve_request_runtime_context(create=True)
            result = get_agent_state_response(
                session_id=context.session_id if context is not None else None,
                run_id=context.run_id if context is not None else None,
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 502

        return jsonify(result)

    @app.route("/edit_json", methods=["POST"])
    def edit_json():
        paths = _request_runtime_paths()
        payload = request.json or {}
        resp, err_json, status = _handle_edit_json(
            payload,
            default_input_path=paths["placements_path"],
            default_image_path=paths["render_path"],
        )
        if err_json:
            return err_json, status
        return jsonify(resp)

    @app.route("/real2sim/start", methods=["POST"])
    def real2sim_start():
        paths = _request_runtime_paths(create=True)
        payload = dict(request.json or {})
        payload.setdefault("image_path", str(paths["latest_input_image"]))
        payload.setdefault("scene_graph_path", str(paths["scene_graph_path"]))
        payload.setdefault("real2sim_root_dir", str(paths["real2sim_root_dir"]))
        payload.setdefault("mask_output", str(paths["real2sim_mask_output_dir"]))
        payload.setdefault("mesh_output_dir", str(paths["real2sim_mesh_output_dir"]))
        payload.setdefault("reuse_mesh_dir", str(paths["real2sim_reuse_mesh_dir"]))
        payload.setdefault("scene_results_dir", str(paths["real2sim_scene_results_dir"]))
        payload.setdefault("log_path", str(paths["real2sim_log_path"]))
        context = paths.get("context")
        if context is not None:
            payload.setdefault("session_id", context.session_id)
            payload.setdefault("run_id", context.run_id)
        job_info = _start_real2sim_job(payload)
        job_id = str(job_info["job_id"])
        job = _get_real2sim_job_status(job_id)
        response = {
            "status": "ok",
            "job_id": job_id,
            "log_start_offset": int(job_info.get("log_start_offset", 0) or 0),
            "log_path": str(job_info.get("log_path") or paths["real2sim_log_path"]),
        }
        if context is not None:
            response["session_id"] = context.session_id
            response["run_id"] = context.run_id
        return jsonify(response)

    @app.route("/real2sim/status/<job_id>")
    def real2sim_status(job_id):
        job = _get_real2sim_job_status(job_id)
        if not job:
            return jsonify({"status": "error", "msg": "job not found"}), 404
        return jsonify({"status": "ok", "job": job})

    @app.route("/real2sim/log")
    def real2sim_log():
        paths = _request_runtime_paths()
        try:
            offset = int(request.args.get("offset", 0))
        except (TypeError, ValueError):
            return jsonify({"status": "error", "msg": "offset must be an integer"}), 400

        try:
            limit = int(request.args.get("limit", 65536))
        except (TypeError, ValueError):
            return jsonify({"status": "error", "msg": "limit must be an integer"}), 400

        log_path = str(paths["real2sim_log_path"])
        data = read_real2sim_log(offset=offset, limit=limit, log_path=log_path)
        return jsonify({"status": "ok", **data, "path": log_path})

    @app.route("/real2sim/assignment", methods=["GET", "POST"])
    def real2sim_assignment():
        paths = _request_runtime_paths(create=request.method == "POST")
        context = paths.get("context")

        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            try:
                review = save_assignment_review(
                    assignments=payload.get("assignments") or [],
                    scene_graph_path=paths["scene_graph_path"],
                    masks_dir=paths["real2sim_mask_output_dir"],
                    results_dir=paths["real2sim_scene_results_dir"],
                    latest_input_image=paths["latest_input_image"],
                )
            except (FileNotFoundError, ValueError) as e:
                return jsonify({"error": str(e)}), 400
            except RuntimeError as e:
                return jsonify({"error": str(e)}), 500

            response = {
                "status": "ok",
                "review": _annotate_assignment_review_urls(review),
            }
            if context is not None:
                response["session_id"] = context.session_id
                response["run_id"] = context.run_id
            return jsonify(response)

        try:
            review = load_assignment_review(
                scene_graph_path=paths["scene_graph_path"],
                masks_dir=paths["real2sim_mask_output_dir"],
                results_dir=paths["real2sim_scene_results_dir"],
                latest_input_image=paths["latest_input_image"],
            )
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500

        response = {
            "status": "ok",
            "review": _annotate_assignment_review_urls(review),
        }
        if context is not None:
            response["session_id"] = context.session_id
            response["run_id"] = context.run_id
        return jsonify(response)
