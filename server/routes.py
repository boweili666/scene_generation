import json
import os
import subprocess
from pathlib import Path

from flask import jsonify, request, send_file, send_from_directory

try:
    from .config import (
        DEFAULT_MODEL,
        DEFAULT_PLACEMENTS_PATH,
        DEFAULT_RENDER_PATH,
        SCENE_GRAPH_PATH,
        WEB_DIR,
    )
    from .openai_service import (
        call_gpt_json_editor_with_image,
        encode_image_b64,
        parse_scene_graph_from_image,
        parse_scene_graph_from_text,
        read_json_file,
        write_json_file,
    )
    from .pipeline_service import (
        log_pipeline_failure,
        run_generate,
        run_real2sim,
        run_real2sim_option2,
    )
except ImportError:
    from config import (
        DEFAULT_MODEL,
        DEFAULT_PLACEMENTS_PATH,
        DEFAULT_RENDER_PATH,
        SCENE_GRAPH_PATH,
        WEB_DIR,
    )
    from openai_service import (
        call_gpt_json_editor_with_image,
        encode_image_b64,
        parse_scene_graph_from_image,
        parse_scene_graph_from_text,
        read_json_file,
        write_json_file,
    )
    from pipeline_service import (
        log_pipeline_failure,
        run_generate,
        run_real2sim,
        run_real2sim_option2,
    )


def _write_scene_graph(scene_graph_json):
    with open(SCENE_GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(scene_graph_json, f, indent=2)


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
                scene_graph_json = parse_scene_graph_from_image(file.read(), class_names_raw)
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

    @app.route("/generate")
    def generate():
        try:
            run_generate()
            return jsonify({"status": "ok"})
        except subprocess.TimeoutExpired:
            return jsonify({"status": "error", "msg": "Isaac Sim timed out"}), 500
        except subprocess.CalledProcessError as e:
            return jsonify(
                {"status": "error", "msg": "Isaac Sim failed", "detail": e.stderr}
            ), 500

    @app.route("/real2sim")
    def real2sim():
        try:
            run_real2sim()
            return jsonify({"status": "ok"})
        except subprocess.TimeoutExpired:
            return jsonify({"status": "error", "msg": "Real2Sim pipeline timed out"}), 500
        except subprocess.CalledProcessError as e:
            log_pipeline_failure(e)
            return jsonify(
                {
                    "status": "error",
                    "msg": "Real2Sim pipeline failed",
                    "detail": e.stderr,
                }
            ), 500

    @app.route("/real2sim_option2", methods=["POST"])
    def real2sim_option2():
        payload = request.json or {}
        try:
            result = run_real2sim_option2(payload)
            return jsonify({"status": "ok", "option": 2, "artifacts": result})
        except ValueError as e:
            return jsonify({"status": "error", "msg": str(e)}), 400
        except subprocess.TimeoutExpired:
            return jsonify({"status": "error", "msg": "Real2Sim option2 timed out"}), 500
        except subprocess.CalledProcessError as e:
            log_pipeline_failure(e)
            return jsonify(
                {
                    "status": "error",
                    "msg": "Real2Sim option2 failed",
                    "detail": e.stderr,
                }
            ), 500
