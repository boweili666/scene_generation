import os
import traceback

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from .api.routes import register_routes
from .config import WEB_DIR, ensure_runtime_layout


def create_app() -> Flask:
    ensure_runtime_layout()
    app = Flask(
        __name__,
        static_folder=os.path.join(WEB_DIR, "assets"),
        static_url_path="/assets",
    )
    # Stop the browser from caching the SPA shell or its JS/CSS. Without this,
    # Flask's default Cache-Control: max-age=43200 makes every code change
    # require a manual Ctrl+Shift+R. Set FLASK_STATIC_CACHE_SECONDS to a
    # positive value when serving real users to re-enable caching.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = int(
        os.environ.get("FLASK_STATIC_CACHE_SECONDS", "0")
    )

    @app.after_request
    def _no_cache_frontend(response):  # noqa: ANN001
        req_path = request.path or ""
        if req_path == "/" or req_path.endswith(".html") or req_path.startswith("/assets/"):
            # `no-store` does not evict pre-existing cache entries, so we use
            # `no-cache, must-revalidate, max-age=0` instead. With Flask's
            # built-in ETag/Last-Modified, browsers will always issue a
            # conditional GET and unchanged files come back as cheap 304s.
            response.headers["Cache-Control"] = "no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.errorhandler(Exception)
    def _json_error(exc):  # noqa: ANN001
        # Werkzeug HTTPExceptions already produce sensible responses; let
        # them through for non-API paths but coerce to JSON for API calls.
        if isinstance(exc, HTTPException):
            req_path = request.path or ""
            looks_like_api = req_path != "/" and not req_path.startswith("/assets/") and not req_path.endswith(".html")
            if not looks_like_api:
                return exc
            return jsonify({
                "error": f"{exc.name}: {exc.description}",
                "status_code": exc.code,
            }), exc.code or 500
        # Any other unhandled exception: return JSON instead of Flask's
        # default HTML 500 page so the frontend never has to JSON.parse HTML.
        app.logger.exception("Unhandled exception in %s %s", request.method, request.path)
        return jsonify({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "path": request.path,
            "method": request.method,
        }), 500

    register_routes(app)
    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
