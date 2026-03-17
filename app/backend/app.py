import os

from flask import Flask

from .api.routes import register_routes
from .config import WEB_DIR, ensure_runtime_layout


def create_app() -> Flask:
    ensure_runtime_layout()
    app = Flask(
        __name__,
        static_folder=os.path.join(WEB_DIR, "assets"),
        static_url_path="/assets",
    )
    register_routes(app)
    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
