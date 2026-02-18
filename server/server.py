import os

from flask import Flask

try:
    from .config import WEB_DIR
    from .routes import register_routes
except ImportError:
    from config import WEB_DIR
    from routes import register_routes


app = Flask(
    __name__,
    static_folder=os.path.join(WEB_DIR, "assets"),
    static_url_path="/assets",
)
register_routes(app)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
