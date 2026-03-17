from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend.app import app


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
