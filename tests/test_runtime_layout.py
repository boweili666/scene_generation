from pathlib import Path
import unittest

from app.backend.config.settings import (
    DEFAULT_RENDER_PATH,
    FRONTEND_DIR,
    LOGS_DIR,
    RUNTIME_DIR,
    SCENE_GRAPH_PATH,
)


class RuntimeLayoutTest(unittest.TestCase):
    def test_runtime_directories_exist(self) -> None:
        self.assertTrue(RUNTIME_DIR.exists())
        self.assertTrue(LOGS_DIR.exists())
        self.assertTrue(FRONTEND_DIR.exists())

    def test_runtime_paths_are_normalized(self) -> None:
        self.assertEqual(DEFAULT_RENDER_PATH.parent, RUNTIME_DIR / "renders")
        self.assertEqual(Path(SCENE_GRAPH_PATH).parent, RUNTIME_DIR / "scene_graph")


if __name__ == "__main__":
    unittest.main()
