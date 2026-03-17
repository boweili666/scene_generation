from pathlib import Path
import unittest

from app.backend.config.settings import (
    DEFAULT_PLACEMENTS_PATH,
    DEFAULT_RENDER_PATH,
    FRONTEND_DIR,
    LOGS_DIR,
    RUNTIME_DIR,
    REAL2SIM_PREDICT_STREAM_CLIENT,
    REAL2SIM_SEGMENT_SCRIPT,
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
        self.assertEqual(DEFAULT_PLACEMENTS_PATH.parent, RUNTIME_DIR / "scene_service" / "placements")

    def test_real2sim_scripts_use_current_flow(self) -> None:
        self.assertEqual(Path(REAL2SIM_SEGMENT_SCRIPT), Path("pipelines/real2sim/segment_objects.py"))
        self.assertEqual(Path(REAL2SIM_PREDICT_STREAM_CLIENT), Path("pipelines/real2sim/predict_stream_client.py"))


if __name__ == "__main__":
    unittest.main()
