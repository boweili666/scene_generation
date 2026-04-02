import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.backend.services import runtime_context


class RuntimeContextTest(unittest.TestCase):
    def test_create_session_builds_isolated_run_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            sessions_root = runtime_root / "sessions"
            with (
                mock.patch.object(runtime_context, "RUNTIME_DIR", runtime_root),
                mock.patch.object(runtime_context, "SESSIONS_DIR", sessions_root),
            ):
                context = runtime_context.create_session(session_id="sess_demo", run_id="run_demo")
                self.assertEqual(context.session_id, "sess_demo")
                self.assertEqual(context.run_id, "run_demo")
                self.assertEqual(context.run_root, sessions_root / "sess_demo" / "runs" / "run_demo")
                self.assertTrue(context.scene_graph_path.parent.exists())
                self.assertTrue(context.latest_input_image.parent.exists())
                self.assertTrue(context.default_placements_path.exists())
                self.assertEqual(context.real2sim_log_path.name, "real2sim.log")
                self.assertEqual(context.real2sim_assignment_path.name, "assignment.json")
                self.assertEqual(context.real2sim_poses_path.name, "poses.json")
                self.assertEqual(context.real2sim_object_usd_dir.name, "usd_objects")
                self.assertEqual(
                    (context.session_root / "current_run.txt").read_text(encoding="utf-8").strip(),
                    "run_demo",
                )

    def test_resolve_runtime_context_uses_current_run_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            sessions_root = runtime_root / "sessions"
            with (
                mock.patch.object(runtime_context, "RUNTIME_DIR", runtime_root),
                mock.patch.object(runtime_context, "SESSIONS_DIR", sessions_root),
            ):
                created = runtime_context.create_session(session_id="sess_demo", run_id="run_demo")
                resolved = runtime_context.resolve_runtime_context(session_id="sess_demo")
                self.assertIsNotNone(resolved)
                assert resolved is not None
                self.assertEqual(resolved.session_id, created.session_id)
                self.assertEqual(resolved.run_id, created.run_id)
                self.assertEqual(resolved.scene_graph_path, created.scene_graph_path)


if __name__ == "__main__":
    unittest.main()
