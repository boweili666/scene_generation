import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.backend.services import agent_service, runtime_context


REAL2SIM_SCENE_GRAPH = {
    "scene": {
        "room_type": "office",
        "dimensions": {"length": 5, "width": 4, "height": 3, "unit": "m"},
        "materials": {"floor": "wood", "walls": "paint"},
    },
    "obj": {
        "/World/table_0": {"id": 0, "class": "table", "caption": "table", "source": "real2sim"},
    },
    "edges": {"obj-obj": [], "obj-wall": []},
}


class AgentServiceTest(unittest.TestCase):
    def _create_context(self) -> runtime_context.RuntimeContext:
        self.tmpdir = tempfile.TemporaryDirectory()
        runtime_root = Path(self.tmpdir.name) / "runtime"
        sessions_root = runtime_root / "sessions"
        self.addCleanup(self.tmpdir.cleanup)
        with (
            mock.patch.object(runtime_context, "RUNTIME_DIR", runtime_root),
            mock.patch.object(runtime_context, "SESSIONS_DIR", sessions_root),
        ):
            context = runtime_context.create_session(session_id="sess_demo")
        return context

    def test_sync_real2sim_job_to_session_records_artifacts(self) -> None:
        context = self._create_context()
        context.real2sim_scene_results_dir.mkdir(parents=True, exist_ok=True)
        context.real2sim_assignment_path.write_text(json.dumps({"matches": []}), encoding="utf-8")
        context.real2sim_poses_path.write_text(json.dumps({"obj_00": {}}), encoding="utf-8")
        context.real2sim_manifest_path.write_text(json.dumps({"objects": {}}), encoding="utf-8")
        context.real2sim_objects_dir.mkdir(parents=True, exist_ok=True)
        context.real2sim_object_usd_dir.mkdir(parents=True, exist_ok=True)
        (context.real2sim_objects_dir / "obj_00.glb").write_bytes(b"glb")
        (context.real2sim_object_usd_dir / "obj_00.usd").write_bytes(b"usd")

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
        ):
            snapshot = agent_service.sync_real2sim_job_to_session(
                {
                    "job_id": "job123",
                    "status": "succeeded",
                    "payload": {"session_id": context.session_id},
                    "artifacts": {
                        "real2sim_root_dir": str(context.session_root),
                        "scene_results_dir": str(context.real2sim_scene_results_dir),
                        "assignment_json": "real2sim/scene_results/assignment.json",
                        "poses_json": "real2sim/scene_results/poses.json",
                        "manifest_json": "real2sim/scene_results/real2sim_asset_manifest.json",
                        "object_glbs": ["real2sim/scene_results/objects/obj_00.glb"],
                        "object_usds": ["real2sim/scene_results/usd_objects/obj_00.usd"],
                    },
                }
            )

        self.assertIsNotNone(snapshot)
        current_run = snapshot["current_run"]
        self.assertEqual(current_run["real2sim"]["status"], "succeeded")
        artifacts = current_run["real2sim"]["artifacts"]
        self.assertEqual(artifacts["assignment_json_path"], str(context.real2sim_assignment_path))
        self.assertEqual(artifacts["poses_json_path"], str(context.real2sim_poses_path))
        self.assertEqual(artifacts["manifest_json_path"], str(context.real2sim_manifest_path))
        self.assertEqual(artifacts["object_glb_paths"], [str(context.real2sim_objects_dir / "obj_00.glb")])
        self.assertEqual(artifacts["object_usd_paths"], [str(context.real2sim_object_usd_dir / "obj_00.usd")])

    def test_sync_real2sim_job_to_session_records_error_info(self) -> None:
        context = self._create_context()

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
        ):
            snapshot = agent_service.sync_real2sim_job_to_session(
                {
                    "job_id": "job_failed",
                    "status": "failed",
                    "error": "Connection refused",
                    "error_info": {
                        "code": "remote_server_unavailable",
                        "category": "remote",
                        "step": "remote_predict",
                        "retryable": True,
                        "user_message": "The remote SAM3D service is unreachable.",
                        "technical_detail": "Connection refused",
                    },
                    "payload": {"session_id": context.session_id},
                    "artifacts": {},
                }
            )

        self.assertIsNotNone(snapshot)
        current_run = snapshot["current_run"]
        self.assertEqual(current_run["real2sim"]["status"], "failed")
        self.assertEqual(current_run["real2sim"]["error_info"]["code"], "remote_server_unavailable")
        self.assertEqual(snapshot["history"][-1]["role"], "assistant")
        self.assertIn("remote SAM3D service is unreachable", snapshot["history"][-1]["content"])

    def test_get_agent_state_response_restores_pending_question_and_outputs(self) -> None:
        context = self._create_context()
        context.scene_graph_path.write_text(json.dumps(REAL2SIM_SCENE_GRAPH), encoding="utf-8")
        context.render_path.write_bytes(b"png")
        context.scene_service_usd_path.parent.mkdir(parents=True, exist_ok=True)
        context.scene_service_usd_path.write_text("#usda 1.0\n", encoding="utf-8")
        context.default_placements_path.write_text(json.dumps({"/World/table_0": {"x": 0.0, "y": 0.0, "z": 0.5}}), encoding="utf-8")

        state_path = context.session_root / "agent_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "session_id": context.session_id,
                    "current_state": "await_layout_strategy",
                    "last_intent": "generate_scene",
                    "last_completed_state": "generate_scene",
                    "last_decision": {"reason": "Need a layout strategy before generating the scene."},
                    "pending_question": {
                        "type": "layout_strategy",
                        "question": "Choose joint or lock_real2sim.",
                        "options": [{"id": "joint", "label": "Joint"}],
                        "scene_endpoint": "scene_new",
                    },
                    "scene_generation": {
                        "status": "succeeded",
                        "outputs": {
                            "saved_usd": str(context.scene_service_usd_path),
                            "placements_path": str(context.default_placements_path),
                            "screenshot_path": str(context.render_path),
                            "debug": {"resample_mode": "joint"},
                        },
                    },
                    "real2sim": {
                        "status": "running",
                        "job_id": "job123",
                        "log_path": str(context.real2sim_log_path),
                        "log_start_offset": 12,
                    },
                    "history": [{"role": "assistant", "content": "Choose joint or lock_real2sim."}],
                }
            ),
            encoding="utf-8",
        )

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
        ):
            result = agent_service.get_agent_state_response(
                session_id=context.session_id,
            )

        self.assertEqual(result["agent"]["state"], "await_layout_strategy")
        self.assertEqual(result["agent"]["question"], "Choose joint or lock_real2sim.")
        self.assertEqual(result["scene_result"]["saved_usd"], str(context.scene_service_usd_path))
        self.assertEqual(result["real2sim_job"]["job_id"], "job123")
        self.assertEqual(result["real2sim_job"]["log_path"], str(context.real2sim_log_path))
        self.assertEqual(result["session_state"]["history"][0]["content"], "Choose joint or lock_real2sim.")

    def test_get_agent_state_response_refreshes_live_real2sim_artifacts_from_disk(self) -> None:
        context = self._create_context()
        runtime_root = Path(self.tmpdir.name) / "runtime"
        context.real2sim_scene_results_dir.mkdir(parents=True, exist_ok=True)
        context.real2sim_objects_dir.mkdir(parents=True, exist_ok=True)
        context.real2sim_assignment_path.write_text(
            json.dumps({"assignments": [{"mask_label": 1, "scene_path": "/World/table_0", "output_name": "1"}]}),
            encoding="utf-8",
        )
        context.real2sim_manifest_path.write_text(json.dumps({"objects": {}}), encoding="utf-8")
        (context.real2sim_objects_dir / "1.glb").write_bytes(b"glb")

        state_path = context.session_root / "agent_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "session_id": context.session_id,
                    "current_state": "run_real2sim",
                    "last_intent": "run_real2sim",
                    "real2sim": {
                        "status": "running",
                        "job_id": "job_live",
                        "log_path": str(context.real2sim_log_path),
                        "log_start_offset": 7,
                    },
                    "history": [],
                }
            ),
            encoding="utf-8",
        )

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
            mock.patch.object(agent_service, "RUNTIME_DIR", runtime_root),
        ):
            result = agent_service.get_agent_state_response(
                session_id=context.session_id,
            )

        self.assertEqual(result["real2sim_job"]["job_id"], "job_live")
        artifacts = result["session_state"]["current_run"]["real2sim"]["artifacts"]
        self.assertEqual(artifacts["assignment_json_path"], str(context.real2sim_assignment_path))
        self.assertEqual(artifacts["manifest_json_path"], str(context.real2sim_manifest_path))
        self.assertEqual(artifacts["object_glb_paths"], [str(context.real2sim_objects_dir / "1.glb")])
        self.assertTrue(str(artifacts["assignment_json_url"]).startswith("/runtime_file/"))
        self.assertEqual(len(artifacts["object_glb_urls"]), 1)
        self.assertTrue(str(artifacts["object_glb_urls"][0]).startswith("/runtime_file/"))


if __name__ == "__main__":
    unittest.main()
