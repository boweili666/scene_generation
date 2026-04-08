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
    def _react(
        self,
        action: str,
        *,
        confidence: float = 0.95,
        instruction: str | None = None,
        scene_endpoint: str | None = None,
        resample_mode: str | None = None,
        question: str | None = None,
        clarification_kind: str = "none",
        reason: str = "Mocked ReAct step.",
    ) -> dict:
        return {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "instruction": instruction,
            "scene_endpoint": scene_endpoint,
            "resample_mode": resample_mode,
            "question": question,
            "clarification_kind": clarification_kind,
        }

    def _create_context(self) -> runtime_context.RuntimeContext:
        self.tmpdir = tempfile.TemporaryDirectory()
        runtime_root = Path(self.tmpdir.name) / "runtime"
        sessions_root = runtime_root / "sessions"
        self.addCleanup(self.tmpdir.cleanup)
        with (
            mock.patch.object(runtime_context, "RUNTIME_DIR", runtime_root),
            mock.patch.object(runtime_context, "SESSIONS_DIR", sessions_root),
        ):
            context = runtime_context.create_session(session_id="sess_demo", run_id="run_demo")
        return context

    def test_generate_scene_without_strategy_returns_question(self) -> None:
        context = self._create_context()
        context.scene_graph_path.write_text(json.dumps(REAL2SIM_SCENE_GRAPH), encoding="utf-8")

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
            mock.patch.object(
                agent_service,
                "react_agent_step",
                side_effect=[
                    self._react("generate_scene", scene_endpoint="scene_new"),
                    self._react(
                        "ask_clarification",
                        question="Choose joint or lock_real2sim.",
                        clarification_kind="layout_strategy",
                        reason="A graph with real2sim objects needs an explicit layout strategy.",
                    ),
                ],
            ),
        ):
            result = agent_service.handle_agent_message(
                session_id=context.session_id,
                run_id=context.run_id,
                text="generate scene",
            )

        self.assertEqual(result["agent"]["state"], "await_layout_strategy")
        self.assertEqual(result["agent"]["intent"], "generate_scene")
        self.assertIn("layout strategy", result["agent"]["message"])
        self.assertTrue(result["agent"]["decision"]["requires_clarification"])
        self.assertEqual(len(result["agent"]["options"]), 2)

    def test_answering_layout_strategy_continues_scene_generation(self) -> None:
        context = self._create_context()
        context.scene_graph_path.write_text(json.dumps(REAL2SIM_SCENE_GRAPH), encoding="utf-8")
        context.render_path.write_bytes(b"png")
        context.default_placements_path.write_text(json.dumps({"/World/table_0": {"x": 0.0, "y": 0.0, "z": 0.5}}), encoding="utf-8")

        state_path = context.session_root / "agent_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "session_id": context.session_id,
                    "current_run_id": context.run_id,
                    "pending_question": {
                        "type": "layout_strategy",
                        "question": "Choose strategy",
                        "scene_endpoint": "scene_new",
                        "run_id": context.run_id,
                    },
                    "history": [],
                }
            ),
            encoding="utf-8",
        )

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
            mock.patch.object(
                agent_service,
                "react_agent_step",
                return_value=self._react(
                    "generate_scene",
                    scene_endpoint="scene_new",
                    resample_mode="lock_real2sim",
                ),
            ),
            mock.patch.object(
                agent_service,
                "_run_scene_service",
                return_value={
                    "saved_usd": str(context.scene_service_usd_path),
                    "placements": {"/World/table_0": {"x": 0.0, "y": 0.0, "z": 0.5}},
                    "debug": {"resample_mode": "lock_real2sim"},
                    "resample_mode": "lock_real2sim",
                },
            ) as run_scene_mock,
        ):
            result = agent_service.handle_agent_message(
                session_id=context.session_id,
                run_id=context.run_id,
                text="lock real2sim",
            )

        self.assertEqual(result["agent"]["state"], "completed")
        self.assertEqual(result["agent"]["intent"], "generate_scene")
        self.assertEqual(result["agent"]["completed_state"], "generate_scene")
        self.assertEqual(result["scene_result"]["saved_usd"], str(context.scene_service_usd_path))
        self.assertEqual(result["scene_result"]["render_image_path"], str(context.render_path))
        self.assertEqual(result["scene_result"]["placements_path"], str(context.default_placements_path))
        self.assertIn("scene_generation", result["session_state"]["current_run"])
        run_scene_mock.assert_called_once()

    def test_generate_scene_repairs_layout_conflict_by_falling_back_to_joint(self) -> None:
        context = self._create_context()
        context.scene_graph_path.write_text(json.dumps(REAL2SIM_SCENE_GRAPH), encoding="utf-8")
        context.render_path.write_bytes(b"png")
        context.default_placements_path.write_text(
            json.dumps({"/World/table_0": {"x": 0.0, "y": 0.0, "z": 0.5}}),
            encoding="utf-8",
        )

        layout_error = RuntimeError("Scene service failed: failed to generate a collision-free layout after all attempts")
        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
            mock.patch.object(
                agent_service,
                "react_agent_step",
                return_value=self._react(
                    "generate_scene",
                    scene_endpoint="scene_new",
                    resample_mode="lock_real2sim",
                ),
            ),
            mock.patch.object(
                agent_service,
                "_run_scene_service",
                side_effect=[
                    layout_error,
                    layout_error,
                    {
                        "saved_usd": str(context.scene_service_usd_path),
                        "placements": {"/World/table_0": {"x": 0.0, "y": 0.0, "z": 0.5}},
                        "debug": {"resample_mode": "joint"},
                        "resample_mode": "joint",
                    },
                ],
            ) as run_scene_mock,
        ):
            result = agent_service.handle_agent_message(
                session_id=context.session_id,
                run_id=context.run_id,
                text="generate scene with lock real2sim",
            )

        self.assertEqual(result["agent"]["state"], "completed")
        self.assertEqual(result["scene_result"]["resample_mode"], "joint")
        self.assertTrue(any("retried automatically" in warning.lower() for warning in result.get("warnings", [])))
        self.assertEqual(run_scene_mock.call_count, 3)
        self.assertEqual(run_scene_mock.call_args_list[0].kwargs["resample_mode"], "lock_real2sim")
        self.assertEqual(run_scene_mock.call_args_list[2].kwargs["resample_mode"], "joint")

    def test_run_real2sim_starts_background_job(self) -> None:
        context = self._create_context()
        context.scene_graph_path.write_text(json.dumps(REAL2SIM_SCENE_GRAPH), encoding="utf-8")

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
            mock.patch.object(
                agent_service,
                "react_agent_step",
                return_value=self._react("run_real2sim"),
            ),
            mock.patch.object(
                agent_service,
                "start_real2sim_job",
                return_value={"job_id": "job123", "log_start_offset": 0, "log_path": "real2sim.log"},
            ) as start_job_mock,
        ):
            result = agent_service.handle_agent_message(
                session_id=context.session_id,
                run_id=context.run_id,
                text="run real2sim",
            )

        self.assertEqual(result["agent"]["state"], "run_real2sim")
        self.assertEqual(result["agent"]["outcome"], "started_job")
        self.assertEqual(result["real2sim_job"]["job_id"], "job123")
        start_job_mock.assert_called_once()

    def test_ambiguous_graph_request_requires_create_or_edit_clarification(self) -> None:
        context = self._create_context()
        context.scene_graph_path.write_text(json.dumps(REAL2SIM_SCENE_GRAPH), encoding="utf-8")

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
            mock.patch.object(
                agent_service,
                "react_agent_step",
                return_value=self._react(
                    "ask_clarification",
                    confidence=0.92,
                    question="Should I create a new scene graph from this input, or edit the current scene graph?",
                    clarification_kind="graph_mode",
                    reason="Fresh room description could mean replace or edit.",
                ),
            ),
        ):
            result = agent_service.handle_agent_message(
                session_id=context.session_id,
                run_id=context.run_id,
                text="A room with a table and a chair.",
            )

        self.assertEqual(result["agent"]["state"], "needs_clarification")
        self.assertEqual(result["agent"]["intent"], "understand_request")
        self.assertIn("create a new scene graph", result["agent"]["question"].lower())
        option_actions = {option["action"] for option in result["agent"]["options"]}
        self.assertEqual(option_actions, {"create_scene_graph", "edit_scene_graph"})

    def test_low_confidence_top_level_route_returns_generic_clarification(self) -> None:
        context = self._create_context()
        context.scene_graph_path.write_text(json.dumps(REAL2SIM_SCENE_GRAPH), encoding="utf-8")
        context.latest_input_image.write_bytes(b"fake-image-bytes")

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
            mock.patch.object(
                agent_service,
                "react_agent_step",
                return_value=self._react(
                    "generate_scene",
                    confidence=0.42,
                    scene_endpoint="scene_new",
                    reason="The request could mean editing the graph or generating the scene.",
                    clarification_kind="intent",
                ),
            ),
        ):
            result = agent_service.handle_agent_message(
                session_id=context.session_id,
                run_id=context.run_id,
                text="handle this scene",
            )

        self.assertEqual(result["agent"]["state"], "needs_clarification")
        self.assertEqual(result["agent"]["intent"], "understand_request")
        self.assertIn("confidence", result["agent"]["reason"].lower())
        option_actions = {option["action"] for option in result["agent"]["options"]}
        self.assertIn("edit_scene_graph", option_actions)
        self.assertIn("run_real2sim", option_actions)
        self.assertIn("generate_scene", option_actions)

    def test_run_real2sim_without_scene_graph_returns_bootstrap_question(self) -> None:
        context = self._create_context()
        context.latest_input_image.write_bytes(b"fake-image-bytes")

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
            mock.patch.object(
                agent_service,
                "react_agent_step",
                side_effect=[
                    self._react("run_real2sim"),
                    self._react(
                        "ask_clarification",
                        question="Should I create a new scene graph from the current image first?",
                        clarification_kind="intent",
                        reason="Real2Sim is blocked until a scene graph exists.",
                    ),
                ],
            ),
        ):
            result = agent_service.handle_agent_message(
                session_id=context.session_id,
                run_id=context.run_id,
                text="run real2sim",
            )

        self.assertEqual(result["agent"]["state"], "needs_clarification")
        self.assertEqual(result["agent"]["intent"], "understand_request")
        self.assertIn("scene graph", result["agent"]["question"].lower())
        option_actions = {option["action"] for option in result["agent"]["options"]}
        self.assertIn("create_scene_graph", option_actions)

    def test_explicit_create_scene_graph_reuses_saved_image(self) -> None:
        context = self._create_context()
        context.scene_graph_path.write_text(json.dumps(REAL2SIM_SCENE_GRAPH), encoding="utf-8")
        context.latest_input_image.write_bytes(b"saved-image-bytes")

        create_result = {
            "status": "ok",
            "assistant_message": "Created a new scene graph from the current input.",
            "scene_graph": {"scene": {}, "obj": {}, "edges": {"obj-obj": [], "obj-wall": []}},
            "placements": {},
            "warnings": [],
        }

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
            mock.patch.object(agent_service, "create_scene_graph_from_input", return_value=create_result) as create_mock,
        ):
            result = agent_service.handle_agent_message(
                session_id=context.session_id,
                run_id=context.run_id,
                text="",
                action="create_scene_graph",
            )

        self.assertEqual(result["agent"]["state"], "completed")
        self.assertEqual(result["agent"]["completed_state"], "create_scene_graph")
        create_mock.assert_called_once()
        self.assertTrue(create_mock.call_args.kwargs["reuse_saved_image"])

    def test_explicit_edit_scene_graph_uses_apply_instruction(self) -> None:
        context = self._create_context()
        context.scene_graph_path.write_text(json.dumps(REAL2SIM_SCENE_GRAPH), encoding="utf-8")

        edit_result = {
            "status": "ok",
            "assistant_message": "Updated the current scene graph.",
            "scene_graph": REAL2SIM_SCENE_GRAPH,
            "placements": {},
            "warnings": [],
        }

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
            mock.patch.object(agent_service, "apply_instruction", return_value=edit_result) as edit_mock,
        ):
            result = agent_service.handle_agent_message(
                session_id=context.session_id,
                run_id=context.run_id,
                text="move the table slightly left",
                action="edit_scene_graph",
            )

        self.assertEqual(result["agent"]["state"], "completed")
        self.assertEqual(result["agent"]["completed_state"], "edit_scene_graph")
        edit_mock.assert_called_once()
        self.assertEqual(edit_mock.call_args.args[0], "move the table slightly left")

    def test_react_loop_recovers_by_creating_graph_then_generating_scene(self) -> None:
        context = self._create_context()
        context.latest_input_image.write_bytes(b"saved-image-bytes")
        context.render_path.write_bytes(b"png")
        context.default_placements_path.write_text(
            json.dumps({"/World/table_0": {"x": 0.0, "y": 0.0, "z": 0.5}}),
            encoding="utf-8",
        )

        create_result = {
            "status": "ok",
            "assistant_message": "Created a new scene graph from the current input.",
            "scene_graph": REAL2SIM_SCENE_GRAPH,
            "placements": {},
            "warnings": [],
        }

        with (
            mock.patch.object(agent_service, "resolve_runtime_context", return_value=context),
            mock.patch.object(agent_service, "create_session", return_value=context),
            mock.patch.object(
                agent_service,
                "react_agent_step",
                side_effect=[
                    self._react("generate_scene", scene_endpoint="scene_new", reason="Try previewing first."),
                    self._react("create_scene_graph", instruction="Create a scene graph from the current image.", reason="Need a scene graph first."),
                    self._react("generate_scene", scene_endpoint="scene_new", resample_mode="joint", reason="Now the graph exists, generate the preview."),
                ],
            ),
            mock.patch.object(agent_service, "create_scene_graph_from_input", return_value=create_result) as create_mock,
            mock.patch.object(
                agent_service,
                "_run_scene_service",
                return_value={
                    "saved_usd": str(context.scene_service_usd_path),
                    "placements": {"/World/table_0": {"x": 0.0, "y": 0.0, "z": 0.5}},
                    "debug": {"resample_mode": "joint"},
                    "resample_mode": "joint",
                },
            ) as run_scene_mock,
        ):
            result = agent_service.handle_agent_message(
                session_id=context.session_id,
                run_id=context.run_id,
                text="Generate a preview from the current image.",
            )

        self.assertEqual(result["agent"]["state"], "completed")
        self.assertEqual(result["agent"]["completed_state"], "generate_scene")
        self.assertEqual(len(result["react_trace"]), 3)
        self.assertEqual(result["react_trace"][0]["observation"]["code"], "missing_scene_graph")
        self.assertEqual(result["react_trace"][1]["observation"]["code"], "scene_graph_created")
        self.assertEqual(result["react_trace"][2]["observation"]["code"], "scene_generated")
        create_mock.assert_called_once()
        run_scene_mock.assert_called_once()

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
                    "payload": {"session_id": context.session_id, "run_id": context.run_id},
                    "artifacts": {
                        "real2sim_root_dir": str(context.run_root),
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
                    "payload": {"session_id": context.session_id, "run_id": context.run_id},
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
                    "current_run_id": context.run_id,
                    "current_state": "await_layout_strategy",
                    "last_intent": "generate_scene",
                    "last_completed_state": "generate_scene",
                    "last_decision": {"reason": "Need a layout strategy before generating the scene."},
                    "pending_question": {
                        "type": "layout_strategy",
                        "question": "Choose joint or lock_real2sim.",
                        "options": [{"id": "joint", "label": "Joint"}],
                        "scene_endpoint": "scene_new",
                        "run_id": context.run_id,
                    },
                    "runs": {
                        context.run_id: {
                            "run_id": context.run_id,
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
                        }
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
                run_id=context.run_id,
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
                    "current_run_id": context.run_id,
                    "current_state": "run_real2sim",
                    "last_intent": "run_real2sim",
                    "runs": {
                        context.run_id: {
                            "run_id": context.run_id,
                            "real2sim": {
                                "status": "running",
                                "job_id": "job_live",
                                "log_path": str(context.real2sim_log_path),
                                "log_start_offset": 7,
                            },
                        }
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
                run_id=context.run_id,
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
