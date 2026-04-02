import io
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from app.backend.app import create_app
except ModuleNotFoundError as exc:
    create_app = None
    _CREATE_APP_IMPORT_ERROR = exc
else:
    _CREATE_APP_IMPORT_ERROR = None


@unittest.skipIf(create_app is None, f"Flask is not installed: {_CREATE_APP_IMPORT_ERROR}")
class AgentRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = create_app().test_client()

    def test_agent_message_json_forwards_runtime_context(self) -> None:
        runtime_ctx = SimpleNamespace(session_id="sess_demo", run_id="run_demo")
        response_payload = {
            "status": "ok",
            "session_id": "sess_demo",
            "run_id": "run_demo",
            "agent": {"intent": "graph", "state": "completed", "message": "ok", "question": None},
        }

        with (
            mock.patch("app.backend.api.routes._resolve_request_runtime_context", return_value=runtime_ctx),
            mock.patch("app.backend.api.routes.handle_agent_message", return_value=response_payload) as agent_mock,
        ):
            response = self.client.post(
                "/agent/message",
                json={
                    "text": "add a chair",
                    "action": "graph",
                    "session_id": "sess_demo",
                    "run_id": "run_demo",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), response_payload)
        agent_mock.assert_called_once_with(
            session_id="sess_demo",
            run_id="run_demo",
            text="add a chair",
            image_bytes=None,
            class_names_raw="",
            action="graph",
            resample_mode=None,
            scene_endpoint=None,
        )

    def test_agent_message_multipart_forwards_image_and_controls(self) -> None:
        runtime_ctx = SimpleNamespace(session_id="sess_demo", run_id="run_demo")
        response_payload = {
            "status": "ok",
            "session_id": "sess_demo",
            "run_id": "run_demo",
            "agent": {
                "intent": "generate_scene",
                "state": "needs_clarification",
                "message": "Need layout strategy",
                "question": "joint or lock_real2sim?",
            },
        }

        with (
            mock.patch("app.backend.api.routes._resolve_request_runtime_context", return_value=runtime_ctx),
            mock.patch("app.backend.api.routes.handle_agent_message", return_value=response_payload) as agent_mock,
        ):
            response = self.client.post(
                "/agent/message",
                data={
                    "text": "generate the scene",
                    "action": "generate_scene",
                    "resample_mode": "joint",
                    "scene_endpoint": "scene_new",
                    "class_names": "[\"chair\"]",
                    "session_id": "sess_demo",
                    "run_id": "run_demo",
                    "image": (io.BytesIO(b"fake-image-bytes"), "reference.png"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), response_payload)
        agent_mock.assert_called_once_with(
            session_id="sess_demo",
            run_id="run_demo",
            text="generate the scene",
            image_bytes=b"fake-image-bytes",
            class_names_raw='["chair"]',
            action="generate_scene",
            resample_mode="joint",
            scene_endpoint="scene_new",
        )

    def test_agent_state_forwards_runtime_context(self) -> None:
        runtime_ctx = SimpleNamespace(session_id="sess_demo", run_id="run_demo")
        response_payload = {
            "status": "ok",
            "session_id": "sess_demo",
            "run_id": "run_demo",
            "agent": {"intent": "generate_scene", "state": "await_layout_strategy", "message": "Need layout strategy"},
        }

        with (
            mock.patch("app.backend.api.routes._resolve_request_runtime_context", return_value=runtime_ctx),
            mock.patch("app.backend.api.routes.get_agent_state_response", return_value=response_payload) as state_mock,
        ):
            response = self.client.get(
                "/agent/state",
                query_string={"session_id": "sess_demo", "run_id": "run_demo"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), response_payload)
        state_mock.assert_called_once_with(session_id="sess_demo", run_id="run_demo")


if __name__ == "__main__":
    unittest.main()
