import subprocess
import tempfile
import unittest
from pathlib import Path

from app.backend.services.pipeline_service import classify_real2sim_failure, get_real2sim_log_size, read_real2sim_log


class PipelineServiceTest(unittest.TestCase):
    def test_classify_real2sim_failure_for_missing_real2sim_objects(self) -> None:
        error = ValueError("No object prompts found in scene graph. Generate scene graph first.")

        info = classify_real2sim_failure(error)

        self.assertEqual(info["code"], "no_real2sim_objects")
        self.assertFalse(info["retryable"])
        self.assertEqual(info["category"], "input")

    def test_classify_real2sim_failure_for_remote_server_unavailable(self) -> None:
        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "-u", "pipelines/real2sim/streaming_generation_client.py"],
            output="requests.exceptions.ConnectionError: Failed to establish a new connection: [Errno 111] Connection refused",
            stderr="",
        )

        info = classify_real2sim_failure(error)

        self.assertEqual(info["code"], "remote_server_unavailable")
        self.assertEqual(info["step"], "remote_predict")
        self.assertTrue(info["retryable"])

    def test_classify_real2sim_failure_for_remote_timeout(self) -> None:
        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "-u", "pipelines/real2sim/streaming_generation_client.py"],
            output="requests.exceptions.ReadTimeout: HTTPConnectionPool(host='example.com', port=8000): Read timed out.",
            stderr="",
        )

        info = classify_real2sim_failure(error)

        self.assertEqual(info["code"], "remote_request_timeout")
        self.assertEqual(info["step"], "remote_predict")
        self.assertTrue(info["retryable"])

    def test_classify_real2sim_failure_for_mask_assignment_issue(self) -> None:
        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "-u", "pipelines/real2sim/streaming_generation_client.py"],
            output="[WARN] VLM mask assignment skipped: OpenAI request failed",
            stderr="",
        )

        info = classify_real2sim_failure(error)

        self.assertEqual(info["code"], "mask_assignment_failed")
        self.assertEqual(info["category"], "assignment")

    def test_read_real2sim_log_uses_custom_run_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            log_path.write_text("line-1\nline-2\n", encoding="utf-8")

            result = read_real2sim_log(log_path=str(log_path))

            self.assertIn("line-1", result["content"])
            self.assertEqual(result["size"], get_real2sim_log_size(log_path=str(log_path)))


if __name__ == "__main__":
    unittest.main()
