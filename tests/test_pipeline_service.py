import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.backend.services import pipeline_service
from app.backend.services.pipeline_service import classify_real2sim_failure, get_real2sim_log_size, read_real2sim_log


class PipelineServiceTest(unittest.TestCase):
    def test_expected_real2sim_object_count_prefers_assignment_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            results_dir = run_root / "real2sim" / "scene_results"
            masks_dir = run_root / "real2sim" / "masks"
            results_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            for name in ("0.png", "1.png", "2.png", "3.png"):
                (masks_dir / name).write_bytes(b"mask")
            (results_dir / "assignment.json").write_text(
                '{"assignments":[{"output_name":"0"},{"output_name":"2"}]}',
                encoding="utf-8",
            )

            count = pipeline_service._expected_real2sim_object_count(
                str(run_root),
                {"assignment_json": "real2sim/scene_results/assignment.json"},
                "real2sim/masks",
            )

            self.assertEqual(count, 2)

    def test_run_real2sim_resolves_local_script_paths_from_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            image_path = run_root / "uploads" / "latest_input.jpg"
            scene_graph_path = run_root / "scene_graph" / "current_scene_graph.json"
            mask_output = run_root / "real2sim" / "masks"
            mesh_output_dir = run_root / "real2sim" / "meshes"
            scene_results_dir = run_root / "real2sim" / "scene_results"

            image_path.parent.mkdir(parents=True, exist_ok=True)
            scene_graph_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            scene_graph_path.write_text(
                '{"obj": {"/World/table_0": {"class": "table", "source": "real2sim"}}}',
                encoding="utf-8",
            )

            recorded_cmds: list[list[str]] = []

            def fake_run_step(cmd, timeout, label, env=None, cwd=None, job_id=None, log_path=None):
                recorded_cmds.append(list(cmd))
                if label == "sam3_segment_objects_only":
                    mask_output.mkdir(parents=True, exist_ok=True)
                    (mask_output / "image.png").write_bytes(b"png")

            with mock.patch.object(pipeline_service, "_run_step", side_effect=fake_run_step):
                with mock.patch.object(pipeline_service, "collect_scene_result_artifacts", return_value={}):
                    pipeline_service.run_real2sim(
                        {
                            "image_path": str(image_path),
                            "scene_graph_path": str(scene_graph_path),
                            "real2sim_root_dir": str(run_root),
                            "mask_output": str(mask_output),
                            "mesh_output_dir": str(mesh_output_dir),
                            "reuse_mesh_dir": str(mesh_output_dir),
                            "scene_results_dir": str(scene_results_dir),
                            "prompts": ["table"],
                        }
                    )

            self.assertEqual(len(recorded_cmds), 2)
            self.assertTrue(recorded_cmds[0][2].endswith("pipelines/real2sim/object_segmentation_pipeline.py"))
            self.assertTrue(Path(recorded_cmds[0][2]).is_absolute())
            self.assertTrue(recorded_cmds[1][2].endswith("pipelines/real2sim/streaming_generation_client.py"))
            self.assertTrue(Path(recorded_cmds[1][2]).is_absolute())

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

    def test_classify_real2sim_failure_for_segment_model_timeout(self) -> None:
        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "-u", "pipelines/real2sim/object_segmentation_pipeline.py"],
            output=(
                "File \"object_segmentation_pipeline.py\", line 150, in main\n"
                "processor = Sam3Processor.from_pretrained(\"facebook/sam3\")\n"
                "huggingface_hub.hf_api.list_repo_tree\n"
                "httpx.ReadTimeout: The read operation timed out"
            ),
            stderr="",
        )

        info = classify_real2sim_failure(error)

        self.assertEqual(info["code"], "segmentation_model_timeout")
        self.assertEqual(info["step"], "segment_masks")
        self.assertEqual(info["category"], "segmentation")

    def test_read_real2sim_log_uses_custom_run_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            log_path.write_text("line-1\nline-2\n", encoding="utf-8")

            result = read_real2sim_log(log_path=str(log_path))

            self.assertIn("line-1", result["content"])
            self.assertEqual(result["size"], get_real2sim_log_size(log_path=str(log_path)))


if __name__ == "__main__":
    unittest.main()
