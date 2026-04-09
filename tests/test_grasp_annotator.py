from pathlib import Path
import sys
import unittest

import numpy as np


SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "scene_robot" / "src"
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from grasp_annotator import PipelineConfig, RunConfig
from grasp_annotator.pose_generator import axes_to_rotation_matrix, make_ring_grasp_poses
from grasp_annotator.schema import build_annotation


class GraspAnnotatorSchemaTest(unittest.TestCase):
    def test_public_api_exposes_configs(self) -> None:
        pipeline = PipelineConfig()
        run = RunConfig(input_dir=Path("/tmp/input"), output_dir=Path("/tmp/output"))
        self.assertEqual(pipeline.max_candidates, 24)
        self.assertEqual(run.pattern, "*.glb")

    def test_build_annotation_emits_axis_band_for_handle_branch(self) -> None:
        result = {
            "object_name": "hammer",
            "classification": {"category": "handle_tool_object"},
            "classification_json": "/tmp/classification.json",
            "handle_tool_stage": {
                "grasp_primitives": [
                    {
                        "type": "axis_band",
                        "point_local": [0.0, 0.0, 0.0],
                        "axis_local": [1.0, 0.0, 0.0],
                        "slide_range": [-0.05, 0.05],
                    }
                ],
                "candidate_overlay": "/tmp/review.png",
            },
        }

        annotation = build_annotation(result, Path("/tmp/hammer.glb"), Path("/tmp/object"))
        self.assertEqual(annotation["schema_version"], "grasp_primitives_v1")
        self.assertEqual(annotation["category"], "handle_tool_object")
        self.assertEqual(len(annotation["grasp_primitives"]), 1)
        self.assertEqual(annotation["grasp_primitives"][0]["type"], "axis_band")

    def test_build_annotation_emits_pose_set_for_graspnet_branch(self) -> None:
        result = {
            "object_name": "toy",
            "classification": {"category": "graspnet_object"},
            "graspnet_object_stage": {
                "grasp_primitives": [
                    {
                        "type": "pose_set",
                        "poses_local": [
                            {
                                "pose_id": 0,
                                "score": 0.8,
                                "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                "translation": [0.0, 0.0, 0.1],
                            }
                        ],
                    }
                ]
            },
        }

        annotation = build_annotation(result, Path("/tmp/toy.glb"), Path("/tmp/object"))
        self.assertEqual(annotation["grasp_primitives"][0]["type"], "pose_set")

    def test_make_ring_grasp_poses_samples_requested_count(self) -> None:
        poses = make_ring_grasp_poses(
            selected_world=np.array([0.0, 0.0, 0.0], dtype=float),
            axis_unit=np.array([0.0, 1.0, 0.0], dtype=float),
            radius=0.05,
            num_poses=8,
        )

        self.assertEqual(len(poses), 8)
        self.assertEqual(poses[0]["pose_id"], 0)
        self.assertAlmostEqual(float(poses[0]["approach_axis"][1]), 0.0, places=6)

    def test_axes_to_rotation_matrix_returns_orthonormal_frame(self) -> None:
        rotation = axes_to_rotation_matrix([0.0, 0.0, -1.0], [1.0, 0.0, 0.0])
        self.assertEqual(rotation.shape, (3, 3))
        self.assertAlmostEqual(float(rotation[:, 0].dot(rotation[:, 1])), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
