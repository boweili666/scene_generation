import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from pipelines.real2sim.postprocess import (
    extract_support_pairs,
    postprocess_real2sim_outputs,
    pose_translation_to_glb,
    resolve_support_penetration,
    upright_rotation_from_current,
    PlacementState,
)


class Real2SimPostprocessTest(unittest.TestCase):
    def test_pose_translation_is_converted_to_glb_y_up(self) -> None:
        translated = pose_translation_to_glb([1.0, 2.0, 3.0])
        np.testing.assert_allclose(translated, np.array([1.0, 3.0, -2.0]), atol=1e-8)

    def test_upright_rotation_removes_tilt_but_keeps_heading(self) -> None:
        rotation = Rotation.from_euler("yxz", [32.0, 7.5, -5.0], degrees=True).as_matrix()
        upright = upright_rotation_from_current(rotation)

        np.testing.assert_allclose(upright @ np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), atol=1e-8)

        original_heading = rotation @ np.array([0.0, 0.0, 1.0])
        original_heading[1] = 0.0
        original_heading /= np.linalg.norm(original_heading)
        snapped_heading = upright @ np.array([0.0, 0.0, 1.0])
        snapped_heading[1] = 0.0
        snapped_heading /= np.linalg.norm(snapped_heading)
        self.assertGreater(float(np.dot(original_heading, snapped_heading)), 0.999)

    def test_extract_support_pairs_deduplicates_bidirectional_edges(self) -> None:
        scene_graph = {
            "obj": {
                "/World/Table_0": {"class": "table"},
                "/World/Bottle_1": {"class": "bottle"},
            },
            "edges": {
                "obj-obj": [
                    {"source": "/World/Bottle_1", "target": "/World/Table_0", "relation": "supported by"},
                    {"source": "/World/Table_0", "target": "/World/Bottle_1", "relation": "supports"},
                ]
            },
        }
        self.assertEqual(extract_support_pairs(scene_graph), [("/World/Bottle_1", "/World/Table_0")])

    def test_resolve_support_penetration_lifts_supported_object(self) -> None:
        base = PlacementState(
            name="obj_00",
            mesh=trimesh.creation.box(extents=[2.0, 1.0, 2.0]),
            rotation=np.eye(3),
            scale=np.ones(3),
            translation=np.array([0.0, 0.5, 0.0]),
        )
        top = PlacementState(
            name="obj_01",
            mesh=trimesh.creation.box(extents=[0.4, 0.4, 0.4]),
            rotation=np.eye(3),
            scale=np.ones(3),
            translation=np.array([0.0, 0.85, 0.0]),
        )
        placements = {"obj_00": base, "obj_01": top}

        adjustments = resolve_support_penetration(placements, [("obj_01", "obj_00")], clearance=1e-4)

        self.assertEqual(adjustments, 1)
        top_min, _ = top.bounds
        _, base_max = base.bounds
        self.assertGreaterEqual(float(top_min[1]), float(base_max[1]) + 1e-4 - 1e-8)

    def test_postprocess_rebuilds_scene_and_solves_support_penetration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "scene_results"
            objects_dir = output_dir / "objects"
            objects_dir.mkdir(parents=True)

            base_mesh = trimesh.creation.box(extents=[2.0, 1.0, 2.0])
            top_mesh = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
            (objects_dir / "obj_00.glb").write_bytes(trimesh.Scene(base_mesh).export(file_type="glb"))
            (objects_dir / "obj_01.glb").write_bytes(trimesh.Scene(top_mesh).export(file_type="glb"))

            poses = {
                "obj_00": {
                    "rotation": [[1.0, 0.0, 0.0, 0.0]],
                    "translation": [[0.0, 0.0, 0.5]],
                    "scale": [[1.0, 1.0, 1.0]],
                },
                "obj_01": {
                    "rotation": [[1.0, 0.0, 0.0, 0.0]],
                    "translation": [[0.0, 0.1, 0.85]],
                    "scale": [[1.0, 1.0, 1.0]],
                },
            }
            (output_dir / "poses.json").write_text(json.dumps(poses), encoding="utf-8")

            scene_graph = {
                "obj": {
                    "/World/Table_0": {"class": "table"},
                    "/World/Bottle_1": {"class": "bottle"},
                },
                "edges": {
                    "obj-obj": [
                        {"source": "/World/Bottle_1", "target": "/World/Table_0", "relation": "supported by"},
                        {"source": "/World/Table_0", "target": "/World/Bottle_1", "relation": "supports"},
                    ]
                },
            }
            scene_graph_path = root / "scene_graph.json"
            scene_graph_path.write_text(json.dumps(scene_graph), encoding="utf-8")

            summary = postprocess_real2sim_outputs(output_dir, scene_graph_path=scene_graph_path)

            self.assertEqual(summary["support_pairs"], 1)
            self.assertTrue((output_dir / "scene_merged.glb").exists())

            rebuilt = trimesh.load(output_dir / "scene_merged.glb", force="scene")
            top_transform, top_geom_name = rebuilt.graph["obj_01"]
            base_transform, base_geom_name = rebuilt.graph["obj_00"]
            top_geom = rebuilt.geometry[top_geom_name]
            base_geom = rebuilt.geometry[base_geom_name]
            top_bounds = trimesh.transform_points(top_geom.vertices, top_transform)
            base_bounds = trimesh.transform_points(base_geom.vertices, base_transform)

            self.assertGreaterEqual(float(top_bounds[:, 1].min()), float(base_bounds[:, 1].max()) + 1e-4 - 1e-8)

            updated_poses = json.loads((output_dir / "poses.json").read_text(encoding="utf-8"))
            self.assertGreater(updated_poses["obj_01"]["translation"][0][1], poses["obj_01"]["translation"][0][1])


if __name__ == "__main__":
    unittest.main()
