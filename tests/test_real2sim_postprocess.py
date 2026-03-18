import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from pipelines.real2sim.postprocess import (
    extract_support_pairs,
    pose_rotation_to_glb,
    pose_translation_to_glb,
    postprocess_real2sim_outputs,
    preserve_relative_support_transforms,
    resolve_support_penetration,
    upright_rotation_from_current,
    PlacementState,
)


class Real2SimPostprocessTest(unittest.TestCase):
    def test_pose_translation_is_used_as_is(self) -> None:
        translated = pose_translation_to_glb([1.0, 2.0, 3.0])
        np.testing.assert_allclose(translated, np.array([1.0, 2.0, 3.0]), atol=1e-8)

    def test_postprocess_prefers_scene_transform_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "scene_results"
            objects_dir = output_dir / "objects"
            objects_dir.mkdir(parents=True)

            mesh = trimesh.creation.box(extents=[0.4, 0.6, 0.4])
            (objects_dir / "obj_00.glb").write_bytes(trimesh.Scene(mesh).export(file_type="glb"))

            matrix = np.array(
                [
                    [1.0, 0.0, 0.0, 0.25],
                    [0.0, 1.0, 0.0, 1.5],
                    [0.0, 0.0, 1.0, -0.4],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            poses = {
                "obj_00": {
                    "rotation": [[1.0, 0.0, 0.0, 0.0]],
                    "translation": [[100.0, 100.0, 100.0]],
                    "scale": [[1.0, 1.0, 1.0]],
                    "scene_transform": matrix.tolist(),
                    "scene_transform_convention": "gltf_y_up",
                    "raw_model_pose": {
                        "rotation": [[1.0, 0.0, 0.0, 0.0]],
                        "translation": [[100.0, 100.0, 100.0]],
                        "scale": [[1.0, 1.0, 1.0]],
                    },
                }
            }
            (output_dir / "poses.json").write_text(json.dumps(poses), encoding="utf-8")

            raw_scene = trimesh.Scene()
            raw_scene.add_geometry(
                mesh,
                node_name="obj_00",
                geom_name="obj_00",
                transform=matrix,
            )
            (output_dir / "scene_merged.glb").write_bytes(raw_scene.export(file_type="glb"))

            scene_graph = {
                "obj": {
                    "/World/Table_0": {"class": "table"},
                }
            }
            scene_graph_path = root / "scene_graph.json"
            scene_graph_path.write_text(json.dumps(scene_graph), encoding="utf-8")

            summary = postprocess_real2sim_outputs(output_dir, scene_graph_path=scene_graph_path)

            self.assertEqual(summary["grounded_roots"], 1)
            updated_poses = json.loads((output_dir / "poses.json").read_text(encoding="utf-8"))
            self.assertIn("scene_transform", updated_poses["obj_00"])
            self.assertEqual(updated_poses["obj_00"]["raw_model_pose"], poses["obj_00"]["raw_model_pose"])
            self.assertEqual(updated_poses["obj_00"]["scene_transform_convention"], "gltf_y_up")
            scene_matrix = np.asarray(updated_poses["obj_00"]["scene_transform"], dtype=float)
            self.assertAlmostEqual(float(scene_matrix[1, 3]), 0.3, places=6)
            self.assertAlmostEqual(float(scene_matrix[0, 3]), 0.25, places=6)
            self.assertAlmostEqual(float(scene_matrix[2, 3]), -0.4, places=6)

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

    def test_postprocess_rebuilds_scene_and_keeps_supported_objects_on_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "scene_results"
            objects_dir = output_dir / "objects"
            objects_dir.mkdir(parents=True)

            base_mesh = trimesh.creation.box(extents=[2.0, 1.0, 2.0])
            top_mesh = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
            (objects_dir / "obj_00.glb").write_bytes(trimesh.Scene(base_mesh).export(file_type="glb"))
            (objects_dir / "obj_01.glb").write_bytes(trimesh.Scene(top_mesh).export(file_type="glb"))

            base_rotation = Rotation.from_euler("x", 40.0, degrees=True)
            relative_rotation = Rotation.from_euler("y", 25.0, degrees=True)
            top_rotation = base_rotation * relative_rotation
            base_translation = np.array([0.0, 0.0, 0.8], dtype=float)
            relative_translation = np.array([0.2, 0.8, 0.1], dtype=float)
            top_translation = base_rotation.as_matrix() @ relative_translation + base_translation

            def as_wxyz(rotation: Rotation) -> list[float]:
                quat_xyzw = rotation.as_quat()
                return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]

            poses = {
                "obj_00": {
                    "rotation": [as_wxyz(base_rotation)],
                    "translation": [[float(v) for v in base_translation]],
                    "scale": [[1.0, 1.0, 1.0]],
                },
                "obj_01": {
                    "rotation": [as_wxyz(top_rotation)],
                    "translation": [[float(v) for v in top_translation]],
                    "scale": [[1.0, 1.0, 1.0]],
                },
            }
            (output_dir / "poses.json").write_text(json.dumps(poses), encoding="utf-8")

            raw_base = PlacementState(
                name="obj_00",
                mesh=base_mesh,
                rotation=pose_rotation_to_glb(poses["obj_00"]["rotation"][0]),
                scale=np.array(poses["obj_00"]["scale"][0], dtype=float),
                translation=pose_translation_to_glb(poses["obj_00"]["translation"][0]),
            )
            raw_top = PlacementState(
                name="obj_01",
                mesh=top_mesh,
                rotation=pose_rotation_to_glb(poses["obj_01"]["rotation"][0]),
                scale=np.array(poses["obj_01"]["scale"][0], dtype=float),
                translation=pose_translation_to_glb(poses["obj_01"]["translation"][0]),
            )
            raw_scene = trimesh.Scene()
            raw_scene.add_geometry(raw_base.mesh, node_name="obj_00", geom_name="obj_00", transform=raw_base.matrix)
            raw_scene.add_geometry(raw_top.mesh, node_name="obj_01", geom_name="obj_01", transform=raw_top.matrix)
            (output_dir / "scene_merged.glb").write_bytes(raw_scene.export(file_type="glb"))

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
            self.assertGreaterEqual(summary["penetration_adjustments"], 0)
            self.assertTrue((output_dir / "scene_merged.glb").exists())
            self.assertTrue((output_dir / "scene_merged_pre.glb").exists())
            self.assertTrue((output_dir / "scene_merged_post.glb").exists())
            self.assertTrue((output_dir / "poses_pre.json").exists())
            self.assertTrue((output_dir / "poses_post.json").exists())

            pre_poses = json.loads((output_dir / "poses_pre.json").read_text(encoding="utf-8"))
            self.assertEqual(pre_poses, poses)

            rebuilt = trimesh.load(output_dir / "scene_merged.glb", force="scene")
            top_transform, top_geom_name = rebuilt.graph["obj_01"]
            base_transform, base_geom_name = rebuilt.graph["obj_00"]
            top_rotation_post = top_transform[:3, :3]
            base_rotation_post = base_transform[:3, :3]
            np.testing.assert_allclose(base_rotation_post @ np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), atol=1e-8)
            top_geom = rebuilt.geometry[top_geom_name]
            base_geom = rebuilt.geometry[base_geom_name]
            top_bounds = trimesh.transform_points(top_geom.vertices, top_transform)
            base_bounds = trimesh.transform_points(base_geom.vertices, base_transform)
            self.assertGreaterEqual(float(top_bounds[:, 1].min()), float(base_bounds[:, 1].max()) - 1e-8)

            raw_rebuilt = trimesh.load(output_dir / "scene_merged_pre.glb", force="scene")
            raw_top_transform, raw_top_geom_name = raw_rebuilt.graph["obj_01"]
            raw_base_transform, raw_base_geom_name = raw_rebuilt.graph["obj_00"]
            raw_top_rotation = raw_top_transform[:3, :3]
            raw_base_rotation = raw_base_transform[:3, :3]

            raw_relative_rotation = raw_base_rotation.T @ raw_top_rotation
            post_relative_rotation = base_rotation_post.T @ top_rotation_post
            np.testing.assert_allclose(post_relative_rotation, raw_relative_rotation, atol=1e-8)

            raw_relative_translation = raw_base_rotation.T @ (
                raw_top_transform[:3, 3] - raw_base_transform[:3, 3]
            )
            post_relative_translation = base_rotation_post.T @ (top_transform[:3, 3] - base_transform[:3, 3])
            np.testing.assert_allclose(post_relative_translation[[0, 2]], raw_relative_translation[[0, 2]], atol=1e-8)

            updated_poses = json.loads((output_dir / "poses.json").read_text(encoding="utf-8"))
            self.assertIn("scene_transform", updated_poses["obj_00"])
            self.assertEqual(updated_poses["obj_00"]["scene_transform_convention"], "gltf_y_up")
            base_up = pose_rotation_to_glb(updated_poses["obj_00"]["rotation"][0]) @ np.array([0.0, 1.0, 0.0])
            np.testing.assert_allclose(base_up, np.array([0.0, 1.0, 0.0]), atol=1e-8)
            post_poses = json.loads((output_dir / "poses_post.json").read_text(encoding="utf-8"))
            self.assertEqual(post_poses, updated_poses)

    def test_postprocess_grounds_unsupported_root_and_moves_supported_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "scene_results"
            objects_dir = output_dir / "objects"
            objects_dir.mkdir(parents=True)

            base_mesh = trimesh.creation.box(extents=[2.0, 1.0, 2.0])
            top_mesh = trimesh.creation.box(extents=[0.4, 0.2, 0.4])
            (objects_dir / "obj_00.glb").write_bytes(trimesh.Scene(base_mesh).export(file_type="glb"))
            (objects_dir / "obj_01.glb").write_bytes(trimesh.Scene(top_mesh).export(file_type="glb"))

            poses = {
                "obj_00": {
                    "rotation": [[1.0, 0.0, 0.0, 0.0]],
                    "translation": [[0.0, 1.2, 0.0]],
                    "scale": [[1.0, 1.0, 1.0]],
                },
                "obj_01": {
                    "rotation": [[1.0, 0.0, 0.0, 0.0]],
                    "translation": [[0.3, 1.8, 0.1]],
                    "scale": [[1.0, 1.0, 1.0]],
                },
            }
            (output_dir / "poses.json").write_text(json.dumps(poses), encoding="utf-8")

            raw_scene = trimesh.Scene()
            raw_scene.add_geometry(
                base_mesh,
                node_name="obj_00",
                geom_name="obj_00",
                transform=np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 1.2],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            )
            raw_scene.add_geometry(
                top_mesh,
                node_name="obj_01",
                geom_name="obj_01",
                transform=np.array(
                    [
                        [1.0, 0.0, 0.0, 0.3],
                        [0.0, 1.0, 0.0, 1.8],
                        [0.0, 0.0, 1.0, 0.1],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            )
            (output_dir / "scene_merged.glb").write_bytes(raw_scene.export(file_type="glb"))

            scene_graph = {
                "obj": {
                    "/World/Table_0": {"class": "table"},
                    "/World/Laptop_1": {"class": "laptop"},
                },
                "edges": {
                    "obj-obj": [
                        {"source": "/World/Laptop_1", "target": "/World/Table_0", "relation": "supported by"},
                    ]
                },
            }
            scene_graph_path = root / "scene_graph.json"
            scene_graph_path.write_text(json.dumps(scene_graph), encoding="utf-8")

            summary = postprocess_real2sim_outputs(output_dir, scene_graph_path=scene_graph_path)

            self.assertEqual(summary["grounded_roots"], 1)

            pre_scene = trimesh.load(output_dir / "scene_merged_pre.glb", force="scene")
            post_scene = trimesh.load(output_dir / "scene_merged.glb", force="scene")
            pre_base_transform, _ = pre_scene.graph["obj_00"]
            pre_top_transform, _ = pre_scene.graph["obj_01"]
            post_base_transform, post_base_geom_name = post_scene.graph["obj_00"]
            post_top_transform, post_top_geom_name = post_scene.graph["obj_01"]

            base_vertices = trimesh.transform_points(
                post_scene.geometry[post_base_geom_name].vertices,
                post_base_transform,
            )
            top_vertices = trimesh.transform_points(
                post_scene.geometry[post_top_geom_name].vertices,
                post_top_transform,
            )
            self.assertAlmostEqual(float(base_vertices[:, 1].min()), 0.0, places=6)
            self.assertGreaterEqual(float(top_vertices[:, 1].min()), float(base_vertices[:, 1].max()) - 1e-8)

            base_delta = post_base_transform[:3, 3] - pre_base_transform[:3, 3]
            top_delta = post_top_transform[:3, 3] - pre_top_transform[:3, 3]
            np.testing.assert_allclose(top_delta, base_delta, atol=2e-4)

    def test_postprocess_grounds_unsupported_root_without_support_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "scene_results"
            objects_dir = output_dir / "objects"
            objects_dir.mkdir(parents=True)

            mesh = trimesh.creation.box(extents=[0.4, 0.6, 0.4])
            (objects_dir / "obj_00.glb").write_bytes(trimesh.Scene(mesh).export(file_type="glb"))

            poses = {
                "obj_00": {
                    "rotation": [[1.0, 0.0, 0.0, 0.0]],
                    "translation": [[0.0, 1.5, 0.0]],
                    "scale": [[1.0, 1.0, 1.0]],
                }
            }
            (output_dir / "poses.json").write_text(json.dumps(poses), encoding="utf-8")

            raw_scene = trimesh.Scene()
            raw_scene.add_geometry(
                mesh,
                node_name="obj_00",
                geom_name="obj_00",
                transform=np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 1.5],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            )
            (output_dir / "scene_merged.glb").write_bytes(raw_scene.export(file_type="glb"))

            scene_graph = {"obj": {"/World/Mug_0": {"class": "mug"}}}
            scene_graph_path = root / "scene_graph.json"
            scene_graph_path.write_text(json.dumps(scene_graph), encoding="utf-8")

            summary = postprocess_real2sim_outputs(output_dir, scene_graph_path=scene_graph_path)

            self.assertEqual(summary["grounded_roots"], 1)
            post_scene = trimesh.load(output_dir / "scene_merged.glb", force="scene")
            post_transform, post_geom_name = post_scene.graph["obj_00"]
            vertices = trimesh.transform_points(post_scene.geometry[post_geom_name].vertices, post_transform)
            self.assertAlmostEqual(float(vertices[:, 1].min()), 0.0, places=6)

    def test_preserve_relative_support_transforms(self) -> None:
        base_original = PlacementState(
            name="obj_00",
            mesh=trimesh.creation.box(extents=[2.0, 1.0, 2.0]),
            rotation=pose_rotation_to_glb(
                [float(v) for v in Rotation.from_euler("x", 40.0, degrees=True).as_quat()[[3, 0, 1, 2]]]
            ),
            scale=np.ones(3),
            translation=pose_translation_to_glb([0.0, 0.0, 0.8]),
        )
        top_original = PlacementState(
            name="obj_01",
            mesh=trimesh.creation.box(extents=[0.4, 0.4, 0.4]),
            rotation=base_original.rotation @ Rotation.from_euler("y", 25.0, degrees=True).as_matrix(),
            scale=np.ones(3),
            translation=base_original.rotation @ np.array([0.2, 0.8, 0.1]) + base_original.translation,
        )
        base_updated = PlacementState(
            name="obj_00",
            mesh=base_original.mesh,
            rotation=upright_rotation_from_current(base_original.rotation),
            scale=np.ones(3),
            translation=base_original.translation.copy(),
        )
        top_updated = PlacementState(
            name="obj_01",
            mesh=top_original.mesh,
            rotation=top_original.rotation.copy(),
            scale=np.ones(3),
            translation=top_original.translation.copy(),
        )

        original = {"obj_00": base_original, "obj_01": top_original}
        updated = {"obj_00": base_updated, "obj_01": top_updated}
        preserve_relative_support_transforms(original, updated, [("obj_01", "obj_00")])

        original_rel_rot = base_original.rotation.T @ top_original.rotation
        updated_rel_rot = updated["obj_00"].rotation.T @ updated["obj_01"].rotation
        original_rel_trans = base_original.rotation.T @ (top_original.translation - base_original.translation)
        updated_rel_trans = updated["obj_00"].rotation.T @ (
            updated["obj_01"].translation - updated["obj_00"].translation
        )

        np.testing.assert_allclose(updated_rel_rot, original_rel_rot, atol=1e-8)
        np.testing.assert_allclose(updated_rel_trans, original_rel_trans, atol=1e-8)

if __name__ == "__main__":
    unittest.main()
