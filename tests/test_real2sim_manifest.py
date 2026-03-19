import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from pipelines.real2sim.manifest import (
    GLTF_TO_USD_UNIT_SCALE,
    build_real2sim_asset_manifest,
    gltf_scene_transform_to_usd_transform,
)


class Real2SimManifestTest(unittest.TestCase):
    def test_gltf_scene_transform_to_usd_transform_converts_basis_and_scale(self) -> None:
        scene_transform = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        usd_transform = gltf_scene_transform_to_usd_transform(scene_transform)

        np.testing.assert_allclose(usd_transform[:3, :3], np.eye(3), atol=1e-8)
        np.testing.assert_allclose(
            usd_transform[:3, 3],
            np.array([1.0, -3.0, 2.0], dtype=float) * GLTF_TO_USD_UNIT_SCALE,
            atol=1e-8,
        )

    def test_build_real2sim_asset_manifest_filters_and_maps_real2sim_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "scene_results"
            objects_dir = output_dir / "objects"
            usd_objects_dir = output_dir / "usd_objects"
            objects_dir.mkdir(parents=True)
            usd_objects_dir.mkdir(parents=True)

            mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
            for name in ("obj_00", "obj_01"):
                (objects_dir / f"{name}.glb").write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
                (usd_objects_dir / f"{name}.usd").write_text("#usda 1.0\n", encoding="utf-8")

            (output_dir / "scene_merged.glb").write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
            (output_dir / "scene_merged_post.glb").write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
            (output_dir / "scene_merged_post.usd").write_text("#usda 1.0\n", encoding="utf-8")

            pose0 = np.array(
                [
                    [1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, -2.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            pose1 = np.array(
                [
                    [1.0, 0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0, 0.25],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            poses = {
                "obj_00": {"scene_transform": pose0.tolist()},
                "obj_01": {"scene_transform": pose1.tolist()},
            }
            (output_dir / "poses.json").write_text(json.dumps(poses), encoding="utf-8")

            scene_graph = {
                "obj": {
                    "/World/table_0": {
                        "id": 0,
                        "class": "table",
                        "caption": "wood table",
                        "source": "real2sim",
                    },
                    "/World/lamp_1": {
                        "id": 1,
                        "class": "lamp",
                        "caption": "desk lamp",
                        "source": "retrieval",
                    },
                    "/World/mug_2": {
                        "id": 2,
                        "class": "mug",
                        "caption": "white mug",
                        "source": "real2sim",
                    },
                },
                "edges": {"obj-obj": [], "obj-wall": []},
            }
            scene_graph_path = root / "scene_graph.json"
            scene_graph_path.write_text(json.dumps(scene_graph), encoding="utf-8")

            manifest_path, manifest = build_real2sim_asset_manifest(
                output_dir,
                scene_graph_path=scene_graph_path,
            )

            self.assertTrue(manifest_path.exists())
            self.assertEqual(set(manifest["objects"].keys()), {"/World/table_0", "/World/mug_2"})
            self.assertEqual(manifest["objects"]["/World/table_0"]["output_name"], "obj_00")
            self.assertEqual(manifest["objects"]["/World/mug_2"]["output_name"], "obj_01")
            self.assertEqual(manifest["objects"]["/World/table_0"]["usd_path"], "usd_objects/obj_00.usd")
            self.assertEqual(manifest["scene_usd"], "scene_merged_post.usd")
            self.assertEqual(manifest["unmatched_scene_paths"], [])
            self.assertEqual(manifest["unmatched_outputs"], [])

            expected_usd_pose0 = gltf_scene_transform_to_usd_transform(pose0)
            np.testing.assert_allclose(
                np.asarray(manifest["objects"]["/World/table_0"]["usd_transform"], dtype=float),
                expected_usd_pose0,
                atol=1e-8,
            )

    def test_build_real2sim_asset_manifest_falls_back_to_scene_usd_internal_prim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "scene_results"
            objects_dir = output_dir / "objects"
            objects_dir.mkdir(parents=True)

            mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
            (objects_dir / "obj_00.glb").write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
            (output_dir / "scene_merged_post.usd").write_text("#usda 1.0\n", encoding="utf-8")
            (output_dir / "scene_merged_post.glb").write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
            (output_dir / "scene_merged.glb").write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
            (output_dir / "poses.json").write_text(
                json.dumps({"obj_00": {"scene_transform": np.eye(4).tolist()}}),
                encoding="utf-8",
            )

            scene_graph = {
                "obj": {
                    "/World/table_0": {
                        "id": 0,
                        "class": "table",
                        "caption": "wood table",
                        "source": "real2sim",
                    }
                },
                "edges": {"obj-obj": [], "obj-wall": []},
            }
            scene_graph_path = root / "scene_graph.json"
            scene_graph_path.write_text(json.dumps(scene_graph), encoding="utf-8")

            _, manifest = build_real2sim_asset_manifest(output_dir, scene_graph_path=scene_graph_path)

            entry = manifest["objects"]["/World/table_0"]
            self.assertEqual(entry["usd_path"], "scene_merged_post.usd")
            self.assertEqual(entry["usd_prim_path"], "/World/obj_00")


if __name__ == "__main__":
    unittest.main()
