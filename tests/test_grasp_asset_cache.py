import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pxr import Gf, Usd, UsdGeom

from app.backend.services.grasp_asset_cache import (
    ASSET_GRASP_CACHE_SCHEMA_VERSION,
    build_asset_grasp_cache,
    cache_entry_path_for_prim,
    default_grasp_annotation_root,
    ensure_asset_grasp_cache_for_prim,
)
from app.backend.services.grasp_scene_adapter import build_scene_grasp_proposals
from app.backend.services.grasp_scene_adapter import build_stage_grasp_proposals


class GraspAssetCacheTest(unittest.TestCase):
    def _write_manifest(self, root: Path, *, object_entries: dict[str, dict]) -> Path:
        results_root = root / "real2sim" / "scene_results"
        objects_root = results_root / "objects"
        objects_root.mkdir(parents=True, exist_ok=True)
        for meta in object_entries.values():
            glb_rel = meta.get("glb_path")
            if isinstance(glb_rel, str):
                glb_path = results_root / glb_rel
                glb_path.parent.mkdir(parents=True, exist_ok=True)
                glb_path.write_bytes(b"glb")
        manifest_path = results_root / "real2sim_asset_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "objects": object_entries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return manifest_path

    def test_build_asset_grasp_cache_writes_stable_cache_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = self._write_manifest(
                root,
                object_entries={
                    "/World/power_drill_1": {
                        "output_name": "5",
                        "class": "power_drill",
                        "caption": "orange cordless drill",
                        "source": "real2sim",
                        "glb_path": "objects/5.glb",
                    },
                    "/World/tool_4": {
                        "output_name": "8",
                        "class": "tool",
                        "caption": "green hex key set",
                        "source": "real2sim",
                        "glb_path": "objects/8.glb",
                    },
                },
            )

            def fake_annotate_single_object(glb_path: Path, run_dir: Path, config, *, resume: bool):
                object_dir = Path(run_dir) / Path(glb_path).stem
                object_dir.mkdir(parents=True, exist_ok=True)
                annotation_path = object_dir / "annotation.json"
                annotation_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "grasp_primitives_v1",
                            "object_name": Path(glb_path).stem,
                            "category": "axis_object",
                            "grasp_primitives": [
                                {
                                    "type": "axis_band",
                                    "point_local": [0.0, 0.0, 0.0],
                                    "axis_local": [1.0, 0.0, 0.0],
                                    "slide_range": [-0.05, 0.05],
                                }
                            ],
                            "artifacts": {"pipeline_result": "pipeline_result.json"},
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return {
                    "status": "ok",
                    "object_dir": str(object_dir),
                    "annotation": str(annotation_path),
                }

            with patch(
                "app.backend.services.grasp_asset_cache.annotate_single_object",
                side_effect=fake_annotate_single_object,
            ):
                manifest_payload = build_asset_grasp_cache(manifest_path)

            annotation_root = default_grasp_annotation_root(manifest_path)
            self.assertEqual(Path(manifest_payload["output_root"]), annotation_root)
            self.assertEqual(len(manifest_payload["objects"]), 2)

            cache_path = annotation_root / "annotations" / "5.json"
            self.assertTrue(cache_path.exists())
            cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(cache_payload["schema_version"], ASSET_GRASP_CACHE_SCHEMA_VERSION)
            self.assertEqual(cache_payload["asset_id"], "5")
            self.assertEqual(cache_payload["class"], "power_drill")
            self.assertEqual(cache_payload["prim_paths"], ["/World/power_drill_1"])
            self.assertEqual(cache_payload["grasp_primitives"][0]["type"], "axis_band")

    def test_build_scene_grasp_proposals_projects_local_primitives_into_world(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = self._write_manifest(
                root,
                object_entries={
                    "/World/tool_0": {
                        "output_name": "5",
                        "class": "tool",
                        "caption": "tool",
                        "source": "real2sim",
                        "glb_path": "objects/5.glb",
                    }
                },
            )

            annotation_root = default_grasp_annotation_root(manifest_path)
            annotation_path = annotation_root / "annotations" / "5.json"
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(
                json.dumps(
                    {
                        "schema_version": ASSET_GRASP_CACHE_SCHEMA_VERSION,
                        "annotation_schema_version": "grasp_primitives_v1",
                        "asset_id": "5",
                        "category": "tool",
                        "grasp_primitives": [
                            {
                                "type": "point_grasp",
                                "point_local": [1.0, 0.0, 0.0],
                                "approach_dirs_local": [[0.0, 0.0, 1.0]],
                                "closing_dirs_local": [[1.0, 0.0, 0.0]],
                                "width_range": [0.02, 0.04],
                            },
                            {
                                "type": "axis_band",
                                "point_local": [1.0, 0.0, 0.0],
                                "axis_local": [1.0, 0.0, 0.0],
                                "slide_range": [-0.10, 0.20],
                                "approach_dirs_local": [[0.0, 0.0, 1.0]],
                                "closing_dirs_local": [[0.0, 1.0, 0.0]],
                                "width_range": [0.02, 0.04],
                                "depth_range": [0.03, 0.05],
                            },
                            {
                                "type": "pose_set",
                                "poses_local": [
                                    {
                                        "pose_id": 0,
                                        "score": 0.8,
                                        "translation": [1.0, 0.0, 0.0],
                                        "rotation_matrix": [
                                            [1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0],
                                        ],
                                        "width": 0.02,
                                        "height": 0.03,
                                        "depth": 0.04,
                                    }
                                ],
                            },
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            scene_usd = root / "scene_service" / "usd" / "scene_latest.usda"
            scene_usd.parent.mkdir(parents=True, exist_ok=True)
            stage = Usd.Stage.CreateNew(str(scene_usd))
            world = stage.DefinePrim("/World", "Xform")
            stage.SetDefaultPrim(world)
            tool = UsdGeom.Xform.Define(stage, "/World/tool_0")
            tool.AddTransformOp().Set(
                Gf.Matrix4d(
                    (
                        (0.0, 1.0, 0.0, 0.0),
                        (-1.0, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (10.0, 20.0, 0.5, 1.0),
                    )
                )
            )
            stage.GetRootLayer().Save()

            payload = build_scene_grasp_proposals(
                scene_usd,
                manifest_path,
                annotation_root=annotation_root,
            )

            self.assertEqual(payload["summary"]["object_count"], 1)
            object_payload = payload["objects"]["/World/tool_0"]
            primitives = object_payload["grasp_primitives_world"]
            point_grasp = primitives[0]
            axis_band = primitives[1]
            pose_set = primitives[2]

            self.assertEqual(point_grasp["point_world"], [10.0, 21.0, 0.5])
            self.assertEqual(axis_band["point_world"], [10.0, 21.0, 0.5])
            self.assertAlmostEqual(axis_band["axis_world"][0], 0.0, places=6)
            self.assertAlmostEqual(axis_band["axis_world"][1], 1.0, places=6)
            self.assertAlmostEqual(axis_band["closing_dirs_world"][0][0], -1.0, places=6)
            self.assertAlmostEqual(axis_band["closing_dirs_world"][0][1], 0.0, places=6)
            self.assertEqual(axis_band["slide_range_world"], [-0.1, 0.2])
            self.assertEqual(pose_set["poses_world"][0]["translation"], [10.0, 21.0, 0.5])
            self.assertAlmostEqual(pose_set["poses_world"][0]["rotation_matrix"][0][0], 0.0, places=6)
            self.assertAlmostEqual(pose_set["poses_world"][0]["rotation_matrix"][1][0], 1.0, places=6)

    def test_build_stage_grasp_proposals_uses_current_live_stage_transform(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = self._write_manifest(
                root,
                object_entries={
                    "/World/tool_0": {
                        "output_name": "5",
                        "class": "tool",
                        "caption": "tool",
                        "source": "real2sim",
                        "glb_path": "objects/5.glb",
                    }
                },
            )

            annotation_root = default_grasp_annotation_root(manifest_path)
            annotation_path = annotation_root / "annotations" / "5.json"
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(
                json.dumps(
                    {
                        "schema_version": ASSET_GRASP_CACHE_SCHEMA_VERSION,
                        "annotation_schema_version": "grasp_primitives_v1",
                        "asset_id": "5",
                        "category": "tool",
                        "grasp_primitives": [
                            {
                                "type": "point_grasp",
                                "point_local": [1.0, 0.0, 0.0],
                                "approach_dirs_local": [[0.0, 0.0, 1.0]],
                                "closing_dirs_local": [[1.0, 0.0, 0.0]],
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            stage = Usd.Stage.CreateInMemory()
            world = stage.DefinePrim("/World", "Xform")
            stage.SetDefaultPrim(world)
            tool = UsdGeom.Xform.Define(stage, "/World/tool_0")
            xform_op = tool.AddTransformOp()
            xform_op.Set(
                Gf.Matrix4d(
                    (
                        (1.0, 0.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (1.0, 2.0, 0.5, 1.0),
                    )
                )
            )
            payload_before = build_stage_grasp_proposals(
                stage,
                manifest_path,
                annotation_root=annotation_root,
                target_prim_paths=["/World/tool_0"],
            )
            xform_op.Set(
                Gf.Matrix4d(
                    (
                        (1.0, 0.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (3.0, 4.0, 0.5, 1.0),
                    )
                )
            )
            payload_after = build_stage_grasp_proposals(
                stage,
                manifest_path,
                annotation_root=annotation_root,
                target_prim_paths=["/World/tool_0"],
            )

            point_before = payload_before["objects"]["/World/tool_0"]["grasp_primitives_world"][0]["point_world"]
            point_after = payload_after["objects"]["/World/tool_0"]["grasp_primitives_world"][0]["point_world"]
            self.assertEqual(point_before, [2.0, 2.0, 0.5])
            self.assertEqual(point_after, [4.0, 4.0, 0.5])

    def test_build_stage_grasp_proposals_prefers_generated_scene_live_prim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = self._write_manifest(
                root,
                object_entries={
                    "/World/tool_0": {
                        "output_name": "5",
                        "class": "tool",
                        "caption": "tool",
                        "source": "real2sim",
                        "glb_path": "objects/5.glb",
                        "usd_transform": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [30.0, 40.0, 0.5, 1.0],
                        ],
                    }
                },
            )

            annotation_root = default_grasp_annotation_root(manifest_path)
            annotation_path = annotation_root / "annotations" / "5.json"
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(
                json.dumps(
                    {
                        "schema_version": ASSET_GRASP_CACHE_SCHEMA_VERSION,
                        "annotation_schema_version": "grasp_primitives_v1",
                        "asset_id": "5",
                        "category": "tool",
                        "grasp_primitives": [
                            {
                                "type": "point_grasp",
                                "point_local": [1.0, 0.0, 0.0],
                                "approach_dirs_local": [[0.0, 0.0, 1.0]],
                                "closing_dirs_local": [[1.0, 0.0, 0.0]],
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            stage = Usd.Stage.CreateInMemory()
            world = stage.DefinePrim("/World", "Xform")
            stage.SetDefaultPrim(world)
            envs = UsdGeom.Xform.Define(stage, "/World/envs")
            env_0 = UsdGeom.Xform.Define(stage, "/World/envs/env_0")
            generated = UsdGeom.Xform.Define(stage, "/World/envs/env_0/GeneratedScene")
            _ = envs, env_0, generated
            live_tool = UsdGeom.Xform.Define(stage, "/World/envs/env_0/GeneratedScene/tool_0")
            live_tool.AddTransformOp().Set(
                Gf.Matrix4d(
                    (
                        (1.0, 0.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (3.0, 4.0, 0.5, 1.0),
                    )
                )
            )

            payload = build_stage_grasp_proposals(
                stage,
                manifest_path,
                annotation_root=annotation_root,
                target_prim_paths=["/World/tool_0"],
            )

            point_world = payload["objects"]["/World/tool_0"]["grasp_primitives_world"][0]["point_world"]
            self.assertEqual(point_world, [4.0, 4.0, 0.5])

    def test_build_stage_grasp_proposals_applies_asset_ref_axis_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = self._write_manifest(
                root,
                object_entries={
                    "/World/tool_0": {
                        "output_name": "5",
                        "class": "tool",
                        "caption": "tool",
                        "source": "real2sim",
                        "glb_path": "objects/5.glb",
                    }
                },
            )

            annotation_root = default_grasp_annotation_root(manifest_path)
            annotation_path = annotation_root / "annotations" / "5.json"
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(
                json.dumps(
                    {
                        "schema_version": ASSET_GRASP_CACHE_SCHEMA_VERSION,
                        "annotation_schema_version": "grasp_primitives_v1",
                        "asset_id": "5",
                        "category": "tool",
                        "grasp_primitives": [
                            {
                                "type": "axis_band",
                                "point_local": [0.0, 1.0, 0.0],
                                "axis_local": [0.0, 1.0, 0.0],
                                "approach_dirs_local": [[1.0, 0.0, 0.0]],
                                "closing_dirs_local": [[0.0, 0.0, 1.0]],
                                "width_range": [0.02, 0.04],
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            stage = Usd.Stage.CreateInMemory()
            world = stage.DefinePrim("/World", "Xform")
            stage.SetDefaultPrim(world)
            UsdGeom.Xform.Define(stage, "/World/envs")
            UsdGeom.Xform.Define(stage, "/World/envs/env_0")
            UsdGeom.Xform.Define(stage, "/World/envs/env_0/GeneratedScene")
            live_tool = UsdGeom.Xform.Define(stage, "/World/envs/env_0/GeneratedScene/tool_0")
            live_tool.AddTransformOp().Set(
                Gf.Matrix4d(
                    (
                        (1.0, 0.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (1.0, 2.0, 0.5, 1.0),
                    )
                )
            )
            asset_ref = UsdGeom.Xform.Define(stage, "/World/envs/env_0/GeneratedScene/tool_0/AssetRef")
            asset_ref.AddTransformOp(opSuffix="assetNormalization").Set(
                Gf.Matrix4d(
                    (
                        (0.01, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.01, 0.0),
                        (0.0, -0.01, 0.0, 0.0),
                        (0.0, 0.0, 0.0, 1.0),
                    )
                )
            )

            payload = build_stage_grasp_proposals(
                stage,
                manifest_path,
                annotation_root=annotation_root,
                target_prim_paths=["/World/tool_0"],
            )

            primitive = payload["objects"]["/World/tool_0"]["grasp_primitives_world"][0]
            self.assertEqual(primitive["point_world"], [1.0, 2.0, 1.5])
            self.assertAlmostEqual(primitive["axis_world"][0], 0.0, places=6)
            self.assertAlmostEqual(primitive["axis_world"][1], 0.0, places=6)
            self.assertAlmostEqual(primitive["axis_world"][2], 1.0, places=6)

    def test_cache_entry_path_for_target_prim_uses_asset_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = self._write_manifest(
                root,
                object_entries={
                    "/World/power_drill_1": {
                        "output_name": "5",
                        "class": "power_drill",
                        "caption": "orange cordless drill",
                        "source": "real2sim",
                        "glb_path": "objects/5.glb",
                    }
                },
            )
            annotation_root = default_grasp_annotation_root(manifest_path)
            resolved = cache_entry_path_for_prim(manifest_path, "/World/power_drill_1")
            self.assertEqual(resolved, annotation_root / "annotations" / "5.json")

    def test_ensure_asset_grasp_cache_for_prim_builds_only_requested_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = self._write_manifest(
                root,
                object_entries={
                    "/World/power_drill_1": {
                        "output_name": "5",
                        "class": "power_drill",
                        "caption": "orange cordless drill",
                        "source": "real2sim",
                        "glb_path": "objects/5.glb",
                    },
                    "/World/tool_4": {
                        "output_name": "8",
                        "class": "tool",
                        "caption": "green hex key set",
                        "source": "real2sim",
                        "glb_path": "objects/8.glb",
                    },
                },
            )

            built_glb_names: list[str] = []

            def fake_annotate_single_object(glb_path: Path, run_dir: Path, config, *, resume: bool):
                built_glb_names.append(Path(glb_path).name)
                object_dir = Path(run_dir) / Path(glb_path).stem
                object_dir.mkdir(parents=True, exist_ok=True)
                annotation_path = object_dir / "annotation.json"
                annotation_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "grasp_primitives_v1",
                            "object_name": Path(glb_path).stem,
                            "category": "axis_object",
                            "grasp_primitives": [
                                {
                                    "type": "axis_band",
                                    "point_local": [0.0, 0.0, 0.0],
                                    "axis_local": [1.0, 0.0, 0.0],
                                    "slide_range": [-0.05, 0.05],
                                }
                            ],
                            "artifacts": {},
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return {
                    "status": "ok",
                    "object_dir": str(object_dir),
                    "annotation": str(annotation_path),
                }

            with patch(
                "app.backend.services.grasp_asset_cache.annotate_single_object",
                side_effect=fake_annotate_single_object,
            ):
                result = ensure_asset_grasp_cache_for_prim(
                    manifest_path,
                    "/World/power_drill_1",
                )

            self.assertEqual(built_glb_names, ["5.glb"])
            self.assertTrue(result["cache_exists_now"])
            self.assertTrue(Path(result["cache_path"]).exists())
            self.assertFalse((default_grasp_annotation_root(manifest_path) / "annotations" / "8.json").exists())


if __name__ == "__main__":
    unittest.main()
