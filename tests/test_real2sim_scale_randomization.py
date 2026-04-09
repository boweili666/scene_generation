import json
from pathlib import Path
import sys
import tempfile
import unittest

from pxr import Gf, Usd, UsdGeom


SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "scene_robot" / "src"
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from scene_robot_apps.real2sim_scale_randomization import randomize_real2sim_asset_scales


class Real2SimScaleRandomizationTest(unittest.TestCase):
    def _make_scene_run(self, root: Path) -> tuple[Path, Path]:
        table_asset = root / "table_asset.usda"
        tool_asset = root / "tool_asset.usda"
        self._build_box_asset(table_asset, size_xyz=(2.0, 1.0, 1.0))
        self._build_box_asset(tool_asset, size_xyz=(0.4, 0.4, 0.4))

        scene_usd = root / "scene_latest.usda"
        stage = Usd.Stage.CreateNew(str(scene_usd))
        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)
        self._define_object(stage, "/World/table_0", table_asset, translation_xyz=(0.0, 0.0, 0.5))
        self._define_object(stage, "/World/tool_0", tool_asset, translation_xyz=(0.3, 0.0, 1.2))
        stage.GetRootLayer().Save()

        run_root = root / "session_run"
        (run_root / "scene_service" / "usd").mkdir(parents=True, exist_ok=True)
        scene_run_usd = run_root / "scene_service" / "usd" / "scene_latest.usda"
        scene_run_usd.write_bytes(scene_usd.read_bytes())

        scene_graph_path = run_root / "scene_graph" / "current_scene_graph.json"
        scene_graph_path.parent.mkdir(parents=True, exist_ok=True)
        scene_graph_path.write_text(
            json.dumps(
                {
                    "obj": {
                        "/World/table_0": {"class": "table", "caption": "table", "source": "real2sim"},
                        "/World/tool_0": {"class": "tool", "caption": "tool", "source": "real2sim"},
                    },
                    "edges": {
                        "obj-obj": [
                            {"source": "/World/tool_0", "target": "/World/table_0", "relation": "supported by"},
                            {"source": "/World/table_0", "target": "/World/tool_0", "relation": "supports"},
                        ]
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return scene_run_usd, scene_graph_path

    def _build_box_asset(self, usd_path: Path, *, size_xyz: tuple[float, float, float]) -> None:
        stage = Usd.Stage.CreateNew(str(usd_path))
        root = stage.DefinePrim("/Asset", "Xform")
        stage.SetDefaultPrim(root)
        cube = UsdGeom.Cube.Define(stage, "/Asset/geometry")
        cube.CreateSizeAttr(1.0)
        half = [float(value) * 0.5 for value in size_xyz]
        cube.CreateExtentAttr([(-half[0], -half[1], -half[2]), (half[0], half[1], half[2])])
        stage.GetRootLayer().Save()

    def _define_object(
        self,
        stage,
        prim_path: str,
        asset_path: Path,
        *,
        translation_xyz: tuple[float, float, float],
    ) -> None:
        object_xform = UsdGeom.Xform.Define(stage, prim_path)
        object_xform.AddTransformOp().Set(
            Gf.Matrix4d(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (float(translation_xyz[0]), float(translation_xyz[1]), float(translation_xyz[2]), 1.0),
                )
            )
        )
        asset_ref = UsdGeom.Xform.Define(stage, f"{prim_path}/AssetRef")
        asset_ref.AddTransformOp(opSuffix="assetNormalization").Set(
            Gf.Matrix4d(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )
        )
        asset_ref.GetPrim().GetReferences().AddReference(str(asset_path))

    def test_randomization_scales_assets_and_preserves_support_contact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_run_usd, scene_graph_path = self._make_scene_run(root)
            run_root = root / "session_run"

            output_usd = run_root / "scene_service" / "usd" / "scene_latest.scale_randomized.usda"
            output_json = output_usd.with_suffix(".randomization.json")

            result = randomize_real2sim_asset_scales(
                scene_run_usd,
                output_usd_path=output_usd,
                output_metadata_path=output_json,
                scene_graph_path=scene_graph_path,
                min_scale=1.0,
                max_scale=1.0,
                scale_overrides={"/World/table_0": 1.5, "/World/tool_0": 0.5},
            )

            self.assertEqual(result["object_count"], 2)
            saved_stage = Usd.Stage.Open(str(output_usd))
            self.assertIsNotNone(saved_stage)

            table_bbox = self._world_bbox(saved_stage, "/World/table_0")
            tool_bbox = self._world_bbox(saved_stage, "/World/tool_0")

            self.assertAlmostEqual(table_bbox["min"][2], 0.0, places=6)
            self.assertAlmostEqual(tool_bbox["min"][2], table_bbox["max"][2], places=6)

            table_center_x = float((table_bbox["min"][0] + table_bbox["max"][0]) * 0.5)
            tool_center_x = float((tool_bbox["min"][0] + tool_bbox["max"][0]) * 0.5)
            self.assertAlmostEqual(tool_center_x - table_center_x, 0.45, places=6)

            metadata = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertAlmostEqual(float(metadata["objects"]["/World/table_0"]["scale"]), 1.5, places=6)
            self.assertAlmostEqual(float(metadata["objects"]["/World/tool_0"]["scale"]), 0.5, places=6)

    def test_global_scale_applies_one_shared_scale_to_all_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_run_usd, scene_graph_path = self._make_scene_run(root)
            run_root = root / "session_run"

            output_usd = run_root / "scene_service" / "usd" / "scene_latest.global_scale.usda"
            output_json = output_usd.with_suffix(".randomization.json")

            result = randomize_real2sim_asset_scales(
                scene_run_usd,
                output_usd_path=output_usd,
                output_metadata_path=output_json,
                scene_graph_path=scene_graph_path,
                global_scale=1.25,
            )

            self.assertEqual(result["scale_mode"], "global_fixed")
            self.assertAlmostEqual(float(result["shared_scale"]), 1.25, places=6)
            self.assertAlmostEqual(float(result["scales"]["/World/table_0"]), 1.25, places=6)
            self.assertAlmostEqual(float(result["scales"]["/World/tool_0"]), 1.25, places=6)

            saved_stage = Usd.Stage.Open(str(output_usd))
            self.assertIsNotNone(saved_stage)

            table_bbox = self._world_bbox(saved_stage, "/World/table_0")
            tool_bbox = self._world_bbox(saved_stage, "/World/tool_0")

            self.assertAlmostEqual(table_bbox["min"][2], 0.0, places=6)
            self.assertAlmostEqual(tool_bbox["min"][2], table_bbox["max"][2], places=6)

            table_center_x = float((table_bbox["min"][0] + table_bbox["max"][0]) * 0.5)
            tool_center_x = float((tool_bbox["min"][0] + tool_bbox["max"][0]) * 0.5)
            self.assertAlmostEqual(tool_center_x - table_center_x, 0.375, places=6)

            metadata = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(metadata["scale_mode"], "global_fixed")
            self.assertAlmostEqual(float(metadata["shared_scale"]), 1.25, places=6)
            self.assertAlmostEqual(float(metadata["objects"]["/World/table_0"]["scale"]), 1.25, places=6)
            self.assertAlmostEqual(float(metadata["objects"]["/World/tool_0"]["scale"]), 1.25, places=6)

    def _world_bbox(self, stage, prim_path: str) -> dict[str, list[float]]:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
        rng = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
        bmin = rng.GetMin()
        bmax = rng.GetMax()
        return {
            "min": [float(bmin[0]), float(bmin[1]), float(bmin[2])],
            "max": [float(bmax[0]), float(bmax[1]), float(bmax[2])],
        }


if __name__ == "__main__":
    unittest.main()
