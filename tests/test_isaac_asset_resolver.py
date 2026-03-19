import tempfile
import unittest
from pathlib import Path

from pipelines.isaac.asset_resolver import build_asset_match_lookup, build_real2sim_uniform_scale_lookup


class IsaacAssetResolverTest(unittest.TestCase):
    def test_build_real2sim_uniform_scale_lookup_reads_column_norms(self) -> None:
        manifest = {
            "objects": {
                "/World/table_0": {
                    "usd_transform": [
                        [0.0, -2.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 2.0, 0.0],
                        [1.0, 2.0, 3.0, 1.0],
                    ]
                },
                "/World/laptop_1": {
                    "usd_transform": [
                        [0.25, 0.0, 0.0, 0.0],
                        [0.0, 0.25, 0.0, 0.0],
                        [0.0, 0.0, 0.25, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                },
                "/World/invalid_2": {"usd_transform": []},
            }
        }

        lookup = build_real2sim_uniform_scale_lookup(manifest)

        self.assertAlmostEqual(lookup["/World/table_0"], 2.0)
        self.assertAlmostEqual(lookup["/World/laptop_1"], 0.25)
        self.assertNotIn("/World/invalid_2", lookup)

    def test_build_asset_match_lookup_routes_by_source_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retrieval_root = root / "testusd"
            fallback_root = root / "legacy"
            retrieval_root.mkdir(parents=True)
            fallback_root.mkdir(parents=True)

            retrieval_usd = retrieval_root / "chair_asset.usd"
            fallback_usd = fallback_root / "cup_asset.usd"
            scene_usd = root / "scene_merged_post.usd"
            retrieval_usd.write_text("#usda 1.0\n", encoding="utf-8")
            fallback_usd.write_text("#usda 1.0\n", encoding="utf-8")
            scene_usd.write_text("#usda 1.0\n", encoding="utf-8")

            data = {
                "obj": {
                    "/World/chair_0": {"class": "chair", "source": "retrieval"},
                    "/World/table_1": {"class": "table", "source": "real2sim"},
                    "/World/cup_2": {"class": "cup"},
                }
            }
            manifest = {
                "results_root": str(root.resolve()),
                "scene_usd": "scene_merged_post.usd",
                "objects": {
                    "/World/table_1": {
                        "usd_path": "scene_merged_post.usd",
                        "usd_prim_path": "/World/obj_01",
                    }
                },
            }

            lookup = build_asset_match_lookup(
                data,
                [fallback_usd],
                retrieval_usd_paths=[retrieval_usd],
                real2sim_manifest=manifest,
            )

            self.assertEqual(lookup["/World/chair_0"].asset_path, retrieval_usd.resolve())
            self.assertEqual(lookup["/World/chair_0"].source, "retrieval")
            self.assertIsNone(lookup["/World/chair_0"].reference_prim_path)

            self.assertEqual(lookup["/World/table_1"].asset_path, scene_usd.resolve())
            self.assertEqual(lookup["/World/table_1"].source, "real2sim")
            self.assertEqual(lookup["/World/table_1"].reference_prim_path, "/World/obj_01")

            self.assertEqual(lookup["/World/cup_2"].asset_path, fallback_usd.resolve())
            self.assertEqual(lookup["/World/cup_2"].source, "fallback")
            self.assertIsNone(lookup["/World/cup_2"].reference_prim_path)


if __name__ == "__main__":
    unittest.main()
