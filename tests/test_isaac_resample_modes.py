import unittest

import numpy as np

from pipelines.isaac.resample_modes import (
    apply_lock_real2sim_relative_transforms,
    column_transform_to_row_major,
    row_transform_to_column_major,
)


def _translate(tx: float, ty: float, tz: float) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = np.array([tx, ty, tz], dtype=float)
    return matrix


class IsaacResampleModesTest(unittest.TestCase):
    def test_lock_real2sim_keeps_supported_children_rigid_relative_to_root(self) -> None:
        data = {
            "obj": {
                "/World/table_0": {"class": "table", "source": "real2sim"},
                "/World/mug_1": {"class": "mug", "source": "real2sim"},
                "/World/chair_2": {"class": "chair", "source": "retrieval"},
            },
            "edges": {
                "obj-obj": [
                    {
                        "source": "/World/mug_1",
                        "target": "/World/table_0",
                        "relation": "supported by",
                    }
                ]
            },
        }
        manifest = {
            "objects": {
                "/World/table_0": {"usd_transform": _translate(0.0, 0.0, 0.0).tolist()},
                "/World/mug_1": {"usd_transform": _translate(0.25, 0.0, 0.8).tolist()},
            }
        }
        root_sampled = _translate(2.0, -1.5, 0.0)
        child_joint_sampled = _translate(9.0, 9.0, 9.0)
        object_entries = [
            {
                "prim": "/World/table_0",
                "name": "table",
                "transform": column_transform_to_row_major(root_sampled),
            },
            {
                "prim": "/World/mug_1",
                "name": "mug",
                "transform": column_transform_to_row_major(child_joint_sampled),
            },
        ]
        placements = {
            "/World/table_0": (2.0, -1.5, 0.0),
            "/World/mug_1": (9.0, 9.0, 9.0),
        }
        asset_bbox_lookup = {
            "/World/table_0": {"center": (0.0, 0.0, 0.0), "size": (1.0, 1.0, 1.0)},
            "/World/mug_1": {"center": (0.0, 0.0, 0.0), "size": (0.2, 0.2, 0.4)},
        }

        updated_entries, updated_placements, debug = apply_lock_real2sim_relative_transforms(
            data,
            object_entries,
            placements,
            asset_bbox_lookup=asset_bbox_lookup,
            real2sim_manifest=manifest,
        )

        entry_lookup = {entry["prim"]: entry for entry in updated_entries}
        child_world = row_transform_to_column_major(entry_lookup["/World/mug_1"]["transform"])
        np.testing.assert_allclose(child_world[:3, 3], np.array([2.25, -1.5, 0.8]), atol=1e-8)
        self.assertEqual(updated_placements["/World/mug_1"], (2.25, -1.5, 0.8))
        self.assertTrue(debug["mode_applied"])
        self.assertEqual(debug["real2sim_roots"], ["/World/table_0"])
        self.assertEqual(debug["locked_real2sim_children"], ["/World/mug_1"])

    def test_lock_real2sim_skips_chain_supported_by_non_real2sim_parent(self) -> None:
        data = {
            "obj": {
                "/World/table_0": {"class": "table", "source": "retrieval"},
                "/World/mug_1": {"class": "mug", "source": "real2sim"},
            },
            "edges": {
                "obj-obj": [
                    {
                        "source": "/World/mug_1",
                        "target": "/World/table_0",
                        "relation": "supported by",
                    }
                ]
            },
        }
        manifest = {
            "objects": {
                "/World/mug_1": {"usd_transform": _translate(0.25, 0.0, 0.8).tolist()},
            }
        }
        child_joint_sampled = _translate(4.0, 5.0, 6.0)
        object_entries = [
            {
                "prim": "/World/mug_1",
                "name": "mug",
                "transform": column_transform_to_row_major(child_joint_sampled),
            }
        ]
        placements = {"/World/mug_1": (4.0, 5.0, 6.0)}

        updated_entries, updated_placements, debug = apply_lock_real2sim_relative_transforms(
            data,
            object_entries,
            placements,
            asset_bbox_lookup=None,
            real2sim_manifest=manifest,
        )

        self.assertEqual(updated_entries, object_entries)
        self.assertEqual(updated_placements, placements)
        self.assertFalse(debug["mode_applied"])
        self.assertEqual(debug["skipped_real2sim_prims"], ["/World/mug_1"])
        self.assertEqual(debug["locked_real2sim_children"], [])


if __name__ == "__main__":
    unittest.main()
