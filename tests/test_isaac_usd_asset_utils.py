import tempfile
import unittest
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from pipelines.isaac.usd_asset_utils import (
    column_transform_to_row_major,
    compute_asset_local_to_scene_matrix,
    transform_aligned_bbox,
)


class IsaacUsdAssetUtilsTest(unittest.TestCase):
    def test_compute_asset_local_to_scene_matrix_converts_y_up_centimeters_to_z_up_meters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            usd_path = Path(tmp) / "asset.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            world = stage.DefinePrim("/World", "Xform")
            stage.SetDefaultPrim(world)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            UsdGeom.SetStageMetersPerUnit(stage, 0.01)
            stage.GetRootLayer().Save()

            matrix = compute_asset_local_to_scene_matrix(usd_path)

            expected = np.array(
                [
                    [0.01, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.01, 0.0],
                    [0.0, 0.01, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            np.testing.assert_allclose(matrix, expected, atol=1e-8)

    def test_transform_aligned_bbox_applies_axis_swap_and_unit_scale(self) -> None:
        info = {"size": (100.0, 60.0, 70.0), "center": (0.0, 0.0, 0.0)}
        matrix = np.array(
            [
                [0.01, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.01, 0.0],
                [0.0, 0.01, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        transformed = transform_aligned_bbox(info, matrix)

        np.testing.assert_allclose(transformed["size"], np.array([1.0, 0.7, 0.6], dtype=float), atol=1e-8)
        np.testing.assert_allclose(transformed["center"], np.zeros(3, dtype=float), atol=1e-8)

    def test_column_transform_to_row_major_transposes_for_usd_xform_ops(self) -> None:
        matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.01, 0.0],
                [0.0, 0.01, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        converted = column_transform_to_row_major(matrix)

        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.01, 0.0],
                [0.0, -0.01, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(converted, expected, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
