from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pxr import Usd, UsdGeom

from app.backend.services.robot_placement import RobotPlacementPlan
from app.backend.services.robot_scene import (
    compute_robot_floor_offset_z,
    embed_robot_in_scene_usd,
)


class RobotSceneTest(unittest.TestCase):
    def test_compute_robot_floor_offset_is_non_negative(self) -> None:
        self.assertGreaterEqual(compute_robot_floor_offset_z("agibot"), 0.0)
        self.assertGreaterEqual(compute_robot_floor_offset_z("kinova"), 0.0)

    def test_embed_robot_in_scene_writes_robot_prim(self) -> None:
        plan = RobotPlacementPlan(
            robot="agibot",
            target_prim="/World/mug_0",
            support_prim="/World/table_0",
            support_center_xy=(0.0, 0.0),
            support_z=0.75,
            support_yaw_deg=0.0,
            support_half_extents_xy=(0.5, 0.4),
            support_shape="rect",
            chosen_side="front",
            base_pose=(1.25, -0.4, 0.0, 135.0),
            room_bounds=(-2.0, 2.0, -2.0, 2.0),
            supported_objects=["/World/mug_0"],
            floor_obstacles=[],
            candidates=[],
        )
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "base_scene.usda"
            output = Path(tmpdir) / "scene_with_robot.usda"
            stage = Usd.Stage.CreateNew(str(source))
            world = stage.DefinePrim("/World", "Xform")
            stage.SetDefaultPrim(world)
            stage.GetRootLayer().Save()

            result = embed_robot_in_scene_usd(source, plan, output_usd_path=output, floor_z=0.0)

            saved = Usd.Stage.Open(str(output))
            self.assertIsNotNone(saved)
            wrapper = saved.GetPrimAtPath("/World/RobotPlacement")
            self.assertTrue(wrapper.IsValid())
            prim = saved.GetPrimAtPath("/World/RobotPlacement/RobotAsset")
            self.assertTrue(prim.IsValid())
            refs = prim.GetMetadata("references")
            self.assertIsNotNone(refs)

            xform = UsdGeom.XformCommonAPI(wrapper)
            translate = xform.GetXformVectors(Usd.TimeCode.Default())[0]
            rotate = xform.GetXformVectors(Usd.TimeCode.Default())[1]
            self.assertAlmostEqual(float(translate[0]), 1.25, places=6)
            self.assertAlmostEqual(float(translate[1]), -0.4, places=6)
            self.assertAlmostEqual(float(translate[2]), result.robot_floor_offset_z, places=6)
            self.assertAlmostEqual(float(rotate[2]), 135.0, places=6)


if __name__ == "__main__":
    unittest.main()
