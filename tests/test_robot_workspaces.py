from pathlib import Path
import sys
import tempfile
import unittest

from pxr import Usd


SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "scene_robot" / "src"
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from scene_robot_apps.robot_workspaces import (
    add_robot_workspace_visuals_to_stage,
    project_workspace_box_to_support,
    default_robot_workspace_specs,
    save_robot_workspace_overview,
)


class RobotWorkspaceVisualizationTest(unittest.TestCase):
    def test_default_robot_workspace_specs_cover_three_robots(self) -> None:
        specs = default_robot_workspace_specs()
        self.assertEqual(set(specs.keys()), {"kinova", "agibot", "r1lite"})
        self.assertEqual(specs["kinova"].working_area.name, "working_area")
        self.assertEqual(specs["agibot"].working_area.name, "working_area")
        self.assertEqual(specs["r1lite"].working_area.name, "working_area")
        self.assertGreater(float(specs["agibot"].working_area.size_xy[1]), 0.0)

    def test_save_robot_workspace_overview_writes_png_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "robot_workspaces.png"
            json_path = root / "robot_workspaces.json"
            result = save_robot_workspace_overview(
                output_image_path=image_path,
                output_json_path=json_path,
            )

            self.assertTrue(image_path.exists())
            self.assertGreater(image_path.stat().st_size, 0)
            self.assertTrue(json_path.exists())
            self.assertEqual(len(result["robots"]), 3)

    def test_add_robot_workspace_visuals_to_stage_creates_workspace_root(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)
        stage.DefinePrim("/World/Robot", "Xform")

        result = add_robot_workspace_visuals_to_stage(
            stage,
            robot="agibot",
            robot_prim_path="/World/Robot",
        )

        self.assertEqual(result["workspace_root_path"], "/World/Robot/WorkspaceVisuals")
        self.assertTrue(stage.GetPrimAtPath("/World/Robot/WorkspaceVisuals").IsValid())
        self.assertGreaterEqual(len(result["created_paths"]), 1)

    def test_project_workspace_box_to_support_clips_to_support_surface(self) -> None:
        projected = project_workspace_box_to_support(
            robot="agibot",
            base_pose=(0.0, 0.0, 0.0, 0.0),
            support_center_xy=(0.60, 0.0),
            support_half_extents_xy=(0.45, 0.30),
            support_yaw_deg=0.0,
        )
        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertAlmostEqual(float(projected["center_xy"][1]), 0.0, places=6)
        self.assertLessEqual(float(projected["size_xy"][0]), 0.90)
        self.assertLessEqual(float(projected["size_xy"][1]), 0.60)
        self.assertGreater(float(projected["size_xy"][0]), 0.0)
        self.assertGreater(float(projected["size_xy"][1]), 0.0)


if __name__ == "__main__":
    unittest.main()
