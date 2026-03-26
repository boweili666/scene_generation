import math
import re
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from app.backend.services.robot_placement import (
    find_support_prim,
    plan_robot_base_pose,
    render_plan_svg,
    resolve_target_prim,
)


SCENE_GRAPH = {
    "scene": {
        "room_type": "studio",
        "dimensions": {"length": 6.0, "width": 6.0, "height": 3.0, "unit": "m"},
        "materials": {"floor": "wood", "walls": "white"},
    },
    "obj": {
        "/World/table_0": {"id": 0, "class": "table", "caption": "round wooden table", "source": "real2sim"},
        "/World/mug_0": {"id": 1, "class": "mug", "caption": "ceramic mug", "source": "real2sim"},
        "/World/chair_0": {"id": 2, "class": "chair", "caption": "desk chair", "source": "retrieval"},
        "/World/plant_0": {"id": 3, "class": "plant", "caption": "floor plant", "source": "retrieval"},
        "/World/lamp_0": {"id": 4, "class": "lamp", "caption": "standing lamp", "source": "retrieval"},
    },
    "edges": {
        "obj-obj": [
            {"source": "/World/mug_0", "target": "/World/table_0", "relation": "supported by"},
            {"source": "/World/table_0", "target": "/World/mug_0", "relation": "supports"},
        ],
        "obj-wall": [],
    },
}


class RobotPlacementTest(unittest.TestCase):
    def test_find_support_from_scene_graph(self) -> None:
        self.assertEqual(find_support_prim(SCENE_GRAPH, "/World/mug_0"), "/World/table_0")

    def test_resolve_default_target_prefers_supported_tabletop_object(self) -> None:
        placements = {
            "/World/table_0": [0.0, 0.0, 0.75, 0.0],
            "/World/mug_0": [0.18, 0.05, 0.87, 0.0],
            "/World/chair_0": [1.0, 0.0, 0.5, 0.0],
        }
        self.assertEqual(resolve_target_prim(SCENE_GRAPH, placements), "/World/mug_0")

    def test_plan_prefers_the_clearest_side(self) -> None:
        placements = {
            "/World/table_0": [0.0, 0.0, 0.75, 0.0],
            "/World/mug_0": [0.18, 0.05, 0.87, 0.0],
            "/World/chair_0": [1.10, 0.0, 0.5, 0.0],
            "/World/plant_0": [0.0, 1.10, 0.4, 0.0],
            "/World/lamp_0": [0.0, -1.10, 0.4, 0.0],
        }

        plan = plan_robot_base_pose(SCENE_GRAPH, placements, target_prim="/World/mug_0", robot="agibot")

        self.assertEqual(plan.chosen_side, "back")
        self.assertLess(plan.base_pose[0], 0.0)
        self.assertAlmostEqual(plan.base_pose[1], 0.0, places=3)
        self.assertAlmostEqual(plan.base_pose[3], 0.0, places=3)

    def test_plan_yaw_points_back_to_support_center(self) -> None:
        placements = {
            "/World/table_0": [0.3, -0.8, 0.75, 0.0],
            "/World/mug_0": [0.45, -0.75, 0.87, 0.0],
            "/World/chair_0": [1.2, -0.8, 0.5, 0.0],
            "/World/plant_0": [0.3, 0.3, 0.4, 0.0],
            "/World/lamp_0": [0.3, -1.9, 0.4, 0.0],
        }

        plan = plan_robot_base_pose(SCENE_GRAPH, placements, target_prim="/World/mug_0", robot="agibot")

        dx = plan.support_center_xy[0] - plan.base_pose[0]
        dy = plan.support_center_xy[1] - plan.base_pose[1]
        expected = math.degrees(math.atan2(dy, dx))
        self.assertAlmostEqual(plan.base_pose[3], expected, places=6)

    def test_plan_prefers_nearest_target_when_multiple_sides_are_clear(self) -> None:
        placements = {
            "/World/table_0": [0.0, 0.0, 0.75, 0.0],
            "/World/mug_0": [0.30, 0.02, 0.87, 0.0],
        }

        plan = plan_robot_base_pose(SCENE_GRAPH, placements, target_prim="/World/mug_0", robot="agibot")

        self.assertEqual(plan.chosen_side, "front")
        self.assertGreater(plan.base_pose[0], 0.0)
        self.assertAlmostEqual(plan.base_pose[1], 0.0, places=3)

    def test_plan_uses_floor_obstacle_aabb_not_outer_circle(self) -> None:
        scene_graph = deepcopy(SCENE_GRAPH)
        scene_graph["obj"]["/World/bed_0"] = {
            "id": 5,
            "class": "bed",
            "caption": "bed against wall",
            "source": "retrieval",
        }
        placements = {
            "/World/table_0": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.4,
                "yaw": 0.0,
                "aabb": {
                    "min": [-0.55, -0.55, 0.0],
                    "max": [0.55, 0.55, 0.8],
                    "center": [0.0, 0.0, 0.4],
                    "size": [1.1, 1.1, 0.8],
                },
            },
            "/World/mug_0": {
                "x": -0.05,
                "y": 0.38,
                "z": 0.9,
                "yaw": 0.0,
                "aabb": {
                    "min": [-0.10, 0.33, 0.84],
                    "max": [0.00, 0.43, 0.96],
                    "center": [-0.05, 0.38, 0.9],
                    "size": [0.10, 0.10, 0.12],
                },
            },
            "/World/chair_0": {
                "x": 1.2,
                "y": 0.0,
                "z": 0.5,
                "yaw": 0.0,
                "aabb": {
                    "min": [0.9, -0.3, 0.0],
                    "max": [1.5, 0.3, 1.0],
                    "center": [1.2, 0.0, 0.5],
                    "size": [0.6, 0.6, 1.0],
                },
            },
            "/World/plant_0": [0.0, -1.6, 0.4, 0.0],
            "/World/lamp_0": [0.0, 1.9, 0.4, 0.0],
            "/World/bed_0": {
                "x": -1.75,
                "y": 1.75,
                "z": 0.5,
                "yaw": 0.0,
                "aabb": {
                    "min": [-2.25, -0.30, 0.0],
                    "max": [-1.14, 2.25, 1.30],
                    "center": [-1.695, 0.975, 0.65],
                    "size": [1.11, 2.55, 1.30],
                },
            },
        }

        plan = plan_robot_base_pose(scene_graph, placements, target_prim="/World/mug_0", robot="agibot")

        self.assertEqual(plan.chosen_side, "right")
        self.assertTrue(next(c for c in plan.candidates if c.side == "right").overlap_free)

    def test_plan_uses_support_aabb_instead_of_surface_prior(self) -> None:
        scene_graph = deepcopy(SCENE_GRAPH)
        scene_graph["obj"]["/World/table_0"]["caption"] = "rectangular work table"
        placements = {
            "/World/table_0": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.4,
                "yaw": 0.0,
                "aabb": {
                    "min": [-1.0, -0.5, 0.0],
                    "max": [1.0, 0.5, 0.8],
                    "center": [0.0, 0.0, 0.4],
                    "size": [2.0, 1.0, 0.8],
                },
            },
            "/World/mug_0": {
                "x": 0.25,
                "y": 0.05,
                "z": 0.9,
                "yaw": 0.0,
                "aabb": {
                    "min": [0.20, 0.0, 0.84],
                    "max": [0.30, 0.10, 0.96],
                    "center": [0.25, 0.05, 0.9],
                    "size": [0.10, 0.10, 0.12],
                },
            },
            "/World/chair_0": [1.4, 0.0, 0.5, 0.0],
            "/World/plant_0": [0.0, 1.3, 0.4, 0.0],
            "/World/lamp_0": [0.0, -1.3, 0.4, 0.0],
        }

        plan = plan_robot_base_pose(scene_graph, placements, target_prim="/World/mug_0", robot="agibot")

        self.assertEqual(plan.chosen_side, "back")
        self.assertAlmostEqual(plan.support_half_extents_xy[0], 1.0, places=3)
        self.assertAlmostEqual(plan.support_half_extents_xy[1], 0.5, places=3)
        self.assertAlmostEqual(plan.base_pose[0], -1.45, places=3)
        self.assertAlmostEqual(plan.base_pose[1], 0.0, places=3)

    def test_plan_keeps_robot_bbox_gap_near_point_one_meter(self) -> None:
        scene_graph = deepcopy(SCENE_GRAPH)
        scene_graph["obj"]["/World/table_0"]["caption"] = "rectangular work table"
        placements = {
            "/World/table_0": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.4,
                "yaw": 0.0,
                "aabb": {
                    "min": [-1.0, -0.5, 0.0],
                    "max": [1.0, 0.5, 0.8],
                    "center": [0.0, 0.0, 0.4],
                    "size": [2.0, 1.0, 0.8],
                },
            },
            "/World/mug_0": [0.25, 0.05, 0.9, 0.0],
            "/World/chair_0": [1.5, 0.0, 0.5, 0.0],
            "/World/plant_0": [0.0, 1.5, 0.4, 0.0],
            "/World/lamp_0": [0.0, -1.5, 0.4, 0.0],
        }

        plan = plan_robot_base_pose(scene_graph, placements, target_prim="/World/mug_0", robot="agibot")

        support_half_extent_x = plan.support_half_extents_xy[0]
        robot_radius = 0.35
        gap = abs(plan.base_pose[0] - plan.support_center_xy[0]) - support_half_extent_x - robot_radius
        self.assertAlmostEqual(gap, 0.1, places=6)

    def test_render_plan_svg_prefers_aabb_rectangles(self) -> None:
        placements = {
            "/World/table_0": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.4,
                "yaw": 0.0,
                "aabb": {
                    "min": [-1.0, -0.5, 0.0],
                    "max": [1.0, 0.5, 0.8],
                    "center": [0.0, 0.0, 0.4],
                    "size": [2.0, 1.0, 0.8],
                },
            },
            "/World/mug_0": {
                "x": 0.25,
                "y": 0.05,
                "z": 0.9,
                "yaw": 0.0,
                "aabb": {
                    "min": [0.20, 0.0, 0.84],
                    "max": [0.30, 0.10, 0.96],
                    "center": [0.25, 0.05, 0.9],
                    "size": [0.10, 0.10, 0.12],
                },
            },
            "/World/chair_0": {
                "x": 1.1,
                "y": 0.0,
                "z": 0.5,
                "yaw": 0.0,
                "aabb": {
                    "min": [0.8, -0.3, 0.0],
                    "max": [1.4, 0.3, 1.0],
                    "center": [1.1, 0.0, 0.5],
                    "size": [0.6, 0.6, 1.0],
                },
            },
            "/World/plant_0": [0.0, 1.1, 0.4, 0.0],
            "/World/lamp_0": [0.0, -1.1, 0.4, 0.0],
        }
        plan = plan_robot_base_pose(SCENE_GRAPH, placements, target_prim="/World/mug_0", robot="agibot")

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "plan.svg"
            render_plan_svg(SCENE_GRAPH, placements, plan, svg_path)
            svg = svg_path.read_text(encoding="utf-8")

        self.assertRegex(svg, r'<rect[^>]*fill="#b8c7ff"')
        self.assertRegex(svg, r'<rect[^>]*fill="#ff6b6b"')
        self.assertRegex(svg, r'<rect[^>]*fill="#7f8da8"')
        self.assertNotRegex(svg, r'<circle[^>]*fill="#b8c7ff"')


if __name__ == "__main__":
    unittest.main()
