import math
import unittest

from pipelines.isaac.layout_utils import clamp_center_to_room_bounds, effective_xy_half_extents


class IsaacLayoutUtilsTest(unittest.TestCase):
    def test_effective_xy_half_extents_respects_yaw(self) -> None:
        hx, hy = effective_xy_half_extents((4.0, 2.0, 1.0), math.radians(90.0))
        self.assertAlmostEqual(hx, 1.0)
        self.assertAlmostEqual(hy, 2.0)

    def test_clamp_center_to_room_bounds_respects_closed_front_wall(self) -> None:
        center = clamp_center_to_room_bounds(
            (4.5, 0.0, 0.0),
            (4.0, 2.0, 1.0),
            math.radians(90.0),
            (-5.0, 5.0, -5.0, 5.0),
            {"behind": True, "left": True, "right": True, "front": True},
        )
        self.assertAlmostEqual(center[0], 4.0)
        self.assertAlmostEqual(center[1], 0.0)

    def test_clamp_center_to_room_bounds_allows_open_front_wall(self) -> None:
        center = clamp_center_to_room_bounds(
            (4.5, 0.0, 0.0),
            (4.0, 2.0, 1.0),
            math.radians(90.0),
            (-5.0, 5.0, -5.0, 5.0),
            {"behind": True, "left": True, "right": True, "front": False},
        )
        self.assertAlmostEqual(center[0], 4.5)
        self.assertAlmostEqual(center[1], 0.0)


if __name__ == "__main__":
    unittest.main()
