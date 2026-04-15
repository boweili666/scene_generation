from pathlib import Path
import sys
import unittest


SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "scene_robot" / "src"
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from scene_robot_apps.grasp_execution import (
    FilteredGraspExecution,
    GraspExecutionPose,
    expand_grasp_candidates,
    filter_grasp_candidates_geometry,
    infer_arm_side,
    rank_filtered_grasp_candidates_by_start_pose,
)


class GraspExecutionTest(unittest.TestCase):
    def test_expand_grasp_candidates_handles_point_axis_and_pose_set(self) -> None:
        payload = {
            "objects": {
                "/World/tool_0": {
                    "grasp_primitives_world": [
                        {
                            "type": "point_grasp",
                            "point_world": [0.6, 0.2, 0.9],
                            "approach_dirs_world": [[0.0, 0.0, -1.0]],
                            "closing_dirs_world": [[1.0, 0.0, 0.0]],
                            "width_range": [0.02, 0.04],
                            "score": 0.9,
                        },
                        {
                            "type": "axis_band",
                            "point_world": [0.6, -0.2, 0.9],
                            "axis_world": [1.0, 0.0, 0.0],
                            "slide_range_world": [-0.05, 0.05],
                            "approach_dirs_world": [[0.0, 0.0, -1.0]],
                            "closing_dirs_world": [[0.0, 1.0, 0.0]],
                            "width_range": [0.03, 0.05],
                            "score": 0.8,
                        },
                        {
                            "type": "pose_set",
                            "poses_world": [
                                {
                                    "translation": [0.55, 0.0, 0.95],
                                    "rotation_matrix": [
                                        [1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                    ],
                                    "width": 0.04,
                                    "score": 0.7,
                                }
                            ],
                        },
                    ]
                }
            }
        }
        candidates = expand_grasp_candidates(payload, target_prim="/World/tool_0", axis_band_slide_samples=3)
        self.assertEqual(len(candidates), 5)
        self.assertEqual(candidates[0].object_prim, "/World/tool_0")
        self.assertIn(candidates[0].primitive_type, {"point_grasp", "axis_band", "pose_set"})

    def test_infer_arm_side_uses_base_frame_y(self) -> None:
        base_pose = (0.0, 0.0, 0.0, 0.0)
        self.assertEqual(infer_arm_side("agibot", base_pose, (0.5, 0.2, 0.9)), "left")
        self.assertEqual(infer_arm_side("agibot", base_pose, (0.5, -0.2, 0.9)), "right")
        self.assertEqual(infer_arm_side("kinova", base_pose, (0.5, -0.2, 0.9)), "left")

    def test_geometric_filter_keeps_workspace_candidate_and_rejects_body_overlap(self) -> None:
        payload = {
            "objects": {
                "/World/tool_0": {
                    "grasp_primitives_world": [
                        {
                            "type": "point_grasp",
                            "point_world": [0.60, 0.20, 0.92],
                            "approach_dirs_world": [[0.0, 0.0, -1.0]],
                            "closing_dirs_world": [[1.0, 0.0, 0.0]],
                            "width_range": [0.02, 0.04],
                            "score": 0.9,
                        },
                        {
                            "type": "point_grasp",
                            "point_world": [0.10, 0.01, 0.92],
                            "approach_dirs_world": [[0.0, 0.0, -1.0]],
                            "closing_dirs_world": [[1.0, 0.0, 0.0]],
                            "width_range": [0.02, 0.04],
                            "score": 0.8,
                        },
                    ]
                }
            }
        }
        candidates = expand_grasp_candidates(payload, target_prim="/World/tool_0")
        filtered = filter_grasp_candidates_geometry(
            candidates,
            robot="agibot",
            base_pose=(0.0, 0.0, 0.0, 0.0),
            support_center_xy=(0.65, 0.0),
            support_half_extents_xy=(0.5, 0.4),
            support_yaw_deg=0.0,
            support_top_z=0.75,
            target_bbox_world={
                "min": [0.55, 0.15, 0.84],
                "max": [0.65, 0.25, 1.00],
            },
            preferred_arm_side="left",
            pre_grasp_distance=0.08,
            lift_height=0.10,
            retreat_distance=0.08,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].arm_side, "left")
        self.assertAlmostEqual(filtered[0].grasp.position_world[0], 0.60, places=6)

    def test_rank_candidates_by_start_pose_prefers_closer_pre_grasp(self) -> None:
        base_grasp = GraspExecutionPose(
            object_prim="/World/tool_0",
            primitive_type="point_grasp",
            candidate_id="near",
            score=0.90,
            position_world=(0.60, 0.20, 0.92),
            quat_wxyz_world=(1.0, 0.0, 0.0, 0.0),
            approach_axis_world=(0.0, 0.0, -1.0),
            closing_axis_world=(1.0, 0.0, 0.0),
            width=0.03,
            source_branch=None,
            source_primitive_index=0,
        )
        near = FilteredGraspExecution(
            grasp=base_grasp,
            arm_side="left",
            pre_grasp_pos_world=(0.40, 0.10, 0.92),
            pre_grasp_quat_world=(1.0, 0.0, 0.0, 0.0),
            lift_pos_world=(0.60, 0.20, 1.02),
            lift_quat_world=(1.0, 0.0, 0.0, 0.0),
            retreat_pos_world=(0.60, 0.20, 1.10),
            retreat_quat_world=(1.0, 0.0, 0.0, 0.0),
            base_frame_xy=(0.60, 0.20),
            pre_grasp_base_frame_xy=(0.40, 0.10),
            workspace_margin_xy=(0.02, 0.02),
            support_clearance=0.17,
            score=0.90,
        )
        far = FilteredGraspExecution(
            grasp=GraspExecutionPose(
                object_prim="/World/tool_0",
                primitive_type="point_grasp",
                candidate_id="far",
                score=0.92,
                position_world=(0.70, 0.22, 0.92),
                quat_wxyz_world=(1.0, 0.0, 0.0, 0.0),
                approach_axis_world=(0.0, 0.0, -1.0),
                closing_axis_world=(1.0, 0.0, 0.0),
                width=0.03,
                source_branch=None,
                source_primitive_index=1,
            ),
            arm_side="left",
            pre_grasp_pos_world=(0.95, 0.45, 0.92),
            pre_grasp_quat_world=(1.0, 0.0, 0.0, 0.0),
            lift_pos_world=(0.70, 0.22, 1.02),
            lift_quat_world=(1.0, 0.0, 0.0, 0.0),
            retreat_pos_world=(0.70, 0.22, 1.10),
            retreat_quat_world=(1.0, 0.0, 0.0, 0.0),
            base_frame_xy=(0.70, 0.22),
            pre_grasp_base_frame_xy=(0.95, 0.45),
            workspace_margin_xy=(0.02, 0.02),
            support_clearance=0.17,
            score=0.92,
        )
        ranked = rank_filtered_grasp_candidates_by_start_pose(
            [far, near],
            current_pos_world=(0.35, 0.08, 0.92),
            current_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            position_weight=0.30,
            rotation_weight=0.10,
        )
        self.assertEqual(ranked[0].grasp.candidate_id, "near")
        self.assertIsNotNone(ranked[0].ranking_score)
        self.assertLess(ranked[0].start_pose_position_error, ranked[1].start_pose_position_error)

    def test_rank_candidates_can_use_grasp_orientation(self) -> None:
        canonical_identity = (1.0, 0.0, 0.0, 0.0)
        canonical_rot_z_90 = (0.70710678, 0.0, 0.0, 0.70710678)
        candidate_a = FilteredGraspExecution(
            grasp=GraspExecutionPose(
                object_prim="/World/tool_0",
                primitive_type="point_grasp",
                candidate_id="canonical_match",
                score=0.8,
                position_world=(0.5, 0.0, 0.9),
                quat_wxyz_world=canonical_identity,
                approach_axis_world=(1.0, 0.0, 0.0),
                closing_axis_world=(0.0, 1.0, 0.0),
                width=0.03,
                source_branch=None,
                source_primitive_index=0,
            ),
            arm_side="left",
            pre_grasp_pos_world=(0.45, 0.0, 0.9),
            pre_grasp_quat_world=canonical_rot_z_90,
            lift_pos_world=(0.5, 0.0, 1.0),
            lift_quat_world=canonical_rot_z_90,
            retreat_pos_world=(0.5, 0.0, 1.1),
            retreat_quat_world=canonical_rot_z_90,
            base_frame_xy=(0.5, 0.0),
            pre_grasp_base_frame_xy=(0.45, 0.0),
            workspace_margin_xy=(0.02, 0.02),
            support_clearance=0.1,
            score=0.8,
        )
        candidate_b = FilteredGraspExecution(
            grasp=GraspExecutionPose(
                object_prim="/World/tool_0",
                primitive_type="point_grasp",
                candidate_id="execution_match_only",
                score=0.8,
                position_world=(0.5, 0.0, 0.9),
                quat_wxyz_world=canonical_rot_z_90,
                approach_axis_world=(1.0, 0.0, 0.0),
                closing_axis_world=(0.0, 1.0, 0.0),
                width=0.03,
                source_branch=None,
                source_primitive_index=1,
            ),
            arm_side="left",
            pre_grasp_pos_world=(0.45, 0.0, 0.9),
            pre_grasp_quat_world=canonical_identity,
            lift_pos_world=(0.5, 0.0, 1.0),
            lift_quat_world=canonical_identity,
            retreat_pos_world=(0.5, 0.0, 1.1),
            retreat_quat_world=canonical_identity,
            base_frame_xy=(0.5, 0.0),
            pre_grasp_base_frame_xy=(0.45, 0.0),
            workspace_margin_xy=(0.02, 0.02),
            support_clearance=0.1,
            score=0.8,
        )
        ranked = rank_filtered_grasp_candidates_by_start_pose(
            [candidate_b, candidate_a],
            current_pos_world=(0.45, 0.0, 0.9),
            current_quat_wxyz=canonical_identity,
            position_weight=0.30,
            rotation_weight=0.10,
            use_grasp_orientation=True,
        )
        self.assertEqual(ranked[0].grasp.candidate_id, "canonical_match")

    def test_axis_band_radial_symmetry_generates_both_roll_directions(self) -> None:
        payload = {
            "objects": {
                "/World/bolt_0": {
                    "grasp_primitives_world": [
                        {
                            "type": "axis_band",
                            "point_world": [0.0, 0.0, 0.0],
                            "axis_world": [0.0, 0.0, 1.0],
                            "slide_range_world": [0.0, 0.0],
                            "width_range": [0.02, 0.02],
                            "radial_symmetry": "full",
                            "score": 1.0,
                        }
                    ]
                }
            }
        }
        candidates = expand_grasp_candidates(
            payload,
            target_prim="/World/bolt_0",
            axis_band_slide_samples=1,
            axis_band_ring_samples=4,
        )
        self.assertEqual(len(candidates), 8)
        first = candidates[0]
        same_position = [
            candidate for candidate in candidates
            if all(abs(a - b) < 1.0e-6 for a, b in zip(candidate.position_world, first.position_world))
        ]
        self.assertEqual(len(same_position), 2)
        dot = sum(float(a) * float(b) for a, b in zip(same_position[0].closing_axis_world, same_position[1].closing_axis_world))
        self.assertAlmostEqual(dot, -1.0, places=6)

if __name__ == "__main__":
    unittest.main()
