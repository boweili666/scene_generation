from pathlib import Path
import sys
import unittest

from pxr import Usd


SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "scene_robot" / "src"
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from scene_robot_apps.grasp.execution import FilteredGraspExecution, GraspExecutionPose
from scene_robot_apps.grasp.visualization import add_grasp_candidates_visuals_to_stage


class GraspVisualizationTest(unittest.TestCase):
    def test_add_grasp_candidates_visuals_to_stage_creates_selected_and_candidate_prims(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        base_pose = GraspExecutionPose(
            object_prim="/World/tool_0",
            primitive_type="point_grasp",
            candidate_id="cand_0",
            score=0.9,
            position_world=(0.6, 0.2, 0.9),
            quat_wxyz_world=(1.0, 0.0, 0.0, 0.0),
            approach_axis_world=(1.0, 0.0, 0.0),
            closing_axis_world=(0.0, 1.0, 0.0),
            width=0.03,
            source_branch=None,
            source_primitive_index=0,
        )
        candidate = FilteredGraspExecution(
            grasp=base_pose,
            arm_side="left",
            pre_grasp_pos_world=(0.5, 0.2, 0.9),
            pre_grasp_quat_world=(1.0, 0.0, 0.0, 0.0),
            lift_pos_world=(0.6, 0.2, 1.0),
            lift_quat_world=(1.0, 0.0, 0.0, 0.0),
            retreat_pos_world=(0.55, 0.2, 1.0),
            retreat_quat_world=(1.0, 0.0, 0.0, 0.0),
            base_frame_xy=(0.6, 0.2),
            pre_grasp_base_frame_xy=(0.5, 0.2),
            workspace_margin_xy=(0.02, 0.02),
            support_clearance=0.15,
            score=0.9,
            ranking_score=0.85,
        )

        result = add_grasp_candidates_visuals_to_stage(
            stage,
            root_prim_path="/World/AutoGraspVisuals",
            ranked_candidates=[candidate],
            selected_candidate=candidate,
        )

        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["selected_candidate_id"], "cand_0")
        self.assertEqual(result["selected_grasp_path"], "/World/AutoGraspVisuals/Selected/GraspPose")
        self.assertTrue(stage.GetPrimAtPath("/World/AutoGraspVisuals/Selected/GraspPose").IsValid())
        self.assertFalse(stage.GetPrimAtPath("/World/AutoGraspVisuals/Selected/CanonicalGraspPose").IsValid())
        self.assertFalse(stage.GetPrimAtPath("/World/AutoGraspVisuals/Candidates/Candidate_00").IsValid())
        self.assertFalse(stage.GetPrimAtPath("/World/AutoGraspVisuals/Selected/PreToGrasp").IsValid())

    def test_add_grasp_candidates_visuals_to_stage_can_show_candidate_frames(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        base_pose = GraspExecutionPose(
            object_prim="/World/tool_0",
            primitive_type="point_grasp",
            candidate_id="cand_0",
            score=0.9,
            position_world=(0.6, 0.2, 0.9),
            quat_wxyz_world=(1.0, 0.0, 0.0, 0.0),
            approach_axis_world=(1.0, 0.0, 0.0),
            closing_axis_world=(0.0, 1.0, 0.0),
            width=0.03,
            source_branch=None,
            source_primitive_index=0,
        )
        candidate = FilteredGraspExecution(
            grasp=base_pose,
            arm_side="left",
            pre_grasp_pos_world=(0.5, 0.2, 0.9),
            pre_grasp_quat_world=(1.0, 0.0, 0.0, 0.0),
            lift_pos_world=(0.6, 0.2, 1.0),
            lift_quat_world=(1.0, 0.0, 0.0, 0.0),
            retreat_pos_world=(0.55, 0.2, 1.0),
            retreat_quat_world=(1.0, 0.0, 0.0, 0.0),
            base_frame_xy=(0.6, 0.2),
            pre_grasp_base_frame_xy=(0.5, 0.2),
            workspace_margin_xy=(0.02, 0.02),
            support_clearance=0.15,
            score=0.9,
            ranking_score=0.85,
        )

        result = add_grasp_candidates_visuals_to_stage(
            stage,
            root_prim_path="/World/AutoGraspVisuals",
            ranked_candidates=[candidate],
            selected_candidate=candidate,
            max_candidates=1,
        )

        self.assertEqual(result["candidate_count"], 1)
        self.assertTrue(stage.GetPrimAtPath("/World/AutoGraspVisuals/Candidates/Candidate_00/GraspPose").IsValid())


if __name__ == "__main__":
    unittest.main()
