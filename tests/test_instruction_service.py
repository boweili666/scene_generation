import unittest
from unittest import mock

from app.backend.services import instruction_service


SCENE_GRAPH = {
    "scene": {
        "room_type": "office",
        "dimensions": {"length": 5, "width": 4, "height": 3, "unit": "m"},
        "materials": {"floor": "wood", "walls": "paint"},
    },
    "obj": {
        "/World/cup_0": {"id": 0, "class": "cup", "caption": "coffee cup", "source": "real2sim"},
    },
    "edges": {"obj-obj": [], "obj-wall": []},
}


class InstructionServiceTest(unittest.TestCase):
    def test_bootstrap_generation_when_no_scene_graph_exists(self) -> None:
        with (
            mock.patch.object(instruction_service, "_load_scene_graph", return_value=None),
            mock.patch.object(instruction_service, "_generate_scene_graph", return_value=SCENE_GRAPH) as generate_mock,
            mock.patch.object(instruction_service, "write_json_file") as write_graph_mock,
            mock.patch.object(instruction_service, "_write_placements") as write_placements_mock,
        ):
            result = instruction_service.apply_instruction("Add a cup on the table.")

        self.assertEqual(result["route"]["mode"], "graph")
        self.assertEqual(result["scene_graph"], SCENE_GRAPH)
        generate_mock.assert_called_once()
        write_graph_mock.assert_called_once()
        write_placements_mock.assert_called_once_with(instruction_service.DEFAULT_PLACEMENTS_PATH, {})

    def test_placement_edit_requires_existing_placements(self) -> None:
        with (
            mock.patch.object(instruction_service, "_load_scene_graph", return_value=SCENE_GRAPH),
            mock.patch.object(instruction_service, "_load_placements", return_value={}),
            mock.patch.object(
                instruction_service,
                "route_scene_instruction",
                return_value={"mode": "placement", "confidence": 0.9, "reason": "move object"},
            ),
        ):
            with self.assertRaisesRegex(ValueError, "existing sampled placements"):
                instruction_service.apply_instruction("Move the cup 20 cm left.")

    def test_save_scene_graph_state_prunes_removed_object_placements(self) -> None:
        edited_graph = {
            **SCENE_GRAPH,
            "obj": {},
            "edges": {"obj-obj": [], "obj-wall": []},
        }
        with (
            mock.patch.object(instruction_service, "_load_placements", return_value={"/World/cup_0": [0.0, 0.0, 0.8]}),
            mock.patch.object(instruction_service, "write_json_file") as write_graph_mock,
            mock.patch.object(instruction_service, "_write_placements") as write_placements_mock,
        ):
            result = instruction_service.save_scene_graph_state(edited_graph)

        self.assertEqual(result["placements"], {})
        self.assertEqual(result["invalidated_placements"], ["/World/cup_0"])
        write_graph_mock.assert_called_once()
        write_placements_mock.assert_called_once_with(instruction_service.DEFAULT_PLACEMENTS_PATH, {})


if __name__ == "__main__":
    unittest.main()
