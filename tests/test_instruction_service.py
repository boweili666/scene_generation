import json
import tempfile
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
        "/World/lamp_0": {"id": 1, "class": "lamp", "caption": "desk lamp", "source": "retrieval"},
    },
    "edges": {"obj-obj": [], "obj-wall": []},
}

RELATION_SCENE_GRAPH = {
    "scene": {
        "room_type": "office",
        "dimensions": {"length": 5, "width": 4, "height": 3, "unit": "m"},
        "materials": {"floor": "wood", "walls": "paint"},
    },
    "obj": {
        "/World/mug_0": {"id": 0, "class": "mug", "caption": "ceramic mug", "source": "real2sim"},
        "/World/table_0": {"id": 1, "class": "table", "caption": "round table", "source": "real2sim"},
    },
    "edges": {
        "obj-obj": [
            {"source": "/World/mug_0", "target": "/World/table_0", "relation": "supported by"},
            {"source": "/World/table_0", "target": "/World/mug_0", "relation": "supports"},
        ],
        "obj-wall": [],
    },
}


class InstructionServiceTest(unittest.TestCase):
    def test_normalize_placements_payload_accepts_structured_entries_with_aabb(self) -> None:
        payload = {
            "/World/cup_0": {
                "x": 1.0,
                "y": 2.0,
                "z": 3.0,
                "yaw": 45.0,
                "aabb": {
                    "min": [0.5, 1.5, 2.5],
                    "max": [1.5, 2.5, 3.5],
                    "center": [1.0, 2.0, 3.0],
                    "size": [1.0, 1.0, 1.0],
                },
            }
        }

        normalized = instruction_service._normalize_placements_payload(payload)

        self.assertEqual(normalized, {"/World/cup_0": [1.0, 2.0, 3.0, 45.0]})

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

    def test_single_object_placement_edit_ignores_unmentioned_updates(self) -> None:
        placements = {
            "/World/cup_0": [0.0, 0.0, 0.8],
            "/World/lamp_0": [1.0, 0.5, 1.2],
        }
        with (
            mock.patch.object(instruction_service, "_load_scene_graph", return_value=SCENE_GRAPH),
            mock.patch.object(instruction_service, "_load_placements", return_value=placements),
            mock.patch.object(
                instruction_service,
                "route_scene_instruction",
                return_value={"mode": "placement", "confidence": 0.9, "reason": "move one object"},
            ),
            mock.patch.object(
                instruction_service,
                "edit_placements_with_instruction",
                return_value={
                    "message": "Moved the cup slightly to the right.",
                    "updates": [
                        {"path": "/World/cup_0", "x": 0.2, "y": 0.0, "z": 0.8, "yaw_deg": None},
                        {"path": "/World/lamp_0", "x": 1.5, "y": 0.5, "z": 1.2, "yaw_deg": None},
                    ]
                },
            ),
            mock.patch.object(instruction_service, "write_json_file"),
            mock.patch.object(instruction_service, "_write_placements"),
        ):
            result = instruction_service.apply_instruction("Move the cup right a little.")

        self.assertEqual(result["updated_paths"], ["/World/cup_0"])
        self.assertEqual(result["placements"]["/World/cup_0"], [0.2, 0.0, 0.8])
        self.assertEqual(result["placements"]["/World/lamp_0"], [1.0, 0.5, 1.2])
        self.assertEqual(result["assistant_message"], "Moved the cup slightly to the right.")

    def test_relation_move_passes_aabb_to_llm_and_only_updates_source_object(self) -> None:
        placements = {
            "/World/mug_0": [0.0, 0.0, 0.55, 0.0],
            "/World/table_0": [0.0, 0.0, 0.35, 0.0],
        }
        placements_raw = {
            "/World/mug_0": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.55,
                "yaw": 0.0,
                "aabb": {
                    "min": [-0.05, -0.05, 0.45],
                    "max": [0.05, 0.05, 0.65],
                    "center": [0.0, 0.0, 0.55],
                    "size": [0.1, 0.1, 0.2],
                },
            },
            "/World/table_0": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.35,
                "yaw": 0.0,
                "aabb": {
                    "min": [-0.5, -0.4, 0.0],
                    "max": [0.5, 0.4, 0.7],
                    "center": [0.0, 0.0, 0.35],
                    "size": [1.0, 0.8, 0.7],
                },
            },
        }
        with (
            mock.patch.object(instruction_service, "_load_scene_graph", return_value=RELATION_SCENE_GRAPH),
            mock.patch.object(instruction_service, "_load_placements", return_value=placements),
            mock.patch.object(instruction_service, "_load_placements_payload_raw", return_value=placements_raw),
            mock.patch.object(
                instruction_service,
                "route_scene_instruction",
                return_value={"mode": "placement", "confidence": 0.9, "reason": "move object"},
            ),
            mock.patch.object(
                instruction_service,
                "edit_placements_with_instruction",
                return_value={
                    "message": "I moved the mug to the right side of the table without moving the table.",
                    "updates": [
                        {"path": "/World/mug_0", "x": 0.25, "y": 0.32, "z": 0.8, "yaw_deg": 0.0},
                        {"path": "/World/table_0", "x": 1.0, "y": 1.0, "z": 0.35, "yaw_deg": 0.0},
                    ]
                },
            ) as placement_editor_mock,
            mock.patch.object(instruction_service, "write_json_file"),
            mock.patch.object(instruction_service, "_write_placements"),
        ):
            result = instruction_service.apply_instruction("Move the mug to the right side of the table.")

        placement_editor_mock.assert_called_once()
        _, kwargs = placement_editor_mock.call_args
        self.assertIsNone(kwargs["image_b64"])
        self.assertEqual(kwargs["placement_context"], placements_raw)
        self.assertIn("Movable object: /World/mug_0", kwargs["editing_hint"])
        self.assertIn("Reference object: /World/table_0", kwargs["editing_hint"])
        self.assertEqual(result["route"]["mode"], "placement")
        self.assertEqual(result["updated_paths"], ["/World/mug_0"])
        self.assertEqual(result["placements"]["/World/table_0"], [0.0, 0.0, 0.35, 0.0])
        self.assertEqual(result["placements"]["/World/mug_0"], [0.25, 0.32, 0.8, 0.0])
        self.assertEqual(
            result["assistant_message"],
            "I moved the mug to the right side of the table without moving the table.",
        )

    def test_write_placements_preserves_and_translates_aabb(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            placements_path = instruction_service.Path(tmpdir) / "placements.json"
            placements_path.write_text(
                json.dumps(
                    {
                        "/World/cup_0": {
                            "x": 1.0,
                            "y": 2.0,
                            "z": 3.0,
                            "yaw": 45.0,
                            "aabb": {
                                "min": [0.5, 1.5, 2.5],
                                "max": [1.5, 2.5, 3.5],
                                "center": [1.0, 2.0, 3.0],
                                "size": [1.0, 1.0, 1.0],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            instruction_service._write_placements(placements_path, {"/World/cup_0": [2.0, 4.0, 6.0, 45.0]})
            payload = json.loads(placements_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["/World/cup_0"]["aabb"]["center"], [2.0, 4.0, 6.0])
        self.assertEqual(payload["/World/cup_0"]["aabb"]["min"], [1.5, 3.5, 5.5])
        self.assertEqual(payload["/World/cup_0"]["aabb"]["max"], [2.5, 4.5, 6.5])


if __name__ == "__main__":
    unittest.main()
