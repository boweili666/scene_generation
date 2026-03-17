import unittest

from app.backend.services.openai_service import _normalize_scene_graph_payload


class OpenAIServiceSceneGraphNormalizationTest(unittest.TestCase):
    def test_normalizes_objects_array_into_obj_map(self) -> None:
        payload = {
            "scene": {
                "room_type": "office",
                "dimensions": {"length": 5, "width": 4, "height": 3, "unit": "m"},
                "materials": {"floor": "wood", "walls": "paint"},
            },
            "objects": [
                {"path": "/World/table_0", "id": 0, "class": "table", "caption": "wood table"},
                {"path": "/World/chair_1", "id": 1, "class": "chair", "caption": "office chair"},
            ],
            "edges": {
                "obj-obj": [
                    {"source": "/World/chair_1", "target": "/World/table_0", "relation": "adjacent"}
                ],
                "obj-wall": [
                    {"source": "/World/table_0", "relation": "against wall"}
                ],
            },
        }

        normalized = _normalize_scene_graph_payload(payload)

        self.assertIn("obj", normalized)
        self.assertNotIn("objects", normalized)
        self.assertEqual(
            normalized["obj"]["/World/table_0"],
            {"id": 0, "class": "table", "caption": "wood table"},
        )
        self.assertEqual(
            normalized["obj"]["/World/chair_1"],
            {"id": 1, "class": "chair", "caption": "office chair"},
        )

    def test_rejects_edge_reference_to_unknown_object_path(self) -> None:
        payload = {
            "scene": {
                "room_type": "office",
                "dimensions": {"length": 5, "width": 4, "height": 3, "unit": "m"},
                "materials": {"floor": "wood", "walls": "paint"},
            },
            "objects": [
                {"path": "/World/table_0", "id": 0, "class": "table", "caption": "wood table"}
            ],
            "edges": {
                "obj-obj": [
                    {"source": "/World/chair_1", "target": "/World/table_0", "relation": "adjacent"}
                ],
                "obj-wall": [],
            },
        }

        with self.assertRaisesRegex(ValueError, "obj-obj"):
            _normalize_scene_graph_payload(payload)


if __name__ == "__main__":
    unittest.main()
