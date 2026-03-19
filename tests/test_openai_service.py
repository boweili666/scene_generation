import unittest
from unittest import mock

from app.backend.services import openai_service
from app.backend.services.openai_service import (
    SYSTEM_PROMPT,
    _normalize_scene_graph_payload,
    parse_scene_graph_from_image,
)


class OpenAIServiceSceneGraphNormalizationTest(unittest.TestCase):
    def test_normalizes_objects_array_into_obj_map(self) -> None:
        payload = {
            "scene": {
                "room_type": "office",
                "dimensions": {"length": 5, "width": 4, "height": 3, "unit": "m"},
                "materials": {"floor": "wood", "walls": "paint"},
            },
            "objects": [
                {
                    "path": "/World/table_0",
                    "id": 0,
                    "class": "table",
                    "caption": "wood table",
                    "source": "real2sim",
                },
                {
                    "path": "/World/chair_1",
                    "id": 1,
                    "class": "chair",
                    "caption": "office chair",
                    "source": "retrieval",
                },
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
            {"id": 0, "class": "table", "caption": "wood table", "source": "real2sim"},
        )
        self.assertEqual(
            normalized["obj"]["/World/chair_1"],
            {"id": 1, "class": "chair", "caption": "office chair", "source": "retrieval"},
        )

    def test_rejects_edge_reference_to_unknown_object_path(self) -> None:
        payload = {
            "scene": {
                "room_type": "office",
                "dimensions": {"length": 5, "width": 4, "height": 3, "unit": "m"},
                "materials": {"floor": "wood", "walls": "paint"},
            },
            "objects": [
                {
                    "path": "/World/table_0",
                    "id": 0,
                    "class": "table",
                    "caption": "wood table",
                    "source": "real2sim",
                }
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

    def test_accepts_valid_source_in_obj_map(self) -> None:
        payload = {
            "scene": {
                "room_type": "office",
                "dimensions": {"length": 5, "width": 4, "height": 3, "unit": "m"},
                "materials": {"floor": "wood", "walls": "paint"},
            },
            "obj": {
                "/World/table_0": {
                    "id": 0,
                    "class": "table",
                    "caption": "wood table",
                    "source": "real2sim",
                }
            },
            "edges": {"obj-obj": [], "obj-wall": []},
        }

        normalized = _normalize_scene_graph_payload(payload)

        self.assertEqual(normalized["obj"]["/World/table_0"]["source"], "real2sim")

    def test_rejects_invalid_source_value(self) -> None:
        payload = {
            "scene": {
                "room_type": "office",
                "dimensions": {"length": 5, "width": 4, "height": 3, "unit": "m"},
                "materials": {"floor": "wood", "walls": "paint"},
            },
            "obj": {
                "/World/table_0": {
                    "id": 0,
                    "class": "table",
                    "caption": "wood table",
                    "source": "unknown",
                }
            },
            "edges": {"obj-obj": [], "obj-wall": []},
        }

        with self.assertRaisesRegex(ValueError, "invalid source"):
            _normalize_scene_graph_payload(payload)

    def test_rejects_missing_source_value(self) -> None:
        payload = {
            "scene": {
                "room_type": "office",
                "dimensions": {"length": 5, "width": 4, "height": 3, "unit": "m"},
                "materials": {"floor": "wood", "walls": "paint"},
            },
            "obj": {
                "/World/table_0": {
                    "id": 0,
                    "class": "table",
                    "caption": "wood table",
                }
            },
            "edges": {"obj-obj": [], "obj-wall": []},
        }

        with self.assertRaisesRegex(ValueError, "missing required source"):
            _normalize_scene_graph_payload(payload)


class OpenAIServiceImagePromptTest(unittest.TestCase):
    def test_parse_scene_graph_from_image_includes_text_instruction(self) -> None:
        captured = {}

        class _FakeResponses:
            def create(self, **kwargs):
                captured.update(kwargs)
                response = mock.Mock()
                response.output_text = """
                {
                  "scene": {
                    "room_type": "office",
                    "dimensions": {"length": 5, "width": 4, "height": 3, "unit": "m"},
                    "materials": {"floor": "wood", "walls": "paint"}
                  },
                  "objects": [
                    {
                      "path": "/World/table_0",
                      "id": 0,
                      "class": "table",
                      "caption": "wood table",
                      "source": "real2sim"
                    },
                    {
                      "path": "/World/chair_1",
                      "id": 1,
                      "class": "chair",
                      "caption": "office chair",
                      "source": "retrieval"
                    }
                  ],
                  "edges": {"obj-obj": [], "obj-wall": []}
                }
                """
                return response

        class _FakeClient:
            responses = _FakeResponses()

        with mock.patch.object(openai_service, "_get_openai_client", return_value=_FakeClient()):
            result = parse_scene_graph_from_image(
                b"fake-image-bytes",
                text="Add a chair in front of the table and a bed against the wall.",
            )

        self.assertIn("/World/chair_1", result["obj"])
        content = captured["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        prompt_text = content[0]["text"]
        self.assertIn("Add a chair in front of the table", prompt_text)
        self.assertIn("Objects explicitly requested in text but not present in the uploaded image should use source=retrieval.", prompt_text)
        self.assertEqual(content[1]["type"], "input_image")


class OpenAIServicePromptGuardrailTest(unittest.TestCase):
    def test_system_prompt_forbids_duplicate_nodes_for_same_visible_instance(self) -> None:
        self.assertIn("One physical object instance must map to exactly one node.", SYSTEM_PROMPT)
        self.assertIn("Do NOT create duplicate nodes for the same visible instance because of class ambiguity or synonyms.", SYSTEM_PROMPT)
        self.assertIn("jar", SYSTEM_PROMPT)
        self.assertIn("glass", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
