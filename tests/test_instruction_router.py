import unittest

from app.backend.services.instruction_router import (
    reconcile_placements_after_graph_edit,
    validate_route_decision,
)


SCENE_GRAPH = {
    "scene": {
        "room_type": "office",
        "dimensions": {"length": 5, "width": 4, "height": 3, "unit": "m"},
        "materials": {"floor": "wood", "walls": "paint"},
    },
    "obj": {
        "/World/cup_0": {"id": 0, "class": "cup", "caption": "coffee cup", "source": "real2sim"},
        "/World/table_0": {"id": 1, "class": "table", "caption": "work table", "source": "retrieval"},
        "/World/lamp_0": {"id": 2, "class": "lamp", "caption": "desk lamp", "source": "retrieval"},
    },
    "edges": {
        "obj-obj": [
            {"source": "/World/cup_0", "target": "/World/table_0", "relation": "adjacent"},
        ],
        "obj-wall": [],
    },
}


class InstructionRouterTest(unittest.TestCase):
    def test_create_instruction_overrides_llm_to_graph(self) -> None:
        route = validate_route_decision(
            "Add a chair next to the table.",
            SCENE_GRAPH,
            {"/World/cup_0": [0.0, 0.0, 0.8]},
            {"mode": "placement", "confidence": 0.91, "reason": "object should move"},
        )

        self.assertEqual(route["mode"], "graph")
        self.assertIn("create", route["signals"])

    def test_numeric_move_prefers_placement(self) -> None:
        route = validate_route_decision(
            "Move the cup 20 cm to the left.",
            SCENE_GRAPH,
            {"/World/cup_0": [0.0, 0.0, 0.8]},
            {"mode": "graph", "confidence": 0.2, "reason": "weak guess"},
        )

        self.assertEqual(route["mode"], "placement")
        self.assertFalse(route["requires_existing_placements"])

    def test_relation_change_becomes_both(self) -> None:
        route = validate_route_decision(
            "Put the cup on the table.",
            SCENE_GRAPH,
            {"/World/cup_0": [0.0, 0.0, 0.8], "/World/table_0": [0.0, 0.0, 0.4]},
            {"mode": "placement", "confidence": 0.88, "reason": "move one object"},
        )

        self.assertEqual(route["mode"], "both")
        self.assertTrue(route["needs_resample"])

    def test_reconcile_placements_drops_affected_paths(self) -> None:
        next_graph = {
            **SCENE_GRAPH,
            "edges": {
                "obj-obj": [
                    {"source": "/World/cup_0", "target": "/World/table_0", "relation": "supported by"},
                    {"source": "/World/table_0", "target": "/World/cup_0", "relation": "supports"},
                ],
                "obj-wall": [],
            },
        }
        placements = {
            "/World/cup_0": [0.0, 0.0, 0.8],
            "/World/table_0": [0.0, 0.0, 0.4],
            "/World/lamp_0": [1.0, 0.0, 1.1],
        }

        reconciled, invalidated = reconcile_placements_after_graph_edit(
            SCENE_GRAPH,
            next_graph,
            placements,
            {"objects": ["/World/cup_0", "/World/table_0"]},
        )

        self.assertEqual(reconciled, {"/World/lamp_0": [1.0, 0.0, 1.1]})
        self.assertIn("/World/cup_0", invalidated)
        self.assertIn("/World/table_0", invalidated)


if __name__ == "__main__":
    unittest.main()
