import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.backend.services.real2sim_assignment_visualization import (
    ASSIGNMENT_BBOX_OVERLAY_FILENAME,
    ASSIGNMENT_REVIEW_HTML_FILENAME,
    build_assignment_visualization,
)


SCENE_GRAPH = {
    "obj": {
        "/World/table_0": {
            "id": 0,
            "class": "table",
            "caption": "wood table",
            "source": "real2sim",
        },
        "/World/mug_1": {
            "id": 1,
            "class": "mug",
            "caption": "blue mug",
            "source": "real2sim",
        },
    },
    "edges": {"obj-obj": [], "obj-wall": []},
}


class Real2SimAssignmentVisualizationTest(unittest.TestCase):
    def _build_runtime(self) -> tuple[Path, Path]:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        masks_dir = root / "real2sim" / "masks"
        results_dir = root / "real2sim" / "scene_results"
        scene_graph_path = root / "scene_graph.json"
        masks_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        scene_graph_path.write_text(json.dumps(SCENE_GRAPH), encoding="utf-8")

        Image.new("RGB", (96, 96), color=(240, 240, 240)).save(masks_dir / "image.png")

        mask0 = Image.new("RGBA", (96, 96), color=(0, 0, 0, 0))
        mask1 = Image.new("RGBA", (96, 96), color=(0, 0, 0, 0))
        for x in range(8, 40):
            for y in range(10, 44):
                mask0.putpixel((x, y), (255, 255, 255, 255))
        for x in range(42, 88):
            for y in range(38, 88):
                mask1.putpixel((x, y), (255, 255, 255, 255))
        mask0.save(masks_dir / "0.png")
        mask1.save(masks_dir / "1.png")

        mask_metadata = {
            "0": {"prompt": "table", "bbox_xyxy": [8, 10, 39, 43]},
            "1": {"prompt": "mug", "bbox_xyxy": [42, 38, 87, 87]},
        }
        (results_dir / "mask_metadata.json").write_text(json.dumps(mask_metadata), encoding="utf-8")
        assignment_path = results_dir / "assignment.json"
        assignment_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "source": "vlm_mask_assignment",
                    "scene_graph_path": str(scene_graph_path),
                    "image_path": "../masks/image.png",
                    "overlay_image_path": "numbered_masks_overlay.png",
                    "mask_labels": [
                        {"mask_label": 1, "output_name": "0", "mask_path": "../masks/0.png", "prompt": "table"},
                        {"mask_label": 2, "output_name": "1", "mask_path": "../masks/1.png", "prompt": "mug"},
                    ],
                    "assignments": [
                        {
                            "scene_path": "/World/table_0",
                            "mask_label": 1,
                            "output_name": "0",
                            "confidence": 0.95,
                            "reason": "Good match.",
                        }
                    ],
                    "unmatched_scene_paths": ["/World/mug_1"],
                    "unmatched_mask_labels": [2],
                }
            ),
            encoding="utf-8",
        )
        return assignment_path, results_dir

    def test_build_assignment_visualization_writes_overlay_and_html(self) -> None:
        assignment_path, results_dir = self._build_runtime()

        result = build_assignment_visualization(assignment_path)

        bbox_path = Path(result["bbox_output_path"])
        html_path = Path(result["html_output_path"])
        self.assertEqual(bbox_path.name, ASSIGNMENT_BBOX_OVERLAY_FILENAME)
        self.assertEqual(html_path.name, ASSIGNMENT_REVIEW_HTML_FILENAME)
        self.assertTrue(bbox_path.exists())
        self.assertTrue(html_path.exists())
        self.assertTrue((results_dir / "numbered_masks_overlay.png").exists())

        html_text = html_path.read_text(encoding="utf-8")
        self.assertIn("Real2Sim Assignment Review", html_text)
        self.assertIn("/World/table_0", html_text)
        self.assertIn("assignment_bbox_overlay.png", html_text)
        self.assertIn("numbered_masks_overlay.png", html_text)


if __name__ == "__main__":
    unittest.main()
