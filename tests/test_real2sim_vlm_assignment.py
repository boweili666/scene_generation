import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from pipelines.real2sim.vlm_assignment import (
    build_mask_label_index,
    generate_vlm_mask_assignment,
    render_numbered_masks,
)


class Real2SimVlmAssignmentTest(unittest.TestCase):
    def test_build_mask_label_index_and_render_numbered_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            mask0 = root / "0.png"
            mask1 = root / "1.png"
            metadata_path = root / "mask_metadata.json"
            overlay_path = root / "numbered_masks_overlay.png"

            Image.new("RGB", (128, 96), color=(240, 240, 240)).save(image_path)
            Image.new("RGBA", (128, 96), color=(0, 0, 0, 0)).save(mask0)
            Image.new("RGBA", (128, 96), color=(0, 0, 0, 0)).save(mask1)

            mask0_img = Image.open(mask0).convert("RGBA")
            mask1_img = Image.open(mask1).convert("RGBA")
            for x in range(8, 48):
                for y in range(10, 40):
                    mask0_img.putpixel((x, y), (255, 255, 255, 255))
            for x in range(60, 112):
                for y in range(34, 82):
                    mask1_img.putpixel((x, y), (255, 255, 255, 255))
            mask0_img.save(mask0)
            mask1_img.save(mask1)

            metadata_path.write_text(
                json.dumps(
                    {
                        "0": {"prompt": "table", "prompt_key": "table", "bbox_xyxy": [8, 10, 47, 39]},
                        "1": {"prompt": "cup", "prompt_key": "cup", "bbox_xyxy": [60, 34, 111, 81]},
                    }
                ),
                encoding="utf-8",
            )

            mask_index = build_mask_label_index([mask1, mask0], mask_metadata_path=metadata_path)

            self.assertEqual([row["mask_label"] for row in mask_index], [1, 2])
            self.assertEqual([row["output_name"] for row in mask_index], ["0", "1"])
            self.assertEqual(mask_index[0]["prompt"], "table")

            render_numbered_masks(image_path, mask_index, overlay_path)

            self.assertTrue(overlay_path.exists())
            with Image.open(overlay_path) as overlay_image:
                self.assertEqual(overlay_image.size, (128, 96))

    def test_generate_vlm_mask_assignment_writes_assignment_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "scene_results"
            output_dir.mkdir(parents=True)
            image_path = root / "image.png"
            mask0 = root / "0.png"
            mask1 = root / "1.png"
            metadata_path = output_dir / "mask_metadata.json"
            scene_graph_path = root / "scene_graph.json"

            Image.new("RGB", (96, 96), color=(240, 240, 240)).save(image_path)
            Image.new("RGBA", (96, 96), color=(0, 0, 0, 0)).save(mask0)
            Image.new("RGBA", (96, 96), color=(0, 0, 0, 0)).save(mask1)

            metadata_path.write_text(
                json.dumps(
                    {
                        "0": {"prompt": "table", "bbox_xyxy": [8, 8, 40, 40]},
                        "1": {"prompt": "table", "bbox_xyxy": [42, 42, 90, 90]},
                    }
                ),
                encoding="utf-8",
            )
            scene_graph_path.write_text(
                json.dumps(
                    {
                        "obj": {
                            "/World/table_0": {
                                "id": 0,
                                "class": "table",
                                "caption": "large wood table",
                                "source": "real2sim",
                            },
                            "/World/lamp_1": {
                                "id": 1,
                                "class": "lamp",
                                "caption": "desk lamp",
                                "source": "retrieval",
                            },
                        },
                        "edges": {"obj-obj": [], "obj-wall": []},
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "app.backend.services.openai_service.assign_real2sim_masks_with_images",
                return_value={
                    "assignments": [
                        {
                            "scene_path": "/World/table_0",
                            "mask_label": 2,
                            "confidence": 0.95,
                            "reason": "matches the main tabletop",
                        }
                    ],
                    "unmatched_scene_paths": [],
                    "unmatched_mask_labels": [1],
                },
            ), mock.patch(
                "app.backend.services.openai_service.encode_image_b64",
                side_effect=lambda path: f"data:image/png;base64,{Path(path).name}",
            ):
                assignment_path = generate_vlm_mask_assignment(
                    image_path,
                    [mask0, mask1],
                    scene_graph_path,
                    output_dir,
                    mask_metadata_path=metadata_path,
                )

            self.assertIsNotNone(assignment_path)
            assignment = json.loads(Path(assignment_path).read_text(encoding="utf-8"))
            self.assertEqual(assignment["assignments"][0]["scene_path"], "/World/table_0")
            self.assertEqual(assignment["assignments"][0]["mask_label"], 2)
            self.assertEqual(assignment["assignments"][0]["output_name"], "1")
            self.assertEqual(assignment["unmatched_mask_labels"], [1])
            self.assertEqual(len(assignment["mask_labels"]), 2)


if __name__ == "__main__":
    unittest.main()
