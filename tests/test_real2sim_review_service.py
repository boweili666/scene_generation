import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from app.backend.services.real2sim_review_service import load_assignment_review, save_assignment_review


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


class Real2SimReviewServiceTest(unittest.TestCase):
    def _build_runtime(self) -> tuple[Path, Path, Path, Path]:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        masks_dir = root / "real2sim" / "masks"
        results_dir = root / "real2sim" / "scene_results"
        scene_graph_path = root / "scene_graph.json"
        latest_input_image = root / "latest_input.jpg"
        masks_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        scene_graph_path.write_text(json.dumps(SCENE_GRAPH), encoding="utf-8")

        Image.new("RGB", (96, 96), color=(240, 240, 240)).save(masks_dir / "image.png")
        Image.new("RGB", (96, 96), color=(240, 240, 240)).save(latest_input_image)

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

        (results_dir / "mask_metadata.json").write_text(
            json.dumps(
                {
                    "0": {"prompt": "table", "bbox_xyxy": [8, 10, 39, 43]},
                    "1": {"prompt": "mug", "bbox_xyxy": [42, 38, 87, 87]},
                }
            ),
            encoding="utf-8",
        )
        return scene_graph_path, masks_dir, results_dir, latest_input_image

    def test_load_assignment_review_builds_overlay_and_attention_summary(self) -> None:
        scene_graph_path, masks_dir, results_dir, latest_input_image = self._build_runtime()

        review = load_assignment_review(
            scene_graph_path=scene_graph_path,
            masks_dir=masks_dir,
            results_dir=results_dir,
            latest_input_image=latest_input_image,
        )

        self.assertEqual(len(review["scene_objects"]), 2)
        self.assertEqual(len(review["mask_labels"]), 2)
        self.assertTrue(review["needs_attention"])
        self.assertEqual(review["summary"]["unmatched_scene_paths"], 2)
        self.assertTrue((results_dir / "numbered_masks_overlay.png").exists())

    def test_save_assignment_review_writes_manual_assignment_and_manifest(self) -> None:
        scene_graph_path, masks_dir, results_dir, latest_input_image = self._build_runtime()
        objects_dir = results_dir / "objects"
        usd_objects_dir = results_dir / "usd_objects"
        objects_dir.mkdir(parents=True, exist_ok=True)
        usd_objects_dir.mkdir(parents=True, exist_ok=True)

        mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        for name in ("0", "1"):
            (objects_dir / f"{name}.glb").write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
            (usd_objects_dir / f"{name}.usd").write_text("#usda 1.0\n", encoding="utf-8")

        (results_dir / "scene_merged.glb").write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
        (results_dir / "scene_merged_post.glb").write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
        (results_dir / "scene_merged_post.usd").write_text("#usda 1.0\n", encoding="utf-8")
        (results_dir / "poses.json").write_text(
            json.dumps(
                {
                    "0": {"scene_transform": np.eye(4).tolist()},
                    "1": {"scene_transform": np.eye(4).tolist()},
                }
            ),
            encoding="utf-8",
        )

        review = save_assignment_review(
            assignments=[
                {"mask_label": 1, "scene_path": "/World/table_0"},
                {"mask_label": 2, "scene_path": "/World/mug_1"},
            ],
            scene_graph_path=scene_graph_path,
            masks_dir=masks_dir,
            results_dir=results_dir,
            latest_input_image=latest_input_image,
        )

        assignment_payload = json.loads((results_dir / "assignment.json").read_text(encoding="utf-8"))
        manifest_payload = json.loads((results_dir / "real2sim_asset_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(assignment_payload["source"], "manual_review")
        self.assertEqual(len(assignment_payload["assignments"]), 2)
        self.assertEqual(review["summary"]["unmatched_scene_paths"], 0)
        self.assertEqual(manifest_payload["unmatched_scene_paths"], [])


if __name__ == "__main__":
    unittest.main()
