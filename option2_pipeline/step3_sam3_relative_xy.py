from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


DEFAULT_IMAGE = Path("/home/lbw/3dgen-project/scene_graph_ui_test/option2_pipeline/runtime/scene_60_top_view.png")
DEFAULT_PROMPTS = ["table", "desk lamp", "alarm clock", "notebook", "pen", "glass cup"]
DEFAULT_OUTPUT = Path("/home/lbw/3dgen-project/scene_graph_ui_test/option2_pipeline/runtime/sam3_bbox_relative.csv")


@dataclass
class Detection:
    label: str
    instance_id: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def bbox_width(self) -> int:
        return self.x_max - self.x_min + 1

    @property
    def bbox_height(self) -> int:
        return self.y_max - self.y_min + 1

    @property
    def bbox_area(self) -> int:
        return self.bbox_width * self.bbox_height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SAM3 to get bboxes and relative XY centered at table."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="input image path")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        help="object prompts, include 'table' as reference",
    )
    parser.add_argument("--reference", default="table", help="reference object label")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output csv path")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="load model/processor only from local cache",
    )
    return parser.parse_args()


def as_numpy_mask(mask: Any) -> np.ndarray:
    # SAM3 returns torch tensors; keep this helper generic to avoid hard dependency in module import.
    if hasattr(mask, "detach") and hasattr(mask, "cpu") and hasattr(mask, "numpy"):
        mask = mask.detach().cpu().numpy()
    return mask.astype(np.uint8)


def find_largest_instance(
    masks: Iterable[torch.Tensor | np.ndarray],
    label: str,
) -> Detection | None:
    best: Detection | None = None
    best_area = -1

    for idx, mask in enumerate(masks):
        mask_np = as_numpy_mask(mask)
        if mask_np.sum() == 0:
            continue

        ys, xs = np.where(mask_np > 0)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        area = int((x_max - x_min + 1) * (y_max - y_min + 1))

        if area > best_area:
            best_area = area
            best = Detection(
                label=label,
                instance_id=idx,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )

    return best


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    try:
        import torch
        from transformers import Sam3Model, Sam3Processor
    except ImportError as e:
        raise SystemExit(
            "Missing dependency. Please install: pip install torch transformers pillow numpy"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")
    print(f"[INFO] image: {args.image}")
    print(f"[INFO] prompts: {args.prompts}")

    model = Sam3Model.from_pretrained("facebook/sam3", local_files_only=args.local_files_only).to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3", local_files_only=args.local_files_only)

    image = Image.open(args.image).convert("RGB")

    detections: list[Detection] = []
    for prompt in args.prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        result = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        masks = result.get("masks", [])
        best = find_largest_instance(masks, label=prompt)
        if best is None:
            print(f"[WARN] no instance found for: {prompt}")
            continue

        detections.append(best)
        print(
            f"[OK] {prompt}: bbox=({best.x_min},{best.y_min})-({best.x_max},{best.y_max}), "
            f"center=({best.center_x:.1f},{best.center_y:.1f})"
        )

    ref_candidates = [d for d in detections if d.label.lower() == args.reference.lower()]
    if not ref_candidates:
        raise RuntimeError(
            f"Reference object '{args.reference}' not found. "
            f"Detected labels: {[d.label for d in detections]}"
        )

    reference = ref_candidates[0]
    ref_cx, ref_cy = reference.center_x, reference.center_y

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "instance_id",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "bbox_width",
                "bbox_height",
                "bbox_area",
                "center_x",
                "center_y",
                "rel_x_to_table",
                "rel_y_to_table",
            ]
        )

        for d in detections:
            rel_x = d.center_x - ref_cx
            rel_y = d.center_y - ref_cy
            writer.writerow(
                [
                    d.label,
                    d.instance_id,
                    d.x_min,
                    d.y_min,
                    d.x_max,
                    d.y_max,
                    d.bbox_width,
                    d.bbox_height,
                    d.bbox_area,
                    round(d.center_x, 2),
                    round(d.center_y, 2),
                    round(rel_x, 2),
                    round(rel_y, 2),
                ]
            )

    # Print a quick table for direct terminal inspection.
    print(
        "\nlabel         bbox_w  bbox_h  bbox_area  center_x  center_y  rel_x_to_table  rel_y_to_table"
    )
    for d in detections:
        rel_x = d.center_x - ref_cx
        rel_y = d.center_y - ref_cy
        print(
            f"{d.label[:12]:<12}  {d.bbox_width:>6d}  {d.bbox_height:>6d}  {d.bbox_area:>9d}  "
            f"{d.center_x:>8.1f}  {d.center_y:>8.1f}  {rel_x:>14.1f}  {rel_y:>14.1f}"
        )

    print(f"\n[DONE] csv saved to: {args.output}")
    print("[NOTE] rel_y_to_table > 0 means below table center in image coordinates.")


if __name__ == "__main__":
    main()
