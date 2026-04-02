from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_IMAGE_PATH = Path("runtime/uploads/latest_input.jpg")
DEFAULT_SCENE_GRAPH = Path("runtime/scene_graph/current_scene_graph.json")
DEFAULT_OUTPUT_ROOT = Path("runtime/real2sim/masks")
DEFAULT_MESH_OUTPUT = Path("runtime/real2sim/meshes")
DEFAULT_REUSE_MESH_DIR = Path("runtime/real2sim/meshes")
DEFAULT_PROMPTS = ["table", "desk lamp", "alarm clock", "notebook", "pen", "glass cup"]
MASK_METADATA_FILENAME = "mask_metadata.json"


def normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment scene objects with SAM3. Prompts are read from scene graph by default. "
            "Can optionally skip SAM3D and reuse existing GLB meshes."
        )
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--scene-graph", type=Path, default=DEFAULT_SCENE_GRAPH)
    parser.add_argument("--prompts", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mesh-output-dir", type=Path, default=DEFAULT_MESH_OUTPUT)
    parser.add_argument("--reuse-mesh-dir", type=Path, default=DEFAULT_REUSE_MESH_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def extract_prompts_from_scene_graph(scene_graph_path: Path) -> list[str]:
    if not scene_graph_path.exists():
        return []

    with scene_graph_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    names: list[str] = []

    # Format A: {"objects": [{"class_name": "Table"}, ...]}
    objects = data.get("objects")
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            val = obj.get("class_name") or obj.get("class")
            if isinstance(val, str) and val.strip():
                names.append(val.strip().lower())

    # Format B: {"obj": {"/World/SM_Table_0": {"class": "table"}}}
    obj_map = data.get("obj")
    if isinstance(obj_map, dict):
        for obj in obj_map.values():
            if not isinstance(obj, dict):
                continue
            val = obj.get("class") or obj.get("class_name")
            if isinstance(val, str) and val.strip():
                names.append(val.strip().lower())

    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)

    return deduped


def choose_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts:
        return args.prompts

    graph_prompts = extract_prompts_from_scene_graph(args.scene_graph)
    if graph_prompts:
        return graph_prompts

    return DEFAULT_PROMPTS

def copy_reused_meshes(prompts: list[str], reuse_mesh_dir: Path, mesh_output_dir: Path) -> int:
    if not reuse_mesh_dir.exists():
        print(f"[WARN] Reuse mesh directory does not exist: {reuse_mesh_dir}")
        return 0

    if reuse_mesh_dir.resolve() == mesh_output_dir.resolve():
        return 0

    copied = 0
    prompt_keys = [normalize_label(p) for p in prompts]
    for mesh_path in sorted(reuse_mesh_dir.glob("*.glb")):
        name = mesh_path.name.lower()
        if any(key and key in name for key in prompt_keys):
            dst = mesh_output_dir / mesh_path.name
            if mesh_path.resolve() == dst.resolve():
                continue
            shutil.copy2(mesh_path, dst)
            copied += 1

    return copied


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    try:
        import torch
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install in sam3 env: pip install torch transformers pillow numpy requests"
        ) from exc

    prompts = choose_prompts(args)
    if not prompts:
        raise RuntimeError("No prompts available. Provide --prompts or a valid scene graph.")

    # Keep table first, because downstream relative-XY step usually uses table as reference.
    table_like = [p for p in prompts if p.strip().lower() == "table"]
    others = [p for p in prompts if p.strip().lower() != "table"]
    prompts = table_like + others

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Image: {args.image}")
    print(f"[INFO] Prompts: {prompts}")
    print(f"[INFO] Scene graph: {args.scene_graph}")

    model = Sam3Model.from_pretrained("facebook/sam3", local_files_only=args.local_files_only).to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3", local_files_only=args.local_files_only)

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.mesh_output_dir.mkdir(parents=True, exist_ok=True)
    image.save(args.output_root / "image.png")

    total_masks = 0
    total_meshes = 0
    global_mask_idx = 0
    mask_metadata: dict[str, dict[str, Any]] = {}

    for prompt in prompts:
        prompt_key = normalize_label(prompt)
        print(f"\n[SEGMENT] {prompt}")

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        masks = results.get("masks", [])
        print(f"[INFO] Found {len(masks)} masks")

        saved_idx = 0
        for idx, mask in enumerate(masks):
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()

            mask = mask.astype(np.uint8)
            if mask.sum() == 0:
                continue

            ys, xs = np.where(mask > 0)
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())

            alpha_full = (mask * 255).astype(np.uint8)
            rgba_full = np.dstack((image_np, alpha_full))

            rgb_crop = image_np[y_min : y_max + 1, x_min : x_max + 1]
            mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1]
            alpha_crop = (mask_crop * 255).astype(np.uint8)
            rgba_crop = np.dstack((rgb_crop, alpha_crop))

            # Save only full-size RGBA masks into a single flat folder:
            # runtime/real2sim/masks/0.png, 1.png, 2.png, ...
            mask_path = args.output_root / f"{global_mask_idx}.png"
            Image.fromarray(rgba_full).save(mask_path)
            mask_metadata[str(global_mask_idx)] = {
                "mask_path": mask_path.name,
                "prompt": prompt.strip().lower(),
                "prompt_key": prompt_key,
                "bbox_xyxy": [x_min, y_min, x_max, y_max],
            }

            total_masks += 1
            saved_idx += 1
            global_mask_idx += 1

            print(
                f"[SAVE] {mask_path} | prompt={prompt_key} | bbox=({x_min},{y_min})-({x_max},{y_max})"
            )

    existing = list(args.mesh_output_dir.glob("*_mesh.glb"))
    if existing:
        print(f"[INFO] Using existing meshes in {args.mesh_output_dir} (count={len(existing)})")
    else:
        reused = copy_reused_meshes(prompts, args.reuse_mesh_dir, args.mesh_output_dir)
        print(f"[INFO] Reused meshes copied: {reused}")

    (args.output_root / MASK_METADATA_FILENAME).write_text(
        json.dumps(mask_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Saved mask metadata: {args.output_root / MASK_METADATA_FILENAME}")

    print(f"\n[DONE] Total saved masks: {total_masks}")
    print(f"[DONE] Mask output root: {args.output_root}")
    print(f"[DONE] Mesh output dir: {args.mesh_output_dir}")
    print(f"[DONE] Generated meshes: {total_meshes}")


if __name__ == "__main__":
    main()
