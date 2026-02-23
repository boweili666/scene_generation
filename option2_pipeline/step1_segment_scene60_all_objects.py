from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_IMAGE_PATH = Path("../sam3/scene_60.jpeg")
DEFAULT_SCENE_GRAPH = Path(
    "/home/lbw/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/bowei/my_viewer/full_scene_graph.json"
)
DEFAULT_OUTPUT_ROOT = Path("option2_pipeline/runtime/masks")
DEFAULT_MESH_OUTPUT = Path("option2_pipeline/runtime/meshes")
DEFAULT_REUSE_MESH_DIR = Path("option2_pipeline/runtime/meshes")
DEFAULT_RTS_SOURCE_DIR = Path("../scene60_mesh_rts")
DEFAULT_SAM3D_URL = "http://128.2.204.116:8000/generate_mesh"
DEFAULT_PROMPTS = ["table", "desk lamp", "alarm clock", "notebook", "pen", "glass cup"]


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
    parser.add_argument("--sam3d-url", default=DEFAULT_SAM3D_URL)
    parser.add_argument("--skip-sam3d", action="store_true")
    parser.add_argument("--reuse-mesh-dir", type=Path, default=DEFAULT_REUSE_MESH_DIR)
    parser.add_argument("--rts-source-dir", type=Path, default=DEFAULT_RTS_SOURCE_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--sam3d-timeout", type=float, default=120.0)
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


def post_to_sam3d(rgba_pil: Image.Image, sam3d_url: str, timeout: float) -> bytes | None:
    import requests

    buf = io.BytesIO()
    rgba_pil.save(buf, format="PNG")
    buf.seek(0)
    files = {"image": ("mask.png", buf, "image/png")}
    params = {
        "texture_resolution": 1024,
        "remesh_option": "none",
        "target_vertex_count": -1,
    }
    response = requests.post(sam3d_url, files=files, params=params, timeout=timeout)
    if response.status_code != 200:
        return None
    return response.content


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


def copy_rts_files(rts_source_dir: Path, runtime_dir: Path) -> int:
    if not rts_source_dir.exists():
        print(f"[WARN] RTS source directory does not exist: {rts_source_dir}")
        return 0

    copied = 0
    for src in sorted(rts_source_dir.glob("*_rts.json")):
        dst = runtime_dir / src.name
        if src.resolve() == dst.resolve():
            continue
        shutil.copy2(src, dst)
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
    print(f"[INFO] skip_sam3d: {args.skip_sam3d}")

    model = Sam3Model.from_pretrained("facebook/sam3", local_files_only=args.local_files_only).to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3", local_files_only=args.local_files_only)

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.mesh_output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = args.output_root.parent
    runtime_dir.mkdir(parents=True, exist_ok=True)

    total_masks = 0
    total_meshes = 0

    for prompt in prompts:
        prompt_key = normalize_label(prompt)
        class_dir = args.output_root / prompt_key
        class_dir.mkdir(parents=True, exist_ok=True)
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

            mask_img = (mask * 255).astype(np.uint8)

            instance_key = f"{prompt_key}_{saved_idx}"
            mask_path = class_dir / f"{instance_key}.png"
            rgba_full_path = class_dir / f"{instance_key}_rgba_fullsize.png"
            rgba_crop_path = class_dir / f"{instance_key}_rgba_crop.png"

            Image.fromarray(mask_img).save(mask_path)
            Image.fromarray(rgba_full).save(rgba_full_path)
            Image.fromarray(rgba_crop).save(rgba_crop_path)

            if saved_idx == 0:
                flat_rgba = runtime_dir / f"{prompt_key}_0_rgba.png"
                flat_full = runtime_dir / f"{prompt_key}_0_rgba_fullsize.png"
                flat_white = runtime_dir / f"{prompt_key}_0_rgba_whitebg.png"

                Image.fromarray(rgba_crop).save(flat_rgba)
                Image.fromarray(rgba_full).save(flat_full)

                white_bg = np.full_like(image_np, 255, dtype=np.uint8)
                white_bg[mask > 0] = image_np[mask > 0]
                Image.fromarray(white_bg).save(flat_white)

            total_masks += 1
            saved_idx += 1

            print(
                f"[SAVE] {mask_path} | bbox=({x_min},{y_min})-({x_max},{y_max})"
            )

            if args.skip_sam3d:
                continue

            mesh_bytes = post_to_sam3d(
                Image.fromarray(rgba_crop, mode="RGBA"),
                sam3d_url=args.sam3d_url,
                timeout=args.sam3d_timeout,
            )
            if not mesh_bytes:
                print(f"[WARN] SAM3D failed for {prompt_key}_{idx}")
                continue

            mesh_path = args.mesh_output_dir / f"{prompt_key}_{idx}_mesh.glb"
            mesh_path.write_bytes(mesh_bytes)
            total_meshes += 1
            print(f"[SAVE] Mesh: {mesh_path}")

    reused = 0
    if args.skip_sam3d:
        existing = list(args.mesh_output_dir.glob("*_mesh.glb"))
        if existing:
            print(f"[INFO] Using existing meshes in {args.mesh_output_dir} (count={len(existing)})")
        else:
            reused = copy_reused_meshes(prompts, args.reuse_mesh_dir, args.mesh_output_dir)
            print(f"[INFO] Reused meshes copied: {reused}")

    copied_rts = copy_rts_files(args.rts_source_dir, runtime_dir)
    print(f"[INFO] RTS files copied: {copied_rts}")

    print(f"\n[DONE] Total saved masks: {total_masks}")
    print(f"[DONE] Mask output root: {args.output_root}")
    print(f"[DONE] Mesh output dir: {args.mesh_output_dir}")
    print(f"[DONE] Generated meshes: {total_meshes}")


if __name__ == "__main__":
    main()
