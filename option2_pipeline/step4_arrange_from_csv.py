#!/usr/bin/env python3
"""
Arrange objects on table using:
1) yaw estimated from per-object reference images,
2) XY offsets from sam3_bbox_relative.csv (converted to XZ in Y-up),
3) Y alignment by bbox bottom-to-table-top.

Notes:
- Scene uses Y-up for placement.
- Table is the reference frame at identity transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import trimesh
except ImportError as exc:
    raise SystemExit("Missing dependency 'trimesh'. Install with: pip install trimesh") from exc

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
try:
    import pyrender
except ImportError as exc:
    raise SystemExit("Missing dependency 'pyrender'. Install with: pip install pyrender") from exc

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Missing dependency 'Pillow'. Install with: pip install pillow") from exc

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError as exc:
    raise SystemExit("Missing dependency 'scikit-image'. Install with: pip install scikit-image") from exc


@dataclass
class CsvRow:
    label: str
    rel_x_to_table: float
    rel_y_to_table: float
    bbox_width: float
    bbox_height: float


def normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz.astype(float)
    n = np.linalg.norm([w, x, y, z])
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def yaw_matrix_y(yaw_deg: float) -> np.ndarray:
    r = math.radians(yaw_deg)
    c, s = math.cos(r), math.sin(r)
    m = np.eye(4, dtype=float)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def load_single_mesh(mesh_path: Path) -> trimesh.Trimesh:
    scene_or_mesh = trimesh.load(mesh_path, force="scene")
    if scene_or_mesh.is_empty:
        raise ValueError(f"Empty mesh: {mesh_path}")
    if hasattr(scene_or_mesh, "to_geometry"):
        g = scene_or_mesh.to_geometry()
        if isinstance(g, trimesh.Trimesh):
            mesh = g
        else:
            mesh = scene_or_mesh.dump(concatenate=True)
    else:
        mesh = scene_or_mesh.dump(concatenate=True)
    return mesh


def parse_csv(csv_path: Path) -> Dict[str, CsvRow]:
    rows: Dict[str, CsvRow] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = normalize_name(r["label"])
            rows[key] = CsvRow(
                label=r["label"],
                rel_x_to_table=float(r["rel_x_to_table"]),
                rel_y_to_table=float(r["rel_y_to_table"]),
                bbox_width=float(r["bbox_width"]),
                bbox_height=float(r["bbox_height"]),
            )
    return rows


def load_rts(input_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in sorted(input_dir.glob("*_rts.json")):
        with p.open("r", encoding="utf-8") as f:
            j = json.load(f)
        key = p.name.replace("_rts.json", "")
        out[key] = j
    if not out:
        raise FileNotFoundError(f"No *_rts.json found in {input_dir}")
    return out


def find_table_key(rts: Dict[str, dict]) -> str:
    candidates = [k for k in rts if "table" in k]
    if not candidates:
        raise ValueError("No table object found in *_rts.json")
    return sorted(candidates)[0]


def key_to_label_key(obj_key: str) -> str:
    base = obj_key.replace("_0_rgba_fullsize", "")
    parts = base.split("_")
    if len(parts) >= 2:
        # desk_lamp_desk_lamp -> desk_lamp
        half = len(parts) // 2
        if parts[:half] == parts[half:]:
            base = "_".join(parts[:half])
    return normalize_name(base)


def find_ref_image(input_dir: Path, label_key: str) -> Optional[Path]:
    candidates = [
        input_dir / f"{label_key}_0_rgba_whitebg.png",
        input_dir / f"{label_key}_0_rgba_fullsize.png",
        input_dir / f"{label_key}_0_rgba.png",
        input_dir / "masks" / label_key / f"{label_key}_0_rgba_fullsize.png",
        input_dir / "masks" / label_key / f"{label_key}_0_rgba_crop.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def estimate_bg_color_from_corners(img: np.ndarray, patch: int = 24) -> np.ndarray:
    h, w = img.shape[:2]
    p = max(2, min(patch, max(2, h // 4), max(2, w // 4)))
    corners = np.concatenate(
        [
            img[:p, :p].reshape(-1, 3),
            img[:p, -p:].reshape(-1, 3),
            img[-p:, :p].reshape(-1, 3),
            img[-p:, -p:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(corners.astype(np.float32), axis=0)


def mask_from_auto_bg(img: np.ndarray, thr: float = 12.0) -> np.ndarray:
    bg = estimate_bg_color_from_corners(img)
    dist = np.linalg.norm(img.astype(np.float32) - bg[None, None, :], axis=2)
    return dist > thr


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def crop_and_center(img: np.ndarray, mask: np.ndarray, out_size: int = 384, fill_value: int = 255) -> np.ndarray:
    box = bbox_from_mask(mask)
    if box is None:
        return np.full((out_size, out_size, 3), fill_value, dtype=np.uint8)
    y0, y1, x0, x1 = box
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), fill_value, dtype=np.uint8)
    oy = (side - h) // 2
    ox = (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    return np.array(Image.fromarray(canvas).resize((out_size, out_size), Image.BILINEAR))


def normalize_mesh_for_render(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    bmin, bmax = m.bounds
    center = (bmin + bmax) * 0.5
    size = (bmax - bmin).max()
    size = float(size) if size > 1e-8 else 1.0
    m.apply_translation(-center)
    m.apply_scale(1.0 / size)
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = eye - target
    z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = x
    m[:3, 1] = y
    m[:3, 2] = z
    m[:3, 3] = eye
    return m


def render_for_yaw(mesh: trimesh.Trimesh, yaw_deg: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    m = mesh.copy()
    m.apply_transform(yaw_matrix_y(yaw_deg))
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.25, 0.25, 0.25])
    scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(35.0))
    cam_pose = look_at(
        eye=np.array([0.0, 0.1, 2.2], dtype=np.float32),
        target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    scene.add(cam, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    alpha = color[..., 3]
    return color[..., :3], alpha > 5


def estimate_yaw_for_object(
    mesh: trimesh.Trimesh,
    ref_img: np.ndarray,
    samples: int,
    seed: int,
    object_name: str = "",
) -> float:
    random.seed(seed)
    np.random.seed(seed)
    h, w = ref_img.shape[:2]
    ref_mask = mask_from_auto_bg(ref_img)
    ref_crop = crop_and_center(ref_img, ref_mask, out_size=384, fill_value=255)
    render_mesh_norm = normalize_mesh_for_render(mesh)
    best_score = -1e9
    best_yaw = 0.0
    progress_step = max(1, samples // 10)
    for idx in range(samples):
        yaw = random.uniform(-180.0, 180.0)
        r_img, r_mask = render_for_yaw(render_mesh_norm, yaw, w, h)
        r_crop = crop_and_center(r_img, r_mask, out_size=384, fill_value=255)
        score = float(ssim(r_crop, ref_crop, channel_axis=2, data_range=255))
        if score > best_score:
            best_score = score
            best_yaw = yaw
        if ((idx + 1) % progress_step == 0) or (idx + 1 == samples):
            name = object_name or "object"
            print(
                f"[progress][yaw] {name}: {idx + 1}/{samples} "
                f"best_score={best_score:.4f} best_yaw={best_yaw:.2f}"
            )
    return float(best_yaw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="option2_pipeline/runtime")
    parser.add_argument(
        "--mesh-dir",
        default="option2_pipeline/runtime/meshes",
        help="Directory containing *_mesh.glb files. Relative paths are resolved from current working directory.",
    )
    parser.add_argument("--csv-path", default="option2_pipeline/runtime/sam3_bbox_relative.csv")
    parser.add_argument("--output-glb", default="option2_pipeline/runtime/scene_from_csv_yaw.glb")
    parser.add_argument("--output-json", default="option2_pipeline/runtime/scene_from_csv_yaw_transforms.json")
    parser.add_argument("--y-offset", type=float, default=1e-4)
    parser.add_argument("--yaw-samples", type=int, default=120)
    parser.add_argument("--yaw-seed", type=int, default=42)
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=0.0,
        help="Global yaw offset (deg) applied to all non-table objects.",
    )
    parser.add_argument(
        "--invert-image-y",
        action="store_true",
        default=True,
        help="Map image +y(down) to world -z when using rel_y_to_table.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    mesh_dir_arg = Path(args.mesh_dir)
    mesh_dir = mesh_dir_arg if mesh_dir_arg.is_absolute() else (Path.cwd() / mesh_dir_arg)
    mesh_dir = mesh_dir.resolve()
    csv_path = (input_dir / args.csv_path).resolve() if not Path(args.csv_path).is_absolute() else Path(args.csv_path)
    rows = parse_csv(csv_path)
    rts = load_rts(input_dir)
    table_key = find_table_key(rts)
    table_label_key = key_to_label_key(table_key)
    if table_label_key not in rows:
        table_label_key = "table"
    if table_label_key not in rows:
        raise ValueError("Cannot find table row in csv")
    table_row = rows[table_label_key]

    table_mesh = load_single_mesh(input_dir / f"{table_key}_mesh.glb")
    table_size_x = float(table_mesh.bounds[1, 0] - table_mesh.bounds[0, 0])
    table_size_z = float(table_mesh.bounds[1, 2] - table_mesh.bounds[0, 2])
    table_top_y = float(table_mesh.bounds[1, 1])

    table_scale = np.array(rts[table_key]["scale"], dtype=float)
    merged = trimesh.Scene()
    merged.add_geometry(table_mesh.copy(), node_name=table_key)

    results: Dict[str, dict] = {
        table_key: {
            "translation": [0.0, 0.0, 0.0],
            "yaw_deg": 0.0,
            "scale": [1.0, 1.0, 1.0],
            "note": "table reference frame",
        }
    }

    obj_keys = [k for k in sorted(rts.keys()) if k != table_key]
    total_objects = len(obj_keys)
    print(f"[progress] start step4: {total_objects} objects (excluding table)")

    for obj_idx, obj_key in enumerate(obj_keys, start=1):
        print(f"[progress] [{obj_idx}/{total_objects}] start: {obj_key}")
        label_key = key_to_label_key(obj_key)
        if label_key not in rows:
            print(f"[progress] [{obj_idx}/{total_objects}] skip {obj_key}: no csv row for '{label_key}'")
            continue
        obj_row = rows[label_key]
        mesh_path = mesh_dir / f"{obj_key}_mesh.glb"
        if not mesh_path.exists():
            print(f"[progress] [{obj_idx}/{total_objects}] skip {obj_key}: missing mesh")
            continue

        print(f"[progress] [{obj_idx}/{total_objects}] load mesh: {mesh_path.name}")
        obj_mesh = load_single_mesh(mesh_path)
        obj_scale_abs = np.array(rts[obj_key]["scale"], dtype=float)
        obj_scale_rel = obj_scale_abs / np.where(np.abs(table_scale) < 1e-8, 1.0, table_scale)
        obj_mesh.apply_scale(obj_scale_rel)

        ref_img_path = find_ref_image(input_dir, label_key)
        if ref_img_path is not None:
            print(f"[progress] [{obj_idx}/{total_objects}] estimate yaw from ref image")
            ref_img = np.array(Image.open(ref_img_path).convert("RGB"))
            yaw_deg = estimate_yaw_for_object(
                obj_mesh,
                ref_img=ref_img,
                samples=args.yaw_samples,
                seed=args.yaw_seed,
                object_name=obj_key,
            )
        else:
            print(f"[progress] [{obj_idx}/{total_objects}] no ref image, use yaw=0")
            yaw_deg = 0.0
        yaw_deg = ((yaw_deg + float(args.yaw_offset_deg) + 180.0) % 360.0) - 180.0

        obj_mesh.apply_transform(yaw_matrix_y(yaw_deg))

        tx = (obj_row.rel_x_to_table / table_row.bbox_width) * table_size_x
        # Front/back position fix: flip previous mapping sign.
        z_sign = 1.0 if args.invert_image_y else -1.0
        tz = z_sign * (obj_row.rel_y_to_table / table_row.bbox_height) * table_size_z
        obj_min_y = float(obj_mesh.bounds[0, 1])
        ty = table_top_y - obj_min_y + float(args.y_offset)

        obj_mesh.apply_translation([tx, ty, tz])
        merged.add_geometry(obj_mesh, node_name=obj_key)
        print(
            f"[progress] [{obj_idx}/{total_objects}] done: {obj_key} "
            f"xyz=({tx:.4f},{ty:.4f},{tz:.4f}) yaw={yaw_deg:.2f}"
        )

        results[obj_key] = {
            "label_key": label_key,
            "translation_xyz_yup": [float(tx), float(ty), float(tz)],
            "yaw_deg_about_y": float(yaw_deg),
            "scale_rel_to_table": obj_scale_rel.astype(float).tolist(),
            "from_csv_rel_pixels": [obj_row.rel_x_to_table, obj_row.rel_y_to_table],
            "constraints": ["y_up", "bbox_min_y_on_table_max_y"],
            "ref_image": str(ref_img_path) if ref_img_path else None,
        }

    out_glb = Path(args.output_glb).resolve()
    out_json = Path(args.output_json).resolve()
    merged.export(out_glb)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "csv_path": str(csv_path),
                "table_key": table_key,
                "table_bbox_pixels": {
                    "width": table_row.bbox_width,
                    "height": table_row.bbox_height,
                },
                "table_mesh_size_yup": {
                    "x": table_size_x,
                    "y": float(table_mesh.bounds[1, 1] - table_mesh.bounds[0, 1]),
                    "z": table_size_z,
                },
                "objects": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved merged GLB: {out_glb}")
    print(f"Saved transforms: {out_json}")
    print(f"[progress] step4 complete: {len(results) - 1}/{total_objects} objects arranged")


if __name__ == "__main__":
    main()
