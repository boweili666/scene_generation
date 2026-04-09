from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_scene_from_glb(glb_path: Path) -> pyrender.Scene:
    import pyrender
    import trimesh

    tm = trimesh.load(glb_path, force="scene")
    if isinstance(tm, trimesh.Scene):
        return pyrender.Scene.from_trimesh_scene(tm, bg_color=[255, 255, 255, 255])
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
    scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))
    return scene


def scene_bounds(scene: pyrender.Scene) -> Tuple[np.ndarray, np.ndarray]:
    pts = []
    for node in scene.mesh_nodes:
        pose = scene.get_pose(node)
        for prim in node.mesh.primitives:
            pos = prim.positions
            homog = np.hstack([pos, np.ones((pos.shape[0], 1), dtype=pos.dtype)])
            pts.append((pose @ homog.T).T[:, :3])
    if not pts:
        return np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])
    all_pts = np.vstack(pts)
    return all_pts.min(axis=0), all_pts.max(axis=0)


def camera_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up = new_up / np.linalg.norm(new_up)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = new_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def apply_brightness(img: Image.Image, brightness: float, contrast: float) -> Image.Image:
    out = ImageEnhance.Brightness(img).enhance(brightness)
    return ImageEnhance.Contrast(out).enhance(contrast)


def render_top_view(
    glb_path: Path,
    image_size: int = 768,
    fill_ratio: float = 0.95,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> tuple[Image.Image, dict]:
    import pyrender

    scene = load_scene_from_glb(glb_path)
    mn, mx = scene_bounds(scene)
    center = (mn + mx) / 2.0
    ext = mx - mn
    radius = float(np.linalg.norm(ext) + 1e-6)
    dist = max(radius * 1.8, 1.0)

    fit = min(max(fill_ratio, 0.1), 0.98)
    xmag = max(float(ext[0]), 1e-5) / (2.0 * fit)
    ymag = max(float(ext[1]), 1e-5) / (2.0 * fit)

    renderer = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)
    scene.ambient_light = np.array([0.30, 0.30, 0.30, 1.0])
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=8.0)
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    rim_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    eye = center + np.array([0.0, 0.0, dist])
    up = np.array([0.0, 1.0, 0.0])
    cam = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)

    cnode = scene.add(cam, pose=camera_pose(eye=eye, target=center, up=up))
    lnode_key = scene.add(key_light, pose=camera_pose(eye=eye, target=center, up=up))
    lnode_fill = scene.add(
        fill_light,
        pose=camera_pose(
            eye=center + np.array([-dist * 0.8, dist * 0.6, dist * 0.9]),
            target=center,
            up=np.array([0.0, 0.0, 1.0]),
        ),
    )
    lnode_rim = scene.add(
        rim_light,
        pose=camera_pose(
            eye=center + np.array([dist * 0.9, -dist * 0.7, dist * 0.6]),
            target=center,
            up=np.array([0.0, 0.0, 1.0]),
        ),
    )

    color, _ = renderer.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
    scene.remove_node(cnode)
    scene.remove_node(lnode_key)
    scene.remove_node(lnode_fill)
    scene.remove_node(lnode_rim)
    renderer.delete()

    img = apply_brightness(Image.fromarray(color).convert("RGB"), brightness=brightness, contrast=contrast)
    meta = {
        "u_axis": "X",
        "v_axis": "Y",
        "u_min_world": float(center[0] - xmag),
        "u_max_world": float(center[0] + xmag),
        "v_min_world": float(center[1] - ymag),
        "v_max_world": float(center[1] + ymag),
        "world_per_pixel_u": float((2.0 * xmag) / max(image_size - 1, 1)),
        "world_per_pixel_v": float((2.0 * ymag) / max(image_size - 1, 1)),
        "camera_world": eye.tolist(),
    }
    return img, meta


def world_xy_to_image_px(meta: dict, x_world: float, y_world: float, image_size: tuple[int, int]) -> tuple[int, int]:
    w, h = image_size
    x_px = (x_world - meta["u_min_world"]) / meta["world_per_pixel_u"]
    y_px = (meta["v_max_world"] - y_world) / meta["world_per_pixel_v"]
    x_px = int(round(np.clip(x_px, 0, w - 1)))
    y_px = int(round(np.clip(y_px, 0, h - 1)))
    return x_px, y_px


def draw_axis_triad(
    image: Image.Image,
    origin_xy: tuple[int, int] | None = None,
    axis_len: int = 84,
) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = load_font(24)
    w, h = out.size
    if origin_xy is None:
        ox, oy = w // 2, h // 2
    else:
        ox, oy = origin_xy

    draw.ellipse((ox - 6, oy - 6, ox + 6, oy + 6), fill=(20, 20, 20))
    draw.line((ox, oy, ox + axis_len, oy), fill=(220, 40, 40), width=5)
    draw.polygon([(ox + axis_len, oy), (ox + axis_len - 14, oy - 7), (ox + axis_len - 14, oy + 7)], fill=(220, 40, 40))
    draw.text((ox + axis_len + 10, oy - 16), "X", fill=(220, 40, 40), font=font)

    draw.line((ox, oy, ox, oy - axis_len), fill=(40, 170, 60), width=5)
    draw.polygon([(ox, oy - axis_len), (ox - 7, oy - axis_len + 14), (ox + 7, oy - axis_len + 14)], fill=(40, 170, 60))
    draw.text((ox - 10, oy - axis_len - 34), "Y", fill=(40, 170, 60), font=font)

    draw.ellipse((ox - 18, oy - 18, ox + 18, oy + 18), outline=(50, 110, 220), width=4)
    draw.ellipse((ox - 4, oy - 4, ox + 4, oy + 4), fill=(50, 110, 220))
    draw.text((ox - 12, oy + 24), "Z", fill=(50, 110, 220), font=font)
    return out


def _scene_points_world(scene: pyrender.Scene) -> np.ndarray:
    pts = []
    for node in scene.mesh_nodes:
        pose = scene.get_pose(node)
        for prim in node.mesh.primitives:
            pos = prim.positions
            homog = np.hstack([pos, np.ones((pos.shape[0], 1), dtype=pos.dtype)])
            pts.append((pose @ homog.T).T[:, :3])
    if not pts:
        return np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=float,
        )
    return np.vstack(pts)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v.copy()
    return v / n


def _camera_basis(forward: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = _normalize(forward)
    r = np.cross(f, up_hint)
    if np.linalg.norm(r) < 1e-8:
        backup = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(f, backup))) > 0.95:
            backup = np.array([0.0, 1.0, 0.0], dtype=float)
        r = np.cross(f, backup)
    r = _normalize(r)
    u = _normalize(np.cross(r, f))
    return r, u


def render_orthographic_view(
    glb_path: Path,
    view_name: str,
    look_dir: tuple[float, float, float],
    up_hint: tuple[float, float, float],
    image_size: int = 768,
    fill_ratio: float = 0.90,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> tuple[Image.Image, dict]:
    import pyrender

    scene = load_scene_from_glb(glb_path)
    pts = _scene_points_world(scene)
    center = pts.mean(axis=0)
    ext = pts.max(axis=0) - pts.min(axis=0)
    radius = float(np.linalg.norm(ext) + 1e-6)
    dist = max(radius * 2.0, 1.0)

    forward = np.array(look_dir, dtype=float)
    right, up = _camera_basis(forward=forward, up_hint=np.array(up_hint, dtype=float))
    rel = pts - center[None, :]
    u_coords = rel @ right
    v_coords = rel @ up

    fit = min(max(fill_ratio, 0.10), 0.98)
    u_min, u_max = float(np.min(u_coords)), float(np.max(u_coords))
    v_min, v_max = float(np.min(v_coords)), float(np.max(v_coords))
    xmag = max((u_max - u_min) * 0.5 / fit, 1e-5)
    ymag = max((v_max - v_min) * 0.5 / fit, 1e-5)

    eye = center - _normalize(forward) * dist

    renderer = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)
    scene.ambient_light = np.array([0.30, 0.30, 0.30, 1.0])
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=8.0)
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    rim_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    cam = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)

    cnode = scene.add(cam, pose=camera_pose(eye=eye, target=center, up=up))
    lnode_key = scene.add(key_light, pose=camera_pose(eye=eye, target=center, up=up))
    lnode_fill = scene.add(
        fill_light,
        pose=camera_pose(
            eye=center + right * (-dist * 0.8) + up * (dist * 0.6),
            target=center,
            up=up,
        ),
    )
    lnode_rim = scene.add(
        rim_light,
        pose=camera_pose(
            eye=center + right * (dist * 0.9) + up * (-dist * 0.7),
            target=center,
            up=up,
        ),
    )

    color, _ = renderer.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
    scene.remove_node(cnode)
    scene.remove_node(lnode_key)
    scene.remove_node(lnode_fill)
    scene.remove_node(lnode_rim)
    renderer.delete()

    img = apply_brightness(Image.fromarray(color).convert("RGB"), brightness=brightness, contrast=contrast)
    meta = {
        "view_name": view_name,
        "center_world": center.tolist(),
        "camera_world": eye.tolist(),
        "look_dir_world": _normalize(forward).tolist(),
        "right_axis_world": right.tolist(),
        "up_axis_world": up.tolist(),
        "u_min_world": float(-xmag),
        "u_max_world": float(xmag),
        "v_min_world": float(-ymag),
        "v_max_world": float(ymag),
        "world_per_pixel_u": float((2.0 * xmag) / max(image_size - 1, 1)),
        "world_per_pixel_v": float((2.0 * ymag) / max(image_size - 1, 1)),
    }
    return img, meta


def draw_projected_world_axes(
    image: Image.Image,
    view_meta: dict,
    origin_xy: tuple[int, int] | None = None,
    axis_len_px: int = 90,
) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = load_font(22)
    w, h = out.size
    ox, oy = origin_xy if origin_xy is not None else (int(w * 0.15), int(h * 0.85))

    right = np.array(view_meta["right_axis_world"], dtype=float)
    up = np.array(view_meta["up_axis_world"], dtype=float)
    world_axes = {
        "X": (np.array([1.0, 0.0, 0.0], dtype=float), (220, 40, 40)),
        "Y": (np.array([0.0, 1.0, 0.0], dtype=float), (40, 170, 60)),
        "Z": (np.array([0.0, 0.0, 1.0], dtype=float), (50, 110, 220)),
    }

    draw.ellipse((ox - 6, oy - 6, ox + 6, oy + 6), fill=(20, 20, 20))

    for axis_name, (vec_world, color) in world_axes.items():
        du = float(np.dot(vec_world, right))
        dv = float(np.dot(vec_world, up))
        d = np.array([du, dv], dtype=float)
        n = float(np.linalg.norm(d))
        if n < 1e-8:
            continue
        d = d / n
        ex = int(round(ox + d[0] * axis_len_px))
        ey = int(round(oy - d[1] * axis_len_px))
        draw.line((ox, oy, ex, ey), fill=color, width=5)
        draw.text((ex + 6, ey - 18), axis_name, fill=color, font=font)

    return out


def render_three_views(
    glb_path: Path,
    renders_dir: Path,
    debug_dir: Path,
    image_size: int,
    fill_ratio: float,
) -> dict:
    renders_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    views_cfg = {
        "front_view": {"look_dir": (0.0, 0.0, -1.0), "up_hint": (0.0, 1.0, 0.0)},
        "side_view": {"look_dir": (-1.0, 0.0, 0.0), "up_hint": (0.0, 1.0, 0.0)},
        "top_view": {"look_dir": (0.0, -1.0, 0.0), "up_hint": (0.0, 0.0, -1.0)},
    }

    rendered: dict = {}
    for view_name, cfg in views_cfg.items():
        image, meta = render_orthographic_view(
            glb_path=glb_path,
            view_name=view_name,
            look_dir=cfg["look_dir"],
            up_hint=cfg["up_hint"],
            image_size=image_size,
            fill_ratio=fill_ratio,
        )
        annotated = draw_projected_world_axes(image=image, view_meta=meta)

        raw_path = renders_dir / f"{glb_path.stem}_{view_name}.png"
        axes_path = renders_dir / f"{glb_path.stem}_{view_name}_axes.png"
        meta_path = debug_dir / f"{glb_path.stem}_{view_name}_meta.json"

        image.save(raw_path)
        annotated.save(axes_path)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        rendered[view_name] = {
            "image_path": raw_path,
            "axes_image_path": axes_path,
            "meta_path": meta_path,
        }
    return rendered
