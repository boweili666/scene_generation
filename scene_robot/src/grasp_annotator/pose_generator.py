from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from .render import load_font


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v, dtype=float)
    return v / n


def pixel_to_world(view_meta: dict, x_px: float, y_px: float) -> np.ndarray:
    center = np.array(view_meta["center_world"], dtype=float)
    right = np.array(view_meta["right_axis_world"], dtype=float)
    up = np.array(view_meta["up_axis_world"], dtype=float)

    u = float(view_meta["u_min_world"]) + float(x_px) * float(view_meta["world_per_pixel_u"])
    v = float(view_meta["v_max_world"]) - float(y_px) * float(view_meta["world_per_pixel_v"])
    return center + right * u + up * v


def image_axis_to_world(view_meta: dict, axis_xy: tuple[float, float]) -> np.ndarray:
    right = np.array(view_meta["right_axis_world"], dtype=float)
    up = np.array(view_meta["up_axis_world"], dtype=float)
    axis = np.array(axis_xy, dtype=float)
    world = right * axis[0] + up * (-axis[1])
    n = np.linalg.norm(world)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return world / n


def axis_label_to_vec(axis_label: str) -> np.ndarray:
    mapping = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    return mapping.get(str(axis_label).upper(), np.array([0.0, 1.0, 0.0]))


def direction_label_to_vec(direction_label: str) -> np.ndarray:
    label = str(direction_label).upper().replace(" ", "")
    base = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    if label in base:
        return base[label].astype(float)
    if "+" in label:
        a, b = label.split("+", 1)
        if a in base and b in base:
            return normalize(base[a] + base[b])
    if "-" in label:
        a, b = label.split("-", 1)
        if a in base and b in base:
            return normalize(base[a] - base[b])
    return np.array([1.0, 0.0, 0.0], dtype=float)


def load_scene_points_world(glb_path: Path) -> np.ndarray:
    tm = trimesh.load(glb_path, force="scene")
    if isinstance(tm, trimesh.Scene):
        if not tm.geometry:
            return np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=float)
        pts = []
        for geom in tm.geometry.values():
            if hasattr(geom, "vertices") and len(geom.vertices) > 0:
                pts.append(np.asarray(geom.vertices, dtype=float))
        if not pts:
            return np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=float)
        return np.vstack(pts)
    if hasattr(tm, "vertices") and len(tm.vertices) > 0:
        return np.asarray(tm.vertices, dtype=float)
    return np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=float)


def project_world_to_image(view_meta: dict, p_world: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int]:
    center = np.array(view_meta["center_world"], dtype=float)
    right = np.array(view_meta["right_axis_world"], dtype=float)
    up = np.array(view_meta["up_axis_world"], dtype=float)
    rel = p_world - center
    u = float(np.dot(rel, right))
    v = float(np.dot(rel, up))
    x = (u - float(view_meta["u_min_world"])) / float(view_meta["world_per_pixel_u"])
    y = (float(view_meta["v_max_world"]) - v) / float(view_meta["world_per_pixel_v"])
    w, h = image_size
    x = int(round(np.clip(x, 0, w - 1)))
    y = int(round(np.clip(y, 0, h - 1)))
    return x, y


def choose_best_axis_view(rendered: dict, axis_vec_world: np.ndarray) -> str:
    best_view = "front_view"
    best_score = -1.0
    axis_vec_world = normalize(axis_vec_world)
    for view_name in ("front_view", "side_view", "top_view"):
        meta = json.loads(Path(rendered[view_name]["meta_path"]).read_text(encoding="utf-8"))
        right = np.array(meta["right_axis_world"], dtype=float)
        up = np.array(meta["up_axis_world"], dtype=float)
        du = float(np.dot(axis_vec_world, right))
        dv = float(np.dot(axis_vec_world, up))
        score = math.hypot(du, dv)
        if score > best_score:
            best_score = score
            best_view = view_name
    return best_view


def make_axis_samples(
    axis_vec_world: np.ndarray,
    center_world: np.ndarray,
    points_world: np.ndarray,
    num_points: int,
) -> list[dict]:
    axis = normalize(axis_vec_world)
    rel = points_world - center_world[None, :]
    t = rel @ axis
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    if t_max <= t_min:
        t_min, t_max = -0.2, 0.2
    span = t_max - t_min
    lo = t_min + 0.15 * span
    hi = t_max - 0.15 * span
    if hi <= lo:
        lo, hi = t_min, t_max

    ts = np.linspace(lo, hi, max(2, int(num_points)))
    out = []
    for i, tv in enumerate(ts):
        point = center_world + axis * float(tv)
        out.append({"point_id": i, "t_on_axis": float(tv), "world_xyz": point.tolist()})
    return out


def draw_axis_sample_points(
    base_image_path: Path,
    view_meta: dict,
    samples: list[dict],
    out_path: Path,
) -> list[dict]:
    img = Image.open(base_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = load_font(20)
    w, h = img.size

    projected = []
    for sample in samples:
        point = np.array(sample["world_xyz"], dtype=float)
        x, y = project_world_to_image(view_meta, point, (w, h))
        projected.append({"point_id": sample["point_id"], "pixel_xy": [int(x), int(y)], "world_xyz": sample["world_xyz"]})
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 90, 0), outline=(255, 255, 255), width=2)
        label = str(sample["point_id"])
        l, t, r, b = draw.textbbox((0, 0), label, font=font)
        tw, th = r - l, b - t
        tx, ty = x + 8, y - th - 4
        draw.rounded_rectangle((tx - 2, ty - 1, tx + tw + 2, ty + th + 1), radius=4, fill=(255, 255, 255))
        draw.text((tx, ty), label, fill=(200, 30, 30), font=font)

    if len(projected) >= 2:
        p0 = projected[0]["pixel_xy"]
        p1 = projected[-1]["pixel_xy"]
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(40, 200, 200), width=3)

    img.save(out_path)
    return projected


def orthonormal_perp_basis(axis_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(np.dot(axis_unit, ref))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    e1 = normalize(np.cross(axis_unit, ref))
    e2 = normalize(np.cross(axis_unit, e1))
    return e1, e2


def estimate_axis_radius(points_world: np.ndarray, center_world: np.ndarray, axis_unit: np.ndarray) -> float:
    rel = points_world - center_world[None, :]
    proj = (rel @ axis_unit)[:, None] * axis_unit[None, :]
    perp = rel - proj
    d = np.linalg.norm(perp, axis=1)
    if len(d) == 0:
        return 0.05
    return max(0.01, float(np.percentile(d, 70)))


def make_ring_grasp_poses(
    selected_world: np.ndarray,
    axis_unit: np.ndarray,
    radius: float,
    num_poses: int,
) -> list[dict]:
    e1, e2 = orthonormal_perp_basis(axis_unit)
    poses = []
    n = max(4, int(num_poses))
    for i in range(n):
        theta = (2.0 * math.pi * float(i)) / float(n)
        radial = normalize(math.cos(theta) * e1 + math.sin(theta) * e2)
        tangential = normalize(np.cross(axis_unit, radial))
        if np.linalg.norm(tangential) < 1e-8:
            tangential = e1
        pos = selected_world + radial * float(radius)
        poses.append(
            {
                "pose_id": i,
                "position": pos.tolist(),
                "approach_axis": (-radial).tolist(),
                "closing_axis": tangential.tolist(),
                "angle_rad": theta,
            }
        )
    return poses


def save_axis_effect_image(
    base_image_path: Path,
    view_meta: dict,
    selected_point_world: np.ndarray,
    axis_unit: np.ndarray,
    radius: float,
    out_path: Path,
) -> None:
    img = Image.open(base_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = load_font(20)
    w, h = img.size

    cx, cy = project_world_to_image(view_meta, selected_point_world, (w, h))
    draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(255, 90, 0), outline=(255, 255, 255), width=2)
    draw.text((cx + 12, cy - 26), "selected grasp height", fill=(200, 30, 30), font=font)

    e1, e2 = orthonormal_perp_basis(axis_unit)
    ring_px = []
    for i in range(72):
        theta = (2.0 * math.pi * i) / 72.0
        pw = selected_point_world + (math.cos(theta) * e1 + math.sin(theta) * e2) * radius
        x, y = project_world_to_image(view_meta, pw, (w, h))
        ring_px.append((x, y))
    if ring_px:
        draw.line(ring_px + [ring_px[0]], fill=(40, 220, 40), width=3)

    img.save(out_path)


def prepare_graspnet_sys_path(repo_root: Path) -> None:
    add_paths = [
        repo_root,
        repo_root / "models",
        repo_root / "dataset",
        repo_root / "utils",
        repo_root / "graspnetAPI",
        repo_root / "pointnet2",
        repo_root / "knn",
    ]
    for path in add_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def sample_cloud_from_glb(glb_path: Path, num_points: int) -> np.ndarray:
    tm = trimesh.load(glb_path, force="scene")
    mesh = tm.dump(concatenate=True) if isinstance(tm, trimesh.Scene) else tm
    if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise RuntimeError(f"Failed to get mesh vertices from: {glb_path}")

    target_n = max(4096, int(num_points))
    if hasattr(mesh, "faces") and len(mesh.faces) > 0:
        pts, _ = trimesh.sample.sample_surface(mesh, target_n)
        cloud = np.asarray(pts, dtype=np.float32)
    else:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        if len(verts) >= target_n:
            idx = np.random.choice(len(verts), target_n, replace=False)
            cloud = verts[idx]
        else:
            idx1 = np.arange(len(verts))
            idx2 = np.random.choice(len(verts), target_n - len(verts), replace=True)
            cloud = verts[np.concatenate([idx1, idx2], axis=0)]
    return cloud


def run_graspnet_inference(
    glb_path: Path,
    *,
    repo_root: Path,
    checkpoint_path: Path,
    num_points: int,
    max_poses: int,
    collision_thresh: float,
    voxel_size: float,
) -> tuple[list[dict], np.ndarray]:
    prepare_graspnet_sys_path(repo_root)

    import torch
    from collision_detector import ModelFreeCollisionDetector
    from graspnet import GraspNet, pred_decode
    from graspnetAPI import GraspGroup

    cloud = sample_cloud_from_glb(glb_path, num_points=num_points)
    if len(cloud) >= num_points:
        idx = np.random.choice(len(cloud), num_points, replace=False)
    else:
        idx1 = np.arange(len(cloud))
        idx2 = np.random.choice(len(cloud), num_points - len(cloud), replace=True)
        idx = np.concatenate([idx1, idx2], axis=0)
    sampled = cloud[idx]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    net.to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    end_points = {"point_clouds": torch.from_numpy(sampled[None, ...].astype(np.float32)).to(device)}
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    if collision_thresh > 0:
        try:
            mfcdetector = ModelFreeCollisionDetector(sampled, voxel_size=voxel_size)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
            gg = gg[~collision_mask]
        except Exception:
            pass

    try:
        gg = gg.nms()
    except Exception:
        pass
    gg.sort_by_score()
    gg = gg[: max(1, int(max_poses))]

    poses = []
    arr = gg.grasp_group_array
    for i in range(len(arr)):
        grasp = arr[i]
        poses.append(
            {
                "pose_id": int(i),
                "score": float(grasp[0]),
                "width": float(grasp[1]),
                "height": float(grasp[2]),
                "depth": float(grasp[3]),
                "rotation_matrix": grasp[4:13].reshape(3, 3).tolist(),
                "translation": grasp[13:16].tolist(),
                "object_id": int(grasp[16]),
            }
        )
    return poses, sampled


def save_graspnet_effect_image(
    base_image_path: Path,
    view_meta: dict,
    grasp_poses: list[dict],
    out_path: Path,
) -> None:
    img = Image.open(base_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = load_font(18)
    w, h = img.size

    def proj(pt: np.ndarray) -> tuple[int, int]:
        return project_world_to_image(view_meta, pt, (w, h))

    for pose in grasp_poses[:20]:
        center = np.array(pose["translation"], dtype=float)
        rotation = np.array(pose["rotation_matrix"], dtype=float)
        width = float(pose.get("width", 0.05))
        depth = float(pose.get("depth", 0.02))
        score = float(pose.get("score", 0.0))

        tip_left = np.array([0.0, -width * 0.5, 0.0], dtype=float)
        tip_right = np.array([0.0, width * 0.5, 0.0], dtype=float)
        root_left = np.array([-depth, -width * 0.5, 0.0], dtype=float)
        root_right = np.array([-depth, width * 0.5, 0.0], dtype=float)
        tail = np.array([-(depth + 0.04), 0.0, 0.0], dtype=float)

        pts_world = {
            "tip_left": center + rotation @ tip_left,
            "tip_right": center + rotation @ tip_right,
            "root_left": center + rotation @ root_left,
            "root_right": center + rotation @ root_right,
            "tail": center + rotation @ tail,
            "center": center,
        }
        pts_px = {key: proj(value) for key, value in pts_world.items()}

        color = (int(40 + 180 * max(0.0, min(1.0, score))), 60, int(220 - 160 * max(0.0, min(1.0, score))))
        draw.line((*pts_px["tip_left"], *pts_px["root_left"]), fill=color, width=3)
        draw.line((*pts_px["tip_right"], *pts_px["root_right"]), fill=color, width=3)
        draw.line((*pts_px["root_left"], *pts_px["root_right"]), fill=color, width=3)
        draw.line((*pts_px["tail"], *pts_px["center"]), fill=color, width=2)
        draw.ellipse(
            (pts_px["center"][0] - 3, pts_px["center"][1] - 3, pts_px["center"][0] + 3, pts_px["center"][1] + 3),
            fill=(255, 120, 0),
            outline=(255, 255, 255),
            width=1,
        )
        draw.text((pts_px["center"][0] + 6, pts_px["center"][1] - 9), str(pose["pose_id"]), fill=(200, 30, 30), font=font)

    img.save(out_path)


def build_graspnet_group_array(grasp_poses: list[dict]) -> np.ndarray:
    arr = np.zeros((len(grasp_poses), 17), dtype=np.float64)
    for i, pose in enumerate(grasp_poses):
        arr[i, 0] = float(pose.get("score", 0.0))
        arr[i, 1] = float(pose.get("width", 0.02))
        arr[i, 2] = float(pose.get("height", 0.02))
        arr[i, 3] = float(pose.get("depth", 0.02))
        arr[i, 4:13] = np.array(pose["rotation_matrix"], dtype=np.float64).reshape(9)
        arr[i, 13:16] = np.array(pose["translation"], dtype=np.float64).reshape(3)
        arr[i, 16] = float(pose.get("object_id", -1))
    return arr


def axes_to_rotation_matrix(approach_axis: np.ndarray, closing_axis: np.ndarray) -> np.ndarray:
    x = normalize(np.array(approach_axis, dtype=float))
    if np.linalg.norm(x) < 1e-8:
        x = np.array([1.0, 0.0, 0.0], dtype=float)
    y = np.array(closing_axis, dtype=float) - x * float(np.dot(closing_axis, x))
    y = normalize(y)
    if np.linalg.norm(y) < 1e-8:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(ref, x))) > 0.95:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
        y = normalize(np.cross(ref, x))
    z = normalize(np.cross(x, y))
    y = normalize(np.cross(z, x))
    return np.stack([x, y, z], axis=1)


def save_open3d_grippers_screenshot(
    repo_root: Path,
    cloud_points: np.ndarray,
    grasp_poses: list[dict],
    out_path: Path,
    show_window: bool,
    selected_point_world: np.ndarray | None = None,
) -> bool:
    prepare_graspnet_sys_path(repo_root)
    import open3d as o3d
    from graspnetAPI import GraspGroup

    gg = GraspGroup(build_graspnet_group_array(grasp_poses))
    try:
        gg = gg.nms()
    except Exception:
        pass
    gg.sort_by_score()
    gg = gg[:50]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_points.astype(np.float64))
    cloud.paint_uniform_color([0.7, 0.7, 0.7])
    geoms = [cloud, *gg.to_open3d_geometry_list()]

    if selected_point_world is not None:
        bb_min = cloud_points.min(axis=0)
        bb_max = cloud_points.max(axis=0)
        diag = float(np.linalg.norm(bb_max - bb_min))
        radius = max(diag * 0.015, 0.005)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([1.0, 0.45, 0.1])
        sphere.translate(np.array(selected_point_world, dtype=float))
        geoms.append(sphere)

    if show_window:
        try:
            o3d.visualization.draw_geometries(geoms)
        except Exception:
            pass

    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window(window_name="grasp_result", width=1280, height=720, visible=False)
        for geom in geoms:
            vis.add_geometry(geom)
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out_path), do_render=True)
    except Exception:
        return False
    finally:
        vis.destroy_window()
    return out_path.exists()


def save_graspnet_open3d_screenshot(
    repo_root: Path,
    cloud_points: np.ndarray,
    grasp_poses: list[dict],
    out_path: Path,
    show_window: bool,
) -> bool:
    return save_open3d_grippers_screenshot(
        repo_root=repo_root,
        cloud_points=cloud_points,
        grasp_poses=grasp_poses,
        out_path=out_path,
        show_window=show_window,
        selected_point_world=None,
    )


def save_final_effect_image(base_image_path: Path, selected_candidate: dict, out_path: Path) -> None:
    img = Image.open(base_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = load_font(22)

    cx, cy = selected_candidate["center_xy"]
    axis = np.array(selected_candidate["gripper_axis_xy"], dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    axis = np.array([1.0, 0.0], dtype=float) if axis_norm < 1e-8 else axis / axis_norm

    half = max(10, int(round(float(selected_candidate["width_px"]) * 0.25)))
    p0 = np.array([cx, cy], dtype=float) - axis * half
    p1 = np.array([cx, cy], dtype=float) + axis * half
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))

    draw.line((x0, y0, x1, y1), fill=(40, 220, 40), width=5)
    draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(255, 90, 0), outline=(255, 255, 255), width=2)

    label = f"GRASP ID {selected_candidate['id']}"
    tx = cx + 12
    ty = cy - 30
    l, t, r, b = draw.textbbox((0, 0), label, font=font)
    tw, th = r - l, b - t
    draw.rounded_rectangle((tx - 3, ty - 2, tx + tw + 3, ty + th + 2), radius=5, fill=(255, 255, 255))
    draw.text((tx, ty), label, fill=(200, 30, 30), font=font)

    img.save(out_path)
