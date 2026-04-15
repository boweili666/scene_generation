#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_ROBOT_SRC = PROJECT_ROOT / "scene_robot" / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from app.backend.services.grasp_scene_adapter import default_scene_grasp_proposals_path
from scene_robot_apps.grasp_execution import expand_grasp_candidates


DEFAULT_SELECTED_PATH = PROJECT_ROOT / "runtime" / "robot_placement" / "selected_grasp_proposal.json"
DEFAULT_SHORTLIST_PATH = PROJECT_ROOT / "runtime" / "robot_placement" / "grasp_candidate_shortlist.json"


def _import_open3d():
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "open3d is not installed in the current Python environment. "
            "Install it first, then rerun this script."
        ) from exc
    return o3d


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_default_scene_grasp_proposals_path(
    scene_usd_path: Path | None,
    scene_grasp_proposals_path: Path | None,
) -> Path:
    if scene_grasp_proposals_path is not None:
        return scene_grasp_proposals_path.resolve()
    if scene_usd_path is None:
        raise ValueError("Provide either --scene-grasp-proposals-path or --scene-usd-path.")
    return default_scene_grasp_proposals_path(scene_usd_path).resolve()


def _quat_wxyz_to_matrix(quat_wxyz: list[float] | tuple[float, ...]) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
    return R.from_quat(quat_xyzw).as_matrix()


def _pose_matrix(position_world: list[float] | tuple[float, ...], quat_wxyz_world: list[float] | tuple[float, ...]) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _quat_wxyz_to_matrix(quat_wxyz_world)
    matrix[:3, 3] = np.asarray(position_world, dtype=float).reshape(3)
    return matrix


def _row_to_column_transform(row_transform: list[list[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(row_transform, dtype=float).reshape(4, 4)
    return matrix.T


def _trimesh_to_open3d(mesh: trimesh.Trimesh):
    o3d = _import_open3d()
    geom = o3d.geometry.TriangleMesh()
    geom.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    geom.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    vertex_colors = None
    if hasattr(mesh.visual, "vertex_colors"):
        colors = np.asarray(mesh.visual.vertex_colors)
        if colors.ndim == 2 and colors.shape[0] == len(mesh.vertices) and colors.shape[1] >= 3:
            vertex_colors = colors[:, :3].astype(np.float64) / 255.0
    if vertex_colors is not None:
        geom.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    else:
        geom.paint_uniform_color([0.78, 0.78, 0.78])
    geom.compute_vertex_normals()
    return geom


def _load_world_mesh(object_payload: dict[str, Any]):
    glb_path = Path(str(object_payload.get("source_glb") or "")).resolve()
    if not glb_path.exists():
        raise FileNotFoundError(f"Source GLB not found: {glb_path}")
    loaded = trimesh.load(glb_path, force="scene")
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Failed to load a triangle mesh from {glb_path}")
    mesh = mesh.copy()
    world_transform = object_payload.get("world_transform")
    if world_transform is not None:
        mesh.apply_transform(_row_to_column_transform(world_transform))
    return mesh


def _candidate_entry_from_execution_payload(payload: dict[str, Any]) -> dict[str, Any]:
    grasp = payload.get("grasp", {})
    return {
        "candidate_id": grasp.get("candidate_id", payload.get("candidate_id")),
        "primitive_type": grasp.get("primitive_type"),
        "score": float(payload.get("ranking_score", payload.get("score", grasp.get("score", 0.0)))),
        "position_world": grasp.get("position_world"),
        "quat_wxyz_world": grasp.get("quat_wxyz_world"),
        "approach_axis_world": grasp.get("approach_axis_world"),
        "source_branch": grasp.get("source_branch"),
        "arm_side": payload.get("arm_side"),
    }


def _load_selected_candidate(selected_path: Path | None, target_prim: str) -> dict[str, Any] | None:
    if selected_path is None or not selected_path.exists():
        return None
    payload = _load_json(selected_path)
    if str(payload.get("target_prim")) != str(target_prim):
        return None
    selected = payload.get("selected_grasp_proposal")
    if not isinstance(selected, dict):
        return None
    return _candidate_entry_from_execution_payload(selected)


def _load_shortlist_candidates(shortlist_path: Path | None, target_prim: str, max_count: int) -> list[dict[str, Any]]:
    if shortlist_path is None or not shortlist_path.exists():
        return []
    payload = _load_json(shortlist_path)
    if str(payload.get("target_prim")) != str(target_prim):
        return []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [_candidate_entry_from_execution_payload(entry) for entry in candidates[: max(0, int(max_count))] if isinstance(entry, dict)]


def _expand_scene_grasp_candidates(
    scene_grasp_payload: dict[str, Any],
    *,
    target_prim: str,
    axis_band_slide_samples: int,
    axis_band_ring_samples: int,
    max_count: int,
) -> list[dict[str, Any]]:
    expanded = expand_grasp_candidates(
        scene_grasp_payload,
        target_prim=target_prim,
        axis_band_slide_samples=axis_band_slide_samples,
        axis_band_ring_samples=axis_band_ring_samples,
    )
    candidates: list[dict[str, Any]] = []
    for candidate in expanded[: max(0, int(max_count))]:
        candidates.append(
            {
                "candidate_id": candidate.candidate_id,
                "primitive_type": candidate.primitive_type,
                "score": float(candidate.score),
                "position_world": list(candidate.position_world),
                "quat_wxyz_world": list(candidate.quat_wxyz_world),
                "approach_axis_world": list(candidate.approach_axis_world),
                "source_branch": candidate.source_branch,
                "arm_side": None,
            }
        )
    return candidates


def _make_coordinate_frame(position_world: list[float] | tuple[float, ...], quat_wxyz_world: list[float] | tuple[float, ...], *, size: float):
    o3d = _import_open3d()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(size), origin=[0.0, 0.0, 0.0])
    frame.transform(_pose_matrix(position_world, quat_wxyz_world))
    return frame


def _make_sphere(center_world: list[float] | tuple[float, ...], *, radius: float, color: tuple[float, float, float]):
    o3d = _import_open3d()
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([float(color[0]), float(color[1]), float(color[2])])
    sphere.translate(np.asarray(center_world, dtype=float).reshape(3))
    return sphere


def _make_approach_line(position_world: list[float] | tuple[float, ...], approach_axis_world: list[float] | tuple[float, ...] | None, *, length: float):
    if not isinstance(approach_axis_world, (list, tuple)) or len(approach_axis_world) != 3:
        return None
    o3d = _import_open3d()
    start = np.asarray(position_world, dtype=float).reshape(3)
    axis = np.asarray(approach_axis_world, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-8:
        return None
    end = start + (axis / norm) * float(length)
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(np.stack([start, end], axis=0))
    lines.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    lines.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.55, 0.1]], dtype=np.float64))
    return lines


def _collect_visual_geometries(
    object_payload: dict[str, Any],
    *,
    all_candidates: list[dict[str, Any]],
    selected_candidate: dict[str, Any] | None,
    frame_size: float,
    selected_frame_size: float,
    approach_length: float,
) -> list[Any]:
    geoms: list[Any] = []
    mesh = _load_world_mesh(object_payload)
    geoms.append(_trimesh_to_open3d(mesh))

    bbox = mesh.bounds
    diag = float(np.linalg.norm(np.asarray(bbox[1], dtype=float) - np.asarray(bbox[0], dtype=float)))
    marker_radius = max(0.003, diag * 0.03)

    for candidate in all_candidates:
        geoms.append(
            _make_coordinate_frame(
                candidate["position_world"],
                candidate["quat_wxyz_world"],
                size=float(frame_size),
            )
        )
        approach_line = _make_approach_line(
            candidate["position_world"],
            candidate.get("approach_axis_world"),
            length=float(approach_length),
        )
        if approach_line is not None:
            geoms.append(approach_line)

    if selected_candidate is not None:
        geoms.append(
            _make_coordinate_frame(
                selected_candidate["position_world"],
                selected_candidate["quat_wxyz_world"],
                size=float(selected_frame_size),
            )
        )
        geoms.append(
            _make_sphere(
                selected_candidate["position_world"],
                radius=marker_radius,
                color=(1.0, 0.45, 0.05),
            )
        )
        approach_line = _make_approach_line(
            selected_candidate["position_world"],
            selected_candidate.get("approach_axis_world"),
            length=float(approach_length) * 1.2,
        )
        if approach_line is not None:
            geoms.append(approach_line)

    world_frame = _make_coordinate_frame([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], size=max(0.05, float(frame_size) * 1.2))
    geoms.append(world_frame)
    return geoms


def _save_screenshot(geoms: list[Any], screenshot_path: Path) -> None:
    o3d = _import_open3d()
    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window(window_name="grasp_debug", width=1440, height=960, visible=False)
        for geom in geoms:
            vis.add_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(screenshot_path), do_render=True)
    finally:
        vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize scene mesh and grasp poses in Open3D.")
    parser.add_argument("--target", required=True, help="Target prim path, e.g. /World/bolt_2")
    parser.add_argument("--scene-usd-path", type=Path, default=None)
    parser.add_argument("--scene-grasp-proposals-path", type=Path, default=None)
    parser.add_argument("--selected-proposal-path", type=Path, default=DEFAULT_SELECTED_PATH)
    parser.add_argument("--shortlist-path", type=Path, default=DEFAULT_SHORTLIST_PATH)
    parser.add_argument("--selected-only", action="store_true", help="Only show selected grasp pose.")
    parser.add_argument("--max-grasps", type=int, default=24, help="Max all-grasp frames to render.")
    parser.add_argument("--axis-band-slide-samples", type=int, default=3)
    parser.add_argument("--axis-band-ring-samples", type=int, default=12)
    parser.add_argument("--frame-size", type=float, default=0.03)
    parser.add_argument("--selected-frame-size", type=float, default=0.06)
    parser.add_argument("--approach-length", type=float, default=0.06)
    parser.add_argument("--screenshot-path", type=Path, default=None)
    parser.add_argument("--headless", action="store_true", help="Do not open an interactive Open3D window.")
    args = parser.parse_args()

    scene_grasp_path = _resolve_default_scene_grasp_proposals_path(args.scene_usd_path, args.scene_grasp_proposals_path)
    if not scene_grasp_path.exists():
        raise FileNotFoundError(f"Scene grasp proposals JSON not found: {scene_grasp_path}")

    scene_grasp_payload = _load_json(scene_grasp_path)
    objects = scene_grasp_payload.get("objects", {})
    if not isinstance(objects, dict) or args.target not in objects:
        available = sorted(objects)[:20] if isinstance(objects, dict) else []
        raise KeyError(f"Target '{args.target}' not found in {scene_grasp_path}. Available targets: {available}")

    object_payload = objects[args.target]
    selected_candidate = _load_selected_candidate(args.selected_proposal_path, args.target)
    all_candidates = [] if args.selected_only else _load_shortlist_candidates(args.shortlist_path, args.target, args.max_grasps)
    if not args.selected_only and not all_candidates:
        all_candidates = _expand_scene_grasp_candidates(
            scene_grasp_payload,
            target_prim=args.target,
            axis_band_slide_samples=args.axis_band_slide_samples,
            axis_band_ring_samples=args.axis_band_ring_samples,
            max_count=args.max_grasps,
        )
    if selected_candidate is None:
        selected_candidate = all_candidates[0] if all_candidates else None

    geoms = _collect_visual_geometries(
        object_payload,
        all_candidates=all_candidates,
        selected_candidate=selected_candidate,
        frame_size=args.frame_size,
        selected_frame_size=args.selected_frame_size,
        approach_length=args.approach_length,
    )

    print(f"[INFO] Scene grasp proposals: {scene_grasp_path}")
    print(f"[INFO] Target: {args.target}")
    print(f"[INFO] Source GLB: {object_payload.get('source_glb')}")
    print(f"[INFO] Rendered all candidates: {len(all_candidates)}")
    if selected_candidate is not None:
        print(f"[INFO] Selected candidate: {selected_candidate.get('candidate_id')}")
        print(f"[INFO] Selected primitive type: {selected_candidate.get('primitive_type')}")
        print(f"[INFO] Selected score: {selected_candidate.get('score')}")

    if args.screenshot_path is not None:
        _save_screenshot(geoms, args.screenshot_path.resolve())
        print(f"[INFO] Screenshot saved to: {args.screenshot_path.resolve()}")

    if not args.headless:
        o3d = _import_open3d()
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"Open3D Grasp Debug: {args.target}",
            width=1440,
            height=960,
        )


if __name__ == "__main__":
    main()
