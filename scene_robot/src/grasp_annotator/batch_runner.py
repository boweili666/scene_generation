from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .classifier import (
    classify_object,
    extract_skeleton_view,
    pick_axis_sample,
    pick_handle_candidate,
    pick_handle_orientation,
)
from .pose_generator import (
    axes_to_rotation_matrix,
    axis_label_to_vec,
    choose_best_axis_view,
    direction_label_to_vec,
    draw_axis_sample_points,
    estimate_axis_radius,
    image_axis_to_world,
    load_scene_points_world,
    make_axis_samples,
    make_ring_grasp_poses,
    normalize,
    pixel_to_world,
    prepare_graspnet_sys_path,
    project_world_to_image,
    run_graspnet_inference,
    sample_cloud_from_glb,
    save_axis_effect_image,
    save_final_effect_image,
    save_graspnet_open3d_screenshot,
    save_open3d_grippers_screenshot,
)
from .render import render_three_views
from .schema import build_annotation
from .skeleton import analyze_grasp_image, save_grasp_candidate_artifacts


@dataclass
class PipelineConfig:
    size: int = 768
    fill_ratio: float = 0.90
    model_classify: str = "gpt-5"
    model_pick: str = "gpt-5"
    max_candidates: int = 24
    min_width_px: float = 12.0
    max_width_px: float = 220.0
    min_dist_px: float = 1.25
    min_branch_len: float = 10.0
    max_gap_px: float = 10.0
    axis_sample_points: int = 7
    axis_ring_poses: int = 16
    graspnet_repo: Path = Path("/home/lbw/3dgen-project/GraspNet-PointNet2-Pytorch-General-Upgrade")
    graspnet_checkpoint: Path = Path("/home/lbw/3dgen-project/GraspNet-PointNet2-Pytorch-General-Upgrade/logs/log_kn/checkpoint.tar")
    graspnet_num_point: int = 20000
    graspnet_max_poses: int = 50
    graspnet_collision_thresh: float = 0.01
    graspnet_voxel_size: float = 0.01
    graspnet_open3d_vis: bool = False
    render_only: bool = False


@dataclass
class RunConfig:
    input_dir: Path
    output_dir: Path
    pattern: str = "*.glb"
    resume: bool = False
    workers: int = 1
    limit: int | None = None
    run_id: str | None = None


@dataclass
class ObjectLayout:
    object_dir: Path
    renders_dir: Path
    debug_dir: Path
    review_dir: Path


def _now_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _create_layout(run_dir: Path, object_id: str) -> ObjectLayout:
    object_dir = run_dir / object_id
    renders_dir = object_dir / "renders"
    debug_dir = object_dir / "debug"
    review_dir = object_dir / "review"
    renders_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    return ObjectLayout(object_dir=object_dir, renders_dir=renders_dir, debug_dir=debug_dir, review_dir=review_dir)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _select_candidate(candidates: list[dict], pick: dict) -> tuple[dict, dict]:
    selected_id = pick.get("selected_candidate_id")
    selected = None
    if selected_id is not None:
        for candidate in candidates:
            if int(candidate["id"]) == int(selected_id):
                selected = candidate
                break
    if selected is None:
        selected = max(candidates, key=lambda item: float(item.get("score", 0.0)))
        pick["selected_candidate_id"] = int(selected["id"])
        pick["fallback_used"] = True
    return selected, pick


def _select_axis_sample(samples: list[dict], pick: dict) -> tuple[dict, dict]:
    selected_id = pick.get("selected_point_id")
    selected = None
    if selected_id is not None:
        for sample in samples:
            if int(sample["point_id"]) == int(selected_id):
                selected = sample
                break
    if selected is None:
        selected = samples[len(samples) // 2]
        pick["selected_point_id"] = int(selected["point_id"])
        pick["fallback_used"] = True
    return selected, pick


def _width_range(center_width: float, *, spread_ratio: float = 0.15, min_width: float = 0.01) -> list[float]:
    base = max(float(min_width), float(center_width))
    delta = max(float(min_width) * 0.25, base * float(spread_ratio))
    return [float(max(min_width, base - delta)), float(base + delta)]


def _trimmed_axis_interval(projections: np.ndarray, trim_ratio: float = 0.15) -> tuple[float, float]:
    if projections.size == 0:
        return -0.05, 0.05
    t_min = float(np.min(projections))
    t_max = float(np.max(projections))
    if t_max <= t_min:
        return t_min - 0.05, t_max + 0.05
    span = t_max - t_min
    lo = t_min + float(trim_ratio) * span
    hi = t_max - float(trim_ratio) * span
    if hi <= lo:
        return t_min, t_max
    return lo, hi


def _run_handle_branch(
    glb_path: Path,
    layout: ObjectLayout,
    rendered: dict,
    classification: dict,
    config: PipelineConfig,
) -> tuple[str | None, dict]:
    skeleton_view = extract_skeleton_view(classification)
    view_image_path = Path(rendered[skeleton_view]["image_path"])
    view_meta = _read_json(Path(rendered[skeleton_view]["meta_path"]))
    image = Image.open(view_image_path).convert("RGB")
    rgb = np.array(image)

    mask, skeleton_map, candidates = analyze_grasp_image(
        rgb,
        max_candidates=config.max_candidates,
        min_width_px=config.min_width_px,
        max_width_px=config.max_width_px,
        min_dist_px=config.min_dist_px,
        min_branch_len=config.min_branch_len,
        max_gap_px=config.max_gap_px,
    )
    stem_prefix = f"{glb_path.stem}_{skeleton_view}"
    artifacts = save_grasp_candidate_artifacts(
        image,
        mask,
        skeleton_map,
        candidates,
        layout.debug_dir / stem_prefix,
        review_prefix=layout.review_dir / stem_prefix,
    )
    if not candidates:
        return "no_grasp_candidates", {
            "skeleton_view": skeleton_view,
            "candidate_overlay": str(artifacts["candidates_on_original_path"]),
            "candidate_json": str(artifacts["candidates_json_path"]),
        }

    primitives = classification.get("primitives", {})
    handle_primitive = primitives.get("handle_tool_object", {}) if isinstance(primitives, dict) else {}
    skeleton_graph = handle_primitive.get("skeleton_graph", "")
    candidate_dicts = [_candidate_to_dict(c) for c in candidates]
    pick = pick_handle_candidate(
        model=config.model_pick,
        object_name=glb_path.stem,
        skeleton_view=skeleton_view,
        skeleton_graph=skeleton_graph,
        candidates=candidate_dicts,
        candidate_overlay_path=artifacts["candidates_on_original_path"],
    )
    selected, pick = _select_candidate(candidate_dicts, pick)

    cx, cy = selected["center_xy"]
    center_world = pixel_to_world(view_meta, float(cx), float(cy))
    approach_axis_world = normalize(np.array(view_meta["look_dir_world"], dtype=float))
    if np.linalg.norm(approach_axis_world) < 1e-8:
        approach_axis_world = np.array([0.0, 0.0, -1.0], dtype=float)

    orientation_pick = None
    close_axis_world = image_axis_to_world(view_meta, tuple(selected["gripper_axis_xy"]))
    tangent_world = normalize(image_axis_to_world(view_meta, tuple(selected.get("tangent_xy", (1.0, 0.0)))))
    if np.linalg.norm(tangent_world) < 1e-8:
        tangent_world = normalize(np.cross(close_axis_world, approach_axis_world))
    try:
        orientation_pick = pick_handle_orientation(
            model=config.model_pick,
            object_name=glb_path.stem,
            skeleton_view=skeleton_view,
            selected_xy=(int(cx), int(cy)),
            axes_image_path=Path(rendered[skeleton_view]["axes_image_path"]),
            candidate_overlay_path=artifacts["candidates_on_original_path"],
        )
        direction_label = str(orientation_pick.get("direction_label", "X"))
        direction_world = direction_label_to_vec(direction_label)
        direction_world = direction_world - approach_axis_world * float(np.dot(direction_world, approach_axis_world))
        direction_world = normalize(direction_world)
        if np.linalg.norm(direction_world) >= 1e-8:
            parallel_score = abs(float(np.dot(direction_world, tangent_world)))
            if parallel_score > 0.55:
                corrected = normalize(np.cross(approach_axis_world, tangent_world))
                if np.linalg.norm(corrected) >= 1e-8:
                    direction_world = corrected
                    orientation_pick["corrected_by_rule"] = True
                    orientation_pick["correction_reason"] = "direction too parallel to local handle tangent"
                    orientation_pick["parallel_score_before"] = parallel_score
            close_axis_world = direction_world
        else:
            orientation_pick["fallback_used"] = True
            orientation_pick["fallback_reason"] = "chosen direction parallel to approach axis"
    except Exception as exc:
        orientation_pick = {"error": str(exc), "fallback_used": True}

    wppu = float(view_meta["world_per_pixel_u"])
    wppv = float(view_meta["world_per_pixel_v"])
    axis_xy = np.array(selected["gripper_axis_xy"], dtype=float)
    per_px_world = np.linalg.norm(np.array([axis_xy[0] * wppu, axis_xy[1] * wppv], dtype=float))
    opening_width_world = float(selected["width_px"]) * float(per_px_world)

    final_effect_path = layout.review_dir / f"{glb_path.stem}_{skeleton_view}_final_grasp_effect.png"
    save_final_effect_image(view_image_path, selected, final_effect_path)

    handle_rotation = axes_to_rotation_matrix(approach_axis_world, close_axis_world)
    handle_depth = max(0.02, min(0.08, opening_width_world * 0.6))
    handle_grasp_pose = {
        "pose_id": 0,
        "score": float(selected.get("score", 1.0)),
        "width": float(max(0.01, opening_width_world)),
        "height": 0.02,
        "depth": float(handle_depth),
        "rotation_matrix": handle_rotation.tolist(),
        "translation": center_world.tolist(),
        "object_id": -1,
    }
    handle_slide_half_extent = max(0.01, min(0.08, opening_width_world * 0.75))
    handle_grasp_primitive = {
        "type": "axis_band",
        "point_local": center_world.tolist(),
        "axis_local": tangent_world.tolist(),
        "slide_range": [-float(handle_slide_half_extent), float(handle_slide_half_extent)],
        "approach_dirs_local": [approach_axis_world.tolist()],
        "closing_dirs_local": [close_axis_world.tolist()],
        "width_range": _width_range(opening_width_world),
        "depth_range": [float(max(0.02, handle_depth * 0.8)), float(max(0.02, handle_depth * 1.2))],
        "score": float(selected.get("score", 1.0)),
        "source_branch": "handle_tool_object",
        "selection": {
            "view_name": view_meta.get("view_name", skeleton_view),
            "candidate_id": int(selected["id"]),
        },
    }
    handle_open3d_path = layout.review_dir / f"{glb_path.stem}_{skeleton_view}_handle_open3d.png"
    handle_open3d_ok = False
    handle_open3d_error = None
    try:
        vis_cloud = sample_cloud_from_glb(glb_path, num_points=max(12000, config.graspnet_num_point // 2))
        handle_open3d_ok = save_open3d_grippers_screenshot(
            repo_root=config.graspnet_repo,
            cloud_points=vis_cloud,
            grasp_poses=[handle_grasp_pose],
            out_path=handle_open3d_path,
            show_window=config.graspnet_open3d_vis,
            selected_point_world=center_world,
        )
    except Exception as exc:
        handle_open3d_error = str(exc)

    stage = {
        "skeleton_view": skeleton_view,
        "skeleton_view_image": str(view_image_path),
        "candidate_overlay": str(artifacts["candidates_on_original_path"]),
        "candidate_json": str(artifacts["candidates_json_path"]),
        "final_effect_image": str(final_effect_path),
        "open3d_effect_image": str(handle_open3d_path) if handle_open3d_ok else None,
        "open3d_error": handle_open3d_error,
        "candidate_count": len(candidate_dicts),
        "gpt_pick": pick,
        "orientation_pick": orientation_pick,
        "selected_candidate": selected,
        "grasp_pose_world": {
            "position": center_world.tolist(),
            "approach_axis": approach_axis_world.tolist(),
            "closing_axis": close_axis_world.tolist(),
            "opening_width_world": opening_width_world,
            "view_name": view_meta.get("view_name", skeleton_view),
        },
        "grasp_primitives": [handle_grasp_primitive],
    }
    return None, stage


def _run_axis_branch(
    glb_path: Path,
    layout: ObjectLayout,
    rendered: dict,
    classification: dict,
    config: PipelineConfig,
) -> tuple[str | None, dict]:
    primitives = classification.get("primitives", {})
    axis_primitive = primitives.get("axis_object", {}) if isinstance(primitives, dict) else {}
    axis_label = str(axis_primitive.get("object_axis", "Y")).upper()
    axis_unit = normalize(axis_label_to_vec(axis_label))

    center_world = np.array(axis_primitive.get("center_point", [0.0, 0.0, 0.0]), dtype=float)
    if center_world.shape != (3,):
        center_world = np.array([0.0, 0.0, 0.0], dtype=float)

    points_world = load_scene_points_world(glb_path)
    axis_view = choose_best_axis_view(rendered, axis_unit)
    view_image_path = Path(rendered[axis_view]["image_path"])
    view_meta = _read_json(Path(rendered[axis_view]["meta_path"]))

    samples = make_axis_samples(
        axis_vec_world=axis_unit,
        center_world=center_world,
        points_world=points_world,
        num_points=config.axis_sample_points,
    )
    sample_overlay_path = layout.review_dir / f"{glb_path.stem}_{axis_view}_axis_samples.png"
    projected_samples = draw_axis_sample_points(view_image_path, view_meta, samples, sample_overlay_path)
    sample_json_path = layout.debug_dir / f"{glb_path.stem}_{axis_view}_axis_samples.json"
    _write_json(sample_json_path, {"samples": projected_samples})

    pick = pick_axis_sample(
        model=config.model_pick,
        object_name=glb_path.stem,
        axis_label=axis_label,
        projected_samples=projected_samples,
        sample_overlay_path=sample_overlay_path,
    )
    selected, pick = _select_axis_sample(samples, pick)

    selected_world = np.array(selected["world_xyz"], dtype=float)
    rel = points_world - center_world[None, :]
    lo_t, hi_t = _trimmed_axis_interval(rel @ axis_unit)
    selected_t = float(selected.get("t_on_axis", 0.0))
    radius = estimate_axis_radius(points_world=points_world, center_world=center_world, axis_unit=axis_unit) * 1.05
    ring_poses = make_ring_grasp_poses(selected_world, axis_unit, radius, config.axis_ring_poses)
    axis_grasp_primitive = {
        "type": "axis_band",
        "point_local": selected_world.tolist(),
        "axis_local": axis_unit.tolist(),
        "slide_range": [float(lo_t - selected_t), float(hi_t - selected_t)],
        "radial_symmetry": "full",
        "width_range": _width_range(radius * 2.0, spread_ratio=0.20),
        "depth_range": [float(max(0.02, radius * 0.6)), float(max(0.02, radius * 1.0))],
        "score": 1.0,
        "source_branch": "axis_object",
        "selection": {
            "axis_label": axis_label,
            "selected_point_id": int(selected["point_id"]),
        },
    }

    final_effect_path = layout.review_dir / f"{glb_path.stem}_{axis_view}_final_grasp_effect.png"
    save_axis_effect_image(view_image_path, view_meta, selected_world, axis_unit, radius, final_effect_path)

    axis_grasp_poses = []
    for i, ring_pose in enumerate(ring_poses):
        approach = np.array(ring_pose["approach_axis"], dtype=float)
        closing = np.array(ring_pose["closing_axis"], dtype=float)
        rotation = axes_to_rotation_matrix(approach, closing)
        axis_grasp_poses.append(
            {
                "pose_id": int(i),
                "score": 1.0 - (float(i) / max(1.0, float(len(ring_poses)))),
                "width": float(max(0.01, radius * 2.0)),
                "height": 0.02,
                "depth": float(max(0.02, radius * 0.8)),
                "rotation_matrix": rotation.tolist(),
                "translation": ring_pose["position"],
                "object_id": -1,
            }
        )

    axis_open3d_path = layout.review_dir / f"{glb_path.stem}_{axis_view}_axis_open3d.png"
    axis_open3d_ok = False
    axis_open3d_error = None
    try:
        vis_cloud = sample_cloud_from_glb(glb_path, num_points=max(12000, config.graspnet_num_point // 2))
        axis_open3d_ok = save_open3d_grippers_screenshot(
            repo_root=config.graspnet_repo,
            cloud_points=vis_cloud,
            grasp_poses=axis_grasp_poses,
            out_path=axis_open3d_path,
            show_window=config.graspnet_open3d_vis,
            selected_point_world=selected_world,
        )
    except Exception as exc:
        axis_open3d_error = str(exc)

    stage = {
        "axis_label": axis_label,
        "axis_view": axis_view,
        "axis_samples_overlay": str(sample_overlay_path),
        "axis_samples_json": str(sample_json_path),
        "gpt_pick": pick,
        "selected_axis_point": selected,
        "ring_radius_world": radius,
        "ring_grasp_poses_world": ring_poses,
        "grasp_primitives": [axis_grasp_primitive],
        "final_effect_image": str(final_effect_path),
        "open3d_effect_image": str(axis_open3d_path) if axis_open3d_ok else None,
        "open3d_error": axis_open3d_error,
    }
    return None, stage


def _run_graspnet_branch(glb_path: Path, layout: ObjectLayout, config: PipelineConfig) -> tuple[str | None, dict]:
    if not config.graspnet_repo.exists():
        return "graspnet_repo_not_found", {"repo": str(config.graspnet_repo)}
    if not config.graspnet_checkpoint.exists():
        return "graspnet_checkpoint_not_found", {"checkpoint": str(config.graspnet_checkpoint)}

    try:
        grasp_poses, sampled_cloud = run_graspnet_inference(
            glb_path=glb_path,
            repo_root=config.graspnet_repo,
            checkpoint_path=config.graspnet_checkpoint,
            num_points=config.graspnet_num_point,
            max_poses=config.graspnet_max_poses,
            collision_thresh=config.graspnet_collision_thresh,
            voxel_size=config.graspnet_voxel_size,
        )
        final_effect_path = layout.review_dir / f"{glb_path.stem}_graspnet_open3d.png"
        screenshot_ok = save_graspnet_open3d_screenshot(
            repo_root=config.graspnet_repo,
            cloud_points=sampled_cloud,
            grasp_poses=grasp_poses,
            out_path=final_effect_path,
            show_window=config.graspnet_open3d_vis,
        )
        pose_set_primitive = {
            "type": "pose_set",
            "poses_local": grasp_poses,
            "score": float(max((pose.get("score", 0.0) for pose in grasp_poses), default=0.0)),
            "source_branch": "graspnet_object",
        }
        stage = {
            "graspnet_repo": str(config.graspnet_repo),
            "graspnet_checkpoint": str(config.graspnet_checkpoint),
            "num_output_poses": len(grasp_poses),
            "grasp_poses_world": grasp_poses,
            "grasp_primitives": [pose_set_primitive],
            "open3d_visualization": {
                "interactive_requested": bool(config.graspnet_open3d_vis),
                "screenshot_saved": bool(screenshot_ok),
            },
            "final_effect_image": str(final_effect_path) if screenshot_ok else None,
        }
        return None, stage
    except Exception as exc:
        return "graspnet_inference_failed", {
            "graspnet_repo": str(config.graspnet_repo),
            "graspnet_checkpoint": str(config.graspnet_checkpoint),
            "error": str(exc),
        }


def _candidate_to_dict(candidate) -> dict:
    if isinstance(candidate, dict):
        return candidate
    return {
        "id": candidate.id,
        "center_xy": list(candidate.center_xy),
        "gripper_axis_xy": list(candidate.gripper_axis_xy),
        "tangent_xy": list(candidate.tangent_xy),
        "width_px": candidate.width_px,
        "score": candidate.score,
        "branch_id": candidate.branch_id,
        "candidate_type": candidate.candidate_type,
    }


def annotate_single_object(
    glb_path: Path,
    run_dir: Path,
    config: PipelineConfig,
    *,
    resume: bool = False,
) -> dict:
    if glb_path.suffix.lower() != ".glb":
        raise ValueError(f"Expected a .glb file, got: {glb_path}")

    layout = _create_layout(run_dir, glb_path.stem)
    pipeline_result_path = layout.object_dir / "pipeline_result.json"
    annotation_path = layout.object_dir / "annotation.json"
    if resume and pipeline_result_path.exists() and annotation_path.exists():
        return {
            "object_name": glb_path.stem,
            "status": "skipped_existing",
            "object_dir": str(layout.object_dir),
            "pipeline_result": str(pipeline_result_path),
            "annotation": str(annotation_path),
        }

    rendered = render_three_views(glb_path, layout.renders_dir, layout.debug_dir, config.size, config.fill_ratio)
    classification = classify_object(config.model_classify, glb_path.stem, rendered)
    classification_path = layout.debug_dir / "classification.json"
    _write_json(classification_path, classification)

    result = {
        "object_name": glb_path.stem,
        "classification": classification,
        "classification_json": str(classification_path),
    }

    if not config.render_only:
        category = classification.get("category")
        status = None
        if category == "handle_tool_object":
            status, stage = _run_handle_branch(glb_path, layout, rendered, classification, config)
            result["handle_tool_stage"] = stage
        elif category == "axis_object":
            status, stage = _run_axis_branch(glb_path, layout, rendered, classification, config)
            result["axis_object_stage"] = stage
        elif category == "graspnet_object":
            status, stage = _run_graspnet_branch(glb_path, layout, config)
            result["graspnet_object_stage"] = stage
        else:
            status = "skip_unsupported_category"
        if status:
            result["status"] = status

    _write_json(pipeline_result_path, result)
    annotation = build_annotation(result, glb_path, layout.object_dir)
    _write_json(annotation_path, annotation)
    return {
        "object_name": glb_path.stem,
        "status": result.get("status", "ok"),
        "object_dir": str(layout.object_dir),
        "pipeline_result": str(pipeline_result_path),
        "annotation": str(annotation_path),
    }


def annotate_dataset(run_config: RunConfig, pipeline_config: PipelineConfig) -> dict:
    run_id = run_config.run_id or _now_run_id()
    run_dir = run_config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    glb_files = sorted(run_config.input_dir.glob(run_config.pattern))
    if run_config.limit is not None:
        glb_files = glb_files[: max(0, int(run_config.limit))]

    results = []
    if run_config.workers <= 1:
        for glb_path in glb_files:
            results.append(annotate_single_object(glb_path, run_dir, pipeline_config, resume=run_config.resume))
    else:
        with ThreadPoolExecutor(max_workers=run_config.workers) as executor:
            futures = {
                executor.submit(annotate_single_object, glb_path, run_dir, pipeline_config, resume=run_config.resume): glb_path
                for glb_path in glb_files
            }
            for future in as_completed(futures):
                glb_path = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        {
                            "object_name": glb_path.stem,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

    manifest = {
        "run_id": run_id,
        "input_dir": str(run_config.input_dir),
        "output_dir": str(run_config.output_dir),
        "pattern": run_config.pattern,
        "resume": run_config.resume,
        "workers": run_config.workers,
        "limit": run_config.limit,
        "pipeline_config": {
            **asdict(pipeline_config),
            "graspnet_repo": str(pipeline_config.graspnet_repo),
            "graspnet_checkpoint": str(pipeline_config.graspnet_checkpoint),
        },
        "objects": results,
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return manifest
