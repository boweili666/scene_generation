#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scene_robot.scripts.visualize_grasp_proposals_open3d import (
    _collect_visual_geometries,
    _expand_scene_grasp_candidates,
    _import_open3d,
    _load_json,
    _load_selected_candidate,
    _load_shortlist_candidates,
    _resolve_default_scene_grasp_proposals_path,
    _save_screenshot,
)


DEFAULT_SCENE_USD_PATH = (
    PROJECT_ROOT
    / "runtime"
    / "sessions"
    / "sess_37daed605d8c"
    / "runs"
    / "run_ab267fae7ae8"
    / "scene_service"
    / "usd"
    / "scene_latest.usd"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open3D debug viewer for bolt grasp poses.")
    parser.add_argument("--bolt", type=str, default="bolt_2", choices=["bolt_2", "bolt_3"])
    parser.add_argument("--scene-usd-path", type=Path, default=DEFAULT_SCENE_USD_PATH)
    parser.add_argument("--scene-grasp-proposals-path", type=Path, default=None)
    parser.add_argument("--selected-proposal-path", type=Path, default=None)
    parser.add_argument("--shortlist-path", type=Path, default=None)
    parser.add_argument("--selected-only", action="store_true")
    parser.add_argument("--max-grasps", type=int, default=24)
    parser.add_argument("--axis-band-slide-samples", type=int, default=3)
    parser.add_argument("--axis-band-ring-samples", type=int, default=12)
    parser.add_argument("--frame-size", type=float, default=0.03)
    parser.add_argument("--selected-frame-size", type=float, default=0.06)
    parser.add_argument("--approach-length", type=float, default=0.06)
    parser.add_argument("--screenshot-path", type=Path, default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    target_prim = f"/World/{args.bolt}"
    scene_grasp_path = _resolve_default_scene_grasp_proposals_path(args.scene_usd_path, args.scene_grasp_proposals_path)
    if not scene_grasp_path.exists():
        raise FileNotFoundError(f"Scene grasp proposals JSON not found: {scene_grasp_path}")

    scene_grasp_payload = _load_json(scene_grasp_path)
    objects = scene_grasp_payload.get("objects", {})
    if not isinstance(objects, dict) or target_prim not in objects:
        raise KeyError(f"Target '{target_prim}' not found in {scene_grasp_path}")

    object_payload = objects[target_prim]
    selected_candidate = _load_selected_candidate(args.selected_proposal_path, target_prim)
    all_candidates = [] if args.selected_only else _load_shortlist_candidates(args.shortlist_path, target_prim, args.max_grasps)
    if not args.selected_only and not all_candidates:
        all_candidates = _expand_scene_grasp_candidates(
            scene_grasp_payload,
            target_prim=target_prim,
            axis_band_slide_samples=args.axis_band_slide_samples,
            axis_band_ring_samples=args.axis_band_ring_samples,
            max_count=args.max_grasps,
        )
    if selected_candidate is None and args.selected_only:
        expanded = _expand_scene_grasp_candidates(
            scene_grasp_payload,
            target_prim=target_prim,
            axis_band_slide_samples=args.axis_band_slide_samples,
            axis_band_ring_samples=args.axis_band_ring_samples,
            max_count=1,
        )
        if expanded:
            selected_candidate = expanded[0]
    if selected_candidate is None and all_candidates:
        selected_candidate = all_candidates[0]

    geoms = _collect_visual_geometries(
        object_payload,
        all_candidates=all_candidates,
        selected_candidate=selected_candidate,
        frame_size=args.frame_size,
        selected_frame_size=args.selected_frame_size,
        approach_length=args.approach_length,
    )

    print(f"[INFO] Target: {target_prim}")
    print(f"[INFO] Scene grasp proposals: {scene_grasp_path}")
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
            window_name=f"Bolt Grasp Debug: {args.bolt}",
            width=1440,
            height=960,
        )


if __name__ == "__main__":
    main()
