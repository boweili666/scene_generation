#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from app.backend.services.grasp_asset_cache import build_asset_grasp_cache
from grasp_annotator import PipelineConfig


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--fill-ratio", type=float, default=0.90)
    parser.add_argument("--model-classify", type=str, default="gpt-5")
    parser.add_argument("--model-pick", type=str, default="gpt-5")
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--min-width-px", type=float, default=12.0)
    parser.add_argument("--max-width-px", type=float, default=220.0)
    parser.add_argument("--min-dist-px", type=float, default=1.25)
    parser.add_argument("--min-branch-len", type=float, default=10.0)
    parser.add_argument("--max-gap-px", type=float, default=10.0)
    parser.add_argument("--axis-sample-points", type=int, default=7)
    parser.add_argument("--axis-ring-poses", type=int, default=16)
    parser.add_argument(
        "--graspnet-repo",
        type=Path,
        default=Path("/home/lbw/3dgen-project/GraspNet-PointNet2-Pytorch-General-Upgrade"),
    )
    parser.add_argument(
        "--graspnet-checkpoint",
        type=Path,
        default=Path("/home/lbw/3dgen-project/GraspNet-PointNet2-Pytorch-General-Upgrade/logs/log_kn/checkpoint.tar"),
    )
    parser.add_argument("--graspnet-num-point", type=int, default=20000)
    parser.add_argument("--graspnet-max-poses", type=int, default=50)
    parser.add_argument("--graspnet-collision-thresh", type=float, default=0.01)
    parser.add_argument("--graspnet-voxel-size", type=float, default=0.01)
    parser.add_argument("--graspnet-open3d-vis", action="store_true")
    parser.add_argument("--render-only", action="store_true")


def _pipeline_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        size=args.size,
        fill_ratio=args.fill_ratio,
        model_classify=args.model_classify,
        model_pick=args.model_pick,
        max_candidates=args.max_candidates,
        min_width_px=args.min_width_px,
        max_width_px=args.max_width_px,
        min_dist_px=args.min_dist_px,
        min_branch_len=args.min_branch_len,
        max_gap_px=args.max_gap_px,
        axis_sample_points=args.axis_sample_points,
        axis_ring_poses=args.axis_ring_poses,
        graspnet_repo=args.graspnet_repo,
        graspnet_checkpoint=args.graspnet_checkpoint,
        graspnet_num_point=args.graspnet_num_point,
        graspnet_max_poses=args.graspnet_max_poses,
        graspnet_collision_thresh=args.graspnet_collision_thresh,
        graspnet_voxel_size=args.graspnet_voxel_size,
        graspnet_open3d_vis=args.graspnet_open3d_vis,
        render_only=args.render_only,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline asset -> grasp_primitives cache from a Real2Sim manifest.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--prim-path", action="append", default=None, help="Limit to one or more prim paths.")
    parser.add_argument("--no-resume", action="store_true")
    _add_pipeline_args(parser)
    args = parser.parse_args()

    payload = build_asset_grasp_cache(
        args.manifest_path,
        output_root=args.output_root,
        pipeline_config=_pipeline_config_from_args(args),
        selected_prim_paths=args.prim_path,
        resume=not args.no_resume,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
