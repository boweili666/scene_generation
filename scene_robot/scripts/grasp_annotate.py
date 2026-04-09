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

from grasp_annotator.batch_runner import PipelineConfig, RunConfig, annotate_dataset, annotate_single_object


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=768, help="Render image size")
    parser.add_argument("--fill-ratio", type=float, default=0.90, help="Object occupancy ratio")
    parser.add_argument("--model-classify", type=str, default="gpt-5", help="OpenAI model for classification")
    parser.add_argument("--model-pick", type=str, default="gpt-5", help="OpenAI model for candidate picking")
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
    parser = argparse.ArgumentParser(description="Automated grasp annotation CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser("annotate-dataset", help="Annotate a directory of GLB files")
    dataset_parser.add_argument("--input-dir", type=Path, required=True)
    dataset_parser.add_argument("--output-dir", type=Path, required=True)
    dataset_parser.add_argument("--glob", dest="pattern", type=str, default="*.glb")
    dataset_parser.add_argument("--resume", action="store_true")
    dataset_parser.add_argument("--workers", type=int, default=1)
    dataset_parser.add_argument("--limit", type=int, default=None)
    dataset_parser.add_argument("--run-id", type=str, default=None)
    _add_pipeline_args(dataset_parser)

    one_parser = subparsers.add_parser("annotate-one", help="Annotate a single GLB file")
    one_parser.add_argument("--file", type=Path, required=True)
    one_parser.add_argument("--output-dir", type=Path, required=True)
    one_parser.add_argument("--run-id", type=str, default=None)
    one_parser.add_argument("--resume", action="store_true")
    _add_pipeline_args(one_parser)

    args = parser.parse_args()
    pipeline_config = _pipeline_config_from_args(args)

    if args.command == "annotate-dataset":
        manifest = annotate_dataset(
            RunConfig(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                pattern=args.pattern,
                resume=args.resume,
                workers=args.workers,
                limit=args.limit,
                run_id=args.run_id,
            ),
            pipeline_config,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    run_id = args.run_id or "single_run"
    run_dir = args.output_dir / run_id
    result = annotate_single_object(args.file, run_dir, pipeline_config, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
