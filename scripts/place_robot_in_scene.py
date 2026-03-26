from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.services.robot_placement import (
    DEFAULT_OUTPUT_DIR,
    load_scene_state,
    plan_robot_base_pose,
    save_plan_outputs,
)
from app.backend.services.robot_scene import (
    DEFAULT_ROBOT_SCENE_META_PATH,
    DEFAULT_ROBOT_SCENE_PATH,
    DEFAULT_SCENE_USD_PATH,
    embed_robot_in_scene_usd,
    save_robot_scene_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed a robot asset into the current scene USD using the planned base pose.")
    parser.add_argument("--scene-graph", default=None, help="Optional scene graph JSON path.")
    parser.add_argument("--placements", default=None, help="Optional placements JSON path.")
    parser.add_argument("--scene-usd", default=str(DEFAULT_SCENE_USD_PATH), help="Source scene USD to augment.")
    parser.add_argument("--robot", default="agibot", choices=["kinova", "agibot", "r1lite"], help="Robot profile.")
    parser.add_argument("--target", default=None, help="Target object prim, class, or caption substring.")
    parser.add_argument("--support", default=None, help="Optional explicit support object prim.")
    parser.add_argument("--floor-z", type=float, default=0.0, help="World floor height for robot foot alignment.")
    parser.add_argument("--robot-prim-path", default="/World/RobotPlacement", help="Prim path for inserted robot.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for plan and scene outputs.")
    parser.add_argument("--output-usd", default=str(DEFAULT_ROBOT_SCENE_PATH), help="Output USD path.")
    parser.add_argument("--output-meta", default=str(DEFAULT_ROBOT_SCENE_META_PATH), help="Output metadata JSON path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.scene_graph or args.placements:
        scene_graph, placements = load_scene_state(
            args.scene_graph if args.scene_graph else None,
            args.placements if args.placements else None,
        )
    else:
        scene_graph, placements = load_scene_state()
    plan = plan_robot_base_pose(
        scene_graph,
        placements,
        target_prim=args.target,
        support_prim=args.support,
        robot=args.robot,
    )
    save_plan_outputs(scene_graph, placements, plan, output_dir=args.output_dir)
    result = embed_robot_in_scene_usd(
        args.scene_usd,
        plan,
        robot=args.robot,
        output_usd_path=args.output_usd,
        robot_prim_path=args.robot_prim_path,
        floor_z=args.floor_z,
    )
    meta_path = save_robot_scene_result(result, plan, output_meta_path=args.output_meta)
    print(f"robot={result.robot}")
    print(f"scene_input_usd={result.scene_input_usd}")
    print(f"scene_output_usd={result.scene_output_usd}")
    print(f"robot_prim_path={result.robot_prim_path}")
    print(f"robot_asset_path={result.robot_asset_path}")
    print(f"robot_floor_offset_z={result.robot_floor_offset_z:.4f}")
    print(
        "robot_base_pose="
        f"[{result.base_pose[0]:.3f}, {result.base_pose[1]:.3f}, {result.base_pose[2]:.3f}, {result.base_pose[3]:.2f}]"
    )
    print(f"meta={meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
