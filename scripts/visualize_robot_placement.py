from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.config.settings import DEFAULT_PLACEMENTS_PATH, SCENE_GRAPH_PATH
from app.backend.services.robot_placement import (
    DEFAULT_OUTPUT_DIR,
    load_scene_state,
    plan_robot_base_pose,
    plan_to_payload,
    save_plan_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan and visualize a robot base pose for tabletop manipulation.")
    parser.add_argument("--scene-graph", default=str(SCENE_GRAPH_PATH), help="Path to scene graph JSON.")
    parser.add_argument("--placements", default=str(DEFAULT_PLACEMENTS_PATH), help="Path to placements JSON.")
    parser.add_argument("--target", default=None, help="Target object prim, class, or caption substring.")
    parser.add_argument("--support", default=None, help="Optional explicit support object prim.")
    parser.add_argument("--robot", default="agibot", choices=["kinova", "agibot", "r1lite"], help="Robot profile.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON and SVG outputs.")
    parser.add_argument("--print-json", action="store_true", help="Print the computed plan JSON to stdout.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scene_graph, placements = load_scene_state(args.scene_graph, args.placements)
    plan = plan_robot_base_pose(
        scene_graph,
        placements,
        target_prim=args.target,
        support_prim=args.support,
        robot=args.robot,
    )
    outputs = save_plan_outputs(scene_graph, placements, plan, output_dir=args.output_dir)
    print(f"robot={plan.robot}")
    print(f"target={plan.target_prim}")
    print(f"support={plan.support_prim}")
    print(f"chosen_side={plan.chosen_side}")
    print(
        "base_pose="
        f"[{plan.base_pose[0]:.3f}, {plan.base_pose[1]:.3f}, {plan.base_pose[2]:.3f}, {plan.base_pose[3]:.2f}]"
    )
    print(f"json={outputs['json']}")
    print(f"svg={outputs['svg']}")
    if args.print_json:
        print(json.dumps(plan_to_payload(plan), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
