import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from scene_robot_apps.robot_workspaces import save_robot_workspace_overview


DEFAULT_OUTPUT_IMAGE_PATH = PROJECT_ROOT / "runtime" / "robot_placement" / "robot_workspaces.png"
DEFAULT_OUTPUT_JSON_PATH = PROJECT_ROOT / "runtime" / "robot_placement" / "robot_workspaces.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 2D working areas for Kinova, Agibot, and R1Lite.")
    parser.add_argument(
        "--robot",
        action="append",
        default=[],
        choices=["kinova", "agibot", "r1lite"],
        help="Optional robot filter. Repeat to draw multiple robots.",
    )
    parser.add_argument("--output-image-path", type=str, default=str(DEFAULT_OUTPUT_IMAGE_PATH), help="Output PNG path.")
    parser.add_argument("--output-json-path", type=str, default=str(DEFAULT_OUTPUT_JSON_PATH), help="Output JSON path.")
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    args = parser.parse_args()

    selected = tuple(args.robot) if args.robot else None
    result = save_robot_workspace_overview(
        output_image_path=Path(args.output_image_path).resolve(),
        output_json_path=Path(args.output_json_path).resolve() if args.output_json_path else None,
        robots=selected,
        dpi=int(args.dpi),
    )
    print(f"output_image_path={result['output_image_path']}")
    if "output_json_path" in result:
        print(f"output_json_path={result['output_json_path']}")
    print(f"robots={','.join(robot['robot'] for robot in result['robots'])}")


if __name__ == "__main__":
    main()
