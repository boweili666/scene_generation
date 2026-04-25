import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from scene_robot_apps.pipelines.real2sim_scale_randomization import visualize_real2sim_scale_randomization


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize before/after real2sim scale randomization as a PNG.")
    parser.add_argument("--before-scene-usd-path", type=str, required=True, help="Original scene USD path.")
    parser.add_argument("--after-scene-usd-path", type=str, required=True, help="Randomized scene USD path.")
    parser.add_argument("--scene-graph-path", type=str, default=None, help="Optional current_scene_graph.json path.")
    parser.add_argument("--manifest-path", type=str, default=None, help="Optional real2sim_asset_manifest.json path.")
    parser.add_argument("--metadata-path", type=str, default=None, help="Optional randomization debug JSON path.")
    parser.add_argument("--output-image-path", type=str, required=True, help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    args = parser.parse_args()

    result = visualize_real2sim_scale_randomization(
        args.before_scene_usd_path,
        args.after_scene_usd_path,
        output_image_path=Path(args.output_image_path).resolve(),
        scene_graph_path=args.scene_graph_path,
        manifest_path=args.manifest_path,
        metadata_path=args.metadata_path,
        dpi=int(args.dpi),
    )
    print(f"before_scene_usd={result['before_scene_usd']}")
    print(f"after_scene_usd={result['after_scene_usd']}")
    print(f"output_image_path={result['output_image_path']}")
    print(f"object_count={result['object_count']}")


if __name__ == "__main__":
    main()
