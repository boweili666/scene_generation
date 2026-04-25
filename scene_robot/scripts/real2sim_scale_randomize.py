import argparse
from pathlib import Path
import sys
import termios
import tty


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from scene_robot_apps.pipelines.real2sim_scale_randomization import (
    DEFAULT_MAX_SCALE,
    DEFAULT_MIN_SCALE,
    randomize_real2sim_asset_scales,
    visualize_real2sim_scale_randomization,
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


def _parse_scale_overrides(items: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --scale-override '{item}'. Expected format: /World/object=1.12")
        prim_path, scale_text = item.split("=", 1)
        prim_path = prim_path.strip()
        if not prim_path:
            raise ValueError(f"Invalid --scale-override '{item}'. Prim path is empty.")
        overrides[prim_path] = float(scale_text)
    return overrides


def _run_once(
    *,
    scene_usd_path: Path,
    scene_graph_path: str | None,
    manifest_path: str | None,
    output_usd_path: Path,
    output_metadata_path: Path,
    viz_output_path: Path,
    min_scale: float,
    max_scale: float,
    seed: int,
    shared_scale: bool,
    global_scale: float | None,
    scale_overrides: dict[str, float],
    skip_visualization: bool,
) -> None:
    result = randomize_real2sim_asset_scales(
        scene_usd_path,
        output_usd_path=output_usd_path,
        output_metadata_path=output_metadata_path,
        scene_graph_path=scene_graph_path,
        manifest_path=manifest_path,
        min_scale=float(min_scale),
        max_scale=float(max_scale),
        seed=int(seed),
        shared_scale=bool(shared_scale),
        global_scale=float(global_scale) if global_scale is not None else None,
        scale_overrides=scale_overrides,
    )

    print(f"seed={seed}")
    print(f"scale_mode={result['scale_mode']}")
    if result.get("shared_scale") is not None:
        print(f"shared_scale={float(result['shared_scale']):.6f}")
    print(f"scene_input_usd={result['scene_input_usd']}")
    print(f"scene_output_usd={result['scene_output_usd']}")
    print(f"metadata_path={result['metadata_path']}")
    print(f"object_count={result['object_count']}")
    print(f"root_layout_scale={result['root_layout_scale']:.6f}")
    for prim_path, scale in sorted(result["scales"].items()):
        print(f"scale[{prim_path}]={float(scale):.6f}")

    if not skip_visualization:
        viz_result = visualize_real2sim_scale_randomization(
            scene_usd_path,
            output_usd_path,
            output_image_path=viz_output_path,
            scene_graph_path=scene_graph_path,
            manifest_path=manifest_path,
            metadata_path=output_metadata_path,
        )
        print(f"comparison_png={viz_result['output_image_path']}")


def _read_single_key() -> str:
    if not sys.stdin.isatty():
        raise RuntimeError("--interactive requires a TTY terminal.")

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply scale randomization to real2sim assets inside a scene_service scene USD and adjust relative positions."
    )
    parser.add_argument("--scene-usd-path", type=str, default=str(DEFAULT_SCENE_USD_PATH), help="Input scene USD path.")
    parser.add_argument("--scene-graph-path", type=str, default=None, help="Optional current_scene_graph.json path.")
    parser.add_argument("--manifest-path", type=str, default=None, help="Optional real2sim_asset_manifest.json path.")
    parser.add_argument("--output-usd-path", type=str, default=None, help="Randomized output USD path.")
    parser.add_argument("--output-metadata-path", type=str, default=None, help="Debug JSON output path.")
    parser.add_argument("--viz-output-path", type=str, default=None, help="Optional comparison PNG path.")
    parser.add_argument("--min-scale", type=float, default=DEFAULT_MIN_SCALE, help="Minimum sampled isotropic scale.")
    parser.add_argument("--max-scale", type=float, default=DEFAULT_MAX_SCALE, help="Maximum sampled isotropic scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for scale sampling.")
    parser.add_argument(
        "--shared-scale",
        action="store_true",
        help="Sample one scale from [min-scale, max-scale] and apply it to every real2sim object.",
    )
    parser.add_argument(
        "--global-scale",
        type=float,
        default=None,
        help="Apply one fixed global scale to every real2sim object.",
    )
    parser.add_argument("--interactive", action="store_true", help="Press r or Enter to resample in place; press q to quit.")
    parser.add_argument("--seed-step", type=int, default=1, help="Seed increment used between interactive resamples.")
    parser.add_argument(
        "--scale-override",
        action="append",
        default=[],
        help="Optional explicit override like /World/table_0=1.12. Repeatable.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Do not emit the before/after comparison PNG.",
    )
    args = parser.parse_args()

    scene_usd_path = Path(args.scene_usd_path).resolve()
    output_usd_path = (
        Path(args.output_usd_path).resolve()
        if args.output_usd_path
        else scene_usd_path.with_name(f"{scene_usd_path.stem}.scale_randomized{scene_usd_path.suffix}")
    )
    output_metadata_path = (
        Path(args.output_metadata_path).resolve()
        if args.output_metadata_path
        else output_usd_path.with_suffix(".randomization.json")
    )
    viz_output_path = (
        Path(args.viz_output_path).resolve()
        if args.viz_output_path
        else output_usd_path.with_suffix(".comparison.png")
    )
    scale_overrides = _parse_scale_overrides(args.scale_override)

    current_seed = int(args.seed)
    _run_once(
        scene_usd_path=scene_usd_path,
        scene_graph_path=args.scene_graph_path,
        manifest_path=args.manifest_path,
        output_usd_path=output_usd_path,
        output_metadata_path=output_metadata_path,
        viz_output_path=viz_output_path,
        min_scale=float(args.min_scale),
        max_scale=float(args.max_scale),
        seed=current_seed,
        shared_scale=bool(args.shared_scale),
        global_scale=float(args.global_scale) if args.global_scale is not None else None,
        scale_overrides=scale_overrides,
        skip_visualization=bool(args.skip_visualization),
    )

    if not args.interactive:
        return

    print("interactive_mode=1")
    print("hotkeys: [r|Enter]=new randomization, [q]=quit")
    while True:
        key = _read_single_key()
        if key in {"q", "Q"}:
            print("\ninteractive_mode=exit")
            return
        if key in {"r", "R", "\r", "\n"}:
            current_seed += int(args.seed_step)
            print()
            _run_once(
                scene_usd_path=scene_usd_path,
                scene_graph_path=args.scene_graph_path,
                manifest_path=args.manifest_path,
                output_usd_path=output_usd_path,
                output_metadata_path=output_metadata_path,
                viz_output_path=viz_output_path,
                min_scale=float(args.min_scale),
                max_scale=float(args.max_scale),
                seed=current_seed,
                shared_scale=bool(args.shared_scale),
                global_scale=float(args.global_scale) if args.global_scale is not None else None,
                scale_overrides=scale_overrides,
                skip_visualization=bool(args.skip_visualization),
            )


if __name__ == "__main__":
    main()
