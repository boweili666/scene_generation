import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from app.backend.config.settings import DEFAULT_PLACEMENTS_PATH, SCENE_GRAPH_PATH


DEFAULT_SCENE_USD_PATH = PROJECT_ROOT / "runtime" / "scene_service" / "usd" / "scene_latest.usd"


def _maybe_reexec_with_conda_libstdcpp() -> None:
    if os.environ.get("SCENE_AUTO_COLLECT_LD_READY") == "1":
        return

    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    conda_lib = Path(conda_prefix) / "lib"
    libstdcpp = conda_lib / "libstdc++.so.6"
    if not libstdcpp.exists():
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in existing.split(":") if part]
    conda_lib_str = str(conda_lib)
    if parts and parts[0] == conda_lib_str:
        os.environ["SCENE_AUTO_COLLECT_LD_READY"] = "1"
        return

    merged = [conda_lib_str]
    merged.extend(part for part in parts if part != conda_lib_str)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(merged)
    env["SCENE_AUTO_COLLECT_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_maybe_reexec_with_conda_libstdcpp()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Auto-place robot in the current scene and collect teleop data with a mouse UI.")
parser.add_argument("--robot", type=str, default="agibot", choices=["kinova", "agibot", "r1lite"], help="Robot profile.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Only 1 is supported.")
parser.add_argument("--dataset_file", type=str, default="./datasets/scene_mouse_collect.hdf5", help="Output HDF5 dataset path.")
parser.add_argument("--capture_hz", type=float, default=10.0, help="Frame sampling rate for dataset recording.")
parser.add_argument("--append", action="store_true", default=False, help="Append new episodes to an existing dataset.")
parser.add_argument("--lin_step", type=float, default=0.015, help="Mouse UI translation step size in meters.")
parser.add_argument("--ang_step", type=float, default=0.10, help="Mouse UI rotation step size in radians.")
parser.add_argument("--arm_side", type=str, default="left", choices=["left", "right"], help="Active arm to teleoperate for agibot/r1lite.")
parser.add_argument("--scene_usd_path", type=str, default=str(DEFAULT_SCENE_USD_PATH), help="Current scene USD path.")
parser.add_argument("--scene_graph_path", type=str, default=str(SCENE_GRAPH_PATH), help="Current scene graph JSON path.")
parser.add_argument("--placements_path", type=str, default=str(DEFAULT_PLACEMENTS_PATH), help="Current placements JSON path.")
parser.add_argument("--target", type=str, default=None, help="Optional target object prim/class/caption substring.")
parser.add_argument("--support", type=str, default=None, help="Optional explicit support object prim.")
parser.add_argument(
    "--object_collision_approx",
    type=str,
    default="default",
    choices=[
        "default",
        "triangle_mesh",
        "convex_hull",
        "convex_decomposition",
        "mesh_simplification",
        "bounding_cube",
        "bounding_sphere",
        "sdf",
        "sphere_fill",
    ],
    help="Optional collision approximation override applied to every generated scene object.",
)
parser.add_argument(
    "--target_collision_approx",
    type=str,
    default="default",
    choices=[
        "default",
        "triangle_mesh",
        "convex_hull",
        "convex_decomposition",
        "mesh_simplification",
        "bounding_cube",
        "bounding_sphere",
        "sdf",
        "sphere_fill",
    ],
    help="Optional collision approximation override applied only to the resolved target object.",
)
parser.add_argument(
    "--convex_decomp_voxel_resolution",
    type=int,
    default=1000000,
    help="Voxel resolution used when convex decomposition collision is selected.",
)
parser.add_argument(
    "--convex_decomp_max_convex_hulls",
    type=int,
    default=64,
    help="Maximum convex hull count used when convex decomposition collision is selected.",
)
parser.add_argument(
    "--convex_decomp_error_percentage",
    type=float,
    default=2.0,
    help="Allowed decomposition error percentage when convex decomposition collision is selected.",
)
parser.add_argument(
    "--convex_decomp_shrink_wrap",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable shrink-wrap projection for convex decomposition collision.",
)
parser.add_argument(
    "--base_z_bias",
    type=float,
    default=0.0,
    help="Extra robot base z offset in meters after automatic floor alignment. Negative moves the robot down.",
)
parser.add_argument(
    "--plan_output_dir",
    type=str,
    default=str(PROJECT_ROOT / "runtime" / "robot_placement"),
    help="Directory for planned placement artifacts.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    from scene_robot_apps.scene_mouse_collect import SceneMouseCollectArgs, run_scene_mouse_collect

    run_scene_mouse_collect(
        simulation_app,
        args_cli.robot,
        SceneMouseCollectArgs(
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            dataset_file=args_cli.dataset_file,
            capture_hz=args_cli.capture_hz,
            append=args_cli.append,
            lin_step=args_cli.lin_step,
            ang_step=args_cli.ang_step,
            scene_usd_path=args_cli.scene_usd_path,
            scene_graph_path=args_cli.scene_graph_path,
            placements_path=args_cli.placements_path,
            target=args_cli.target,
            support=args_cli.support,
            object_collision_approx=args_cli.object_collision_approx,
            target_collision_approx=args_cli.target_collision_approx,
            convex_decomp_voxel_resolution=args_cli.convex_decomp_voxel_resolution,
            convex_decomp_max_convex_hulls=args_cli.convex_decomp_max_convex_hulls,
            convex_decomp_error_percentage=args_cli.convex_decomp_error_percentage,
            convex_decomp_shrink_wrap=args_cli.convex_decomp_shrink_wrap,
            plan_output_dir=args_cli.plan_output_dir,
            base_z_bias=args_cli.base_z_bias,
            arm_side=args_cli.arm_side,
        ),
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
