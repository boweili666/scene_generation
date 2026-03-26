import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scene_robot_apps import run_stack_cube_demo


parser = argparse.ArgumentParser(description="Kinova stack-cube demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    run_stack_cube_demo(simulation_app, "kinova", device=args_cli.device, num_envs=args_cli.num_envs)


if __name__ == "__main__":
    main()
    simulation_app.close()

