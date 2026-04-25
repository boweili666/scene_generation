import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scene_robot_apps import MouseTeleopRecordArgs, run_mouse_teleop_record


parser = argparse.ArgumentParser(description="Agibot mouse teleop data collection.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Only 1 is supported for recording.")
parser.add_argument("--dataset_file", type=str, default="./datasets/agibot_mouse_dataset.hdf5", help="Output HDF5 dataset path.")
parser.add_argument("--capture_hz", type=float, default=10.0, help="Frame sampling rate for dataset recording.")
parser.add_argument("--append", action="store_true", default=False, help="Append new episodes to an existing dataset.")
parser.add_argument("--lin_step", type=float, default=0.015, help="Mouse UI translation step size in meters.")
parser.add_argument("--ang_step", type=float, default=0.10, help="Mouse UI rotation step size in radians.")
parser.add_argument("--arm_side", type=str, default="left", choices=["left", "right"], help="Active arm to teleoperate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    run_mouse_teleop_record(
        simulation_app,
        "agibot",
        MouseTeleopRecordArgs(
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            dataset_file=args_cli.dataset_file,
            capture_hz=args_cli.capture_hz,
            append=args_cli.append,
            lin_step=args_cli.lin_step,
            ang_step=args_cli.ang_step,
            arm_side=args_cli.arm_side,
        ),
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
