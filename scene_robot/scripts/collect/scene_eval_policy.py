import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCENE_ROBOT_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from app.backend.config.settings import DEFAULT_PLACEMENTS_PATH, SCENE_GRAPH_PATH
from scene_robot_apps.control.ee_frame_remap import EE_FRAME_REMAP_ROTATIONS


DEFAULT_SCENE_USD_PATH = PROJECT_ROOT / "runtime" / "scene_service" / "usd" / "scene_latest.usd"


def _maybe_reexec_with_conda_libstdcpp() -> None:
    if os.environ.get("SCENE_EVAL_LD_READY") == "1":
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
        os.environ["SCENE_EVAL_LD_READY"] = "1"
        return
    merged = [conda_lib_str]
    merged.extend(part for part in parts if part != conda_lib_str)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(merged)
    env["SCENE_EVAL_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_maybe_reexec_with_conda_libstdcpp()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Closed-loop evaluation of a LeRobot policy in Isaac Sim.")
parser.add_argument("--robot", type=str, default="agibot", choices=["agibot"])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--session", type=str, default=None)
parser.add_argument("--run", type=str, default=None)
parser.add_argument("--scene_usd_path", type=str, default=None)
parser.add_argument("--scene_graph_path", type=str, default=None)
parser.add_argument("--placements_path", type=str, default=None)
parser.add_argument("--target", type=str, required=True, help="Target prim path, e.g. /World/bolt_2")
parser.add_argument("--support", type=str, default=None)
parser.add_argument("--base_z_bias", type=float, default=0.0)
parser.add_argument("--plan_output_dir", type=str, default=str(PROJECT_ROOT / "runtime" / "robot_placement"))
parser.add_argument("--arm_side", type=str, default="auto", choices=["auto", "left", "right"])
parser.add_argument("--manifest-path", type=str, default=None)
parser.add_argument(
    "--agibot-ee-frame-remap",
    type=str,
    default="x_forward_z_up",
    choices=sorted(EE_FRAME_REMAP_ROTATIONS),
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the LeRobot policy checkpoint directory (pretrained_model dir)")
parser.add_argument(
    "--dataset-root",
    type=str,
    required=True,
    help="LeRobotDataset root directory (we read meta/stats.json from here to manually normalize observations / denormalize actions around policy.select_action, because LeRobot 0.4.x doesn't bake stats into the model checkpoint).",
)
parser.add_argument("--num-episodes", type=int, default=10)
parser.add_argument("--max-steps-per-episode", type=int, default=120)
parser.add_argument("--target-forward-randomization", type=float, default=0.03)
parser.add_argument("--success-lift-delta", type=float, default=0.03)
parser.add_argument("--fingertip-distance", type=float, default=0.12)
parser.add_argument("--gripper-threshold", type=float, default=0.5, help="Threshold on the 7-th action dim above which the gripper is commanded closed.")
parser.add_argument("--random-seed", type=int, default=None)
parser.add_argument(
    "--num-inference-steps",
    type=int,
    default=10,
    help="Override the DiffusionPolicy denoising loop length. Training uses 100 steps; inference with a reduced DDIM-style 10-20 step schedule is visually identical but much faster (fewer U-Net forwards between env steps = less visible stutter). 0 = keep the model's default.",
)
parser.add_argument(
    "--sim-steps-per-policy-call",
    type=int,
    default=6,
    help="Physics sub-steps to run between policy calls. Training recorded at 10 Hz with 60 Hz physics = 6 sim steps per dataset frame; the policy's delta action represents 0.1s of motion, not 1/60s. Running the policy once per physics step makes the robot move 6x slower than training and never reach the object. Default 6 matches the training cadence.",
)
parser.add_argument(
    "--async-inference",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run diffusion denoising in a background thread so the main loop never blocks on U-Net inference. Eliminates visible stutter at the cost of a small amount of observation staleness.",
)
parser.add_argument(
    "--record-dir",
    type=str,
    default="",
    help="Directory to write per-episode camera videos into. Empty = no recording. Each episode produces episode_NN_head.mp4, episode_NN_left_hand.mp4, episode_NN_right_hand.mp4 (falls back to PNG sequences if imageio/ffmpeg is unavailable).",
)
parser.add_argument(
    "--record-fps",
    type=float,
    default=10.0,
    help="Playback fps for recorded videos. Default 10 matches the training dataset capture_hz.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _resolve_session_paths() -> None:
    session = args_cli.session
    run_id = args_cli.run
    if session or run_id:
        if not (session and run_id):
            raise SystemExit("--session and --run must be provided together")
        session_root = PROJECT_ROOT / "runtime" / "sessions" / session / "runs" / run_id
        if not session_root.exists():
            raise SystemExit(f"Session run directory not found: {session_root}")
        if args_cli.scene_usd_path is None:
            args_cli.scene_usd_path = str(session_root / "scene_service" / "usd" / "scene_latest.usd")
        if args_cli.scene_graph_path is None:
            args_cli.scene_graph_path = str(session_root / "scene_graph" / "current_scene_graph.json")
        if args_cli.placements_path is None:
            args_cli.placements_path = str(session_root / "scene_service" / "placements" / "placements_default.json")
        if args_cli.manifest_path is None:
            args_cli.manifest_path = str(session_root / "real2sim" / "scene_results" / "real2sim_asset_manifest.json")
    if args_cli.scene_usd_path is None:
        args_cli.scene_usd_path = str(DEFAULT_SCENE_USD_PATH)
    if args_cli.scene_graph_path is None:
        args_cli.scene_graph_path = str(SCENE_GRAPH_PATH)
    if args_cli.placements_path is None:
        args_cli.placements_path = str(DEFAULT_PLACEMENTS_PATH)


def main():
    from scene_robot_apps.pipelines.scene_eval_policy import SceneEvalArgs, run_scene_eval

    _resolve_session_paths()

    run_scene_eval(
        simulation_app,
        args_cli.robot,
        SceneEvalArgs(
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            scene_usd_path=args_cli.scene_usd_path,
            scene_graph_path=args_cli.scene_graph_path,
            placements_path=args_cli.placements_path,
            target=args_cli.target,
            support=args_cli.support,
            plan_output_dir=args_cli.plan_output_dir,
            base_z_bias=args_cli.base_z_bias,
            arm_side_preference=args_cli.arm_side,
            manifest_path=args_cli.manifest_path,
            checkpoint=args_cli.checkpoint,
            num_episodes=args_cli.num_episodes,
            max_steps_per_episode=args_cli.max_steps_per_episode,
            target_forward_randomization=args_cli.target_forward_randomization,
            success_lift_delta=args_cli.success_lift_delta,
            fingertip_distance=args_cli.fingertip_distance,
            agibot_ee_frame_remap=args_cli.agibot_ee_frame_remap,
            gripper_threshold=args_cli.gripper_threshold,
            random_seed=args_cli.random_seed,
            dataset_root=args_cli.dataset_root,
            num_inference_steps=args_cli.num_inference_steps,
            sim_steps_per_policy_call=args_cli.sim_steps_per_policy_call,
            async_inference=args_cli.async_inference,
            record_dir=args_cli.record_dir,
            record_fps=args_cli.record_fps,
        ),
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
