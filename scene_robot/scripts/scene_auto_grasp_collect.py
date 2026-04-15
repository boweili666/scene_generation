import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCENE_ROBOT_SRC))

from app.backend.config.settings import DEFAULT_PLACEMENTS_PATH, SCENE_GRAPH_PATH
from scene_robot_apps.ee_frame_remap import EE_FRAME_REMAP_ROTATIONS


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


parser = argparse.ArgumentParser(description="Automatically select a grasp proposal and collect one grasp episode.")
parser.add_argument("--robot", type=str, default="agibot", choices=["kinova", "agibot", "r1lite"])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--session",
    type=str,
    default=None,
    help="Session id (e.g. sess_37daed605d8c). When set together with --run, all four scene/graph/placements/manifest paths (and, unless explicitly provided, --dataset_file) are derived from runtime/sessions/<session>/runs/<run>/...",
)
parser.add_argument(
    "--run",
    type=str,
    default=None,
    help="Run id within the session (e.g. run_ab267fae7ae8). Required with --session.",
)
parser.add_argument("--dataset_file", type=str, default=None)
parser.add_argument("--capture_hz", type=float, default=10.0)
parser.add_argument("--append", action="store_true", default=False)
parser.add_argument("--scene_usd_path", type=str, default=None)
parser.add_argument("--scene_graph_path", type=str, default=None)
parser.add_argument("--placements_path", type=str, default=None)
parser.add_argument("--target", type=str, default=None)
parser.add_argument("--support", type=str, default=None)
parser.add_argument("--base_z_bias", type=float, default=0.0)
parser.add_argument("--plan_output_dir", type=str, default=str(PROJECT_ROOT / "runtime" / "robot_placement"))
parser.add_argument("--arm_side", type=str, default="auto", choices=["auto", "left", "right"])
parser.add_argument("--show-grasp-poses", action="store_true", default=False)
parser.add_argument(
    "--wait-for-run-request",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show the selected grasp first and wait for a mouse-clicked Run button before executing.",
)
parser.add_argument("--manifest-path", type=str, default=None)
parser.add_argument("--annotation-root", type=str, default=None)
parser.add_argument("--scene-grasp-proposals-path", type=str, default=None)
parser.add_argument(
    "--lazy-build-target-annotation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If the selected target asset has no grasp annotation cache, build that single target cache on demand.",
)
parser.add_argument("--axis-band-slide-samples", type=int, default=5)
parser.add_argument("--axis-band-ring-samples", type=int, default=16)
parser.add_argument("--max-geom-candidates", type=int, default=1024)
parser.add_argument("--workspace-margin", type=float, default=0.02)
parser.add_argument("--body-clearance-margin", type=float, default=0.08)
parser.add_argument("--pre-grasp-distance", type=float, default=0.10)
parser.add_argument("--lift-height", type=float, default=0.10)
parser.add_argument("--retreat-distance", type=float, default=0.08)
parser.add_argument("--approach-clearance", type=float, default=0.006)
parser.add_argument("--pre-grasp-steps", type=int, default=90)
parser.add_argument("--approach-steps", type=int, default=90)
parser.add_argument("--close-steps", type=int, default=120)
parser.add_argument("--lift-steps", type=int, default=90)
parser.add_argument("--retreat-steps", type=int, default=90)
parser.add_argument("--pos-tol", type=float, default=0.03)
parser.add_argument("--grasp-pos-tol", type=float, default=0.025)
parser.add_argument("--rot-tol-deg", type=float, default=25.0)
parser.add_argument("--success-lift-delta", type=float, default=0.03)
parser.add_argument("--start-pose-distance-weight", type=float, default=0.30)
parser.add_argument("--start-pose-rotation-weight", type=float, default=0.10)
parser.add_argument(
    "--agibot-ee-frame-remap",
    type=str,
    default="x_forward_z_up",
    choices=sorted(EE_FRAME_REMAP_ROTATIONS),
    help="Execution EE-frame remap used by agibot auto grasp. Canonical grasp proposals stay unchanged.",
)
parser.add_argument(
    "--num-episodes",
    type=int,
    default=1,
    help="Run the refresh→rank→reachability→attempt loop this many times before exiting. Each episode reuses the same HDF5 writer so all rollouts are appended to --dataset_file.",
)
parser.add_argument(
    "--fingertip-distance",
    type=float,
    default=0.0,
    help="Distance (m) from the controller EE origin (wrist) to the fingertip. When non-zero, IK targets are shifted backwards along approach so the fingertip lands at the grasp point instead of the wrist. Typical AgiBot value: 0.12.",
)
parser.add_argument(
    "--phase-linear-speed",
    type=float,
    default=0.0,
    help="Max linear speed (m/s) used to interpolate commanded targets during pre_grasp/approach/lift/retreat. 0 disables interpolation and commands the final target every step (controller's own velocity limits apply).",
)
parser.add_argument(
    "--phase-angular-speed-deg",
    type=float,
    default=0.0,
    help="Max angular speed (deg/s) used to interpolate commanded targets. 0 disables rotation interpolation.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Disable Isaac Sim's built-in collider debug overlay. PhysX's collision
# visualization reads the collision shapes without applying the root prim's
# non-unit xformOp:transform scale (randomization puts ~0.125 on /World/bolt_*),
# so the rainbow hulls are drawn at the wrong world position and can be
# misleading when eyeballing grasp alignment. Physics itself is correct; only
# this debug overlay is off. Force it off via carb settings so the scene stays
# clean during automated grasp collection runs.
try:
    import carb

    _carb_settings = carb.settings.get_settings()
    for _key in (
        "/physics/visualizationCollisionMesh",
        "/persistent/physics/visualizationCollisionMesh",
        "/physics/visualizationDisplayMasses",
        "/persistent/physics/visualizationDisplayMasses",
        "/physics/visualizationDisplayJoints",
        "/persistent/physics/visualizationDisplayJoints",
    ):
        try:
            _carb_settings.set(_key, False)
        except Exception:
            pass
except Exception:
    pass


def _resolve_session_paths() -> None:
    # If the user passed --session/--run, fill in any omitted path args from
    # the runtime/sessions/<session>/runs/<run>/... layout. Explicit paths
    # still win; fallback defaults are used when neither is given.
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
        if args_cli.dataset_file is None:
            target_slug = (args_cli.target or "target").strip("/").replace("/", "_") or "target"
            args_cli.dataset_file = str(
                PROJECT_ROOT
                / "datasets"
                / f"{session}_{run_id}_{args_cli.robot}_{target_slug}.hdf5"
            )

    # Fall back to the legacy defaults so old invocations keep working.
    if args_cli.scene_usd_path is None:
        args_cli.scene_usd_path = str(DEFAULT_SCENE_USD_PATH)
    if args_cli.scene_graph_path is None:
        args_cli.scene_graph_path = str(SCENE_GRAPH_PATH)
    if args_cli.placements_path is None:
        args_cli.placements_path = str(DEFAULT_PLACEMENTS_PATH)
    if args_cli.dataset_file is None:
        args_cli.dataset_file = "./datasets/scene_auto_grasp_collect.hdf5"


def main():
    from scene_robot_apps.scene_auto_grasp_collect import SceneAutoGraspCollectArgs, run_scene_auto_grasp_collect

    _resolve_session_paths()

    run_scene_auto_grasp_collect(
        simulation_app,
        args_cli.robot,
        SceneAutoGraspCollectArgs(
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            dataset_file=args_cli.dataset_file,
            capture_hz=args_cli.capture_hz,
            append=args_cli.append,
            scene_usd_path=args_cli.scene_usd_path,
            scene_graph_path=args_cli.scene_graph_path,
            placements_path=args_cli.placements_path,
            target=args_cli.target,
            support=args_cli.support,
            plan_output_dir=args_cli.plan_output_dir,
            base_z_bias=args_cli.base_z_bias,
            arm_side_preference=args_cli.arm_side,
            show_grasp_poses=args_cli.show_grasp_poses,
            wait_for_run_request=args_cli.wait_for_run_request,
            manifest_path=args_cli.manifest_path,
            annotation_root=args_cli.annotation_root,
            scene_grasp_proposals_path=args_cli.scene_grasp_proposals_path,
            lazy_build_target_annotation=args_cli.lazy_build_target_annotation,
            axis_band_slide_samples=args_cli.axis_band_slide_samples,
            axis_band_ring_samples=args_cli.axis_band_ring_samples,
            max_geom_candidates=args_cli.max_geom_candidates,
            workspace_margin=args_cli.workspace_margin,
            body_clearance_margin=args_cli.body_clearance_margin,
            pre_grasp_distance=args_cli.pre_grasp_distance,
            lift_height=args_cli.lift_height,
            retreat_distance=args_cli.retreat_distance,
            approach_clearance=args_cli.approach_clearance,
            pre_grasp_steps=args_cli.pre_grasp_steps,
            approach_steps=args_cli.approach_steps,
            close_steps=args_cli.close_steps,
            lift_steps=args_cli.lift_steps,
            retreat_steps=args_cli.retreat_steps,
            pos_tol=args_cli.pos_tol,
            grasp_pos_tol=args_cli.grasp_pos_tol,
            rot_tol_deg=args_cli.rot_tol_deg,
            success_lift_delta=args_cli.success_lift_delta,
            start_pose_distance_weight=args_cli.start_pose_distance_weight,
            start_pose_rotation_weight=args_cli.start_pose_rotation_weight,
            agibot_ee_frame_remap=args_cli.agibot_ee_frame_remap,
            num_episodes=args_cli.num_episodes,
            fingertip_distance=args_cli.fingertip_distance,
            phase_linear_speed=args_cli.phase_linear_speed,
            phase_angular_speed_deg=args_cli.phase_angular_speed_deg,
        ),
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
