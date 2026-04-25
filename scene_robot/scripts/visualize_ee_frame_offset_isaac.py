import argparse
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCENE_ROBOT_SRC))


def _maybe_reexec_with_conda_libstdcpp() -> None:
    if os.environ.get("SCENE_EE_FRAME_DEBUG_LD_READY") == "1":
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
        os.environ["SCENE_EE_FRAME_DEBUG_LD_READY"] = "1"
        return

    merged = [conda_lib_str]
    merged.extend(part for part in parts if part != conda_lib_str)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(merged)
    env["SCENE_EE_FRAME_DEBUG_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_maybe_reexec_with_conda_libstdcpp()

from isaaclab.app import AppLauncher

from scene_robot_apps.control.ee_frame_remap import (
    EE_FRAME_REMAP_ROTATIONS,
    apply_local_ee_frame_remap_to_world_quat,
    apply_local_translation_to_world_pos,
)


parser = argparse.ArgumentParser(description="Visualize the current EE frame and a fixed local-offset EE frame in Isaac Sim.")
parser.add_argument("--robot", type=str, default="agibot", choices=["kinova", "agibot", "r1lite"], help="Robot profile.")
parser.add_argument("--arm-side", type=str, default="left", choices=["left", "right"], help="Active arm side for bimanual robots.")
parser.add_argument(
    "--ee-frame-offset",
    type=str,
    default="rot_y_neg_90",
    choices=sorted(EE_FRAME_REMAP_ROTATIONS),
    help="Local EE-frame rotation offset applied on top of the current EE frame.",
)
parser.add_argument(
    "--offset-translation-local",
    type=float,
    nargs=3,
    default=(0.08, 0.0, 0.0),
    metavar=("TX", "TY", "TZ"),
    help="Extra local translation applied to the offset EE frame for visualization separation, in meters.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import isaaclab.sim as sim_utils

    from scene_robot_apps.grasp.visualization import add_pose_frames_to_stage
    from scene_robot_apps.control.stack_cube import build_stack_scene
    from scene_robot_apps.control.robot_spec import resolve_stack_spec

    spec = resolve_stack_spec(args_cli.robot, args_cli.arm_side)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim.set_camera_view(spec.camera_eye, spec.camera_target)

    scene, controller = build_stack_scene(sim, args_cli.robot, num_envs=1, arm_side=args_cli.arm_side)
    controller.ee_marker.set_visibility(False)
    controller.goal_marker.set_visibility(False)

    print(f"[INFO] Robot: {args_cli.robot}")
    print(f"[INFO] Arm side: {args_cli.arm_side}")
    print(f"[INFO] EE frame offset: {args_cli.ee_frame_offset}")
    print(f"[INFO] Offset translation local: {tuple(float(v) for v in args_cli.offset_translation_local)}")
    print("[INFO] Visuals = custom USD pose frames")
    print("[INFO] CurrentEE = current EE frame")
    print("[INFO] OffsetEE = current EE frame after local rotation offset and debug translation")
    print("[INFO] Close the Isaac Sim window to exit.")

    while simulation_app.is_running():
        ee_pos_w, ee_quat_w = controller.current_ee_pose_world()
        current_pos = tuple(float(v) for v in ee_pos_w[0].detach().cpu().tolist())
        current_quat = tuple(float(v) for v in ee_quat_w[0].detach().cpu().tolist())
        offset_quat = apply_local_ee_frame_remap_to_world_quat(current_quat, args_cli.ee_frame_offset)
        offset_pos = apply_local_translation_to_world_pos(
            current_pos,
            offset_quat,
            args_cli.offset_translation_local,
        )
        add_pose_frames_to_stage(
            scene.stage,
            root_prim_path="/Visuals/EEFrameOffsetDebug",
            pose_frames=[
                {
                    "name": "CurrentEE",
                    "position_world": current_pos,
                    "quat_wxyz_world": current_quat,
                    "axis_length": 0.08,
                    "axis_thickness": 0.006,
                    "opacity": 0.92,
                },
                {
                    "name": "OffsetEE",
                    "position_world": offset_pos,
                    "quat_wxyz_world": offset_quat,
                    "axis_length": 0.08,
                    "axis_thickness": 0.006,
                    "opacity": 0.92,
                },
            ],
        )

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
