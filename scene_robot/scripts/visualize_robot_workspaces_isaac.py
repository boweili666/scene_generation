import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_ROBOT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCENE_ROBOT_SRC))


def _maybe_reexec_with_conda_libstdcpp() -> None:
    if os.environ.get("SCENE_WORKSPACE_ISAAC_LD_READY") == "1":
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
        os.environ["SCENE_WORKSPACE_ISAAC_LD_READY"] = "1"
        return

    merged = [conda_lib_str]
    merged.extend(part for part in parts if part != conda_lib_str)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(merged)
    env["SCENE_WORKSPACE_ISAAC_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_maybe_reexec_with_conda_libstdcpp()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize robot working areas directly inside Isaac Sim.")
parser.add_argument("--robot", type=str, default="agibot", choices=["kinova", "agibot", "r1lite"], help="Robot profile.")
parser.add_argument("--arm-side", type=str, default="left", choices=["left", "right"], help="Active arm side for bimanual robots.")
parser.add_argument("--plane-z", type=float, default=0.02, help="Workspace overlay height in the robot base frame.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _find_robot_prim_path(stage) -> str:
    candidates: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.endswith("/Robot"):
            candidates.append(path)
    if not candidates:
        raise ValueError("Unable to find a live /Robot prim in stage.")
    return sorted(candidates, key=len)[0]


def main() -> None:
    import isaaclab.sim as sim_utils

    from scene_robot_apps.robot_workspaces import add_robot_workspace_visuals_to_stage
    from scene_robot_apps.stack_cube import build_stack_scene, resolve_stack_spec

    spec = resolve_stack_spec(args_cli.robot, args_cli.arm_side)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim.set_camera_view(spec.camera_eye, spec.camera_target)

    scene, _controller = build_stack_scene(sim, args_cli.robot, num_envs=1, arm_side=args_cli.arm_side)
    robot_prim_path = _find_robot_prim_path(scene.stage)
    result = add_robot_workspace_visuals_to_stage(
        scene.stage,
        robot=args_cli.robot,
        robot_prim_path=robot_prim_path,
        plane_z=float(args_cli.plane_z),
    )

    print(f"[INFO] Workspace robot: {args_cli.robot}")
    print(f"[INFO] Workspace root: {result['workspace_root_path']}")
    print(f"[INFO] Robot prim: {robot_prim_path}")
    print("[INFO] Close the Isaac Sim window to exit.")

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
