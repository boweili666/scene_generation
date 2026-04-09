# scene_robot

Standalone custom robot assets, controllers, and demo scripts for Isaac Lab.

## Layout

- `src/controller`: local controller package used by the custom demos.
- `src/scene_robot_apps`: shared app logic for stack-cube and mouse teleop recording.
- `src/scene_robot_assets`: local robot asset configs and robot files.
- `scripts`: teleop/demo entrypoints plus scene utilities:
  - `kinova_mouse_teleop_record.py`
  - `agibot_mouse_teleop_record.py`
  - `r1lite_mouse_teleop_record.py`
  - `kinova_stack_cube.py`
  - `agibot_stack_cube.py`
  - `r1lite_stack_cube.py`
  - `scene_auto_mouse_collect.py`
  - `real2sim_scale_randomize.py`
  - `visualize_real2sim_randomization.py`
  - `visualize_robot_workspaces.py`
  - `visualize_robot_workspaces_isaac.py`

## Run

Run commands from the repository root after Isaac Lab and Isaac Sim are available in the current Python environment.

```bash
python scripts/kinova_mouse_teleop_record.py
python scripts/agibot_stack_cube.py
python scripts/r1lite_stack_cube.py
python scene_robot/scripts/real2sim_scale_randomize.py --scene-usd-path runtime/scene_service/usd/scene_latest.usd
python scene_robot/scripts/real2sim_scale_randomize.py --scene-usd-path runtime/scene_service/usd/scene_latest.usd --interactive
python scene_robot/scripts/real2sim_scale_randomize.py --scene-usd-path runtime/scene_service/usd/scene_latest.usd --shared-scale
python scene_robot/scripts/real2sim_scale_randomize.py --scene-usd-path runtime/scene_service/usd/scene_latest.usd --global-scale 1.1
python scene_robot/scripts/visualize_robot_workspaces.py
python scene_robot/scripts/visualize_robot_workspaces_isaac.py --robot agibot
python scene_robot/scripts/scene_auto_mouse_collect.py --robot agibot --show-workspace
```
