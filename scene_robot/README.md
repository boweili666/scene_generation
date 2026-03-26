# scene_robot

Standalone custom robot assets, controllers, and demo scripts for Isaac Lab.

## Layout

- `src/controller`: local controller package used by the custom demos.
- `src/scene_robot_apps`: shared app logic for stack-cube and mouse teleop recording.
- `src/scene_robot_assets`: local robot asset configs and robot files.
- `scripts`: six entrypoints only:
  - `kinova_mouse_teleop_record.py`
  - `agibot_mouse_teleop_record.py`
  - `r1lite_mouse_teleop_record.py`
  - `kinova_stack_cube.py`
  - `agibot_stack_cube.py`
  - `r1lite_stack_cube.py`

## Run

Run commands from the repository root after Isaac Lab and Isaac Sim are available in the current Python environment.

```bash
python scripts/kinova_mouse_teleop_record.py
python scripts/agibot_stack_cube.py
python scripts/r1lite_stack_cube.py
```
