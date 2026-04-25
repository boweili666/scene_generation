# scene_robot

Robot assets, IK controllers, scene-based data collection pipelines, and
the policy training / evaluation glue for Isaac Lab.

## Directory layout

```
scene_robot/
├── pyproject.toml
├── README.md
├── scripts/                              CLI entrypoints (organized by intent)
│   ├── collect/                          scene-based data collection + closed-loop eval
│   │   ├── scene_auto_grasp_collect.py
│   │   ├── scene_auto_mouse_collect.py
│   │   └── scene_eval_policy.py
│   ├── demo/                             single-robot demos + teleop record
│   │   ├── {agibot,kinova,r1lite}_stack_cube.py
│   │   └── {agibot,kinova,r1lite}_mouse_teleop_record.py
│   ├── grasp/                            offline grasp annotation pipeline
│   │   ├── grasp_annotate.py
│   │   ├── build_grasp_asset_cache.py
│   │   └── export_scene_grasp_proposals.py
│   ├── real2sim/
│   │   └── real2sim_scale_randomize.py
│   └── visualize/                        debug visualizers
│       ├── visualize_bolt_grasp_pose.py
│       ├── visualize_ee_frame_offset_isaac.py
│       ├── visualize_grasp_pose_isaac.py
│       ├── visualize_grasp_proposals_open3d.py
│       ├── visualize_real2sim_randomization.py
│       ├── visualize_robot_workspaces.py
│       └── visualize_robot_workspaces_isaac.py
└── src/
    ├── scene_robot_apps/                 application layer (most-edited)
    │   ├── control/                      RobotController + IK + per-robot specs
    │   │   ├── robot_controller.py       single-arm + whole-body bimanual driver
    │   │   ├── phase_runner.py           target-tracking + hold phase runners
    │   │   ├── ee_frame_remap.py         agibot semantic↔controller EE-frame mapping
    │   │   ├── robot_spec.py             RobotSpec, ROBOT_SPECS, normalize_arm_side
    │   │   ├── scene_cfg.py              InteractiveSceneCfg factories
    │   │   └── stack_cube.py             stack-cube task only (build_stack_scene + demo loop)
    │   ├── grasp/                        grasp execution / ranking / proposal I/O
    │   │   ├── execution.py              FilteredGraspExecution, expand_grasp_candidates
    │   │   ├── ranking.py                geometric filter + EE-pose ranking
    │   │   ├── target_state.py           physx snapshot/restore + per-episode randomization
    │   │   ├── proposals_io.py           load/build scene grasp proposals
    │   │   └── visualization.py          stage-side grasp-pose markers
    │   ├── pipelines/                    top-level orchestrators
    │   │   ├── scene_mouse_collect.py    mouse teleop scene data collection
    │   │   ├── scene_auto_grasp_collect.py  automated grasp data collection
    │   │   ├── scene_eval_policy.py      closed-loop policy eval in Isaac Sim
    │   │   ├── mouse_teleop_record.py    standalone mouse teleop recording
    │   │   └── real2sim_scale_randomization.py
    │   ├── record/                       HDF5 episode writers
    │   │   ├── episode_writer.py         SceneTeleopEpisodeWriter (base)
    │   │   └── auto_grasp_writer.py      SceneAutoGraspEpisodeWriter (extends it)
    │   ├── scene/                        physics setup + workspace projection
    │   │   ├── physics.py                collision / BBox / rigid-body / reset
    │   │   └── workspaces.py             reachable-workspace polygon projection
    │   └── ui/                           Omniverse Kit windows
    │       ├── mouse_collect_ui.py       MouseCommandCollectUI (record + episode buttons)
    │       ├── auto_grasp_preview.py     AutoGraspPreviewUI (Run/Close gate)
    │       └── teleop_ui.py              MouseClickTeleopUI + KeyboardTeleop
    ├── controller/                       low-level diff-IK pose controllers
    │   ├── isaaclab_pose_controller.py   IsaacLabPoseController, IsaacLabBimanualPoseController
    │   ├── utils/                        Geometry, VizColor
    │   └── robot/{base,agibot,kinova,r1lite}/{config,kinematics}/  per-robot kinematics
    ├── grasp_annotator/                  offline grasp annotation pipeline
    │   ├── batch_runner.py
    │   ├── pose_generator.py             axes → rotation, ring grasp sampling
    │   ├── classifier.py                 grasp-type classification
    │   ├── segmentation.py
    │   ├── skeleton.py
    │   ├── render.py
    │   └── schema.py                     annotation JSON schema
    └── scene_robot_assets/               per-robot Isaac Lab ArticulationCfg
        ├── agibot.py                     AGIBOT_G1_OMNIPICKER_*
        ├── gen3_robotiq.py               GEN3_7DOF_VISION_ROBOTIQ_2F85_*
        └── r1lite.py                     R1LITE_*
```

## src/ packages — what each one does

**`scene_robot_apps/`** — application layer. Top-level pipelines, scene
construction, recorders, UI windows. This is where the data-collection
and eval business logic lives. Most edits go here.

**`controller/`** — low-level diff-IK pose controllers (Isaac Lab integration).
`scene_robot_apps.control.robot_controller` calls into
`controller.IsaacLabPoseController` / `IsaacLabBimanualPoseController`.
The package also holds per-robot `RobotConfig` / `RobotKinematics`
classes that the IK solver consumes.

**`grasp_annotator/`** — offline grasp annotation. Reads GLB/USD assets,
emits `point_grasp`, `axis_band`, and `pose_set` primitives into a JSON
cache (`runtime/.../grasp_asset_cache/...json`). Driven by
`scripts/grasp/grasp_annotate.py` and `scripts/grasp/build_grasp_asset_cache.py`.
At runtime, `scene_robot_apps.grasp.proposals_io` *reads* this cache and
expands it into FilteredGraspExecution candidates.

**`scene_robot_assets/`** — per-robot Isaac Lab `ArticulationCfg` (USD
path, joint init pos, actuator gains). Imported by
`scene_robot_apps.control.robot_spec` to populate `RobotSpec.robot_cfg`.

## Data → Training → Evaluation pipeline

The full policy loop spans `scripts/`, `tools/`, and the external
`lerobot-train` CLI.

```
[1. Collect raw HDF5]                                   env: env_isaaclab
    scripts/collect/scene_auto_grasp_collect.py
    scripts/collect/scene_auto_mouse_collect.py
    scripts/demo/<robot>_mouse_teleop_record.py
    └─→ datasets/<run>.hdf5

[2. Convert to LeRobotDataset]                          env: env_isaaclab or lerobot
    tools/convert_hdf5_to_lerobot.py
    └─→ datasets/lerobot/<repo_id>/{data, videos, meta}

[3. Train policy]                                       env: lerobot
    lerobot-train  (LeRobot's own CLI; we don't ship a trainer)
      --dataset.repo_id=local/<repo_id>
      --dataset.root=datasets/lerobot/<repo_id>
      --policy.type=diffusion        (or act / tdmpc / ...)
      --output_dir=outputs/train/<job>
    └─→ outputs/train/<job>/checkpoints/last/pretrained_model

[4. Evaluate]
    Offline action-MSE check (fast, no sim):           env: lerobot
        tools/eval_policy_offline.py
        └─→ per-dim MSE / RMSE on the training set

    Closed-loop sim rollout:                           env: env_isaaclab
        scripts/collect/scene_eval_policy.py
        └─→ success rate, optional per-episode camera videos

    Train-vs-eval comparison video:                    env: lerobot
        tools/make_comparison_video.py
        └─→ 2x3 train (top) / eval (bottom) MP4
```

### Why two conda envs?

`env_isaaclab` ships Isaac Sim 5.1 + Isaac Lab pinned with a particular
`numpy` / `torch` / `transformers` set. `lerobot` pulls a newer
`huggingface_hub` and conflicting `transformers`, plus `torchcodec` for
LeRobotDataset video decoding. They cannot coexist in one env without
breaking either side. Common rule:

| Task                                 | Environment      |
| ------------------------------------ | ---------------- |
| Anything that imports `isaaclab`     | `env_isaaclab`   |
| `lerobot-train`, `LeRobotDataset`    | `lerobot`        |
| `convert_hdf5_to_lerobot.py`         | either (works in both) |

`scripts/collect/scene_eval_policy.py` is the awkward one — it needs
both Isaac Sim and a LeRobot Diffusion checkpoint. It runs in
`env_isaaclab` and:

* installs an `is_offline_mode` shim into `huggingface_hub` so the
  bundled `transformers` import doesn't fail;
* stubs `lerobot.policies.groot.*` in `sys.modules` so importing
  `lerobot.policies.diffusion.modeling_diffusion` doesn't drag the
  groot → transformers chain in;
* reads `meta/stats.json` directly with `json.load` (LeRobot 0.4.x does
  not bake the dataset stats into the checkpoint, and we don't want to
  import `LeRobotDataset` here because its `torchcodec` is broken in
  this env).

## Common commands

All commands are run from the repository root.

### Collect data with auto-grasp

```bash
/home/lbw/miniconda3/envs/env_isaaclab/bin/python \
  scene_robot/scripts/collect/scene_auto_grasp_collect.py \
  --session sess_xxx --run run_xxx \
  --target /World/bolt_2 \
  --num-episodes 20 \
  --headless --no-wait-for-run-request \
  --target-forward-randomization 0.03
```

### Convert HDF5 → LeRobotDataset

```bash
python tools/convert_hdf5_to_lerobot.py \
  --hdf5 datasets/<run>.hdf5 \
  --repo-id local/<repo_id> \
  --output-root datasets/lerobot/<repo_id> \
  --task "pick up the bolt from the table" \
  --overwrite
```

### Train a Diffusion Policy

```bash
conda activate lerobot
lerobot-train \
  --dataset.repo_id=local/<repo_id> \
  --dataset.root=datasets/lerobot/<repo_id> \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/<job> \
  --steps=30000 --batch_size=8 --save_freq=5000
```

### Offline action-MSE evaluation

```bash
conda activate lerobot
python tools/eval_policy_offline.py \
  --checkpoint outputs/train/<job>/checkpoints/last/pretrained_model \
  --dataset-repo-id local/<repo_id> \
  --dataset-root datasets/lerobot/<repo_id>
```

### Closed-loop sim rollout

```bash
/home/lbw/miniconda3/envs/env_isaaclab/bin/python \
  scene_robot/scripts/collect/scene_eval_policy.py \
  --session sess_xxx --run run_xxx \
  --target /World/bolt_2 \
  --checkpoint outputs/train/<job>/checkpoints/last/pretrained_model \
  --dataset-root datasets/lerobot/<repo_id> \
  --num-episodes 20 \
  --num-inference-steps 10 \
  --sim-steps-per-policy-call 6 \
  --gripper-threshold 0.8 \
  --headless \
  --record-dir outputs/eval/<job>_runs
```

### Train-vs-eval comparison video

```bash
conda activate lerobot
python tools/make_comparison_video.py \
  --dataset-root datasets/lerobot/<repo_id> \
  --dataset-repo-id local/<repo_id> \
  --eval-dir outputs/eval/<job>_runs \
  --episodes all \
  --output outputs/eval/<job>_runs/comparison_all.mp4
```

### Per-robot demos and offline grasp annotation

```bash
python scene_robot/scripts/demo/agibot_stack_cube.py
python scene_robot/scripts/demo/kinova_mouse_teleop_record.py
python scene_robot/scripts/grasp/grasp_annotate.py annotate-one --file ./object.glb --output-dir ./runs
python scene_robot/scripts/grasp/build_grasp_asset_cache.py --manifest <manifest.json>
python scene_robot/scripts/visualize/visualize_robot_workspaces_isaac.py --robot agibot
python scene_robot/scripts/real2sim/real2sim_scale_randomize.py \
    --scene-usd-path runtime/scene_service/usd/scene_latest.usd
```
