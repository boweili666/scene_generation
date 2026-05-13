# Methods

*scene_graph_ui_test — image → scene → robot policy training pipeline*

## 1. Abstract

This system bridges five independently-developed third-party stacks
(SAM3 segmentation, SAM 3D Objects reconstruction, Isaac Sim / Isaac
Lab, LeRobot training, custom robot grasp planning) into a single
end-to-end loop that turns a real-world reference image into a
trained robot pickup policy. A web UI orchestrates the loop through
an LLM agent that plans and executes long-running pipeline stages as
discrete subprocess jobs, persists state per session, and surfaces
intermediate artifacts back to the operator. The integration is
deliberately **loose** — each stage runs in its own conda environment
under its own GPU constraints, communicating only through filesystem
artifacts and structured-output JSON. This document describes the
architecture, the four pipeline stages on the robot side, the
three-mode agent dispatcher, and the state / audit machinery that
keeps everything consistent across long-running async work.

## 2. System Overview

End-to-end flow, from a single reference image to a trained pickup
policy:

```
┌──────────────────────────────────────────────────────────────────┐
│                  Web UI  (Flask, scene_gen env)                  │
└──────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
   ┌────────────────┐ ┌────────────────┐ ┌──────────────────┐
   │ Scene Graph    │ │   Real2Sim     │ │  Scene Service   │
   │   (LLM)        │ │ (sam3d-obj env)│ │ (env_isaaclab)   │
   │                │ │                │ │                  │
   │ image+text →   │ │ image+graph →  │ │  graph+meshes →  │
   │ JSON graph     │ │   GLBs/USDs    │ │   scene USD      │
   └────────────────┘ └────────────────┘ └──────────────────┘
                                                  │
                                                  ▼
                            ┌─────────────────────────────────────┐
                            │     scene_robot pipeline            │
                            │   (env_isaaclab + lerobot envs)     │
                            │                                     │
                            │  collect → convert → train → eval   │
                            └─────────────────────────────────────┘
```

A single user prompt (e.g. *"build a pickup pipeline for the bottle in
this image"*) drives the agent through multiple of these stages
sequentially, each running for minutes-to-hours, with the operator
seeing live logs and intermediate artifacts in the browser. The agent
itself decides which stages to invoke and in what order; the
deterministic execution machinery handles long-running job lifecycle.

## 3. Architecture

### 3.1 Three Conda Environments

The system is constrained by an unavoidable Python dependency split
between Isaac Lab and LeRobot:

| Environment       | Role                                              |
| ----------------- | ------------------------------------------------- |
| `scene_gen`       | Web backend (Flask), scene service, scene-graph LLM editor |
| `sam3d-objects`   | SAM3 mask segmentation + remote SAM 3D Objects client |
| `env_isaaclab`    | Isaac Sim 5.1 + Isaac Lab, robot collect / eval scripts |
| `lerobot`         | LeRobot training and dataset utilities            |

Isaac Lab pins `numpy` / `torch` / `transformers` to a specific point;
LeRobot's recent `huggingface_hub` and `torchcodec` cannot coexist
without breakage. The repository acknowledges this constraint as
fundamental and architects around it rather than fighting it.

### 3.2 Subprocess + Filesystem Integration

Because the web backend (`scene_gen` env) cannot import code from
`env_isaaclab`, `lerobot`, or `sam3d-objects`, **every cross-env
operation is subprocess + filesystem**. The pattern, codified in
[`pipeline_service.py`](app/backend/services/pipeline_service.py) and
[`scene_robot_service.py`](app/backend/services/scene_robot_service.py):

1. The backend route assembles a `payload` dict from the request +
   session context.
2. A job-runner function builds a CLI command (`[python_bin, "-u",
   script, "--session", sess_id, "--target", ...]`).
3. A daemon thread launches the subprocess via `Popen` with separate
   stdout / stderr pipes.
4. A streaming reader interleaves both pipes into a per-stage log file
   under `runtime/sessions/<sess>/runs/<run>/logs/<stage>.log`.
5. A heartbeat line is appended every 30 s when the child is silent,
   so the UI's log-tail reader can prove liveness even during long
   GPU-bound steps.
6. Exit code is recorded; on nonzero, the job's status flips to
   `failed` and the captured stderr is exposed via the status
   endpoint.

The result: the web backend never imports CUDA, never holds GPU
context, and never has to reconcile conflicting pinned dependencies —
it just runs short orchestration code in `scene_gen`. The actual
heavy lifting happens in the right env every time.

### 3.3 Per-run Runtime Layout

[`runtime_context.py`](app/backend/services/runtime_context.py)
defines a frozen dataclass `RuntimeContext` that maps a (session_id,
run_id) pair to ~25 well-defined directories and files:

```
runtime/sessions/<session_id>/
├── session.json                                  ← session metadata
├── current_run.txt                               ← active run pointer
├── agent_state.json                              ← LLM agent state (per session)
└── runs/<run_id>/
    ├── run.json
    ├── uploads/latest_input.jpg                  ← reference image
    ├── scene_graph/current_scene_graph.json      ← scene graph JSON
    ├── renders/render.png                        ← scene preview
    ├── logs/{real2sim,scene_robot,scene_robot_convert,
    │     scene_robot_train,scene_robot_eval}.log ← per-stage streams
    ├── real2sim/                                 ← R2S workspace
    │   ├── masks/, meshes/
    │   └── scene_results/{objects,usd_objects,assignment.json,
    │         poses.json,real2sim_asset_manifest.json,scene_merged.glb}
    ├── scene_service/{usd/scene_latest.usd,placements/}
    └── robot_placement/                          ← inferred robot base poses
```

Sessions and runs are addressed by short UUIDs (`sess_xxxxxxxxxxxx`,
`run_xxxxxxxxxxxx`). The browser keeps `session_id` and `run_id` in
`localStorage` so reloading the page restores the active context.
This per-run isolation makes parallel experiments cheap: launching a
second run inherits no state from the first beyond what the user
explicitly selects.

## 4. Pipeline Stages

### 4.1 Real2Sim

Inputs: a reference image and a scene graph (JSON describing objects
with class names, captions, and `source: real2sim` flags).

Two-step subprocess pipeline driven by
[`pipeline_service.run_real2sim()`](app/backend/services/pipeline_service.py):

1. **Mask segmentation** — `pipelines/real2sim/object_segmentation_pipeline.py`
   loads SAM3 (in `sam3d-objects` env), aligns class prompts from the
   scene graph against the image, and writes per-object PNG masks plus
   a numbered overlay image into `runtime/.../real2sim/masks/`.
2. **3D reconstruction via streaming client** — `streaming_generation_client.py`
   POSTs each mask + crop to a remote SAM 3D Objects HTTP server
   (typically running on a beefier GPU machine, port 8002 by
   convention). Reconstructed GLBs stream back, are post-processed
   through `pipelines/isaac/mesh_to_usd_converter.py`, and dropped into
   `scene_results/objects/` and `scene_results/usd_objects/`. A
   merged-scene GLB and `assignment.json` (mask → scene-graph node
   mapping) are also produced.

Failure classification in `classify_real2sim_failure()` handles common
modes (segmentation timeout, remote server unavailable, mask
assignment ambiguity, partial outputs) so the UI can surface
actionable error messages rather than a raw stderr dump.

### 4.2 Isaac Scene Service

A separate Flask process (`app/backend/services/scene_service.py`,
default port 8001) running in `env_isaaclab`. It exposes two
endpoints:

* `POST /scene_new` — sample a fresh layout for the current scene
  graph using `pipelines/isaac/layout_utils.py`. Two strategies:
  `joint` (resample everything) or `lock_real2sim` (keep observed
  Real2Sim support chains rigid).
* `POST /scene` — preserve the current layout, just rebuild the USD.

Output: `runtime/.../scene_service/usd/scene_latest.usd`, a complete
Isaac-Sim-loadable scene with all per-object USD assets composed into
a room shell built by `room_shell_builder.py`. Layout failures (e.g.
collision constraints unsatisfiable) automatically retry once with a
fresh seed before falling back to `joint` mode for graphs containing
real2sim objects.

### 4.3 scene_robot Pipeline (4 Stages)

Mirrors the Real2Sim integration shape: each stage is a subprocess
job tracked by a shared `_JOBS` dict in
[`scene_robot_service.py`](app/backend/services/scene_robot_service.py),
with per-stage log files and a stage-agnostic
`/scene_robot/status/<job_id>` endpoint. The stages:

#### 4.3.1 Collect

Driver: `scene_robot/scripts/collect/scene_auto_grasp_collect.py`,
launched in `env_isaaclab`. Inputs: scene USD,
scene graph, target prim path (e.g. `/World/bolt_2`), robot type
(`agibot` / `kinova` / `r1lite`). For each episode:

1. Load the scene USD into Isaac Sim.
2. Place the robot at a workspace-feasible base pose (`robot_placement.py`).
3. Build `FilteredGraspExecution` candidates from the offline grasp
   annotation cache (`grasp_asset_cache_*.json`).
4. Rank candidates by EE-pose feasibility and start-pose distance.
5. Run a phase-based sequence (pre-grasp → approach → close → lift →
   retreat) with diff-IK control until success / failure.
6. Append the trajectory + RGB camera streams to an HDF5 dataset.

Output: `datasets/<sess>_<run>_<robot>_<target>.hdf5`.

#### 4.3.2 Convert

Driver: `tools/convert_hdf5_to_lerobot.py`. Re-encodes the HDF5
trajectory into a `LeRobotDataset` directory layout
(`datasets/lerobot/<repo_id>/{data,videos,meta}/`). Fast (seconds to
minutes); runs in either env.

#### 4.3.3 Train

Driver: `lerobot-train` binary in the `lerobot` env. The agent layer
fires it as a subprocess with structured CLI args
(`--dataset.repo_id`, `--policy.type=diffusion`, `--steps`, etc.)
and lets it run for hours. The eventual checkpoint lands in
`outputs/train/<run_id>_<timestamp>/checkpoints/last/pretrained_model`.

#### 4.3.4 Eval

Driver: `scene_robot/scripts/collect/scene_eval_policy.py` in
`env_isaaclab`. Loads the trained checkpoint, performs a closed-loop
sim rollout in the same scene USD that produced the training data,
and records per-episode camera videos to
`outputs/eval/<run_id>_<timestamp>_runs/`. Reports a success rate
and, optionally, an action-MSE comparison via
`tools/eval_policy_offline.py`.

The `scene_eval_policy.py` script is the only place where Isaac Sim
and LeRobot's diffusion policy share a process; it works around the
env conflict via `is_offline_mode` shims and `sys.modules` stubs to
prevent transitive `transformers` / `torchcodec` imports.

## 5. Agent System

The web UI offers two agent dispatch modes, both backed by a shared
tool registry. The legacy single-shot router was retired (PR #3) in
favour of the loop / plan pair. All tool implementations live in
[`agent_loop.py`](app/backend/services/agent_loop.py) so they are
strictly identical between modes.

### 5.1 Tool Registry

Eight tools, registered as OpenAI Responses-API function definitions:

| Tool                          | Type          | Purpose                                |
| ----------------------------- | ------------- | -------------------------------------- |
| `inspect_state`               | read-only     | Snapshot of scene graph, USD, jobs     |
| `create_scene_graph`          | instant       | LLM-generated scene graph from text/img |
| `run_real2sim`                | long-running  | Launch Real2Sim job, return job_id     |
| `generate_scene`              | instant       | Call Isaac scene service               |
| `run_scene_robot_collect`     | long-running  | Launch collect job                     |
| `run_scene_robot_convert`     | long-running  | HDF5 → LeRobotDataset                  |
| `run_scene_robot_train`       | long-running  | Launch lerobot-train                   |
| `run_scene_robot_eval`        | long-running  | Closed-loop eval rollout               |

Long-running tools fire-and-return-job-id within milliseconds. They
do *not* block the agent loop; the operator sees the launched job
streaming in the log box, and the agent's response carries
`real2sim_job` or `scene_robot_job` payload that the frontend's
existing `monitorReal2SimJob` / `monitorSceneRobotJob` machinery
picks up automatically.

Smart defaults derive arguments from session context: convert finds
the most recent collect HDF5 by glob, train stamps an output
directory `outputs/train/<sess>_<run>_<ts>`, eval picks the latest
train output's `pretrained_model` checkpoint when not specified. So
the LLM can fire a stage with empty args and the right thing
happens.

### 5.2 Loop Mode (Default)

Single user message → multi-turn tool-use loop bounded at
`MAX_TOOL_TURNS=8`. Each iteration:

1. Backend POSTs `messages` + `tools` to OpenAI Responses API.
2. Model returns either `function_call` items or a final text message.
3. For each `function_call`, dispatch to the corresponding handler
   (synchronous if instant, fire-and-job-id if long-running).
4. Append the result as `function_call_output` and re-invoke the API.
5. Stop when the model returns text without further tool calls.

A short system prompt teaches the operating rules: always
`inspect_state` first if state is unknown; do not poll long-running
jobs within one turn; if the user request is ambiguous, ask back via
plain text. Tool results that include `note: "X runs in background"`
nudge the model to terminate the turn after launching a long job.

### 5.3 Plan Mode

Three-phase wrapper around the same tool registry:

1. **Propose** — a structured-output LLM call returns a `Plan` JSON
   matching `PLAN_SCHEMA` (goal + steps + per-step `tool` /
   `args_json` / `why`). The plan is persisted in
   `state["active_plan"]` and rendered in the UI panel for human
   review. Steps are validated against `TOOL_HANDLERS`; invalid steps
   are dropped before the plan is accepted.
2. **Execute** — a deterministic walker iterates the plan steps,
   calling each tool's handler directly. Pure (instant) steps run
   inline. Long-running tool steps trigger a pause: the executor
   records `paused_for_job: {kind, job_id, tool}`, sets plan status to
   `paused`, and returns. The user resumes by re-POSTing
   `/agent/plan/execute` after the long job finishes (or the
   frontend's monitor can auto-resume).
3. **Reflect** — a structured-output LLM call summarizes the plan's
   observations and returns a `Reflection` with `next_action` ∈
   {`done`, `wait_for_long_job`, `ask_user`, `follow_up_plan`}. A
   `follow_up_plan` is shown in the UI as an *Accept as New Plan*
   action that promotes it into the active plan slot.

Plan editing (PR #2) lets the operator modify pending steps before
running. Plan history (PR #2) archives every superseded plan into
`state["plan_history"]` capped at 20.

### 5.4 Pause / Resume on Long Jobs

When `_resume_check()` is called on a paused plan, it consults the
**job audit log** first (`state["job_audit"][job_id]`), which is
updated by every status-poll on `/scene_robot/status/<id>` or
`/real2sim/status/<id>`. This keeps resume logic stage-agnostic — a
plan paused on a `run_scene_robot_train` step resumes correctly even
though the legacy `state["runs"][run_id]["scene_robot"]` slot only
tracks the most recent stage. The audit fallback also lets archived
plans (in plan_history) display their old jobs' final status long
after the active state has moved on.

## 6. State Persistence & Job Audit

Per-session state lives in `runtime/sessions/<sess>/agent_state.json`
and is loaded on every backend operation that needs context. Top-level
fields:

```json
{
  "session_id": "sess_...",
  "current_run_id": "run_...",
  "current_state": "run_real2sim",
  "history": [...messages...],
  "active_plan": { ...plan dict... },
  "plan_history": [ ...up to 20 archived... ],
  "job_audit": {
    "<job_id>": {"kind": "real2sim", "status": "succeeded",
                 "first_seen_at": "...", "finished_at": "...",
                 "error": null}
  },
  "runs": {
    "<run_id>": {
      "real2sim": {"status": "succeeded", "job_id": "...",
                   "artifacts": {...}, "log_path": "..."},
      "scene_robot": {"status": "running", "job_id": "...",
                      "robot": "agibot", "target": "/World/bolt_2"},
      "scene_generation": {"status": "succeeded", "outputs": {...}}
    }
  },
  "latest_real2sim_run_id": "run_...",
  "latest_scene_generation_run_id": "run_...",
  "latest_scene_robot_run_id": "run_..."
}
```

Two write-flow patterns:

1. **Direct mutation by tools** — `_record_real2sim_state()` /
   `_record_scene_robot_state()` / `_record_scene_generation_state()`
   are called inline by tool handlers when they fire a job, then
   `_save_agent_state()` flushes JSON to disk.
2. **Sync via status poll** — the frontend's monitor polls
   `/scene_robot/status/<job_id>` ~every 1.5 s; the route calls
   `sync_scene_robot_job_to_session()` which mirrors job state into
   `state["runs"]` *and* appends to the audit log via
   `_record_job_audit()`. Because the UI keeps polling for as long as
   the job is "running", the audit log ends up with terminal states
   (`succeeded` / `failed`) for every job — which is what makes the
   plan_history time-traveling rendering work.

The audit is capped at `MAX_JOB_AUDIT_ENTRIES=100` to keep the JSON
file bounded; oldest entries are evicted FIFO.

## 7. Web UI Architecture

The frontend is six classic non-module `<script>` tags loaded in
order ([state.js](app/frontend/assets/js/state.js),
[ui.js](app/frontend/assets/js/ui.js),
[graph.js](app/frontend/assets/js/graph.js),
[sim.js](app/frontend/assets/js/sim.js),
[model.js](app/frontend/assets/js/model.js),
[boot.js](app/frontend/assets/js/boot.js)). Top-level `let`/`const`
declarations are shared across script tags via the script-shared
lexical environment, so functions and state objects defined in one
file are accessible from another at runtime.

Three persistent state objects:

* `runtimeSessionState` — current session/run + initialization
  promise; persisted to `localStorage`.
* `latestSessionState` — last `session_state` response from any
  endpoint; consulted by `planStepLiveStatus()` to color plan step
  rows by *current* job status (so a step shows green ✓ as soon as
  its long job finishes, without manual page refresh).
* `planEditState` — current plan + edit-mode draft + reflection
  cache.

The plan step row renderer (`buildPlanStepRowDom`) tags each `<li>`
with `data-step-id` and `data-visual-status`, plus a CSS `transition:
color 0.4s ease`. When the monitor poll updates the live cache,
`updatePlanStepVisualsInPlace()` walks existing DOM nodes and only
modifies color + glyph, triggering the transition animation rather
than a full DOM rebuild.

`localStorage` also stores recent prompts (`scene_ui_recent_plan_prompts`,
last 5 deduped) and named templates
(`scene_ui_named_plan_templates`, max 20) for one-click re-fire.

## 8. Limitations & Known Gaps

* **No automated tests** beyond a single runtime-layout smoke test.
  The three large refactor PRs (scene_robot service, train/eval
  stages, single-mode retirement) all merged on manual review only.
* **`scene_robot` per-run state is single-slot**: convert / train /
  eval all write into the same `state["runs"][...]["scene_robot"]`
  field. Concurrent stages (rare in practice) would clobber each
  other in the UI even though the audit log preserves each job
  separately.
* **No job kill switch**: started jobs run to completion. A runaway
  Isaac Sim collect holds the GPU until the operator drops to a shell
  and `kill -9`s the PID. Adding `cancel_job()` is one of the next
  planned items.
* **No service health indicators**: the scene_service (port 8001)
  and predict_stream_server (port 8002) can be down at request time
  with no prior warning to the operator beyond a 502 toast.
* **localStorage-bound user state**: named templates and recent
  prompts don't follow the user across machines. Plan history is
  per-session, so opening a session_id from a different browser
  recovers the data, but the operator has to know the id.
* **Loose coupling has a debug cost**: when something fails halfway
  through a pipeline, the operator triages by reading the relevant
  per-stage log file. Centralized observability (e.g. structured
  events emitted by every subprocess) would help but is unimplemented.
* **`pipelines/` is glue, not models**: SAM3, SAM 3D Objects, Isaac
  Lab, and LeRobot are external projects pinned to specific commits.
  This system contributes orchestration, the scene-graph LLM editor,
  the 4-stage robot data flow, and the agent layer — not novel ML.

## 9. References

Code locations referenced throughout:

* Scene graph LLM editor: [app/backend/services/openai_service.py](app/backend/services/openai_service.py)
* Real2Sim driver: [pipelines/real2sim/](pipelines/real2sim/), orchestrated by [app/backend/services/pipeline_service.py](app/backend/services/pipeline_service.py)
* Isaac scene service: [app/backend/services/scene_service.py](app/backend/services/scene_service.py), USD assembly in [pipelines/isaac/](pipelines/isaac/)
* scene_robot package: [scene_robot/](scene_robot/), 4-stage runner in [app/backend/services/scene_robot_service.py](app/backend/services/scene_robot_service.py)
* Agent dispatch: [app/backend/services/agent_loop.py](app/backend/services/agent_loop.py) (loop mode) and [app/backend/services/agent_planner.py](app/backend/services/agent_planner.py) (plan mode)
* Shared state machinery: [app/backend/services/agent_service.py](app/backend/services/agent_service.py)
* Frontend: [app/frontend/index.html](app/frontend/index.html) + six JS files in [app/frontend/assets/js/](app/frontend/assets/js/)
