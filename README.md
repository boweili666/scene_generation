# Scene Graph UI Test

This repo contains the scene graph web UI, the Real2Sim pipeline, and the Isaac scene service integration.

## Project Structure

- `app/backend`: Flask app, API routes, config, and service orchestration
- `app/frontend`: active frontend HTML, JS, CSS, and local vendor assets
- `pipelines/real2sim`: active Real2Sim scripts
- `pipelines/isaac`: Isaac helper scripts and static USD assets
- `runtime`: mutable runtime outputs
- `logs`: runtime log files
- `tests`: automated smoke tests
- `docs`: architecture and runtime layout notes

## Entry Points

- Web app: `python -m app.backend.app`
- Scene service: `python -m app.backend.services.scene_service`

## Environment Setup

The recorded setup flow for the `scene_gen` environment is:

```bash
conda create -n scene_gen python=3.11 -y
conda activate scene_gen
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
cd third_party/IsaacLab/
conda install -n scene_gen -c conda-forge cmake=3.27.9
conda run -n scene_gen pip install egl-probe==1.0.2
./isaaclab.sh --install
conda install -n scene_gen -c conda-forge casadi=3.7.0
conda run -n scene_gen pip install \
  openai==2.21.0 \
  pin==2.7.0
conda run -n scene_gen pip install \
  transformers==5.2.0 \
  huggingface-hub==1.4.1 \
  hf-xet==1.2.0 \
  tokenizers==0.22.2 \
  safetensors==0.7.0
pip install flask
```

After `cd third_party/IsaacLab/`, the remaining commands above are intended to be run from that directory unless noted otherwise.

## Runtime Layout

- `runtime/uploads`: uploaded reference images
- `runtime/renders`: rendered preview PNGs
- `runtime/scene_graph`: current scene graph JSON
- `runtime/real2sim`: masks, meshes, GLBs, poses, and pipeline outputs
- `runtime/scene_service/usd`: generated USD scene files
- `runtime/scene_service/placements`: persisted placement JSON
- `logs/real2sim.log`: Real2Sim streaming log
- `logs/scene_service.log`: scene service log

## Real2Sim

Active Real2Sim flow:

1. `python pipelines/real2sim/object_segmentation_pipeline.py`
2. `python pipelines/real2sim/streaming_generation_client.py`

The web UI uses the async endpoints:

- `POST /real2sim/start`
- `GET /real2sim/status/<job_id>`
- `GET /real2sim/log`

## Unified Instruction Flow

The main text input now uses a single unified endpoint:

- `POST /apply_instruction`

The backend routes each instruction through:

1. an LLM router
2. a backend validator
3. separate scene-graph and placement editors

Scene graph and placements remain stored separately under `runtime/scene_graph` and
`runtime/scene_service/placements`.

The Scene Graph panel also supports direct graph editing:

- double-click background to add a node
- double-click a node or edge to edit it
- shift-click two nodes to create/update a relation
- right-click or press Delete/Backspace to remove the selected node/edge

Direct graph edits are saved back to `runtime/scene_graph/current_scene_graph.json`.

## Development

Start the web app:

```bash
python -m app.backend.app
```

Start the scene service in a second shell:

```bash
python -m app.backend.services.scene_service
```

## Remote Web Access

If the services run on the remote server `boweili@iclspiderman.ri.cmu.edu`, use SSH port forwarding from your local machine:

```bash
ssh -L 8000:127.0.0.1:8000 boweili@iclspiderman.ri.cmu.edu
```

Then start the service on the remote machine, for example:

```bash
python -m app.backend.app
```

or:

```bash
uvicorn app.backend.services.scene_service:app --host 127.0.0.1 --port 8000
```

Once the SSH tunnel is active, open this URL in your local browser:

```text
http://127.0.0.1:8000
```

Run the automated smoke test:

```bash
python -m unittest tests.test_runtime_layout
```
