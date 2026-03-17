# Real2Sim Pipeline

This directory now reflects the current Real2Sim logic actually used by the app.
All commands below assume you run them from `/home/lbw/3dgen-project/scene_graph_ui_test`.

## Environment Setup

```bash
conda activate sam3
```

Recommended dependencies:

```bash
pip install torch transformers pillow numpy requests
```

## Current Flow

The active pipeline is only two executable steps:

1. `segment_objects.py`
   - reads the latest input image and scene graph
   - derives object prompts from the scene graph
   - writes masks to `runtime/real2sim/masks`
   - optionally reuses mesh files

2. `predict_stream_client.py`
   - sends `image.png` plus masks to `/predict_stream`
   - streams back per-object GLBs, merged scene GLB, and poses JSON
   - writes outputs to `runtime/real2sim/scene_results`

The old top-view / relative-xy / arrange-from-csv workflow is no longer part of the repo.

## Directory Layout

- active step1: `segment_objects.py`
- active step2: `predict_stream_client.py`
- runtime: `runtime/real2sim/`

Default active output directories:

- masks: `runtime/real2sim/masks`
- meshes: `runtime/real2sim/meshes`
- scene results: `runtime/real2sim/scene_results`

---

## Step1 Standalone Test

```bash
cd /home/lbw/3dgen-project/scene_graph_ui_test
python pipelines/real2sim/segment_objects.py \
  --image runtime/uploads/latest_input.jpg \
  --scene-graph runtime/scene_graph/current_scene_graph.json \
  --output-root runtime/real2sim/masks \
  --mesh-output-dir runtime/real2sim/meshes \
  --reuse-mesh-dir runtime/real2sim/meshes \
  --skip-sam3d
```

Success criteria:

- terminal shows `[SEGMENT] ...` and `[DONE] Total saved masks`
- `runtime/real2sim/masks/image.png` exists
- `runtime/real2sim/masks/*.png` object masks exist

---

## Step2 Standalone Test

```bash
python pipelines/real2sim/predict_stream_client.py \
  --server http://128.2.204.110:8000 \
  --image runtime/real2sim/masks/image.png \
  --mask-dir runtime/real2sim/masks \
  --output-dir runtime/real2sim/scene_results
```

Success criteria:

- terminal shows streamed saves for `object`, `scene`, and `poses`
- `runtime/real2sim/scene_results/objects/*.glb` exists
- `runtime/real2sim/scene_results/scene_merged.glb` exists
- `runtime/real2sim/scene_results/poses.json` exists

---

## Common Issues

- `OSError ... gated repo facebook/sam3`: Hugging Face permission/token issue. Run `hf auth login` and ensure your account has access to that model.
- `Connection refused` on `predict_stream`: remote predict service is not listening or the configured server address is wrong.
- `No object prompts found in scene graph`: regenerate the scene graph first.
