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

1. `object_segmentation_pipeline.py`
   - reads the latest input image and scene graph
   - derives object prompts from the scene graph
   - writes masks to `runtime/real2sim/masks`
   - optionally reuses mesh files

2. `streaming_generation_client.py`
   - sends `image.png` plus masks to `/predict_stream`
   - streams back per-object GLBs, merged scene GLB, and poses JSON
   - writes outputs to `runtime/real2sim/scene_results`

`scene_generation` now owns the binary streaming inference server entrypoint
too, while still importing the actual SAM 3D inference stack from
`third_party/sam-3d-objects`.

The old top-view / relative-xy / arrange-from-csv workflow is no longer part of the repo.

## Directory Layout

- active step1: `object_segmentation_pipeline.py`
- active step2: `streaming_generation_client.py`
- local server: `predict_stream_server.py`
- runtime: `runtime/real2sim/`

Default active output directories:

- masks: `runtime/real2sim/masks`
- meshes: `runtime/real2sim/meshes`
- scene results: `runtime/real2sim/scene_results`

---

## Step1 Standalone Test

```bash
cd /home/lbw/3dgen-project/scene_graph_ui_test
python pipelines/real2sim/object_segmentation_pipeline.py \
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
python pipelines/real2sim/streaming_generation_client.py \
  --server http://iclspiderman.ri.cmu.edu:8000
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

## Minimal Integration Split

Keep the integration split into two layers:

### Submodule: `third_party/sam-3d-objects`

These are the minimal code changes that belong in the `sam-3d-objects`
checkout and are imported by the in-repo server:

- `notebook/inference.py`
  - forwards extra inference flags from the server layer into the pointmap pipeline
- `sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py`
  - adds safe fallbacks for empty / non-finite mask points
- `sam3d_objects/pipeline/inference_pipeline_pointmap.py`
  - enables manual alignment during layout post-optimization
- `sam3d_objects/pipeline/layout_post_optimization_utils.py`
  - filters alignment outliers before optimization
- `pipelines/real2sim/server.py`
  - original reference implementation the main repo server was derived from
- `reconstruct_scene_from_masks.py`
  - standalone reconstruction helper for local debugging

### Main Repo: `scene_generation`

These stay in the app repo:

- `pipelines/real2sim/object_segmentation_pipeline.py`
  - prepares `image.png` and mask PNGs from the uploaded scene
- `pipelines/real2sim/streaming_generation_client.py`
  - posts masks to `/predict_stream`, stores GLBs / poses, runs postprocess + USD conversion
- `pipelines/real2sim/predict_stream_server.py`
  - FastAPI `/predict_stream` server that imports `Inference` and related modules from `third_party/sam-3d-objects`
- `pipelines/real2sim/sam3d_bootstrap.py`
  - lightweight path/bootstrap helpers for locating the third-party SAM 3D checkout

### Local Server Launch

If the submodule is initialized and includes the expected SAM 3D files, you can
launch the local predict server from the main repo with:

```bash
python pipelines/real2sim/predict_stream_server.py
```

By default it imports from `third_party/sam-3d-objects`. You can point to a
different checkout with:

```bash
SAM3D_OBJECTS_ROOT=/home/boweili/sam-3d-objects \
python pipelines/real2sim/predict_stream_server.py
```
