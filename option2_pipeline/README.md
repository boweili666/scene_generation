# Real2Sim Option2 Pipeline (Unified)

This directory is used for step-by-step execution of Real2Sim option2 in `scene_graph_ui_test`.
All commands below assume you run them from `/home/lbw/3dgen-project/scene_graph_ui_test`.

## Environment Setup

```bash
conda activate sam3
```

Recommended dependencies (at least required by step3/step4):

```bash
pip install torch transformers pillow numpy pyrender trimesh scikit-image PyOpenGL pyglet
```

If you want to run step2 (OpenAI image editing):

```bash
export OPENAI_API_KEY=your_key
```

## Directory Layout and Default Inputs/Outputs

- step1: `step1_segment_scene60_all_objects.py`
- step2: `step2_generate_top_view.py`
- step3: `step3_sam3_relative_xy.py`
- step4: `step4_arrange_from_csv.py`
- runtime: `scene_graph_ui_test/option2_pipeline/runtime/`

Default step4 input directory (rts/json + reference images):
- `option2_pipeline/runtime`
Default mesh output directory (generated/copied by step1):
- `option2_pipeline/runtime/meshes`
Default `reuse_mesh_dir`:
- `option2_pipeline/runtime/meshes` (so by default no copy from an external mesh directory)

---

## Step1 Standalone Test (Segmentation)

### Use default prompts (recommended first test)

```bash
cd /home/lbw/3dgen-project/scene_graph_ui_test
python option2_pipeline/step1_segment_scene60_all_objects.py \
  --skip-sam3d
```

### Use scene-graph-derived prompts

```bash
python option2_pipeline/step1_segment_scene60_all_objects.py \
  --scene-graph /home/lbw/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/bowei/my_viewer/full_scene_graph.json \
  --skip-sam3d
```

### Success Criteria

- Terminal shows `[SEGMENT] ...` and `[DONE] Total saved masks`.
- Output directory exists: `scene_graph_ui_test/option2_pipeline/runtime/masks/`.

---

## Step2 Standalone Test (Top View)

```bash
python option2_pipeline/step2_generate_top_view.py \
  --input ../sam3/scene_60.jpeg \
  --output option2_pipeline/runtime/scene_60_top_view.png
```

### Success Criteria

- Output file exists: `scene_graph_ui_test/option2_pipeline/runtime/scene_60_top_view.png`.

---

## Step3 Standalone Test (Relative Position CSV)

```bash
python option2_pipeline/step3_sam3_relative_xy.py \
  --image option2_pipeline/runtime/scene_60_top_view.png \
  --prompts table "desk lamp" "alarm clock" notebook pen "glass cup" \
  --reference table \
  --output option2_pipeline/runtime/sam3_bbox_relative.csv
```

### Success Criteria

- Terminal prints `[DONE] csv saved to ...`.
- CSV exists: `scene_graph_ui_test/option2_pipeline/runtime/sam3_bbox_relative.csv`.

---

## Step4 Standalone Test (Assemble Scene GLB)

```bash
python option2_pipeline/step4_arrange_from_csv.py \
  --input-dir option2_pipeline/runtime \
  --mesh-dir option2_pipeline/runtime/meshes \
  --csv-path option2_pipeline/runtime/sam3_bbox_relative.csv \
  --output-glb option2_pipeline/runtime/scene_from_csv_yaw.glb \
  --output-json option2_pipeline/runtime/scene_from_csv_yaw_transforms.json
```

### Success Criteria

- Terminal shows object progress logs: `[progress] [i/N] ...`.
- Final output contains: `Saved merged GLB`, `Saved transforms`.
- Files exist:
  - `option2_pipeline/runtime/scene_from_csv_yaw.glb`
  - `option2_pipeline/runtime/scene_from_csv_yaw_transforms.json`

---

## Common Issues

- `OSError ... gated repo facebook/sam3`: Hugging Face permission/token issue. Run `hf auth login` and ensure your account has access to that model.
- `OPENAI_API_KEY is not set`: run `export OPENAI_API_KEY=...` before step2.
- `Missing dependency 'pyrender'`: install `pyrender` (and `PyOpenGL`, `pyglet`).
