# Real2Sim Step-by-Step Test

这个目录用于把当前 Real2Sim 拆成可单独执行的步骤，便于定位“卡在哪一步”。

## 0) 环境

在项目根目录执行（推荐）：

```bash
conda activate sam3
cd /home/lbw/3dgen-project/scene_graph_ui_test
```

## 1) Step1: 分割 (生成 masks)

```bash
bash test/real2sim_step_by_step/01_segment.sh
```

预期输出：

- `option2_pipeline/runtime/masks/image.png`
- `option2_pipeline/runtime/masks/*.png`（对象 mask）

## 2) Step2: 发送 masks 到 predict_stream (生成 GLB)

```bash
bash test/real2sim_step_by_step/02_predict_stream.sh
```

预期输出：

- `option2_pipeline/runtime/scene_results/objects/*.glb`
- `option2_pipeline/runtime/scene_results/scene_merged.glb`
- `option2_pipeline/runtime/scene_results/poses.json`

## 3) Step3: 快速检查产物计数

```bash
bash test/real2sim_step_by_step/03_check_outputs.sh
```

## 常用覆盖参数

你可以临时覆盖环境变量（只影响当前命令）：

```bash
IMAGE_PATH=/abs/path/to/input.jpg \
SCENE_GRAPH_PATH=/abs/path/to/full_scene_graph.json \
PREDICT_SERVER=http://128.2.204.110:8000 \
bash test/real2sim_step_by_step/01_segment.sh
```

支持变量：

- `PYTHON_BIN` (默认 `python`)
- `IMAGE_PATH` (默认 `web/assets/uploads/latest_input.jpg`)
- `SCENE_GRAPH_PATH` (默认 `isaac_local/my_viewer/full_scene_graph.json`)
- `MASK_DIR` (默认 `option2_pipeline/runtime/masks`)
- `MESH_DIR` (默认 `option2_pipeline/runtime/meshes`)
- `REUSE_MESH_DIR` (默认 `option2_pipeline/runtime/meshes`)
- `PREDICT_SERVER` (默认 `http://128.2.204.110:8000`)
- `SCENE_RESULTS_DIR` (默认 `option2_pipeline/runtime/scene_results`)

