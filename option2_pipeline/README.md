# Real2Sim Option2 Pipeline (Unified)

本目录用于 `scene_graph_ui_test` 的 Real2Sim option2 分步执行。

## 环境准备

```bash
conda activate sam3
```

建议依赖（至少 step3/4 需要）：

```bash
pip install torch transformers pillow numpy pyrender trimesh scikit-image PyOpenGL pyglet
```

如果需要 step2（OpenAI 图像编辑）：

```bash
export OPENAI_API_KEY=你的key
```

## 目录与默认输入输出

- step1: `step1_segment_scene60_all_objects.py`
- step2: `step2_generate_top_view.py`
- step3: `step3_sam3_relative_xy.py`
- step4: `step4_arrange_from_csv.py`
- runtime: `scene_graph_ui_test/option2_pipeline/runtime/`

默认 mesh 目录：

- `/home/lbw/3dgen-project/scene60_mesh_rts`

---

## Step1 单独测试（分割）

### 用默认 prompts（推荐先测这个）

```bash
python /home/lbw/3dgen-project/scene_graph_ui_test/option2_pipeline/step1_segment_scene60_all_objects.py \
  --skip-sam3d
```

### 用 scene graph 自动 prompts

```bash
python /home/lbw/3dgen-project/scene_graph_ui_test/option2_pipeline/step1_segment_scene60_all_objects.py \
  --scene-graph /home/lbw/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/bowei/my_viewer/full_scene_graph.json \
  --skip-sam3d
```

### 成功判据

- 终端有 `[SEGMENT] ...` 和 `[DONE] Total saved masks`。
- 生成目录：`scene_graph_ui_test/option2_pipeline/runtime/masks/`。

---

## Step2 单独测试（俯视图）

```bash
python /home/lbw/3dgen-project/scene_graph_ui_test/option2_pipeline/step2_generate_top_view.py \
  --input /home/lbw/3dgen-project/sam3/scene_60.jpeg \
  --output /home/lbw/3dgen-project/scene_graph_ui_test/option2_pipeline/runtime/scene_60_top_view.png
```

### 成功判据

- 输出文件存在：`scene_graph_ui_test/option2_pipeline/runtime/scene_60_top_view.png`。

---

## Step3 单独测试（相对位置 CSV）

```bash
python /home/lbw/3dgen-project/scene_graph_ui_test/option2_pipeline/step3_sam3_relative_xy.py \
  --image /home/lbw/3dgen-project/scene_graph_ui_test/option2_pipeline/runtime/scene_60_top_view.png \
  --prompts table "desk lamp" "alarm clock" notebook pen "glass cup" \
  --reference table \
  --output /home/lbw/3dgen-project/scene_graph_ui_test/option2_pipeline/runtime/sam3_bbox_relative.csv
```

### 成功判据

- 终端输出 `[DONE] csv saved to ...`。
- CSV 存在：`scene_graph_ui_test/option2_pipeline/runtime/sam3_bbox_relative.csv`。

---

## Step4 单独测试（拼场景 GLB）

```bash
python /home/lbw/3dgen-project/scene_graph_ui_test/option2_pipeline/step4_arrange_from_csv.py \
  --input-dir /home/lbw/3dgen-project/scene60_mesh_rts \
  --csv-path /home/lbw/3dgen-project/scene_graph_ui_test/option2_pipeline/runtime/sam3_bbox_relative.csv \
  --output-glb /home/lbw/3dgen-project/scene60_mesh_rts/scene_from_csv_yaw.glb \
  --output-json /home/lbw/3dgen-project/scene60_mesh_rts/scene_from_csv_yaw_transforms.json
```

### 成功判据

- 终端有 object 进度日志：`[progress] [i/N] ...`。
- 最后输出：`Saved merged GLB`、`Saved transforms`。
- 文件存在：
  - `/home/lbw/3dgen-project/scene60_mesh_rts/scene_from_csv_yaw.glb`
  - `/home/lbw/3dgen-project/scene60_mesh_rts/scene_from_csv_yaw_transforms.json`

---

## 常见问题

- `OSError ... gated repo facebook/sam3`：HF 权限或 token 问题，先 `hf auth login` 并确认账号已获该模型访问权限。
- `OPENAI_API_KEY is not set`：先 `export OPENAI_API_KEY=...` 再跑 step2。
- `Missing dependency 'pyrender'`：安装 `pyrender`（以及 `PyOpenGL`、`pyglet`）。
