#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
IMAGE_PATH="${IMAGE_PATH:-$ROOT_DIR/runtime/uploads/latest_input.jpg}"
SCENE_GRAPH_PATH="${SCENE_GRAPH_PATH:-$ROOT_DIR/runtime/scene_graph/current_scene_graph.json}"
MASK_DIR="${MASK_DIR:-$ROOT_DIR/runtime/real2sim/masks}"
MESH_DIR="${MESH_DIR:-$ROOT_DIR/runtime/real2sim/meshes}"
REUSE_MESH_DIR="${REUSE_MESH_DIR:-$ROOT_DIR/runtime/real2sim/meshes}"

mkdir -p "$MASK_DIR" "$MESH_DIR"

echo "[STEP1] segment objects"
echo "python: $PYTHON_BIN"
echo "image:  $IMAGE_PATH"
echo "graph:  $SCENE_GRAPH_PATH"
echo "masks:  $MASK_DIR"

"$PYTHON_BIN" option2_pipeline/step1_segment_scene60_all_objects.py \
  --image "$IMAGE_PATH" \
  --scene-graph "$SCENE_GRAPH_PATH" \
  --output-root "$MASK_DIR" \
  --mesh-output-dir "$MESH_DIR" \
  --reuse-mesh-dir "$REUSE_MESH_DIR" \
  --skip-sam3d

echo "[STEP1 DONE] generated masks in $MASK_DIR"
