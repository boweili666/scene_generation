#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MASK_DIR="${MASK_DIR:-$ROOT_DIR/runtime/real2sim/masks}"
IMAGE_PNG="${IMAGE_PNG:-$MASK_DIR/image.png}"
PREDICT_SERVER="${PREDICT_SERVER:-http://128.2.204.110:8000}"
SCENE_RESULTS_DIR="${SCENE_RESULTS_DIR:-$ROOT_DIR/runtime/real2sim/scene_results}"

mkdir -p "$SCENE_RESULTS_DIR"

if [[ ! -f "$IMAGE_PNG" ]]; then
  echo "[ERROR] image png not found: $IMAGE_PNG"
  echo "Run step1 first: bash test/real2sim_step_by_step/01_segment.sh"
  exit 1
fi

echo "[STEP2] predict_stream generate glbs"
echo "python: $PYTHON_BIN"
echo "server: $PREDICT_SERVER"
echo "image:  $IMAGE_PNG"
echo "masks:  $MASK_DIR"
echo "output: $SCENE_RESULTS_DIR"

"$PYTHON_BIN" scripts/send_masks_to_predict.py \
  --server "$PREDICT_SERVER" \
  --image "$IMAGE_PNG" \
  --mask-dir "$MASK_DIR" \
  --output-dir "$SCENE_RESULTS_DIR"

echo "[STEP2 DONE] generated outputs in $SCENE_RESULTS_DIR"
