#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MASK_DIR="${MASK_DIR:-$ROOT_DIR/runtime/real2sim/masks}"
SCENE_RESULTS_DIR="${SCENE_RESULTS_DIR:-$ROOT_DIR/runtime/real2sim/scene_results}"
OBJECTS_DIR="$SCENE_RESULTS_DIR/objects"
MERGED_GLB="$SCENE_RESULTS_DIR/scene_merged.glb"
POSES_JSON="$SCENE_RESULTS_DIR/poses.json"

mask_count=0
if [[ -d "$MASK_DIR" ]]; then
  mask_count="$(find "$MASK_DIR" -maxdepth 1 -type f -name '*.png' ! -name 'image.png' | wc -l | tr -d ' ')"
fi

obj_glb_count=0
if [[ -d "$OBJECTS_DIR" ]]; then
  obj_glb_count="$(find "$OBJECTS_DIR" -maxdepth 1 -type f -name '*.glb' | wc -l | tr -d ' ')"
fi

echo "=== Real2Sim Step Check ==="
echo "mask_dir         : $MASK_DIR"
echo "scene_results_dir: $SCENE_RESULTS_DIR"
echo "mask_count       : $mask_count"
echo "object_glb_count : $obj_glb_count"

if [[ -f "$MERGED_GLB" ]]; then
  echo "merged_scene_glb : OK ($MERGED_GLB)"
else
  echo "merged_scene_glb : MISSING ($MERGED_GLB)"
fi

if [[ -f "$POSES_JSON" ]]; then
  echo "poses_json       : OK ($POSES_JSON)"
else
  echo "poses_json       : MISSING ($POSES_JSON)"
fi
