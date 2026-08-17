#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/render_potted_plant_multiview.sh [GPU_ID]
GPU_ID="${1:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/potted_plant_multiview}"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$PROJECT_ROOT/render_multiview.py" \
  --config_path "$PROJECT_ROOT/configs/render.json" \
  --source_path "$PROJECT_ROOT/assets/potted_plant" \
  --model_path "$OUTPUT_DIR" \
  --gs_ply "$PROJECT_ROOT/assets/potted_plant/model.ply" \
  --radius_set 1.5 \
  --elevation_set 20.0 \
  "${@:2}"
