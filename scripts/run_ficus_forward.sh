#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_ficus_forward.sh [GPU_ID] [extra simulate.py arguments]
GPU_ID="${1:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/ficus}"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$PROJECT_ROOT/simulate.py" \
  --config_path "$PROJECT_ROOT/configs/ficus.json" \
  --source_path "$PROJECT_ROOT/assets/ficus" \
  --model_path "$OUTPUT_DIR" \
  --gs_ply "$PROJECT_ROOT/assets/ficus/model.ply" \
  --white_background \
  --radius_set 4.11 \
  --elevation_set 8.97 \
  --wind_dir 0.0 -0.1 0.0 \
  --wind_scale 100 \
  --render_view 0 25 50 75 12 37 62 87 \
  --frames 100 \
  --wind_frames 50 \
  --mpm_iter_cnt 400 \
  --lbm_steps 20 \
  --warmup_steps 400 \
  "${@:2}"
