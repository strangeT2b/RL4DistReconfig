#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Prepare model-agnostic veRL parquet data.  Usage:
#   bash RL/verl/prepare_verl_data.sh

DATA_PATH="${DATA_PATH:-Dataset/Processed/train_33_69_84_nodes.csv}"
VERL_DATA_DIR="${VERL_DATA_DIR:-Dataset/verl/train_33_69_84_nodes}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"

python RL/verl/prepare_verl_data.py \
  --data_path "${DATA_PATH}" \
  --output_dir "${VERL_DATA_DIR}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_val_samples "${MAX_VAL_SAMPLES}"
