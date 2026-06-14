#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/env_cache_disk2.sh

# Merge a selected LoRA adapter into a full model for veRL or standalone inference.
# Usage:
#   bash SFT/merge_adapter.sh
#
# Override defaults with environment variables, for example:
#   BASE_MODEL="${LLAMA31_INSTRUCT_MODEL}" \
#   ADAPTER_PATH=/mnt/disk2/gzt/RL4DistReconfig/runs/llama31_8b_instruct/sft_ce_ep3/checkpoint-1095 \
#   OUTPUT_DIR=runs/llama31_8b_instruct/merged/sft_ce_v1 \
#   bash SFT/merge_adapter.sh

BASE_MODEL="${BASE_MODEL:-${LLAMA31_INSTRUCT_MODEL:-../models/meta-llama/Llama-3.1-8B-Instruct}}"
ADAPTER_PATH="${ADAPTER_PATH:-runs/llama31_8b_instruct/sft_ce_ep3/checkpoint-1095}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/llama31_8b_instruct/merged/sft_ce_ep3_1}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"

python SFT/merge_adapter.py \
  --base_model "${BASE_MODEL}" \
  --adapter_path "${ADAPTER_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --dtype "${DTYPE}" \
  --device_map "${DEVICE_MAP}" \
  --max_shard_size "${MAX_SHARD_SIZE}"
