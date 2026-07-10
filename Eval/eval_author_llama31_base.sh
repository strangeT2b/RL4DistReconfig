#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/env_cache_disk2.sh

# Evaluate Llama-3.1-8B base with vLLM author-format + system loss.
# Defaults use the merged SFT model; override BASE_MODEL/OUTPUT_DIR/etc. as needed.

DATA_PATH="${DATA_PATH:-Dataset/Processed/train_33_69_84_nodes.csv}"
BASE_MODEL="${BASE_MODEL:-runs/llama31_8b/merged/sft_ce_ep3}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
ADAPTER_NAME="${ADAPTER_NAME:-llama31_8b_sft_ce_merged}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/llama31_8b_sft_ce_merged__on__train_33_69_84_nodes_test}"
SPLIT="${SPLIT:-test}"
PROMPT_FORMAT="${PROMPT_FORMAT:-legacy}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1200}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DTYPE="${DTYPE:-auto}"
SAVE_HARD_SAMPLES="${SAVE_HARD_SAMPLES:-1}"
SIM_LOSS="${SIM_LOSS:-1}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3} python Eval/eval_author_lora.py \
  --data_path "${DATA_PATH}" \
  --base_model "${BASE_MODEL}" \
  --adapter_path "${ADAPTER_PATH}" \
  --adapter_name "${ADAPTER_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --prompt_format "${PROMPT_FORMAT}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --repetition_penalty "${REPETITION_PENALTY}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype "${DTYPE}" \
  --save_hard_samples "${SAVE_HARD_SAMPLES}" \
  --sim_loss "${SIM_LOSS}"
