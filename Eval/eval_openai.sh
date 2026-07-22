#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Evaluate a closed-source model via OpenAI-compatible API.
# Set API_BASE/API_KEY, PJ_API_BASE/PJ_API_KEY, or MY_BASE_URL/MY_API_KEY in .env before running.
#
# Usage:
#   bash Eval/eval_openai.sh

DATA_PATH="${DATA_PATH:-Dataset/Processed_xml/train_33_69_84_nodes_full_xml.csv}"
MODEL_NAME="${MODEL_NAME:-${API_MODEL:-gpt-4o}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${MODEL_NAME//\//_}__test_full}"
SPLIT="${SPLIT:-test}"
PROMPT_FORMAT="${PROMPT_FORMAT:-llama3_chat}"
OUTPUT_XML_FORMAT="${OUTPUT_XML_FORMAT:-full_xml}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
SAMPLE_MODE="${SAMPLE_MODE:-first}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SAVE_HARD_SAMPLES="${SAVE_HARD_SAMPLES:-1}"
CONCURRENCY="${CONCURRENCY:-8}"
TIMEOUT="${TIMEOUT:-120}"
REASONING_EFFORT="${REASONING_EFFORT:-none}"
STREAM="${STREAM:-1}"
FEW_SHOT="${FEW_SHOT:-}"
FEW_SHOT_NUM="${FEW_SHOT_NUM:--1}"
FEW_SHOT_SEED="${FEW_SHOT_SEED:-42}"
API_ENV="${API_ENV:-PJ_API}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cmd=(
  "${PYTHON_BIN}" Eval/eval_openai.py
  --data_path "${DATA_PATH}"
  --model_name "${MODEL_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --prompt_format "${PROMPT_FORMAT}"
  --output_xml_format "${OUTPUT_XML_FORMAT}"
  --num_samples "${NUM_SAMPLES}"
  --sample_mode "${SAMPLE_MODE}"
  --seed "${SEED}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --save_hard_samples "${SAVE_HARD_SAMPLES}"
  --concurrency "${CONCURRENCY}"
  --timeout "${TIMEOUT}"
  --reasoning_effort "${REASONING_EFFORT}"
  --stream "${STREAM}"
  --few_shot_num "${FEW_SHOT_NUM}"
  --few_shot_seed "${FEW_SHOT_SEED}"
  --api_env "${API_ENV}"
)

if [[ -n "${FEW_SHOT}" ]]; then
  cmd+=(--few_shot "${FEW_SHOT}")
fi

"${cmd[@]}"
