#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Evaluate a closed-source model via OpenAI-compatible API.
# Set API_BASE and API_KEY in .env before running.
#
# Usage:
#   bash Eval/eval_openai.sh
#
# Override defaults:
#   MODEL_NAME=gpt-4o DATA_PATH=... OUTPUT_DIR=... bash Eval/eval_openai.sh

DATA_PATH="${DATA_PATH:-Dataset/Processed_xml/train_33_69_84_nodes_full_xml.csv}"
MODEL_NAME="${MODEL_NAME:-${API_MODEL:-gpt-4o}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${MODEL_NAME//\//_}__test_full}"
SPLIT="${SPLIT:-test}"
PROMPT_FORMAT="${PROMPT_FORMAT:-llama3_chat}"
OUTPUT_XML_FORMAT="${OUTPUT_XML_FORMAT:-full_xml}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1200}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SAVE_HARD_SAMPLES="${SAVE_HARD_SAMPLES:-1}"

python Eval/eval_openai.py \
  --data_path "${DATA_PATH}" \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --prompt_format "${PROMPT_FORMAT}" \
  --output_xml_format "${OUTPUT_XML_FORMAT}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --save_hard_samples "${SAVE_HARD_SAMPLES}"
