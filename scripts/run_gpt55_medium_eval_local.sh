#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export API_ENV="${API_ENV:-MY}"
export MODEL_NAME="${MODEL_NAME:-gpt-5.5}"
export DATA_PATH="${DATA_PATH:-Dataset/eval_subset_1000.csv}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/gpt55_my_medium_eval_subset_1000}"
export NUM_SAMPLES="${NUM_SAMPLES:-1000}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
export CONCURRENCY="${CONCURRENCY:-8}"
export TIMEOUT="${TIMEOUT:-900}"
export REASONING_EFFORT="${REASONING_EFFORT:-medium}"
export STREAM="${STREAM:-0}"

mkdir -p logs "${OUTPUT_DIR}"
log="${LOG_PATH:-logs/gpt55_medium_my_eval_subset_1000_$(date +%Y%m%d_%H%M%S).log}"
echo "${log}" > "${OUTPUT_DIR}/log_path.txt"

echo "Writing log to ${log}"
echo "Model=${MODEL_NAME} API_ENV=${API_ENV} samples=${NUM_SAMPLES} concurrency=${CONCURRENCY}"

bash Eval/eval_openai.sh 2>&1 | tee "${log}"
