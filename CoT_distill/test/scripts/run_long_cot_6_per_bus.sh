#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TEST_DIR}/../.." && pwd)"
CODEX_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/camel/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/CoT_distill/config/long_cot_distill.yaml}"
QA_DATA_PATH="${QA_DATA_PATH:-${CODEX_ROOT}/RL4DistReconfig-eval/Datasets/Processed_jsonl/qa_reconfig_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEST_DIR}/outputs}"
PER_BUS="${PER_BUS:-6}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_WORKERS="${MAX_WORKERS:-2}"
REASON_MODEL_NAME="${REASON_MODEL_NAME:-gpt-5.5-none}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${QA_DATA_PATH}" ]]; then
  echo "Missing qa_data_path: ${QA_DATA_PATH}" >&2
  echo "Sync it from H cluster first or set QA_DATA_PATH." >&2
  exit 2
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/select_6_per_bus.py" \
  --input "${QA_DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --per-bus "${PER_BUS}" \
  --seed "${SEED}"

INDICES="$(cat "${OUTPUT_DIR}/long_cot_test_indices_6_per_bus.txt")"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" CoT_distill/long_CoT_distill.py \
  --config "${CONFIG_PATH}" \
  --qa_data_path "${QA_DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --indices "${INDICES}" \
  --reason_model_name "${REASON_MODEL_NAME}" \
  --few-shot-num-samples 2 \
  --few-shot-seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --max-workers "${MAX_WORKERS}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_long_cot_test_output.py" \
  --output-dir "${OUTPUT_DIR}" \
  --per-bus "${PER_BUS}"
