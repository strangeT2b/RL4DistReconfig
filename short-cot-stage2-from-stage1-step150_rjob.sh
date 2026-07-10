#!/usr/bin/env bash
set -euxo pipefail

REPO=/mnt/shared-storage-gpfs2/ai4energy/gouzitang/RL4DistReconfig
ENV=/mnt/shared-storage-gpfs2/ai4energy/gouzitang/envs/verl041_gzt/bin/activate

BASE_MODEL=runs/llama31_8b_instruct/merged/llama31_short_cot_sft_ckpt4710
STAGE1_ADAPTER=runs/llama31_8b_instruct/short-cot-stage1-constraint/global_step_150/actor/lora_adapter
MERGED_STAGE1=runs/llama31_8b_instruct/merged/llama31_short_cot_sft_ckpt4710__stage1_constraint_step150
VERL_DATA_DIR=Dataset/verl/train_33_69_84_nodes_full_xml_clean_shuf42
RUN_NAME=short-cot-stage2-from-stage1-step150

cd "${REPO}"
source "${ENV}"
if [[ -f "${REPO}/.env" ]]; then
  set -a
  source "${REPO}/.env"
  set +a
fi

export HYDRA_FULL_ERROR=1
export REWARD_NORMALIZE_PENALTIES=true

date
hostname
whoami
pwd
nvidia-smi || true

for p in \
  "${BASE_MODEL}/config.json" \
  "${BASE_MODEL}/tokenizer.json" \
  "${STAGE1_ADAPTER}/adapter_config.json" \
  "${STAGE1_ADAPTER}/adapter_model.safetensors" \
  "${VERL_DATA_DIR}/train.parquet" \
  "${VERL_DATA_DIR}/validation.parquet" \
  SFT/merge_adapter.sh \
  SFT/merge_adapter.py \
  RL/verl/train_verl_grpo.sh \
  RL/verl/verl_reward.py
do
  if [[ ! -e "${p}" ]]; then
    echo "MISSING: ${p}" >&2
    exit 2
  fi
done

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  echo "CHECK_ONLY=1: preflight passed."
  exit 0
fi

if [[ ! -f "${MERGED_STAGE1}/config.json" ]]; then
  rm -rf "${MERGED_STAGE1}.tmp"
  mkdir -p "$(dirname "${MERGED_STAGE1}")"
  BASE_MODEL="${BASE_MODEL}" \
  ADAPTER_PATH="${STAGE1_ADAPTER}" \
  OUTPUT_DIR="${MERGED_STAGE1}.tmp" \
  DTYPE=bfloat16 \
  DEVICE_MAP=auto \
  MAX_SHARD_SIZE=5GB \
  bash SFT/merge_adapter.sh
  rm -rf "${MERGED_STAGE1}"
  mv "${MERGED_STAGE1}.tmp" "${MERGED_STAGE1}"
else
  echo "Merged stage1 model already exists: ${MERGED_STAGE1}"
fi

du -sh "${MERGED_STAGE1}" "${VERL_DATA_DIR}" || true

CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH="${MERGED_STAGE1}" \
VERL_DATA_DIR="${VERL_DATA_DIR}" \
VAL_FILES="${VERL_DATA_DIR}/validation.parquet" \
PREPARE_DATA=0 \
OUTPUT_ROOT=runs/llama31_8b_instruct \
RUN_NAME="${RUN_NAME}" \
REWARD_FUNCTION_NAME=compute_score_full_xml_valid_and_iou \
MAX_STEPS=300 \
SAVE_FREQ=50 \
TEST_FREQ=50 \
TRAIN_BATCH_SIZE=16 \
GEN_BATCH_SIZE=32 \
VAL_BATCH_SIZE=64 \
NUM_GENERATIONS=8 \
PPO_MINI_BATCH_SIZE=16 \
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
N_GPUS_PER_NODE=2 \
MAX_PROMPT_LENGTH=4096 \
MAX_RESPONSE_LENGTH=4096 \
LEARNING_RATE=5e-6 \
USE_KL_LOSS=True \
KL_LOSS_COEF=0.001 \
VLLM_GPU_MEMORY_UTILIZATION=0.6 \
LOGGER='[console]' \
bash RL/verl/train_verl_grpo.sh

date
