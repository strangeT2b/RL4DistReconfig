#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source scripts/env_cache_disk2.sh

# Start the vLLM server used by TRL GRPO server mode.
#
# IMPORTANT:
#   MODEL_ID must match the trainer's --model_id base model.  The initial LoRA
#   adapter is loaded by RL/trl/train_grpo_adapter.py via --init_adapter and synced
#   to this server by TRL; do not point this script at the adapter directory.
#
# Example:
#   CUDA_VISIBLE_DEVICES=2,3 MODEL_ID="${QWEN3_8B_MODEL}" bash RL/trl/serve_vllm_grpo.sh

MODEL_ID=${MODEL_ID:?Set MODEL_ID to the same base model path used by --model_id}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
ENFORCE_EAGER=${ENFORCE_EAGER:-true}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

# vLLM 0.10.x defaults to the V1 engine for many models.  In this GRPO server
# setup the V1 multiprocess engine has been brittle on Qwen3, while V0 is the
# established path used by TRL's weight-sync worker extension.
export VLLM_USE_V1=${VLLM_USE_V1:-0}

trl vllm-serve \
  --model "${MODEL_ID}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --dtype "${VLLM_DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enforce-eager "${ENFORCE_EAGER}"
