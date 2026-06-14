#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# veRL GRPO run starting from the merged Llama-3.1-Instruct SFT checkpoint.
# Usage:
#   bash RL/verl/train_verl_grpo.sh
#
# Prepare parquet data separately with:
#   python RL/verl/prepare_verl_data.py \
#     --data_path Dataset/Processed/train_33_69_84_nodes.csv \
#     --output_dir Dataset/verl/train_33_69_84_nodes
#
# Or set PREPARE_DATA=1 to regenerate parquet before training.
#
# For strict comparison with TRL --init_adapter runs, MODEL_PATH should point
# to a merged SFT model directory.  veRL trains a fresh LoRA via lora_rank and
# lora_alpha; this launcher does not directly load an existing PEFT adapter.

export PYTHONPATH="${PWD}:${VERL_ROOT:-/workspace/verl}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

DATA_PATH="${DATA_PATH:-Dataset/Processed/train_33_69_84_nodes.csv}"
VERL_DATA_DIR="${VERL_DATA_DIR:-Dataset/verl/train_33_69_84_nodes}"
VAL_FILES="${VAL_FILES:-${VERL_DATA_DIR}/validation_64.parquet}"
PREPARE_DATA="${PREPARE_DATA:-0}"
MODEL_PATH="${MODEL_PATH:-runs/llama31_8b_instruct/merged/sft_ce_ep3_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/llama31_8b_instruct}"
RUN_NAME="${RUN_NAME:-grpo_verl_from_sft_ce_ep3_1}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-200}"
TEST_FREQ="${TEST_FREQ:-100}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
USE_KL_LOSS="${USE_KL_LOSS:-True}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.2}"
CLIP_RATIO_C="${CLIP_RATIO_C:-3.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
LOGGER="${LOGGER:-[console,wandb]}"
REWARD_FUNCTION_NAME="${REWARD_FUNCTION_NAME:-compute_score}"
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-False}"
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-seq_final_reward}"
MAX_NUM_GEN_BATCHES="${MAX_NUM_GEN_BATCHES:-10}"
RESUME_MODE="${RESUME_MODE:-disable}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-}"
if [[ "${ROLLOUT_DATA_DIR}" == "auto" ]]; then
  ROLLOUT_DATA_DIR="${RUN_DIR}/rollouts"
fi

TRAINING_STEPS_ARGS=()
if [[ -n "${MAX_STEPS}" ]]; then
  TRAINING_STEPS_ARGS+=(trainer.total_training_steps="${MAX_STEPS}")
fi

ROLLOUT_DATA_ARGS=()
if [[ -n "${ROLLOUT_DATA_DIR}" ]]; then
  ROLLOUT_DATA_ARGS+=(trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}")
fi

mkdir -p "${RUN_DIR}"
mkdir -p "${RAY_TMPDIR}"
if [[ -n "${ROLLOUT_DATA_DIR}" ]]; then
  mkdir -p "${ROLLOUT_DATA_DIR}"
fi

if [[ "${PREPARE_DATA}" == "1" ]]; then
  python RL/verl/prepare_verl_data.py \
    --data_path "${DATA_PATH}" \
    --output_dir "${VERL_DATA_DIR}"
fi

if [[ ! -f "${VERL_DATA_DIR}/train.parquet" || ! -f "${VAL_FILES}" ]]; then
  cat >&2 <<EOF
Missing veRL parquet data.
Run:
  python RL/verl/prepare_verl_data.py \\
    --data_path ${DATA_PATH} \\
    --output_dir ${VERL_DATA_DIR}

Expected files:
  ${VERL_DATA_DIR}/train.parquet
  ${VAL_FILES}

Or rerun this launcher with PREPARE_DATA=1.
EOF
  exit 1
fi

cat > "${RUN_DIR}/run_config.json" <<EOF
{
  "backend": "verl",
  "data_path": "${DATA_PATH}",
  "verl_data_dir": "${VERL_DATA_DIR}",
  "val_files": "${VAL_FILES}",
  "prepare_data": "${PREPARE_DATA}",
  "model_path": "${MODEL_PATH}",
  "output_root": "${OUTPUT_ROOT}",
  "run_name": "${RUN_NAME}",
  "max_steps": "${MAX_STEPS}",
  "train_batch_size": "${TRAIN_BATCH_SIZE}",
  "gen_batch_size": "${GEN_BATCH_SIZE}",
  "num_generations": "${NUM_GENERATIONS}",
  "val_batch_size": "${VAL_BATCH_SIZE}",
  "gradient_checkpointing": "${GRADIENT_CHECKPOINTING}",
  "learning_rate": "${LEARNING_RATE}",
  "use_kl_loss": "${USE_KL_LOSS}",
  "kl_loss_coef": "${KL_LOSS_COEF}",
  "clip_ratio": "${CLIP_RATIO}",
  "clip_ratio_low": "${CLIP_RATIO_LOW}",
  "clip_ratio_high": "${CLIP_RATIO_HIGH}",
  "clip_ratio_c": "${CLIP_RATIO_C}",
  "reward_function_name": "${REWARD_FUNCTION_NAME}",
  "filter_groups_enable": "${FILTER_GROUPS_ENABLE}",
  "filter_groups_metric": "${FILTER_GROUPS_METRIC}",
  "max_num_gen_batches": "${MAX_NUM_GEN_BATCHES}",
  "resume_mode": "${RESUME_MODE}",
  "rollout_data_dir": "${ROLLOUT_DATA_DIR}",
  "tag": "verl"
}
EOF

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${VERL_DATA_DIR}/train.parquet" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  ++data.gen_batch_size="${GEN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.prompt_key=prompt \
  data.truncation=left \
  ++algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE}" \
  ++algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC}" \
  ++algorithm.filter_groups.max_num_gen_batches="${MAX_NUM_GEN_BATCHES}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.clip_ratio="${CLIP_RATIO}" \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
  actor_rollout_ref.actor.clip_ratio_c="${CLIP_RATIO_C}" \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.n="${NUM_GENERATIONS}" \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  custom_reward_function.path="${PWD}/RL/verl/verl_reward.py" \
  custom_reward_function.name="${REWARD_FUNCTION_NAME}" \
  trainer.project_name=RL4DistReconfig \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.logger="${LOGGER}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "${TRAINING_STEPS_ARGS[@]}" \
  "${ROLLOUT_DATA_ARGS[@]}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.default_local_dir="${RUN_DIR}" \
  trainer.resume_mode="${RESUME_MODE}"
