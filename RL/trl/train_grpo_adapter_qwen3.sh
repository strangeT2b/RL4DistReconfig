#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source scripts/env_cache_disk2.sh

# RL run on Qwen3-8B starting from Qwen SFT epoch 3 checkpoint.
# vLLM colocate mode — single GPU for both inference and training.
# Usage: bash RL/trl/train_grpo_adapter_qwen3.sh

MODEL_ID="${MODEL_ID:-${QWEN3_8B_MODEL:-../models/Qwen/Qwen3-8B}}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3} torchrun --nproc_per_node=1 --master_port=${MASTER_PORT:-29523} RL/trl/train_grpo_adapter.py \
  --data_path Dataset/Processed/train_33_69_84_nodes.csv \
  --split train \
  --model_id "${MODEL_ID}" \
  --output_root runs/qwen3_8b \
  --run_name grpo_v1 \
  --init_adapter ./runs/qwen3_8b/sft_ce_v1/checkpoint-548 \
  --max_steps 1000 \
  --max_train_samples 3000 \
  --prompts_per_step 8 \
  --num_generations 8 \
  --gradient_accumulation_steps 1 \
  --prompt_format qwen_chat \
  --max_prompt_length 3072 \
  --max_new_tokens 512 \
  --temperature 1.0 \
  --top_p 0.95 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_steps 15 \
  --min_lr_ratio 0.0 \
  --fp16 0 \
  --bf16 1 \
  --load_in_4bit 0 \
  --gradient_checkpointing 1 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --kl_beta 0.04 \
  --invalid_edges_weight 1.0 \
  --cycles_weight 1.0 \
  --subgraphs_weight 1.0 \
  --format_penalty_weight 0.0 \
  --iou_weight 10.0 \
  --use_vllm 1 \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.3 \
  --logging_steps 5 \
  --save_steps 50 \
  --eval_steps 50 \
  --eval_samples 100 \
  --log_responses 0 \
  --report_to wandb \
  --wandb_project RL4DistReconfig \
  --wandb_run_group grpo_qwen3_8b \
  --wandb_tags grpo,qwen3,bf16,iou_reward
