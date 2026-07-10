#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/env_cache_disk2.sh

MODEL_ID="${MODEL_ID:-${LLAMA31_BASE_MODEL:-../models/meta-llama/Llama-3.1-8B}}"

CUDA_VISIBLE_DEVICES=4,5 torchrun \
  --standalone \
  --nproc_per_node=2 \
  SFT/train_lora_adapter.py \
  --data_path Dataset/Processed/train_33_69_84_nodes.csv \
  --model_id "${MODEL_ID}" \
  --output_root runs/llama31_8b \
  --run_name sft_ce_v2 \
  --loss_type ce \
  --num_train_epochs 1 \
  --prompt_format legacy \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --fp16 0 \
  --bf16 1 \
  --load_in_4bit 0 \
  --max_seq_length 4096 \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_steps 100 \
  --eval_strategy no \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --gradient_log_every 10 \
  --log_per_parameter 0 \
  --save_raw_gradients_every 0 \
  --report_to wandb \
  --wandb_run_group sft_llama31_8b \
  --wandb_tags sft,ce,legacy-prompt,lr2e-4,effbs32,batch8,accum2,bf16,epoch1 \
  --wandb_log_gradients 1
