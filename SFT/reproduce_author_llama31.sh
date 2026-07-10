#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/env_cache_disk2.sh

MODEL_ID="${MODEL_ID:-${LLAMA31_BASE_MODEL:-../models/meta-llama/Llama-3.1-8B}}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5} python SFT/reproduce_author_llama.py \
  --data_path Dataset/Processed/train_33_69_84_nodes.csv \
  --model_id "${MODEL_ID}" \
  --output_root runs/llama31_8b \
  --run_name sft_ce_v1 \
  --prompt_format legacy \
  --num_train_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_new_tokens 1200 \
  --model_name_hf local-author-repro-llama31-base-ce \
  --tokenizer_name_hf local-author-repro-llama31-base-ce-tokenizer \
  --custom_loss 0 \
  --custom_loss_config IEL,SUL,CYL \
  --cycles_loss_scaling_factor 1 \
  --model_for_generation_path runs/llama31_8b/sft_ce_v1 \
  --max_seq_length 4096 \
  --save_strategy epoch \
  --save_steps 100 \
  --report_to wandb \
  --wandb_project RL4DistReconfig \
  --wandb_run_group sft_llama31_8b \
  --wandb_tags sft,ce,llama31,base,fp16,epoch3,bs4,accum4,effbs16
