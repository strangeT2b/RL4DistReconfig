#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${MODEL_ID:-${LLAMA31_BASE_MODEL:-../models/meta-llama/Llama-3.1-8B}}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5} python SFT/reproduce_author_llama.py \
  --data_path Dataset/Processed/train_33_nodes.csv \
  --model_id "${MODEL_ID}" \
  --output_root runs/llama31_8b_instruct \
  --run_name sft_custom__on__train_33 \
  --prompt_format llama3_chat \
  --num_train_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_new_tokens 1200 \
  --model_name_hf local-author-repro-llama31-base-custom \
  --tokenizer_name_hf local-author-repro-llama31-instruct-ce-tokenizer \
  --custom_loss 1 \
  --custom_loss_config IEL,SUL,CYL \
  --cycles_loss_scaling_factor 1 \
  --model_for_generation_path runs/llama31_8b_instruct/sft_custom__on__train_33 \
  --max_seq_length 4096 \
  --save_strategy epoch \
  --save_steps 100 \
  --report_to wandb \
  --wandb_project RL4DistReconfig \
  --wandb_run_group sft_llama31_8b_instruct \
  --wandb_tags sft,custom,llama31_8b_instruct,epoch3,train_33
