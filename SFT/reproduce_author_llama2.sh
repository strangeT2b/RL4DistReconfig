#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/env_cache_disk2.sh

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} python SFT/reproduce_author_llama.py \
  --data_path Dataset/Processed/train_33_69_84_nodes.csv \
  --model_id ../models/meta-llama/Llama-2-7B-hf \
  --output_root runs/llama2_7b \
  --run_name sft_custom_v1 \
  --prompt_format legacy \
  --num_train_epochs 10 \
  --batch_size 16 \
  --gradient_accumulation_steps 1 \
  --max_new_tokens 1200 \
  --model_name_hf local-author-repro-llama2-7b-ce \
  --tokenizer_name_hf local-author-repro-llama2-7b-ce-tokenizer \
  --custom_loss 1 \
  --custom_loss_config IEL,SUL,CYL \
  --cycles_loss_scaling_factor 1 \
  --model_for_generation_path runs/llama2_7b/sft_custom_v1 \
  --max_seq_length 4096 \
  --save_strategy epoch \
  --save_steps 100 \
  --report_to wandb \
  --wandb_project RL4DistReconfig \
  --wandb_run_group sft_llama2_7b \
  --wandb_tags sft,custom,llama2,fp16,epoch10,bs16,effbs16,accum1
