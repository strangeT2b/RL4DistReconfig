#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/env_cache_disk2.sh

# SFT with standard CE loss on XML-formatted data.

MODEL_ID="${MODEL_ID:-${LLAMA31_INSTRUCT_MODEL:-../models/meta-llama/Llama-3.1-8B-Instruct}}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} torchrun \
  --standalone \
  --nproc_per_node=2 \
  SFT/train_lora_adapter.py \
  --data_path "${DATA_PATH:-Dataset/Processed_xml/train_33_69_84_nodes_open_lines_xml.csv}" \
  --model_id "${MODEL_ID}" \
  --output_root runs/llama31_8b_instruct \
  --run_name "${RUN_NAME:-sft_xml_open_lines_ce_ep1__on__train_33_69_84_nodes}" \
  --loss_type ce \
  --prompt_format "${PROMPT_FORMAT:-llama3_chat}" \
  --num_train_epochs "${NUM_EPOCHS:-1}" \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 4096 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --fp16 0 \
  --bf16 1 \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-100}" \
  --max_steps "${MAX_STEPS:--1}" \
  --logging_steps 10 \
  --eval_strategy no \
  --report_to wandb \
  --wandb_project RL4DistReconfig \
  --wandb_run_group sft_llama31_8b_instruct
