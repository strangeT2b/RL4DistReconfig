#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/env_cache_disk2.sh

SAVE_STEPS="${SAVE_STEPS:-100}"
MAX_STEPS="${MAX_STEPS:--1}"
MODEL_ID="${MODEL_ID:-${LLAMA31_INSTRUCT_MODEL:-../models/meta-llama/Llama-3.1-8B-Instruct}}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5} python SFT/reproduce_author_llama.py \
  --data_path Dataset/Processed_xml/train_33_69_84_nodes_open_lines_xml.csv \
  --model_id "${MODEL_ID}" \
  --output_root runs/llama31_8b_instruct \
  --run_name sft_xml_open_lines_ep1__on__train_33_69_84_nodes \
  --prompt_format llama3_chat \
  --num_train_epochs 1 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_new_tokens 256 \
  --model_name_hf local-author-repro-llama31-instruct-xml-open-lines \
  --tokenizer_name_hf local-author-repro-llama31-instruct-xml-open-lines-tokenizer \
  --custom_loss 0 \
  --custom_loss_config IEL,SUL,CYL \
  --cycles_loss_scaling_factor 1 \
  --model_for_generation_path runs/llama31_8b_instruct/sft_xml_open_lines_ep1__on__train_33_69_84_nodes \
  --max_seq_length 4096 \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --max_steps "${MAX_STEPS}" \
  --report_to wandb \
  --wandb_project RL4DistReconfig \
  --wandb_run_group sft_llama31_8b_instruct_xml \
  --wandb_tags sft,ce,xml_open_lines,llama31,instruct,step_ckpt,train_33_69_84_nodes
