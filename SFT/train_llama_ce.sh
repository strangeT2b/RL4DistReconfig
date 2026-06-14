#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# SFT with CE loss on full XML data — reproduces our best SFT initialization.
# Override MODEL_ID/DATA_PATH/etc. via environment variables.

MODEL_ID="${MODEL_ID:-${LLAMA31_INSTRUCT_MODEL:-../models/meta-llama/Llama-3.1-8B-Instruct}}"
DATA_PATH="${DATA_PATH:-Dataset/Processed_xml/train_33_69_84_nodes_full_xml.csv}"
RUN_NAME="${RUN_NAME:-sft_full_xml_ce_ep3__on__train_33_69_84_nodes}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} torchrun \
  --standalone \
  --nproc_per_node=2 \
  SFT/train_llama.py \
  --data_path "${DATA_PATH}" \
  --model_id "${MODEL_ID}" \
  --output_root runs/llama31_8b_instruct \
  --run_name "${RUN_NAME}" \
  --prompt_format llama3_chat \
  --num_train_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_new_tokens 1200 \
  --model_name_hf local-repro-llama31-instruct-fullxml-ce \
  --tokenizer_name_hf local-repro-llama31-instruct-fullxml-ce-tokenizer \
  --custom_loss 0 \
  --custom_loss_config IEL,SUL,CYL \
  --cycles_loss_scaling_factor 1 \
  --model_for_generation_path "runs/llama31_8b_instruct/${RUN_NAME}" \
  --max_seq_length 4096 \
  --save_strategy epoch \
  --report_to wandb \
  --wandb_project RL4DistReconfig \
  --wandb_run_group sft_llama31_8b_instruct
