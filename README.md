# RL4DistReconfig

## Reinforcement Learning for Power Distribution Network Reconfiguration via Structured Output

This repository contains the official implementation for training and evaluating
LLMs on distribution network reconfiguration (DNR) using structured XML outputs
and verifier-guided reinforcement learning.

---

## Installation

```bash
pip install torch transformers peft datasets vllm ray verl
```

For SFT training, additionally install:
```bash
pip install accelerate bitsandbytes trl
```

---

## Datasets

Preprocessed datasets are provided under `Dataset/verl/` as Parquet files:

| Dataset | Description |
|---------|-------------|
| `train_33_69_84_nodes/` | Mixed 33/69/84-bus networks (original format) |
| `train_33_69_84_nodes_open_lines_xml/` | Open-lines XML format |
| `train_33_69_84_nodes_full_xml/` | Full XML format with voltages and loss |

To regenerate datasets from CSV files, use:
```bash
python RL/verl/prepare_verl_data.py --data_path <input.csv> --output_dir <output_dir>
```

---

## Training

### Step 1: Supervised Fine-Tuning (SFT)

Train the base model on XML-formatted examples:

```bash
# Llama-3.1-8B-Instruct with custom loss
bash SFT/reproduce_author_llama31_custom.sh

# Llama-3.1-8B-Instruct with standard CE loss
bash SFT/reproduce_author_llama31_instruct.sh
```

Configure paths, hyperparameters, and LoRA settings inside each script.

### Step 2: Merge LoRA Adapter

Before RL training, merge the SFT LoRA adapter into the base model:

```bash
BASE_MODEL=../models/meta-llama/Llama-3.1-8B-Instruct \
ADAPTER_PATH=runs/llama31_8b_instruct/sft_*/checkpoint-* \
OUTPUT_DIR=runs/llama31_8b_instruct/merged/sft_* \
bash SFT/merge_adapter.sh
```

### Step 3: Reinforcement Learning with veRL GRPO

Start from the merged SFT checkpoint and optimize with GRPO:

```bash
source scripts/env_cache_disk2.sh  # optional: set cache paths

CUDA_VISIBLE_DEVICES=0,1 \
DATA_PATH=Dataset/Processed_xml/train_33_69_84_nodes_full_xml.csv \
VERL_DATA_DIR=Dataset/verl/train_33_69_84_nodes_full_xml \
MODEL_PATH=runs/llama31_8b_instruct/merged/sft_full_xml_ce_ep3__on__train_33_69_84_nodes_final \
RUN_NAME=grpo_fullxml_exp \
REWARD_FUNCTION_NAME=compute_score_full_xml_valid_and_iou \
bash RL/verl/train_verl_grpo.sh
```

Key parameters are controlled via environment variables. See `RL/verl/train_verl_grpo.sh`
for the complete list.

---

## Reward Design

The verifier implements a validity-first reward:

- **Invalid outputs** (parse failure or graph violations) receive a scaled
  graph-penalty reward: $R = -\lambda \cdot (n_{\text{inv}} + n_{\text{cyc}} + n_{\text{sub}})$
- **Valid outputs** are scored by topology matching against the reference:
  $R = w_{\text{iou}} \cdot \text{IoU} + w_{\text{imp}} \cdot \Delta + w_{\text{pre}} \cdot \text{Precision} + w_{\text{rec}} \cdot \text{Recall} - w_{\text{copy}} \cdot \mathbf{1}[\hat{E}=E_{\text{cur}}]$

All reward logic is in `RL/reward.py`, with veRL adapters in `RL/verl/verl_reward.py`.

---

## Evaluation

Evaluate a merged model against a test dataset:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=Dataset/Processed_xml/train_33_69_84_nodes_full_xml.csv \
BASE_MODEL=runs/llama31_8b_instruct/merged/<model_name> \
ADAPTER_PATH="" \
OUTPUT_XML_FORMAT=full_xml \
NUM_SAMPLES=-1 \
MAX_NEW_TOKENS=1200 \
bash Eval/eval_xml_lora.sh
```

Set `SIM_LOSS=1 GT_SIM=1` to additionally compute simulator-based loss metrics
(requires pandapower).

---

## Repository Structure

```
Dataset/           Preprocessed Parquet datasets and data preparation scripts
RL/verl/           veRL GRPO training launcher and reward adapter
RL/reward.py       Core reward implementation
SFT/               Supervised fine-tuning and LoRA merging scripts
Eval/              vLLM-based evaluation scripts
utils/             Shared utilities (metrics, parsing, formatting)
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{gou2025rl4distreconfig,
  title={RL4DistReconfig: Reinforcement Learning for Power Distribution
         Network Reconfiguration via Structured Output},
  author={Gou, Zitang and ...},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.
