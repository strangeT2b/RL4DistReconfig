# RL4DistReconfig

## Reinforcement Learning for LLM-based Distribution Network Reconfiguration with Verifiable Structural Rewards

Official implementation. The pipeline fine-tunes Llama-3.1-8B on XML-structured
distribution network reconfiguration (DNR) tasks: supervised fine-tuning on
XML-formatted examples, followed by verifier-guided reinforcement learning
with GRPO. The verifier checks graph validity and topology matching without
requiring a simulator during RL training.

---

## Installation

SFT environment:

```bash
pip install -r requirements_sft.txt
```

RL environment uses [veRL](https://github.com/volcengine/verl) v0.4.1 with vLLM.
See the official documentation for installation and Docker images:

https://verl.readthedocs.io

---

## Datasets

Network configuration datasets originate from
[grid-datasets](https://github.com/panaschristou/grid-datasets).
Download the CSV files and place them under `Dataset/`.

Preprocessed Parquet datasets are provided under `Dataset/verl/`:

| Dataset | Description |
|---------|-------------|
| `train_33_69_84_nodes/` | Mixed 33/69/84-bus (original format) |
| `train_33_69_84_nodes_open_lines_xml/` | Open-lines XML schema |
| `train_33_69_84_nodes_full_xml/` | Full XML with node voltages and system loss |

To generate XML-formatted CSVs from raw samples:

```bash
bash Dataset/build_xml_processed_from_unprocessed.sh         # open-lines XML
OUTPUT_MODE=full_output OUTPUT_NAME=train_33_69_84_nodes_full_xml \
  bash Dataset/build_xml_processed_from_unprocessed.sh       # full XML
```

To convert CSVs to veRL Parquet format:

```bash
bash RL/verl/prepare_verl_data.sh
```

---

## Training

### Step 1: Supervised Fine-Tuning

```bash
bash SFT/train_llama_ce.sh           # standard CE loss
bash SFT/train_llama_custom.sh       # custom graph-penalty loss
```

Override `DATA_PATH`, `MODEL_ID`, `RUN_NAME`, `CUDA_VISIBLE_DEVICES` as needed.

### Step 2: Merge LoRA Adapter

```bash
BASE_MODEL=../models/meta-llama/Llama-3.1-8B-Instruct \
ADAPTER_PATH=runs/llama31_8b_instruct/<run_name>/checkpoint-* \
OUTPUT_DIR=runs/llama31_8b_instruct/merged/<merged_name> \
bash SFT/merge_adapter.sh
```

### Step 3: Reinforcement Learning with veRL-GRPO

```bash
bash RL/verl/train_verl_grpo.sh
```

All parameters are overridable via environment variables. See the script
for the complete list.

---

## Evaluation

```bash
bash Eval/eval_xml_lora.sh
```

---

## Repository Structure

```
Dataset/              Preprocessed Parquet datasets and data preparation scripts
RL/verl/              veRL GRPO training launcher and reward adapter
RL/reward.py          Core reward implementation
SFT/                  Supervised fine-tuning and LoRA merging
Eval/                 vLLM-based evaluation
utils/                Shared utilities (parsing, metrics, model loading)
```
