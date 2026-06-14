# RL4DistReconfig

Reinforcement learning for LLM-based distribution network reconfiguration with
XML-structured outputs and verifiable graph rewards.

The pipeline: SFT on XML-formatted examples → merge LoRA adapter → RL with veRL-GRPO.

---

## Installation

SFT:

```bash
pip install -r requirements_sft.txt
```

RL uses [veRL](https://github.com/volcengine/verl) v0.4.1. A pre-built Docker image
is available at `registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20260323`.

---

## Datasets

Datasets come from [grid-datasets](https://github.com/panaschristou/grid-datasets).
Download CSV files into `Dataset/`, then prepare:

```bash
bash Dataset/build_xml_processed_from_unprocessed.sh        # open-lines XML
bash RL/verl/prepare_verl_data.sh                           # convert to parquet
```

---

## Usage

```bash
# 1. SFT
bash SFT/train_llama_ce.sh

# 2. Merge
BASE_MODEL=../models/meta-llama/Llama-3.1-8B-Instruct \
ADAPTER_PATH=runs/llama31_8b_instruct/<run>/checkpoint-* \
OUTPUT_DIR=runs/llama31_8b_instruct/merged/<name> \
bash SFT/merge_adapter.sh

# 3. RL
bash RL/verl/train_verl_grpo.sh

# 4. Eval
bash Eval/eval_xml_lora.sh
```

All scripts support environment variable overrides for paths, hyperparameters, and GPU configs.
