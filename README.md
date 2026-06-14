# RL4DistReconfig

## Reinforcement Learning for LLM-based Distribution Network Reconfiguration with Verifiable Structural Rewards

🚀 Official implementation  
🔗 **Paper:** [arXiv](https://arxiv.org)  
📡 **Code:** https://github.com/strangeT2b/RL4DistReconfig

---

## Introduction

RL4DistReconfig fine-tunes Llama-3.1-8B to solve power distribution network reconfiguration
via XML-structured outputs and verifier-guided reinforcement learning. The model is
initialized with supervised fine-tuning on XML-formatted examples and then optimized
with GRPO using a verifier that checks graph validity and topology matching — no
simulator dependency during RL training.

Key results on the 33/69/84-bus test set (17,520 samples):

| Model | Exact Match | Mean IoU |
|-------|:----------:|:--------:|
| CE-SFT (3 epochs) | 20.0% | 0.6527 |
| GRPO noKL + norm + uniform | **33.9%** | **0.7516** |

---

## Installation

### SFT Environment

```bash
pip install -r requirements_sft.txt
```

### RL Environment

RL training uses [veRL](https://github.com/volcengine/verl) v0.4.1 with vLLM for rollout.
We provide a pre-built Docker image:

```
registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20260323
```

Or install from source:

```bash
pip install verl>=0.4.1 vllm ray
```

---

## Datasets

Network configuration datasets originate from
[grid-datasets](https://github.com/panaschristou/grid-datasets).
Download CSV files and place them under `Dataset/`.

Preprocessed datasets are provided under `Dataset/verl/` as Parquet files:

| Dataset | Description |
|---------|-------------|
| `train_33_69_84_nodes/` | Mixed 33/69/84-bus (original format) |
| `train_33_69_84_nodes_open_lines_xml/` | Open-lines XML |
| `train_33_69_84_nodes_full_xml/` | Full XML with voltages and loss |

To generate XML-formatted CSVs from raw samples:

```bash
bash Dataset/build_xml_processed_from_unprocessed.sh         # open-lines XML
OUTPUT_MODE=full_output OUTPUT_NAME=train_33_69_84_nodes_full_xml \
  bash Dataset/build_xml_processed_from_unprocessed.sh       # full XML
```

To convert CSVs to veRL Parquet format:

```bash
python RL/verl/prepare_verl_data.py --data_path <input.csv> --output_dir <output_dir>
```

---

## Training

### Step 1: Supervised Fine-Tuning

Train the base model with CE loss on XML-formatted examples:

```bash
bash SFT/train_llama_ce.sh
```

For custom graph-penalty loss:

```bash
bash SFT/train_llama_custom.sh
```

Set `DATA_PATH`, `MODEL_ID`, `RUN_NAME`, `CUDA_VISIBLE_DEVICES` to override defaults.

### Step 2: Merge LoRA Adapter

```bash
BASE_MODEL=../models/meta-llama/Llama-3.1-8B-Instruct \
ADAPTER_PATH=runs/llama31_8b_instruct/<run_name>/checkpoint-* \
OUTPUT_DIR=runs/llama31_8b_instruct/merged/<merged_name> \
bash SFT/merge_adapter.sh
```

### Step 3: RL with veRL-GRPO

```bash
bash RL/verl/train_verl_grpo.sh
```

Default configuration reproduces our best setting: full XML, no KL, normalized
uniform reward, GRPO with 8 samples/response, 300 steps. All parameters are
overridable via environment variables.

---

## Reward Design

The verifier implements a validity-first reward:

- **Invalid** (parse failure or graph violation): $R = -\lambda(n_{\text{inv}} + n_{\text{cyc}} + n_{\text{sub}})$
- **Valid**: $R = \text{IoU} + \Delta + \text{Precision} + \text{Recall} - \mathbf{1}[\hat{E}=E_{\text{cur}}]$

With $\lambda=1$, all coefficients are 1.0 (uniform weights). Set
`REWARD_NORMALIZE_PENALTIES=true` to normalize penalties by network size.
All reward logic is in [`RL/reward.py`](RL/reward.py).

---

## Evaluation

```bash
bash Eval/eval_xml_lora.sh
```

Defaults: full XML test set, 17,520 samples, no simulator. Set `SIM_LOSS=1 GT_SIM=1`
to additionally compute pandapower-based loss metrics.

---

## Repository Structure

```
Dataset/              Preprocessed Parquet datasets and data preparation
RL/verl/              veRL GRPO training launcher and reward adapter
RL/reward.py          Core reward implementation
SFT/                  Supervised fine-tuning and LoRA merging
Eval/                 vLLM-based evaluation
utils/                Shared utilities (parsing, metrics, model loading)
```

---

## Citation

```bibtex
@article{gou2025rl4distreconfig,
  title={Reinforcement Learning for LLM-based Distribution Network
         Reconfiguration with Verifiable Structural Rewards},
  author={Gou, Zitang and others},
  year={2025}
}
```
