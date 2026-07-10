# RL4DistReconfig

This repository keeps the upstream LLM4DistReconfig code separate from the project-owned training and evaluation scripts.

```text
LLM4DistReconfig/        Upstream reference code. Treat as read-only unless intentionally syncing upstream.
SFT/                     Project-owned supervised fine-tuning scripts.
RL/                      Project-owned reinforcement learning scripts and rewards.
Eval/                    Project-owned evaluation scripts.
utils/                   Shared helpers used by project-owned scripts.
tests/                   Unit tests for project-owned helpers and reward logic.
Dataset/                 Prepared and raw project datasets.
runs/ outputs/ wandb/    Training and evaluation outputs.
```

Most commands should be run from the repository root.
