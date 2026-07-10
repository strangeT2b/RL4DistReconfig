# Training Run Directory Convention

All training adapters live under `runs/` using a unified two-level layout:

```
runs/<model>/<run_id>/
```

## Components

| Field | Values | Notes |
|---|---|---|
| `<model>` | `llama2_7b`, `llama31_8b`, `llama31_8b_instruct`, `qwen3_8b` | Base model identifier |
| `<run_id>` | `sft_ce_v1`, `sft_custom_v1`, `grpo_v1`, `grpo_v2`, ... | `<method>[_<loss>]_v<N>` — short label distinguishing runs |

Hyperparameter details (epoch, batch size, lr, dtype, etc.) live in W&B and `run_config.json`, **not** in the path.

## What lives inside `<run_id>/`

The trainer (HF `Trainer` / trl `GRPOTrainer`) emits checkpoints and final artifacts directly under `<run_id>/`:

```
runs/llama31_8b_instruct/sft_ce_v1/
├── checkpoint-365/
├── checkpoint-730/
├── checkpoint-1095/
├── final_adapter/
└── run_config.json    # RL only
```

No extra `checkpoints/` middle layer. No repetition between `output_root` and `run_name`.

## Examples

```
runs/llama2_7b/sft_custom_v1/
runs/llama31_8b/sft_ce_v1/
runs/llama31_8b/sft_ce_v2/                 # second SFT-CE attempt
runs/llama31_8b/sft_custom_v1/
runs/llama31_8b/grpo_v1/
runs/llama31_8b_instruct/sft_ce_v1/
runs/llama31_8b_instruct/sft_custom_v1/
runs/llama31_8b_instruct/grpo_v1/
runs/qwen3_8b/sft_ce_v1/
runs/qwen3_8b/grpo_v1/
```

## How launchers configure it

`SFT/reproduce_author_llama.py`, `SFT/train_lora_adapter.py`, `RL/trl/train_grpo_adapter.py` all accept:

```
--output_root runs/<model>
--run_name <run_id>
```

and compute `run_dir = output_root / run_name` internally.

## Multiple runs of the same method

Use `_v1`, `_v2`, ... — the differences between runs (init checkpoint, hparams, etc.) live in **W&B run config** and the per-run `run_config.json`, not in the directory name.

Example: two GRPO runs starting from different SFT checkpoints on the same model →
`runs/qwen3_8b/grpo_v1/`, `runs/qwen3_8b/grpo_v2/`. Their `--init_adapter` values are visible in W&B.

## Evaluation outputs

Eval CSV/TXT use the mirror convention under `outputs/`:

```
outputs/<adapter_name>__on__<dataset>_<split>/
```

where `<adapter_name>` is `<model>_<run_id>` for adapter-based evals, or `<model>_base` for raw base models.

Examples:
```
outputs/llama31_8b_instruct_sft_ce_v1__on__train_33_69_84_nodes_test/
outputs/qwen3_8b_grpo_v1__on__train_33_69_84_nodes_test/
outputs/llama31_8b_base__on__train_33_69_84_nodes_test/        # base, no adapter
```

## What does *not* go in directory names

- `epoch<N>`, `effbs32`, `bf16`, `lr2e-4` — these belong in `run_config.json` and W&B tags.
- Date stamps — git history + W&B already track when a run happened.
- Init-checkpoint identifiers — kept in W&B run config, not in `<run_id>`.
- Long encoded hparam sweeps — use a short `<run_id>` (e.g. `sweep_a`) plus a config file.
