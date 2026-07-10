# RL4DistReconfig Agent Guide

Last consolidated: 2026-07-08

This is the canonical root guide for agents working in this repository. Keep it stable and concise. Put CoT-distillation details in `CoT_distill/agent.md`; put dated historical notes in `docs/archive/agent-notes/`.

## Project Purpose

RL4DistReconfig builds reasoning-capable LLM systems for distribution-network reconfiguration.

The model should read grid topology, line impedances, current open lines, node voltages, system loss, and load data, then output a valid radial reconfiguration in full XML:

```xml
<open_lines>...</open_lines>
<node_voltages>...</node_voltages>
<system_loss>...</system_loss>
```

Primary goals:

- improve exact match / IoU / MoU against target open-line sets;
- preserve graph validity: valid edges, no cycles, connected radial topology;
- learn domain reasoning around tie loops, branch exchange, heavy-load paths, and weak-voltage tails.

## Repository Map

- `SFT/` — supervised fine-tuning scripts and adapter/merge utilities.
- `RL/` — RL training code, including TRL and veRL paths.
- `RL/verl/` — veRL GRPO launcher, data preparation, and reward adapter.
- `Eval/` — evaluation scripts for author-format and XML/full-output paths.
- `utils/` — dataset, parsing, prompt-format, reward, and metric utilities.
- `Dataset/` — source and processed datasets. Treat generated/processed data carefully.
- `CoT_distill/` — CoT data generation and curation; see `CoT_distill/agent.md` before changing this module.
- `LLM4DistReconfig/` — upstream/reference code. Do not change it unless explicitly asked.
- `runs/` — experiment outputs/checkpoints; see `runs/README.md` for naming conventions.

## Current Durable Status

- The active remote target is the H cluster, not the older bare-metal machine.
- H cluster project path: `/mnt/shared-storage-gpfs2/ai4energy/gouzitang/RL4DistReconfig`.
- H cluster eval/copy path: `/mnt/shared-storage-gpfs2/ai4energy/gouzitang/RL4DistReconfig-eval`.
- Older bare-metal paths such as `/mnt/disk2/gzt/RL4DistReconfig` are historical unless the user explicitly says otherwise.
- H cluster development nodes are not GPU nodes; use rjob/rlaunch-style submitted jobs for GPU work.
- Local CAMEL Python used for CoT/API workflows: `/opt/anaconda3/envs/camel/bin/python`.

## General Run Conventions

- Run commands from the repository root unless a script explicitly says otherwise.
- Never commit API keys, passwords, SSH passwords, or `.env` secrets.
- Copy `.env.example` to `.env` when needed; keep real values local.
- Important API environment variable names include `MY_BASE_URL`, `MY_API_KEY`, `DS_BASE_URL`, `DS_API_KEY`, `PJ_URL_BASE`, and `PJ_API_KEY`.
- `--fp16` and `--bf16` flags are integer flags (`0`/`1`), not strings.
- Use `qwen_chat` prompt format for Qwen models and `legacy` for Llama models unless a script documents otherwise.
- 4-bit QLoRA is the default training setup (`load_in_4bit=1`) unless explicitly overridden.
- Dataset CSVs commonly use `prompt`, `output`, and `split` columns.
- veRL parquet data should stay model-agnostic chat-message data; do not bake Qwen/Llama templates into parquet.

## Training and Evaluation Entry Points

Common scripts include:

- `SFT/train_lora_adapter.py` — SFT LoRA training.
- `SFT/merge_adapter.sh` — merge base model + PEFT adapter when needed.
- `RL/trl/train_grpo_adapter.py` — TRL GRPO path.
- `RL/verl/train_verl_grpo.sh` — veRL GRPO launcher.
- `RL/verl/prepare_verl_data.py` — CSV-to-veRL parquet conversion.
- `RL/verl/verl_reward.py` — veRL reward adapter around shared reward logic.
- `Eval/eval_xml_lora.py` — XML/full-output evaluation path.
- `Eval/eval_author_lora.py` — author-format evaluation path for reproduction/comparison.

Before launching new training or evaluation jobs, check the relevant script defaults and existing `runs/` directories.

## veRL Notes

- veRL selects GRPO with `algorithm.adv_estimator=grpo`.
- The veRL step limit field is `trainer.total_training_steps`, not `max_steps`.
- `trainer.logger` must be a Hydra list such as `[console,wandb]`, not a quoted JSON string.
- veRL parquet output from `RL/verl/prepare_verl_data.py` contains `data_source`, `prompt`, `ability`, `reward_model`, and `extra_info`.
- `prompt` is a list of chat messages; `reward_model["ground_truth"]` stores the target string for reward parsing.
- TRL and veRL should share core reward logic where possible; adapters may differ, but behavior-changing reward experiments must be explicit.
- Preserve launcher environment-variable overrides unless the user approves breaking changes.
- `/simon-stub-path` tokenizer warnings in LoRA/vLLM rollout have historically been expected when vLLM falls back to the base tokenizer.

## Evaluation Rules

- Use the same eval set, prompt format, max tokens, sampling/voting setting, and XML/parser mode when comparing runs.
- Report validity over all samples, not only well-formatted XML samples.
- Key metrics usually include format error/improper XML, graph validity, exact match, mean IoU/MoU, recall, and precision where available.
- Do not mix best-of-N or majority-voting results with single-sample results in the same table without labeling them.
- Historical eval code had a graph-validity bug where old paths could parse prompt open lines instead of generated output. Verify parser targets before trusting old validity numbers.
- For current CoT/full-XML work, prefer the full XML parser path rather than legacy open-lines-only parsing unless reproducing old author-format outputs.

## Data and Artifact Safety

- Treat `Dataset/Unprocessed/` as source data.
- Treat processed/generated JSONL and evaluation outputs as artifacts; do not overwrite canonical artifacts without explicit user approval.
- Check whether a file is generated, curated, or canonical before deleting or replacing it.
- Large outputs/checkpoints under `runs/`, `outputs/`, and generated dataset directories can be expensive to recreate; ask before cleanup unless the user has given specific instructions.
- H cluster storage can become tight; do not delete active/latest resumable checkpoints. Prefer keeping final checkpoints, LoRA adapters, and selected useful step checkpoints.

## Collaboration Rules

- Do not make opportunistic fixes while handling a scoped request. If a nearby bug is found, report it first and wait for approval before changing unrelated behavior.
- For training/evaluation changes, explain whether the change affects comparability with past experiments.
- Preserve existing CLI interfaces and launcher conventions unless a breaking change is explicitly approved.
- Write code in the style of the surrounding files.
- Use `git --no-pager` for git output to avoid interactive pagers.
- Do not invent naming/standardization schemes without proposing options first.
- When inspecting data for the user, avoid artificial truncation unless output size forces a summary.

## CoT Distillation Pointer

`CoT_distill/` is currently the active research bottleneck area. Before changing CoT generation, curation, prompts, or evaluation logic, read:

```text
CoT_distill/agent.md
```

Root-level guidance should stay stable. CoT-specific current status, v2/v3 pipeline diagnosis, and next engineering steps belong in `CoT_distill/agent.md`.

## Documentation Maintenance

- Keep `CLAUDE.md` short; it is the Claude Code entrypoint and should point to this file rather than duplicate it.
- Keep this root `agent.md` focused on durable project-wide guidance.
- Keep module-specific guidance in module-local `agent.md` files.
- Archive old or dated notes under `docs/archive/agent-notes/` instead of leaving stale instructions in active guidance.
