# RL4DistReconfig guidance archive — 2026-07-08

This file preserves historical guidance content from the root guidance consolidation.
Active guidance after consolidation lives in:

- `CLAUDE.md` — Claude Code entrypoint.
- `agent.md` — canonical root project guide.
- `CoT_distill/agent.md` — CoT_distill-specific guide.

The sections below are historical snapshots, not active instructions. They may contain stale remote paths, old priorities, or dated experiment notes.

---

## Historical `CLAUDE.md`

# RL4DistReconfig

GRPO reinforcement learning for power grid reconfiguration, built on Qwen3-8B SFT adapters.

## Key scripts
- `RL/train_grpo_adapter.sh` — GRPO RL training (run this to train)
- `SFT/train_lora_adapter.py` — SFT training
- `Eval/evaluate_vllm_lora.py` — Evaluation

## Conventions
- Run from repo root. Copy `.env.example` to `.env` (WANDB_API_KEY).
- `--fp16` / `--bf16` are int flags (0/1), not strings.
- Prompt format: `qwen_chat` for Qwen, `legacy` for Llama.
- 4-bit QLoRA is default (`load_in_4bit=1`).
- Datasets: CSV with columns `prompt`, `output`, `split`.
- Remote training machine: `ssh root@203.135.105.147`, project at `/mnt/disk2/gzt/RL4DistReconfig`, conda env `rl4dist_gzt`.


---

## Historical `AGENTS.md`

# RL4DistReconfig Codex Notes

This file is maintained for Codex-style coding sessions. Treat it as project
memory plus guardrails for future agents.

## Project Snapshot

RL4DistReconfig is GRPO reinforcement learning for power grid reconfiguration,
currently using Qwen3-8B and Llama-3.1-8B-Instruct SFT/RL experiments.

Key scripts:

- `RL/trl/train_grpo_adapter.py` — TRL GRPO RL training.
- `RL/verl/train_verl_grpo.sh` — veRL GRPO RL training launcher.
- `RL/verl/prepare_verl_data.py` — CSV-to-veRL parquet conversion.
- `RL/verl/verl_reward.py` — veRL reward adapter around `RL.reward`.
- `SFT/train_lora_adapter.py` — SFT training.
- `SFT/merge_adapter.sh` — merge base model + PEFT adapter for veRL starts.
- `Eval/eval_author_lora.py` — author-format vLLM LoRA evaluation.
- `Eval/eval_xml_lora.py` — XML open-lines vLLM evaluation.
- `Eval/eval_xml_lora.sh` — XML open-lines evaluation launcher with
  environment-variable overrides.
- `Eval/eval_author_qwen.sh`, `Eval/eval_author_llama31_instruct.sh`,
  and `Eval/eval_author_llama31_base.sh` — author-format vLLM evaluation
  launchers with environment-variable overrides, defaulting to merged SFT models.

Run conventions:

- Run commands from the repository root.
- Shell launchers source `scripts/env_cache_disk2.sh` to keep Hugging Face,
  torch, triton, vLLM, W&B, pip, matplotlib/numba, Ray, and temp caches off the
  small root filesystem. On the remote host this defaults to
  `/mnt/disk2/gzt/.cache`, `/mnt/disk2/gzt/tmp`, and project-local `.ray_tmp/`.
- Copy `.env.example` to `.env` and set `WANDB_API_KEY` when W&B is needed.
- `--fp16` and `--bf16` are integer flags (`0`/`1`), not strings.
- Use `--prompt_format qwen_chat` for Qwen models and `legacy` for Llama.
- 4-bit QLoRA is the default training setup (`load_in_4bit=1`) unless a script
  explicitly overrides it.
- Dataset CSVs are expected to have `prompt`, `output`, and `split` columns.
- veRL parquet data is model-agnostic chat-message data; do not bake Qwen/Llama
  prompt templates into parquet.

Remote training environment:

- Host: `ssh root@203.135.105.147`
- Project path: `/mnt/disk2/gzt/RL4DistReconfig`
- Conda env: `rl4dist_gzt`

## veRL Migration Notes

Current veRL launcher:

- `RL/verl/train_verl_grpo.sh`
- Run from repo root, usually inside the veRL docker/container.
- Default model path is the merged Llama SFT model:
  `runs/llama31_8b_instruct/merged/sft_ce_ep3_1`
- Default output root/run:
  `runs/llama31_8b_instruct/grpo_verl_from_sft_ce_ep3_1`
- Defaults can be overridden with environment variables, e.g.
  `MODEL_PATH=... RUN_NAME=... bash RL/verl/train_verl_grpo.sh`.

Important veRL launcher behavior:

- It sources `scripts/env_cache_disk2.sh`, which in turn sources project `.env`,
  so W&B keys are exported to veRL/Ray subprocesses.
- It sets `RAY_TMPDIR=${PWD}/.ray_tmp` and `TMPDIR=/mnt/disk2/gzt/tmp` by
  default on the remote host.
  `.ray_tmp/` is git-ignored and can be removed after jobs stop.
- It uses `LOGGER=[console,wandb]` by default. Use `LOGGER=[console]` for local
  debugging without W&B.
- It uses `VLLM_GPU_MEMORY_UTILIZATION=0.6` by default. If vLLM reports no KV
  cache blocks, increase it; if actor/FSDP OOMs, decrease it.
- It saves every `SAVE_FREQ=200` steps and validates every `TEST_FREQ=50`
  steps by default.
- It defaults validation to
  `Dataset/verl/train_33_69_84_nodes/validation_64.parquet`, a stratified
  sample with 33/69/84-node cases represented (22/21/21 rows).
- It supports `GRADIENT_CHECKPOINTING` as an environment override. Default is
  `True`; use `GRADIENT_CHECKPOINTING=False` only for speed experiments when
  H800 memory headroom is sufficient.
- It supports optional rollout dump via `ROLLOUT_DATA_DIR`. When set, the
  launcher passes `trainer.rollout_data_dir` and creates the directory.
  `ROLLOUT_DATA_DIR=auto` maps to `${RUN_DIR}/rollouts`; an explicit path is
  also accepted.
- It supports optional max-step limiting via `MAX_STEPS`. If unset, veRL runs
  according to `TOTAL_EPOCHS`. To mimic TRL `--max_steps 1000`, run:
  `MAX_STEPS=1000 bash RL/verl/train_verl_grpo.sh`.

veRL config details learned during migration:

- GRPO is selected with `algorithm.adv_estimator=grpo`.
- veRL's step limit field is `trainer.total_training_steps`, not `max_steps`.
- `trainer.logger` must be a Hydra list like `[console,wandb]`, not the string
  `'["console","wandb"]'`.
- Recent veRL/vLLM requires
  `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`; setting only
  `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` is insufficient.
- Current minimal setup has actor, rollout/vLLM, and reference log-prob work
  sharing the same visible GPU pool (`CUDA_VISIBLE_DEVICES`, default `2,3`).
- The current veRL path starts from a merged SFT model and trains a fresh LoRA.
  veRL is not being used to load an existing PEFT adapter directly.
- The `/simon-stub-path` tokenizer warning during LoRA/vLLM rollout is expected:
  LoRA tensors are loaded from GPU memory, the path is a dummy non-None value
  for vLLM, and tokenizer loading falls back to the base model tokenizer.

veRL data/reward conventions:

- `RL/verl/prepare_verl_data.py` writes parquet with columns:
  `data_source`, `prompt`, `ability`, `reward_model`, and `extra_info`.
- `prompt` is a list of chat messages, e.g.
  `[{"role": "user", "content": raw_prompt}]`.
- Ground truth is stored as a string in `reward_model["ground_truth"]`; parsed
  inside the custom reward function.
- `extra_info` is for the custom reward/debugging and is ignored by veRL unless
  our reward code reads it.
- TRL and veRL share core reward logic in `RL.reward.compute_reward_iou`, but
  use different adapters:
  - TRL returns a flat reward list from `reward_func(prompts, completions, ...)`.
  - veRL calls `compute_score(data_source, solution_str, ground_truth, extra_info)`.
- Reward modes were prototyped in `tests/reward_mode_prototype.py` before
  changing production reward code. Keep default `compute_reward_iou` numerically
  equivalent to the old valid-gated IoU reward unless the user explicitly asks
  for a behavior-changing experiment.
- Intended reward-mode split:
  - `compute_reward_valid_only`: only graph validity/optional format gate.
  - `compute_reward_iou_only`: only GT Open Lines IoU, graph penalties are
    diagnostics and do not gate.
  - `compute_reward_valid_and_iou`: current default shape; invalid outputs get
    graph penalty, valid outputs get `valid_base + iou_weight * iou`.
- veRL selects reward functions with `custom_reward_function.name`. The launcher
  exposes this as `REWARD_FUNCTION_NAME`, defaulting to `compute_score`.
  Available veRL adapter entries:
  - `compute_score` / `compute_score_valid_and_iou` for current default reward.
  - `compute_score_valid_only` for graph validity only.
  - `compute_score_iou_only` for GT Open Lines IoU only.

Operational notes:

- A successful veRL startup shows W&B initialization, then
  `Training Progress: 0/....`, followed by `training/global_step:1`.
- `/simon-stub-path` tokenizer warnings have appeared but did not stop training;
  veRL/vLLM fell back to the base model tokenizer. This is expected for the
  current LoRA/vLLM integration.
- Ray warnings about `/tmp/ray` being over 95% full indicate Ray temp files are
  on the wrong filesystem or an old run is still using `/tmp`. New launcher runs
  should use project-local `.ray_tmp`.
- If no training is running, old Ray temp files can be cleaned with
  `ray stop` and removing stale `/tmp/ray/*` or Ray temp directory contents.

## Collaboration Rules

- Do not make opportunistic fixes while handling a scoped request. If a nearby
  bug is found, report it first and wait for explicit approval before changing
  behavior.
- For evaluation/training changes, explain whether the change affects past
  experiment comparability before editing.
- Keep large behavioral changes in commits with detailed messages so experiment
  history can be audited.
- Prefer preserving existing CLI interfaces and launch-script conventions unless
  the user explicitly approves a breaking change.

## Current High-Priority Finding

The legacy evaluation metrics path likely over-reports graph validity.

Observed issue:

- Older evaluator/metrics code builds or receives full decoded text containing
  both prompt and generated text.
- It then uses `parse_open_lines(response)` to compute graph penalties.
- Because `parse_open_lines` returns the first `Open Lines=[...]`, it can parse
  the input configuration from the prompt instead of the generated answer.
- This can make `Avg cycles`, `Avg invalid`, `Avg subgraph`, `Valid`,
  `Invalid`, and `is_valid` reflect the original prompt topology rather than
  the model output.

Correct direction:

- Use `available_lines = parse_available_lines(prompt)`.
- Use `predicted_lines = reformatted["Open Lines"]` or the already extracted
  generated open lines.
- Compute graph penalties from those two objects, not from
  `prompt + generated_text`.

Status:

- `Eval/eval_author_lora.py` has been minimally fixed in the local working
  tree to compute graph penalties from generated open lines.
- Author source under `LLM4DistReconfig/` has not been changed and should remain
  untouched for comparison unless the user explicitly asks.
- `utils/metrics_utils.py` still needs a separately approved fix if the legacy
  `Eval/generate_metrics.py` path will be used.
- `Eval/generate_metrics.py` indirectly, because it calls
  `utils.metrics_utils.generate_metrics`

Less likely affected:

- `Eval/evaluate_hf_lora.py`, because it currently decodes completion tokens
  only and records format validity rather than graph metrics.
- RL training reward helpers, because `graph_penalties(prompt, response)` reads
  available lines from the prompt and predicted open lines from the generated
  response separately.

Important consequence:

- Existing evaluation outputs showing `Avg cycles=0`, `Avg invalid=0`,
  `Avg subgraph=0`, and `Valid=100%` may not prove generated topologies are
  valid. Re-run evaluation after fixing the parsing target before comparing CE
  SFT and RL graph validity.

## Recent Committed Change

Commit `6bacbf8` disabled RL `format_penalty` by default:

- `format_penalty_weight` defaults to `0.0`.
- The format penalty is still computed/logged as a diagnostic.
- It only affects reward when explicitly given a nonzero weight.
- The simulator valid gate no longer lets a zero-weight format penalty suppress
  simulator bonus.

## Open Evaluation Diagnostics Work

A requested change is to add simulator diagnostics to evaluator CSV output:

- `sim_converged`
- `sim_original_loss_mw`
- `sim_new_loss_mw`
- `sim_improvement_pct`
- `sim_failure_reason`

Do not combine this with unrelated summary/statistics changes unless the user
explicitly asks for them.


---

## Historical `agent.md`

# RL4DistReconfig Agent Notes

Last updated: 2026-07-08

This file is a working memory for the RL4DistReconfig project. It records project goals, data status, training/evaluation progress, CoT distillation design, known problems, and immediate next steps. Do not write API keys, passwords, SSH passwords, or other credentials here.

## Project Goal

The core goal is to build a reasoning-capable LLM for distribution network reconfiguration.

The model should:

- Read distribution-grid topology, line impedances, current open lines, node voltages, system loss, and load data.
- Output a valid radial reconfiguration in full XML:
  - `<open_lines>`
  - `<node_voltages>`
  - `<system_loss>`
- Improve exact match / IoU / MoU against target reconfiguration while preserving graph validity.
- Learn not only direct mapping but also domain reasoning patterns:
  - radiality constraints;
  - tie-loop and branch-exchange logic;
  - heavy-load path analysis;
  - weak-voltage tail analysis;
  - avoiding invalid edges, cycles, and disconnected subgraphs.

Current strategy:

1. Build CoT SFT data.
2. SFT a reasoning model.
3. Use RL post-training:
   - stage 1: constraint/reasonableness-oriented reward;
   - stage 2: validity + IoU-style objective.
4. Compare against prior direct-mapping baselines and large API models.

## Important Repositories And Paths

Local project:

```text
/Users/town/Codes/RL4DistReconfig
```

Older/eval project:

```text
/Users/town/Codes/RL4DistReconfig-eval
```

H cluster project:

```text
/mnt/shared-storage-gpfs2/ai4energy/gouzitang/RL4DistReconfig
```

H cluster copied/eval project:

```text
/mnt/shared-storage-gpfs2/ai4energy/gouzitang/RL4DistReconfig-eval
```

Bare-metal paths used earlier:

```text
/mnt/disk2/gzt/RL4DistReconfig
/mnt/disk2/gzt/RL4DistReconfig-eval
```

The bare-metal machine later became unavailable, so the active remote target is now the H cluster.

## Environment Notes

Local CAMEL environment:

```text
/opt/anaconda3/envs/camel/bin/python
```

H cluster uses rjob/rlaunch style jobs. GPU jobs are submitted rather than directly using visible GPUs from the development node.

Important environment variable names:

```text
MY_BASE_URL
MY_API_KEY
DS_BASE_URL
DS_API_KEY
PJ_URL_BASE
PJ_API_KEY
```

Never commit actual values.

## Data Overview

Original unprocessed source data:

```text
Dataset/Unprocessed/samples_33bus.csv
Dataset/Unprocessed/samples_69bus.csv
Dataset/Unprocessed/samples_84bus.csv
```

QA JSONL built from unprocessed data:

```text
CoT_distill/build_qa_data.py
Dataset/Processed_jsonl/33_69_84_nodes/
  train.jsonl
  validation.jsonl
  test.jsonl
```

The QA JSONL records contain:

- `task_type`
- `question`
- `answer`
- `meta`
- `raw`

`meta` contains:

- `bus`
- `split`
- `source_file`
- `row_index`
- `sample_id`, formatted like `33bus_train_123`

The `raw` field preserves the original CSV row.

## Clean / Washed QA Data

We discovered that many samples have target/GT system loss worse than input system loss:

```text
updated_system_loss > existing_system_loss
```

These samples should not be used for CoT distillation because they teach the model to rationalize worse reconfigurations.

Cleaning script:

```text
CoT_distill/wash_qa_data.py
```

Command:

```bash
cd /Users/town/Codes/RL4DistReconfig
python CoT_distill/wash_qa_data.py --overwrite --washed-filenames
```

Washed output:

```text
Dataset/Processed_jsonl/33_69_84_nodes_washed/
  stats.json
  train_washed.jsonl
  validation_washed.jsonl
  test_washed.jsonl
```

Washed counts:

| split | total clean | 33 bus | 69 bus | 84 bus |
|---|---:|---:|---:|---:|
| train | 10060 | 3352 | 2782 | 3926 |
| validation | 10149 | 3383 | 2768 | 3998 |
| test | 10228 | 3381 | 2844 | 4003 |

The washed data has been synced to H cluster. On H:

```text
/mnt/shared-storage-gpfs2/ai4energy/gouzitang/RL4DistReconfig/Dataset/Processed_jsonl/33_69_84_nodes_washed/
```

## CoT SFT Data Construction

Current CoT-related scripts:

```text
CoT_distill/build_qa_data.py
CoT_distill/wash_qa_data.py
CoT_distill/short_CoT_distill.py
CoT_distill/long_CoT_distill.py
CoT_distill/cot_sft_curation.py
CoT_distill/mix_sft.py
```

Older or abandoned code is under:

```text
CoT_distill/old_code_archive/
```

Current naming convention:

- `short CoT`: GT-visible rationalization; compact reasoning explaining the correct target.
- `long CoT`: no-GT forward reasoning first, optionally repaired, optionally GT-guided corrected at the end.
- `mixed CoT`: short + long mixture used for SFT.

Previously generated key SFT dataset:

```text
sft_mixed.jsonl
```

This was the key project artifact earlier: a mixed CoT SFT JSONL for distribution network reconfiguration.

## Long CoT Distillation: Current Design

Main script:

```text
CoT_distill/long_CoT_distill.py
```

Config:

```text
CoT_distill/config/long_cot_distill.yaml
```

Current default input:

```yaml
data:
  qa_data_path: ../../Dataset/Processed_jsonl/33_69_84_nodes_washed/train_washed.jsonl
```

Current long CoT pipeline:

1. Load QA data.
2. Sample cross-bus few-shot examples from:

   ```text
   CoT_distill/few_shot/long_few_shot.json
   ```

3. Reason agent generates one long CoT and answer without seeing GT.
4. Local verifier parses and scores:
   - XML parse;
   - graph validity;
   - invalid edges;
   - cycles;
   - subgraphs;
   - hidden IoU;
   - hidden recall;
   - hidden precision;
   - exact match;
   - copied input open lines.
5. Eval agent scores reasoning text, but currently it is not reliable as a hard filter.
6. Optional repair:
   - rewrite repair;
   - reflexion continuation repair.
7. Optional GT-guided final correction:
   - enabled by config or CLI;
   - uses GT only at final polishing stage;
   - prompt forbids mentioning ground truth/reference/verifier/IoU/oracle;
   - final `<answer>` is forced to exactly match GT XML.
8. Output JSON with trace history and metadata.

Current GT correction config:

```yaml
pipeline:
  gt_correction:
    enabled: false
    min_iou: 0.5
    min_recall: 0.6
    require_valid: true
```

CLI can override:

```bash
--use-gt-correction
--gt-correction-min-iou 0.5
--gt-correction-min-recall 0.6
--gt-correction-require-valid
```

Important implementation note:

- There was a bug where leakage guard searched `iou` as a substring and falsely matched words like `previous`.
- This has been fixed to use word-boundary matching for single-word forbidden terms.
- The fix has been synced to H.

## Reflexion Repair

Reflexion was added as an optional repair style:

1. Extract old `<think>`.
2. Reflexion agent writes one self-reflection paragraph starting with `Wait,`.
3. Reason agent continues from old reasoning + reflection.
4. Final trace is assembled as:

```xml
<think>
old reasoning

reflection

continuation reasoning
</think>
<answer>...</answer>
```

Observation:

- Reflexion can produce self-correction style CoT.
- However, it does not reliably improve answer IoU.
- A more structured/longer reflection version was tested and made quality worse.
- Current concise critical reflexion is better, but still not the main bottleneck solution.

Current view:

- Reflexion is useful for reasoning-style diversity.
- It should not be trusted as the main accuracy-improvement mechanism.

## Long CoT 30-Sample Test

Test directory:

```text
CoT_distill/test/outputs/long_cot_10perbus_current_v2/
```

Files:

```text
selected_raw.jsonl
selected_indices.txt
long_cot_10perbus_current_v2.yaml
generated/generated_long_cot_indices30_07-08-18:29:38.json
quality_report.md
quality_metrics.json
representative_cots.md
```

Sampling:

- 10 samples from 33-bus washed train.
- 10 samples from 69-bus washed train.
- 10 samples from 84-bus washed train.

Generation:

- model: `gpt-5.5`
- few-shot: 2 cross-bus examples
- max_iterations: 1
- GT correction enabled:
  - `min_iou=0.5`
  - `min_recall=0.6`
  - `require_valid=true`

Metrics:

| stage | parse | valid | EM | mean IoU | mean recall | leakage |
|---|---:|---:|---:|---:|---:|---:|
| before final correction | 30/30 | 30/30 | 0/30 | 0.4410 | 0.5856 | 0/30 |
| final | 30/30 | 30/30 | 21/30 | 0.7646 | 0.8005 | 0/30 |

By bus:

| bus | pre IoU | pre recall | final EM | final IoU | final recall |
|---:|---:|---:|---:|---:|---:|
| 33 | 0.5159 | 0.6600 | 9/10 | 0.9111 | 0.9200 |
| 69 | 0.2865 | 0.4200 | 5/10 | 0.5722 | 0.6200 |
| 84 | 0.5206 | 0.6769 | 7/10 | 0.8105 | 0.8615 |

Manual CoT quality review:

- EM samples are mostly fluent, coherent, and useful.
- Many EM CoTs are correction-style and say things like `previous candidate` or `I rechecked`.
- This style is useful for teaching self-correction, but the ratio should be controlled.
- Non-EM CoTs are dangerous: many are fluent and plausible but topologically wrong.
- 69-bus is the hardest and most error-prone.

Manual judgment:

- Keep exact/final IoU=1 long CoTs.
- Do not keep non-exact long CoTs without manual or stronger automated filtering.
- Wrong but fluent CoTs are worse than obviously broken CoTs because they teach plausible wrong heuristics.

## Eval Agent Status

Current eval agent does not effectively filter CoT quality.

Observed issues:

- Eval scores do not correlate with hidden IoU or exact match.
- Many exact traces still receive `correctness=0`.
- Feedback sometimes says hidden alignment is low even when local metric is exact.
- It cannot reliably identify fluent but wrong topology reasoning.

Current conclusion:

- Do not use eval agent as hard filter.
- Use local verifier for hard filtering:
  - valid;
  - IoU;
  - recall;
  - exact match;
  - copied input.
- Eval agent may still be useful as a soft reasoning-style critic if redesigned to evaluate only language/reasoning quality:
  - uses problem data;
  - candidate comparison;
  - no GT leakage;
  - no post-hoc rationalization;
  - no logic jumps;
  - no unsupported exact numerical claims.

## Proposed Long CoT Production Strategy

Current best production path:

1. Use washed train data.
2. For each sample, generate K no-GT candidates.
3. Use local verifier to select the best valid candidate by IoU/recall.
4. If best candidate passes threshold:
   - valid;
   - IoU >= 0.5 or recall >= 0.6;
   - not copied input;
   then run GT-guided final correction.
5. Keep only final exact / final IoU=1 outputs for long CoT SFT.
6. Record metadata:
   - bus;
   - sample_id;
   - source_index;
   - candidate count;
   - pre-correction IoU/recall/precision;
   - final exact/IoU;
   - whether GT correction was used.

Why multi-sample rerank is needed:

- A small probe showed best-of-K improves candidate quality.
- Current one-shot no-GT candidate is not good enough, especially on 69-bus.
- Final correction works when the candidate is already close.
- Final correction does not reliably rescue very poor candidates.

Recommended next code addition:

```yaml
pipeline:
  candidate_sampling_n: 4
  candidate_select_metric: iou_then_recall
```

Then implement:

```text
generate K candidates
local verify all
select best valid non-copied candidate
optional repair/correction
save candidate metrics
```

## Short CoT Strategy

Short CoT should not be just a shorter long CoT.

Goal:

- teach compact expert judgment;
- teach format stability;
- teach minimal constraint and branch-exchange reasoning;
- preserve answer correctness.

Short CoT should usually see GT and rationalize it, but should not mention:

- ground truth;
- reference;
- label;
- oracle;
- given answer.

Suggested short CoT structure:

1. Constraint check:
   - bus count;
   - number of open lines;
   - all open lines are in input.
2. Main topology rationale:
   - one or two key branch exchanges;
   - heavy load / weak voltage / impedance path explanation.
3. Validity summary:
   - connected;
   - radial;
   - low-loss candidate.

Short CoT should avoid:

- pretending to enumerate all candidates;
- inventing exact power-flow values;
- saying it computed exact loss if it did not;
- long reflexion-style narratives.

## SFT Progress Summary

Llama3.1-8B-Instruct was the main earlier base model.

Major SFT variants:

- SFT without CoT / full XML mapping.
- SFT with mixed short/long CoT.
- Short-only CoT SFT.
- Long-only CoT SFT.
- Qwen3-8B CoT SFT.

Important earlier finding:

- CoT SFT improved over pure mapping SFT in some comparisons.
- However, later RL on CoT sometimes collapsed response length and reduced actual reasoning usage.
- Long CoT quality is currently the likely bottleneck.

Mixed CoT SFT:

- The selected Llama mixed CoT SFT checkpoint was around epoch 24.
- It was used as initialization for later RL.

Qwen3-8B SFT:

- Qwen3-8B SFT CoT was trained on H.
- Sparse checkpoint eval was used.
- Epoch 15 looked like a good candidate during one eval pass.

Exact path names may vary under:

```text
runs/llama31_8b_instruct/
runs/qwen3_8b/
```

Check local/H runs before launching new jobs.

## RL Progress Summary

RL framework:

```text
RL/verl/
```

Main RL algorithms/jobs used GRPO via veRL.

Reward functions have included:

- full XML valid + IoU reward;
- constraint-only / reasonableness reward;
- norm/uniform reward variants;
- KL on/off variants;
- older mapping rewards.

Important reward conclusion:

- Use XML parser / full XML parser for current CoT models.
- Old `open_lines` parser was for reproducing old author mapping outputs and should not be changed for the current XML path.

Main experiment families:

1. CoT SFT + one-stage joint RL:
   - constraints + IoU together.
2. CoT SFT + two-stage RL:
   - stage 1 constraint-only;
   - stage 2 validity + IoU.
3. Short CoT RL:
   - short CoT SFT then constraint-only / joint RL.
4. Qwen CoT RL:
   - Qwen3-8B SFT then constraint-only and stage2.

Important RL observation:

- RL reward improved some metrics but often by less than expected.
- Some CoT RL outputs became short; response lengths around 400 tokens suggested CoT may be getting compressed away.
- If reward only scores final answer, RL may learn answer mapping instead of preserving reasoning.
- However, mainstream RL often only rewards final answer; the problem here may be CoT data quality and reward/task structure, not simply lack of reasoning reward.

Current conceptual tension:

- We want RL to activate reasoning.
- But final-answer-only reward can pressure the model to drop reasoning if shorter outputs get reward.
- Adding explicit CoT reward is difficult and slow if using rubric/API evaluation.

Potential direction:

- Better CoT SFT data first.
- Then RL with final-answer rewards may work better.
- If necessary, add lightweight structural CoT reward:
  - has `<think>`;
  - mentions candidate comparison;
  - no excessive length;
  - no forbidden leakage;
  - but avoid expensive rubric reward for every rollout.

## Evaluation Notes

Important eval set:

```text
Dataset/eval_subset_1000.csv
```

It was constructed to better match full test distribution and difficulty. Use this for quick controlled comparisons.

Metrics to report:

- Format error / improper XML;
- valid over all samples, not just proper XML;
- EM;
- MoU / mean IoU;
- mean recall / precision if available.

Be careful:

- Do not mix best-of-N / majority voting results with single-sample eval.
- For fair tables, keep:
  - same eval set;
  - same prompt format;
  - same max tokens;
  - same voting setting;
  - same XML parser.

API model few-shot evals were run for:

- Qwen3.5-397B;
- Deepseek-v4-flash;
- Deepseek-v4-pro;
- GPT5.5.

These are few-shot API baselines, not task-finetuned local models.

Table drafts discussed:

| model | Format error/% | valid/% | EM/% | MoU/% |
|---|---:|---:|---:|---:|
| Qwen3.5-397B | 64.4 | 35.5 | 6.5 | 55.87 |
| Deepseek-v4-flash | 47.5 | 52.5 | 6.9 | 54.49 |
| Deepseek-v4-pro | 4.6 | 95.2 | 5.3 | 53.79 |
| GPT5.5 | 11.3 | 88.7 | 10.7 | 61.52 |
| LLM4DistReconfig | 0.01 | 92.55 | 22.12 | 63.49 |
| Ours | 0.0 | 96.3 | 33.8 | 76.37 |

Verify exact source files before using these numbers in a paper.

## Known Issues And Lessons

### 1. GT quality problem

Some original GT targets have worse system loss than input. This is why washed QA data is required.

### 2. 69-bus difficulty

69-bus appears most problematic for long CoT generation:

- one-shot no-GT candidates often have low IoU;
- model over-focuses on bus 61 / weak tail;
- it often chooses plausible but wrong branch exchanges.

### 3. Fluent wrong CoT

The most dangerous data is not malformed data. It is fluent but wrong CoT.

These traces:

- mention heavy loads;
- mention weak voltages;
- mention radiality;
- but choose wrong open lines.

Do not train on non-exact long CoT unless manually vetted or strongly filtered.

### 4. Eval agent weakness

Current eval agent is not a reliable filter. Use local verifier for hard decisions.

### 5. GT-guided correction is useful but should be gated

GT correction can produce high-quality self-correction traces if the candidate is already close. It should not be used to rescue very poor candidates because that creates fake rationalization.

### 6. Too many old checkpoints consume space

H cluster storage has repeatedly become tight. Conservative cleanup policy:

- keep final checkpoint for each important run;
- keep LoRA adapters;
- keep selected step checkpoints like 200/300/500/1000 where needed;
- remove old intermediate full training states when safe;
- do not delete active/latest resumable checkpoints.

## H Cluster Notes

Development node is not a GPU node. Use rjob to request GPU jobs.

Previous helper scripts:

- `submit_rjob.sh` as generic submitter.
- Job scripts for:
  - `stage2-from-stage1`
  - `both-constraint-iou`
  - Qwen SFT/RL
  - eval jobs

Rjob names should be clear:

```text
stage2-from-stage1
both-constraint-iou
qwen3-stage1-constraint-300
qwen3-stage2-from-stage1-step150
mixed-cot-both-constraint-iou-g16
```

Avoid opaque long names unless necessary.

## Current Best Next Steps

Recommended order:

1. Fix long CoT production pipeline by adding candidate sampling/rerank.
2. Use local verifier as hard filter.
3. Keep only final exact / IoU=1 long CoTs.
4. Generate a larger pilot set:
   - maybe 100 per bus;
   - compare K=1, K=4, K=6;
   - inspect exact rate and CoT text quality.
5. Redesign eval agent to judge reasoning text only, not answer correctness.
6. Generate high-quality long CoT set from washed train.
7. Generate short CoT set from washed train.
8. Curate/mix SFT data with clear proportions:
   - short CoT for stable mapping and concise reasoning;
   - long corrected CoT for exploration and self-correction;
   - avoid low-quality long CoT.
9. Run SFT.
10. Re-evaluate on eval_subset_1000.
11. Only then resume RL experiments.

Suggested long CoT acceptance rules:

```text
final_xml_parse == true
final_graph_valid == true
final_exact == true
no_forbidden_leakage == true
reasoning_quality manually/soft-eval acceptable
```

Suggested candidate pre-correction rules:

```text
candidate_valid == true
candidate_copied_input == false
candidate_iou >= 0.5 OR candidate_recall >= 0.6
```

Suggested next experiment:

```text
For each bus: sample 50 washed-train examples.
For each sample: generate K=4 candidates.
Select best valid candidate by IoU then recall.
GT-correct if threshold passes.
Keep exact final traces only.
Report exact yield, cost, latency, and manual CoT quality.
```

## Commands Worth Remembering

Wash QA data:

```bash
cd /Users/town/Codes/RL4DistReconfig
python CoT_distill/wash_qa_data.py --overwrite --washed-filenames
```

Run long CoT with GT correction:

```bash
cd /Users/town/Codes/RL4DistReconfig
/opt/anaconda3/envs/camel/bin/python CoT_distill/long_CoT_distill.py \
  --qa_data_path Dataset/Processed_jsonl/33_69_84_nodes_washed/train_washed.jsonl \
  --use-gt-correction \
  --gt-correction-min-iou 0.5 \
  --gt-correction-min-recall 0.6 \
  --gt-correction-require-valid
```

Sync selected files to H:

```bash
cd /Users/town/Codes/RL4DistReconfig
rsync -av CoT_distill/long_CoT_distill.py \
  h.pjlab.org.cn:/mnt/shared-storage-gpfs2/ai4energy/gouzitang/RL4DistReconfig/CoT_distill/long_CoT_distill.py
```

Sync washed data to H:

```bash
cd /Users/town/Codes/RL4DistReconfig
rsync -av Dataset/Processed_jsonl/33_69_84_nodes_washed/train_washed.jsonl \
  Dataset/Processed_jsonl/33_69_84_nodes_washed/validation_washed.jsonl \
  Dataset/Processed_jsonl/33_69_84_nodes_washed/test_washed.jsonl \
  Dataset/Processed_jsonl/33_69_84_nodes_washed/stats.json \
  h.pjlab.org.cn:/mnt/shared-storage-gpfs2/ai4energy/gouzitang/RL4DistReconfig/Dataset/Processed_jsonl/33_69_84_nodes_washed/
```

## Current Bottom Line

The project is no longer blocked by missing scripts or missing clean data. The main bottleneck is CoT quality.

The best current insight:

- Good long CoT requires a close no-GT candidate.
- GT correction can polish and align a close candidate.
- Wrong but fluent candidates must be filtered out.
- 69-bus needs multi-sample reranking.
- Eval agent is not yet useful as a hard filter.

Therefore, the next high-value engineering task is:

```text
Add candidate_sampling_n + local-verifier rerank to long_CoT_distill.py,
then generate a 150-sample pilot and keep only exact corrected long CoTs.
```
