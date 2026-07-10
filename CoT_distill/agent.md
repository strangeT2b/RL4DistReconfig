# CoT_distill Agent Guide

Last consolidated: 2026-07-08

This guide applies to `CoT_distill/` work. Also follow repository-wide guidance in `../agent.md`.

## Scope

`CoT_distill/` builds and curates chain-of-thought data for distribution-network reconfiguration. The goal is to create training data that teaches valid radial reconfiguration reasoning, not just answer mapping.

Desired model behavior:

- reason about radiality constraints, tie loops, branch exchanges, heavy-load paths, weak-voltage tails, and invalid topology risks;
- output full XML with `<open_lines>`, `<node_voltages>`, and `<system_loss>`;
- avoid fluent but wrong rationalizations.

## Directory Map

Important files and directories:

- `build_qa_data.py` — build QA JSONL from source CSV data.
- `wash_qa_data.py` — remove samples where target system loss is worse than input loss.
- `short_CoT_distill.py` — short CoT generation path.
- `long_CoT_distill.py` — long CoT generation, repair, verifier, and GT-correction path.
- `cot_sft_curation.py` — CoT SFT curation utilities.
- `mix_sft.py` — short/long mixture construction.
- `config/long_cot_distill.yaml` — main long-CoT config.
- `prompts/` — reason/eval/reflexion/correction prompt files.
- `few_shot/long_few_shot.json` — long-CoT few-shot examples.
- `old_code_archive/` — older/abandoned code; do not revive without checking current design.
- `test/outputs/` — local pilot output directories.

## Data Inputs

Original unprocessed data:

```text
Dataset/Unprocessed/samples_33bus.csv
Dataset/Unprocessed/samples_69bus.csv
Dataset/Unprocessed/samples_84bus.csv
```

QA JSONL output from `build_qa_data.py`:

```text
Dataset/Processed_jsonl/33_69_84_nodes/
  train.jsonl
  validation.jsonl
  test.jsonl
```

Washed QA data is the current preferred input for CoT distillation:

```text
Dataset/Processed_jsonl/33_69_84_nodes_washed/
  stats.json
  train_washed.jsonl
  validation_washed.jsonl
  test_washed.jsonl
```

Washed counts from the current cleaned set:

| split | total clean | 33 bus | 69 bus | 84 bus |
|---|---:|---:|---:|---:|
| train | 10060 | 3352 | 2782 | 3926 |
| validation | 10149 | 3383 | 2768 | 3998 |
| test | 10228 | 3381 | 2844 | 4003 |

The core reason for washing: many original targets have `updated_system_loss > existing_system_loss`; these should not teach the model to rationalize worse reconfigurations.

## CoT Types

- `short CoT`: compact expert rationalization of a correct target. It may see the target but must not mention ground truth/reference/label/oracle/given answer.
- `long CoT`: no-GT forward reasoning first, optionally repaired, optionally GT-guided corrected at the final polishing stage.
- `mixed CoT`: curated short + long mixture for SFT.

Short CoT should teach concise expert judgment and format stability. It should avoid pretending to exhaustively enumerate candidates or invent exact power-flow values.

Long CoT should teach exploration and self-correction. It should not train on fluent but wrong reasoning.

## Current Long-CoT Pipeline Diagnosis

Most recent reviewed pilot:

```text
CoT_distill/test/outputs/long_cot_10perbus_current_v2/
```

Config facts from that run:

- `use_reflexion: false`
- `gt_correction.enabled: true`
- `max_iterations: 1`
- `score_threshold: 0.75`
- model: `gpt-5.5`
- sampling: 10 washed-train samples each from 33/69/84-bus

Metrics:

| stage | parse | valid | exact | mean IoU | leakage |
|---|---:|---:|---:|---:|---:|
| pre-correction | 30/30 | 30/30 | 0/30 | 0.4410 | 0/30 |
| final | 30/30 | 30/30 | 21/30 | 0.7646 | 0/30 |

Final exact by bus:

- 33-bus: 9/10
- 69-bus: 5/10
- 84-bus: 7/10

Manual review conclusion:

- final exact is necessary but not sufficient for SFT acceptance;
- non-exact long CoTs are dangerous because they are often fluent but wrong;
- 69-bus is the hardest and most error-prone;
- many exact traces are correction-style and can be useful only if the self-correction chain is causally coherent;
- raw `improvement_history` cannot be concatenated directly because feedback includes training-pipeline/meta wording such as `hidden reference`, `reference open_lines`, `verifier`, `ground truth`, or `IoU`.

## v2 Sample Quality Decisions

For `long_cot_10perbus_current_v2`, final-trace manual review produced:

Direct keep:

```text
0, 2, 3, 4, 6, 7, 10, 9, 12, 19, 22, 26
```

Cautious keep / manual review:

```text
5, 8, 16, 18, 20, 24, 25, 27, 29
```

Reject:

```text
1, 11, 13, 14, 15, 17, 21, 23, 28
```

Sanitized trajectory potential:

- good: `2, 4, 7, 8, 16, 18, 19, 20, 22, 27`
- salvage/manual review: `0, 3, 5, 6, 9, 10, 12, 24, 25, 26, 29`
- reject: `1, 11, 13, 14, 15, 17, 21, 23, 28`

These IDs are only for this pilot directory; do not generalize them to other generated sets.

## Important Failure Modes

1. **Eval-agent early-exit does not protect good answers.**
   - `long_CoT_distill.py` has a `score_threshold` early-exit, but eval-derived scores stayed low even for locally exact candidates.
   - Low scores triggered unrestricted `improve_trace()`, which can change `<answer>` and sometimes degrade exact candidates.

2. **Feedback leakage into model-facing prompts.**
   - Current repair feedback can include `hidden reference alignment`, `reference open_lines`, `verifier`, or similar meta wording.
   - Such wording may be useful internally but must not be shown to reason/reflexion/correction agents or included in SFT text.

3. **Post-hoc / weak causal correction.**
   - GT correction can polish close candidates, but if applied to poor candidates it can create answer-first rationalization.
   - Correction-style traces are acceptable only when the wrong candidate is clearly rejected and the final branch exchanges are causally supported.

4. **Reasoning-answer inconsistency.**
   - Some traces say a branch should be closed/opened but the final `<open_lines>` does not reflect that change.
   - These must be rejected or rewritten with answer locking and consistency checks.

5. **69-bus difficulty.**
   - One-shot no-GT candidates are often weak.
   - The model over-focuses on weak tails such as bus 61 and can choose plausible but wrong branch exchanges.

## Required Direction for v3

Do not scale v2 as-is. First patch the pipeline and rerun the same 30 selected samples as a v3 pilot.

Recommended engineering changes:

1. **Local-verifier hard early-exit**
   - If local verifier says the answer is exact, do not run unrestricted answer-changing improve.
   - If CoT text is weak, use answer-locked reasoning rewrite only.

2. **Answer-locked reasoning rewrite**
   - Keep `<answer>` exactly unchanged.
   - Rewrite only `<think>`.
   - Reject the rewrite if normalized open lines change.

3. **Sanitized model-facing feedback**
   - Separate internal feedback/logging from model-facing repair feedback.
   - Do not show `hidden reference`, `reference`, `verifier`, `ground truth`, `IoU`, `oracle`, or similar training-pipeline terms to generation agents.
   - Convert feedback into task-native language about tie loops, weak tails, invalid edges, cycles, disconnected subgraphs, copied input, or vague reasoning.

4. **Candidate sampling + rerank**
   - Add `candidate_sampling_n`, initially K=4.
   - Generate K no-GT candidates.
   - Local-verify all candidates.
   - Select the best valid non-copied candidate by exact, hit/IoU, recall, and precision.

5. **Tightened GT-correction gate**
   - GT correction should polish close candidates, not rescue poor candidates.
   - Prefer hit-based gates:
     - 33/69-bus: hit >= 4/5.
     - 84-bus: hit >= 10/13.
   - Record pre-correction hit/IoU/recall and whether GT correction was used.

6. **Sanitized trajectory assembly**
   - Do not raw-concat `improvement_history`.
   - Produce one clean self-correction `<think>` and one final `<answer>`.
   - Intermediate wrong candidates should be described only as rejected candidates, not as standalone final answers.

Acceptance checks for v3:

- no exact → wrong degradation;
- no model-facing verifier/GT/reference wording;
- no reasoning-answer contradiction;
- final retained traces are exact and graph-valid;
- non-exact traces are rejected, not used for SFT.

## Acceptance Rules for Long CoT SFT Seeds

Hard requirements:

```text
final_xml_parse == true
final_graph_valid == true
final_exact == true
no_forbidden_leakage == true
```

Additional quality requirements:

- final open lines are supported by the reasoning;
- open/close statements in reasoning match final `<open_lines>`;
- reasoning includes concrete topology/branch-exchange logic rather than only generic “load balancing” statements;
- no raw verifier/hidden-reference/IoU/ground-truth wording;
- correction-style samples clearly identify and reject earlier wrong candidates;
- no non-exact long CoTs enter SFT without manual or stronger automated filtering.

## Commands

Wash QA data:

```bash
cd /Users/town/Codes/RL4DistReconfig
python CoT_distill/wash_qa_data.py --overwrite --washed-filenames
```

Run long CoT with GT correction, example only; check config and current pipeline before scaling:

```bash
cd /Users/town/Codes/RL4DistReconfig
/opt/anaconda3/envs/camel/bin/python CoT_distill/long_CoT_distill.py \
  --qa_data_path Dataset/Processed_jsonl/33_69_84_nodes_washed/train_washed.jsonl \
  --use-gt-correction \
  --gt-correction-require-valid
```

Syncing to H cluster should use the active H path:

```text
/mnt/shared-storage-gpfs2/ai4energy/gouzitang/RL4DistReconfig
```

Never commit API keys or `.env` values.

## Latest v3 Same-30 Pilot (2026-07-09)

Output paths:

```text
Baseline v3 evalview:
CoT_distill/test/outputs/long_cot_10perbus_random_v3_evalview_20260709/generated_long_cot_indices30_07-09-17:04:47.json

GT-correction + reflexion:
CoT_distill/config/CoT_distill/test/outputs/long_cot_10perbus_random_v3_evalview_reflexion_gtcorr_20260709/generated_long_cot_indices30_07-09-22:05:45.json

Comparison report:
CoT_distill/test/outputs/long_cot_10perbus_random_v3_evalview_reflexion_gtcorr_20260709/quality_compare_report.md
```

Same 30 selected washed-train samples, 10 per bus. GT-correction/reflexion command used `--use-reflexion --use-gt-correction --batch-size 4 --max-workers 4` and completed 30/30 in about 1:10:34.

Metrics:

| run | parse | valid | exact | usable_for_sft | mean IoU | mean recall | mean precision | confirmed SFT leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline v3 evalview | 30/30 | 30/30 | 4/30 | 4/30 | 0.5577 | 0.6867 | 0.6867 | 0/30 |
| GT-correction + reflexion | 30/30 | 30/30 | 25/30 | 20/30 | 0.8804 | 0.9046 | 0.9046 | 0/30 |

Paired deltas over 30 common samples:

- mean IoU: `+0.3227`
- exact gains: `22`
- exact losses: `1`
- usable_for_sft gains: `18`
- usable_for_sft losses: `2`

Conclusion: GT-correction/reflexion sharply improves answer-level quality and usable yield. The original leakage scan reported 8/30 `final_sft_trace` and 11/30 `final_trace` hits, but inspection showed every hit was the substring `iou` inside ordinary words such as `previous`; no true model-facing leakage was found in those records.

The detector was fixed in `prompt_policy.py` to use case-insensitive whole-token/whole-phrase regex boundaries. Both feedback sanitization and GT-correction leakage checks now share this detector. Re-scanning the same 30 records produced `0/30` hits for `final_sft_trace`, `final_trace`, and `working_sft_trace`, while standalone `IoU`, `GT`, `ground truth`, and `hidden reference` are still detected. This was a detector-only fix; prompts and pipeline control flow were not changed.

The output-dir resolution issue remains: generated JSON was written under `CoT_distill/config/CoT_distill/...`; prefer absolute output paths until fixed.

Immediate next steps:

1. Investigate the one exact loss and two usable_for_sft losses relative to baseline, including the five exact records rejected by the reasoning-quality gate.
2. Fix or work around output-dir resolution before the next run.
3. Decide whether the same-30 pilot needs an API rerun; the leakage correction itself was verified by re-scanning existing output and does not require regeneration.
4. After degradation and path handling are controlled, add candidate sampling/rerank (`candidate_sampling_n`, initially K=4), especially for 69-bus.

## Documentation Maintenance

- Keep this file focused on CoT_distillation workflow and current durable diagnosis.
- Put long historical run logs and one-off metric tables under `../docs/archive/agent-notes/` or a dated experiment note.
- Keep root-wide conventions in `../agent.md`.
