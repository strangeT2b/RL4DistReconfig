"""Mix short_CoT + long_CoT into the final SFT reasoning dataset.

Rules:
- Deduplicate by idx; if both sources contain the same idx, keep long_CoT
  because it has fuller forward reasoning.
- Do not force a 1:1 ratio; keep all valid rows after deduplication.
- Shuffle all rows before writing sft_mixed.jsonl.
"""
import json
import random
from collections import Counter

S2 = "CoT_distill/outputs/short_CoT/short_CoT.jsonl"
S3 = "CoT_distill/outputs/long_CoT/long_CoT.jsonl"
OUT = "CoT_distill/outputs/sft_mixed/sft_mixed.jsonl"

# short_CoT: GT-conditioned short rationales; keep rows with reasoning.
s2 = [json.loads(l) for l in open(S2) if json.loads(l).get("got_reasoning")]
print(f"short_CoT: valid {len(s2)}")

# long_CoT: template-guided forward reasoning; keep rows with reasoning.
s3 = [json.loads(l) for l in open(S3) if json.loads(l).get("reasoning_len", 0) > 0]
print(f"long_CoT: valid {len(s3)}")

# Deduplicate: prefer long_CoT for the same idx.
s3_idx = {r["idx"] for r in s3}
s2_dedup = [r for r in s2 if r["idx"] not in s3_idx]
dup = len(s2) - len(s2_dedup)
print(f"dedup: dropped {dup} short_CoT rows duplicated by long_CoT, kept {len(s2_dedup)}")

s2_pick = s2_dedup
s3_pick = s3
print(f"natural mix: short_CoT={len(s2_pick)}, long_CoT={len(s3_pick)}, total {len(s2_pick)+len(s3_pick)}")


def norm(r, source):
    return {
        "idx": r.get("idx"),
        "bus": r.get("bus"),
        "source": source,
        "task_type": "reconfig",
        "problem": r.get("problem", ""),
        "gt": r.get("gt", ""),
        "trace": r.get("trace", ""),
        "reasoning_len": r.get("reasoning_len", 0),
        "iou": r.get("iou", 1.0 if source == "short_CoT" else 0.0),
    }


mix = [norm(r, "short_CoT") for r in s2_pick] + [norm(r, "long_CoT") for r in s3_pick]
rng = random.Random(42)
rng.shuffle(mix)

with open(OUT, "w", encoding="utf-8") as f:
    for r in mix:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 统计
from collections import Counter
by_src = Counter(r["source"] for r in mix)
by_bus = Counter(r["bus"] for r in mix)
print(f"\n[done] {OUT} 共 {len(mix)} 条")
print(f"  source: {dict(by_src)}")
print(f"  bus: {dict(sorted(by_bus.items()))}")
