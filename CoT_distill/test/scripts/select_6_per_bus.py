#!/usr/bin/env python3
"""Select a small deterministic long-CoT test set: 6 samples per bus network."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_BUSES = (33, 69, 84)
BUS_RE = re.compile(r"Busses=(\d+)", re.IGNORECASE)


def extract_problem(sample: dict) -> str:
    if sample.get("question"):
        return str(sample["question"])
    messages = sample.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content") or "")
    return ""


def bus_of(sample: dict) -> int | None:
    meta = sample.get("meta")
    if isinstance(meta, dict) and meta.get("bus") is not None:
        return int(meta["bus"])
    m = BUS_RE.search(extract_problem(sample))
    return int(m.group(1)) if m else None


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--per-bus", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    by_bus: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        bus = bus_of(row)
        if bus in DEFAULT_BUSES:
            by_bus[bus].append(idx)

    selected: list[int] = []
    summary = {"input": str(args.input), "seed": args.seed, "per_bus": args.per_bus, "buses": {}}
    rng = random.Random(args.seed)
    for bus in DEFAULT_BUSES:
        candidates = list(by_bus[bus])
        if len(candidates) < args.per_bus:
            raise SystemExit(f"Not enough samples for bus={bus}: {len(candidates)} < {args.per_bus}")
        picked = sorted(rng.sample(candidates, args.per_bus))
        selected.extend(picked)
        summary["buses"][str(bus)] = {
            "available": len(candidates),
            "selected_indices": picked,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices_path = args.output_dir / "long_cot_test_indices_6_per_bus.txt"
    summary_path = args.output_dir / "long_cot_test_indices_6_per_bus.json"
    indices_path.write_text(",".join(map(str, selected)) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {indices_path}")
    print(f"Wrote {summary_path}")
    print(",".join(map(str, selected)))


if __name__ == "__main__":
    main()
