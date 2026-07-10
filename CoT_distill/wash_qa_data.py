#!/usr/bin/env python3
"""Wash QA JSONL by removing rows whose GT loss is worse than input loss."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DISTILL_DIR = Path(__file__).resolve().parent
REPO_ROOT = DISTILL_DIR.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "Dataset" / "Processed_jsonl" / "33_69_84_nodes"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Dataset" / "Processed_jsonl" / "33_69_84_nodes_washed"
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter CoT QA JSONL files, keeping only samples with "
            "raw.updated_system_loss <= raw.existing_system_loss."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="directory containing train/validation/test JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for washed train/validation/test JSONL files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow writing into a non-empty output directory",
    )
    parser.add_argument(
        "--washed-filenames",
        action="store_true",
        help="write split_washed.jsonl filenames instead of split.jsonl",
    )
    return parser.parse_args()


def loss_is_clean(record: dict[str, Any]) -> bool:
    raw = record.get("raw")
    if not isinstance(raw, dict):
        return False
    try:
        existing_loss = float(raw["existing_system_loss"])
        updated_loss = float(raw["updated_system_loss"])
    except (KeyError, TypeError, ValueError):
        return False
    return updated_loss <= existing_loss


def bus_of(record: dict[str, Any]) -> str:
    meta = record.get("meta")
    if isinstance(meta, dict) and meta.get("bus") is not None:
        return str(meta["bus"])
    raw = record.get("raw")
    if isinstance(raw, dict) and raw.get("buses") is not None:
        return str(raw["buses"])
    return "unknown"


def wash_split(input_path: Path, output_path: Path) -> dict[str, Any]:
    total = kept = removed = malformed = 0
    total_by_bus: Counter[str] = Counter()
    kept_by_bus: Counter[str] = Counter()
    removed_by_bus: Counter[str] = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                removed += 1
                removed_by_bus["unknown"] += 1
                continue

            bus = bus_of(record)
            total_by_bus[bus] += 1
            if loss_is_clean(record):
                kept += 1
                kept_by_bus[bus] += 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                removed += 1
                removed_by_bus[bus] += 1

    return {
        "input": str(input_path),
        "output": str(output_path),
        "total": total,
        "kept": kept,
        "removed": removed,
        "malformed_json": malformed,
        "total_by_bus": dict(sorted(total_by_bus.items())),
        "kept_by_bus": dict(sorted(kept_by_bus.items())),
        "removed_by_bus": dict(sorted(removed_by_bus.items())),
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"[FATAL] input dir does not exist: {input_dir}")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"[FATAL] output dir is not empty: {output_dir}\n"
            "Pass --overwrite if you want to replace its JSONL/stat files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {"input_dir": str(input_dir), "output_dir": str(output_dir)}
    aggregate = defaultdict(int)
    for split in SPLITS:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.is_file():
            raise SystemExit(f"[FATAL] missing split file: {input_path}")
        output_name = f"{split}_washed.jsonl" if args.washed_filenames else f"{split}.jsonl"
        split_stats = wash_split(input_path, output_dir / output_name)
        stats[split] = split_stats
        for key in ("total", "kept", "removed", "malformed_json"):
            aggregate[key] += int(split_stats[key])

    stats["aggregate"] = dict(aggregate)
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")

    print("[done] washed QA JSONL")
    for split in SPLITS:
        s = stats[split]
        print(
            f"  {split}: kept {s['kept']} / {s['total']} "
            f"(removed {s['removed']}) -> {s['output']}"
        )
        print(f"    kept_by_bus: {s['kept_by_bus']}")
        print(f"    removed_by_bus: {s['removed_by_bus']}")
    print(f"[stats] {stats_path}")


if __name__ == "__main__":
    main()
