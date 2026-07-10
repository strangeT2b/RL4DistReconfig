#!/usr/bin/env python3
"""Build CoT-distillation QA JSONL directly from unprocessed grid samples.

This is the source-data entry point for CoT construction. It reads
``Dataset/Unprocessed/samples_*bus.csv`` files, reproduces the original
LLM4DistReconfig split assignment per bus system, and writes split-specific
JSONL files with:

    question: CoT-friendly problem prompt built from existing network state
    answer:   GT full XML built from updated_* fields
    meta:     bus/split/source-file/row-index/sample-id
    raw:      the original CSV row, unchanged

The original CSV row is always preserved in ``raw``. Derived fields such as
line counts or loss-improvement flags are intentionally not stored in meta.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DISTILL_DIR = Path(__file__).resolve().parent
REPO_ROOT = DISTILL_DIR.parent
DEFAULT_UNPROCESSED_DIR = REPO_ROOT / "Dataset" / "Unprocessed"
DEFAULT_BUS_SYSTEMS = (33, 69, 84)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Dataset"
    / "Processed_jsonl"
    / ("_".join(str(bus) for bus in DEFAULT_BUS_SYSTEMS) + "_nodes")
)

QUESTION_TEMPLATE = """Solve this distribution network reconfiguration problem.

Find a low-loss radial configuration by choosing the final open lines. The bus
number indicates that buses are labeled from 1 to Busses.

Use the given Lines, Line Impedances, current Open Lines, NodeVoltages, current
System Loss, and System Load. The final Open Lines must include only edges that
appear in the input Lines list, and their given properties should be considered.

The energized graph after removing the final Open Lines must:
- include all buses in one connected component;
- contain no cycles;
- have exactly Busses - 1 closed lines.

Reason explicitly about candidate open-line configurations before giving the
final answer. Consider heavy loads, weak-voltage regions, line impedances, and
radiality constraints. Do not invent exact node voltages or exact system loss
during reasoning; use the final answer fields for the target values.

Return one <think> block followed by one <answer> block:

<think>
reasoning
</think>
<answer>
<open_lines>
[(u1,v1),(u2,v2),...]
</open_lines>
<node_voltages>
[v1,v2,...]
</node_voltages>
<system_loss>
loss
</system_loss>
</answer>

Power Distribution Network: Busses={buses}, Lines={lines}, Line Impedances={line_impedances}, Open Lines={existing_open_lines}
Network Variables: NodeVoltages={existing_node_voltages}, System Loss={existing_system_loss}, System Load={system_load}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CoT QA JSONL from Dataset/Unprocessed samples."
    )
    parser.add_argument(
        "--unprocessed-dir",
        type=Path,
        default=DEFAULT_UNPROCESSED_DIR,
        help="directory containing samples_*bus.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for train/validation/test JSONL outputs",
    )
    parser.add_argument(
        "--bus-systems",
        nargs="+",
        type=int,
        default=list(DEFAULT_BUS_SYSTEMS),
        help="bus systems to include, e.g. --bus-systems 33 69 84",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed used to reproduce LLM4DistReconfig split assignment",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help=(
            "write only rows with updated_system_loss <= existing_system_loss; "
            "split assignment is still computed before filtering"
        ),
    )
    parser.add_argument(
        "--limit-per-bus",
        type=int,
        default=-1,
        help="smoke limit per bus file after reading original order; -1 means all",
    )
    return parser.parse_args()


def split_values(num_samples: int, seed: int = 42) -> list[str]:
    """Return split labels using the original LLM4DistReconfig policy."""
    np.random.seed(seed)
    split_values = (
        ["train"] * (num_samples // 3)
        + ["test"] * (num_samples // 3)
        + ["validation"] * (num_samples // 3)
    )
    remainder = num_samples - len(split_values)
    split_values += np.random.choice(
        ["train", "test", "validation"], size=remainder
    ).tolist()
    np.random.shuffle(split_values)
    return split_values


def parse_literal(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"malformed literal: {value!r}") from exc


def format_open_lines(value: str) -> str:
    open_lines = parse_literal(value)
    return "[" + ",".join(f"({int(u)},{int(v)})" for u, v in open_lines) + "]"


def format_float_list(value: str) -> str:
    values = parse_literal(value)
    return "[" + ",".join(f"{float(v):g}" for v in values) + "]"


def full_xml_answer(row: dict[str, str]) -> str:
    return (
        "<answer>\n"
        "<open_lines>\n"
        f"{format_open_lines(row['updated_open_lines'])}\n"
        "</open_lines>\n"
        "<node_voltages>\n"
        f"{format_float_list(row['updated_node_voltages'])}\n"
        "</node_voltages>\n"
        "<system_loss>\n"
        f"{row['updated_system_loss']}\n"
        "</system_loss>\n"
        "</answer>"
    )


def question(row: dict[str, str]) -> str:
    return QUESTION_TEMPLATE.format(
        buses=row["buses"],
        lines=row["lines"],
        line_impedances=row["line_impedances"],
        existing_open_lines=row["existing_open_lines"],
        existing_node_voltages=row["existing_node_voltages"],
        existing_system_loss=row["existing_system_loss"],
        system_load=row["system_load"],
    ).strip()


def is_clean_loss(row: dict[str, str]) -> bool:
    try:
        return float(row["updated_system_loss"]) <= float(row["existing_system_loss"])
    except (KeyError, TypeError, ValueError):
        return False


def read_rows(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit > 0:
        return rows[:limit]
    return rows


def build_record(
    *,
    row: dict[str, str],
    bus: int,
    split: str,
    source_file: Path,
    row_index: int,
) -> dict[str, Any]:
    sample_id = f"{bus}bus_{split}_{row_index}"
    source_rel = source_file.as_posix()
    try:
        source_rel = source_file.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    return {
        "task_type": "reconfig",
        "question": question(row),
        "answer": full_xml_answer(row),
        "meta": {
            "bus": bus,
            "split": split,
            "source_file": source_rel,
            "row_index": row_index,
            "sample_id": sample_id,
        },
        "raw": dict(row),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_stats: dict[str, Any] = {}

    for bus in args.bus_systems:
        source_file = args.unprocessed_dir / f"samples_{bus}bus.csv"
        if not source_file.is_file():
            sys.exit(f"[FATAL] missing source file: {source_file}")

        rows = read_rows(source_file, args.limit_per_bus)
        splits = split_values(len(rows), seed=args.seed)
        if len(rows) != len(splits):
            raise RuntimeError("internal split length mismatch")

        source_counter: Counter[str] = Counter()
        clean_counter: Counter[str] = Counter()
        for row_index, (row, split) in enumerate(zip(rows, splits)):
            source_counter[split] += 1
            clean = is_clean_loss(row)
            if clean:
                clean_counter[split] += 1
            if args.clean_only and not clean:
                continue

            record = build_record(
                row=row,
                bus=bus,
                split=split,
                source_file=source_file,
                row_index=row_index,
            )
            by_split[split].append(record)

        source_stats[f"{bus}bus"] = {
            "path": source_file.relative_to(REPO_ROOT).as_posix(),
            "rows_read": len(rows),
            "split_counts_before_filter": dict(source_counter),
            "clean_counts_before_filter": dict(clean_counter),
        }

    for split in ("train", "validation", "test"):
        write_jsonl(output_dir / f"{split}.jsonl", by_split[split])

    output_counts = {
        split: len(by_split[split]) for split in ("train", "validation", "test")
    }

    print("[done] wrote:")
    for split in ("train", "validation", "test"):
        path = output_dir / f"{split}.jsonl"
        print(f"  {path} ({output_counts[split]} rows)")
    print("[stats] source:", json.dumps(source_stats, ensure_ascii=False, sort_keys=True))
    print("[stats] output_counts:", json.dumps(output_counts, ensure_ascii=False, sort_keys=True))

    first = next(
        (rows[0] for rows in (by_split["train"], by_split["validation"], by_split["test"]) if rows),
        None,
    )
    if first:
        print("[check] first keys:", sorted(first.keys()))
        print("[check] first meta:", first["meta"])
        print("[check] first raw fields:", sorted(first["raw"].keys()))
        print("[check] question forbids explanations:", "Do not include explanations" in first["question"])
        print("[check] answer has full XML:", all(tag in first["answer"] for tag in ("<open_lines>", "<node_voltages>", "<system_loss>")))


if __name__ == "__main__":
    main()
