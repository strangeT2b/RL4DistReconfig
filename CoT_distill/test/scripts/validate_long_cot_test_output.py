#!/usr/bin/env python3
"""Validate generated long-CoT test outputs under CoT_distill/test/outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from utils.metrics_utils import graph_penalties_from_open_lines, parse_open_lines_full_xml  # noqa: E402


BUS_RE = re.compile(r"Busses=(\d+)", re.IGNORECASE)


def latest_output(output_dir: Path) -> Path:
    files = sorted(output_dir.glob("generated_long_cot_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"No generated_long_cot_*.json found in {output_dir}")
    return files[-1]


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("traces", "records", "data"):
            if isinstance(data.get(key), list):
                return data[key]
    if isinstance(data, list):
        return data
    raise SystemExit(f"Unsupported output JSON shape: {path}")


def bus_of_problem(problem: str) -> int | None:
    m = BUS_RE.search(problem or "")
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--file", type=Path, default=None)
    ap.add_argument("--per-bus", type=int, default=6)
    ap.add_argument("--strict", action="store_true", help="Exit non-zero when XML/graph validation fails.")
    args = ap.parse_args()

    path = args.file or latest_output(args.output_dir)
    records = load_records(path)

    bus_counts: Counter[int] = Counter()
    invalid = []
    missing_xml = []
    for i, rec in enumerate(records):
        problem = str(rec.get("problem") or rec.get("question") or "")
        trace = str(rec.get("final_trace") or rec.get("trace") or "")
        bus = bus_of_problem(problem)
        if bus is not None:
            bus_counts[bus] += 1
        open_lines = parse_open_lines_full_xml(trace)
        if not open_lines:
            missing_xml.append(i)
            continue
        penalties = graph_penalties_from_open_lines(problem, open_lines)
        if any(float(penalties[k]) != 0.0 for k in ("invalid_edges", "cycles", "subgraphs")):
            invalid.append({"row": i, "bus": bus, "penalties": penalties})

    report = {
        "file": str(path),
        "total": len(records),
        "bus_counts": dict(sorted(bus_counts.items())),
        "missing_full_xml": missing_xml,
        "invalid_graph": invalid,
    }
    report_path = args.output_dir / "long_cot_test_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    expected_total = 3 * args.per_bus
    has_count_error = len(records) != expected_total or any(
        bus_counts.get(bus, 0) != args.per_bus for bus in (33, 69, 84)
    )
    has_quality_error = bool(missing_xml or invalid)

    if has_count_error:
        raise SystemExit(
            f"Expected exactly {args.per_bus} generated records for each of buses 33, 69, and 84."
        )
    if args.strict and has_quality_error:
        raise SystemExit("Validation failed: missing XML or invalid graph.")


if __name__ == "__main__":
    main()
