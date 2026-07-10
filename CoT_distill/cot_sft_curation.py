#!/usr/bin/env python3
"""Curate generated CoT data into SFT-ready training files.

This script keeps the original sft_mix.jsonl intact and creates SFT-friendly
views with sample weights. The intent is to reduce bus/source bias during SFT
without throwing away scarce chain-of-thought data.

Default input:
    CoT_distill/outputs/sft_mix.jsonl

Default outputs:
    CoT_distill/outputs/stage4/sft_mix_weighted.jsonl
    CoT_distill/outputs/stage4/sft_train_prompt_response.jsonl
    CoT_distill/outputs/stage4/sft_train_messages.jsonl
    CoT_distill/outputs/stage4/stage4_stats.json

Recommended use:
    python CoT_distill/cot_sft_curation.py

If your trainer does not support sample weights, create a weighted-resampled
plain training file:
    python CoT_distill/cot_sft_curation.py --resample-size 3394
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as stats
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

DISTILL_DIR = Path(__file__).resolve().parent
REPO_ROOT = DISTILL_DIR.parent
DEFAULT_INPUT = DISTILL_DIR / "outputs" / "sft_mix.jsonl"
DEFAULT_OUT_DIR = DISTILL_DIR / "outputs" / "stage4"

BUS_RE = re.compile(r"Busses\s*=\s*(\d+)", re.IGNORECASE)
LINES_RE = re.compile(r"Lines=\[(.*?)\],\s*Line Impedances", re.DOTALL)
OPEN_LINES_RE = re.compile(r"<open_lines>\s*(.*?)\s*</open_lines>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
EDGE_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
INPUT_LOSS_RE = re.compile(r"System Loss=([\d.eE+-]+)")
OUTPUT_LOSS_RE = re.compile(r"<system_loss>\s*([\d.eE+-]+)\s*</system_loss>", re.DOTALL | re.IGNORECASE)

SYSTEM_NOTE = (
    "You are solving a distribution network reconfiguration task. "
    "Choose open lines so the energized network forms a connected, acyclic "
    "spanning tree and system loss is minimized."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create weighted SFT views from sft_mix.jsonl.")
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="source sft_mix jsonl")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="output directory")
    p.add_argument("--seed", type=int, default=42)

    # Weight policy. bus balance is inverse-frequency by default, but softened
    # or disabled with --bus-balance-strength.
    p.add_argument(
        "--bus-balance-strength",
        type=float,
        default=1.0,
        help="0 disables bus balancing; 1 uses full inverse-frequency balancing.",
    )
    p.add_argument("--stage2-weight", type=float, default=0.9)
    p.add_argument("--stage3-weight", type=float, default=1.1)
    p.add_argument("--stage3-low-iou", type=float, default=0.5)
    p.add_argument("--stage3-very-low-iou", type=float, default=0.3)
    p.add_argument("--stage3-low-iou-mult", type=float, default=0.5)
    p.add_argument("--stage3-very-low-iou-mult", type=float, default=0.25)
    p.add_argument(
        "--normalize-mean-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="normalize kept-record weights to mean 1.0",
    )

    # Hard filters should only catch samples that cannot be trained safely.
    p.add_argument(
        "--keep-hard-bad",
        action="store_true",
        help="keep malformed / graph-invalid records but annotate them with weight 0",
    )
    p.add_argument(
        "--skip-graph-check",
        action="store_true",
        help="skip graph validation and only use format/loss checks",
    )

    # Output controls.
    p.add_argument(
        "--formats",
        nargs="+",
        default=["weighted", "prompt_response", "messages"],
        choices=["weighted", "prompt_response", "messages"],
        help="which output jsonl views to write",
    )
    p.add_argument(
        "--resample-size",
        type=int,
        default=0,
        help="if >0, also write weighted-resampled prompt_response/messages files",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            row["_line_no"] = lineno
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def edge_set(text: str) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in EDGE_RE.findall(text or "")}


def edge_list(text: str) -> list[tuple[int, int]]:
    return [tuple(sorted((int(a), int(b)))) for a, b in EDGE_RE.findall(text or "")]


def extract_bus(row: dict[str, Any]) -> int | None:
    if isinstance(row.get("bus"), int):
        return row["bus"]
    if str(row.get("bus", "")).isdigit():
        return int(row["bus"])
    m = BUS_RE.search(row.get("problem", "") or "")
    return int(m.group(1)) if m else None


def extract_open_lines(text: str) -> set[tuple[int, int]] | None:
    m = OPEN_LINES_RE.search(text or "")
    if not m:
        return None
    return edge_set(m.group(1))


def parse_float(pattern: re.Pattern[str], text: str) -> float | None:
    m = pattern.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def graph_status(problem: str, trace: str, bus: int | None) -> dict[str, Any]:
    status = {
        "checked": False,
        "open_subset": False,
        "edge_count_ok": False,
        "connected": False,
        "acyclic": False,
        "valid": False,
    }
    if bus is None:
        return status
    lines_m = LINES_RE.search(problem or "")
    open_edges = extract_open_lines(trace)
    if not lines_m or open_edges is None:
        return status

    all_edges = edge_list(lines_m.group(1))
    all_edge_set = set(all_edges)
    closed_edges = [edge for edge in all_edges if edge not in open_edges]
    status["checked"] = True
    status["open_subset"] = open_edges <= all_edge_set
    status["edge_count_ok"] = len(closed_edges) == bus - 1

    adj: dict[int, list[int]] = {i: [] for i in range(1, bus + 1)}
    for u, v in closed_edges:
        if 1 <= u <= bus and 1 <= v <= bus:
            adj[u].append(v)
            adj[v].append(u)

    seen = {1}
    q: deque[int] = deque([1])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    status["connected"] = len(seen) == bus
    status["acyclic"] = status["connected"] and status["edge_count_ok"]
    status["valid"] = (
        status["open_subset"]
        and status["edge_count_ok"]
        and status["connected"]
        and status["acyclic"]
    )
    status["closed_edges"] = len(closed_edges)
    status["reachable_buses"] = len(seen)
    return status


def format_status(row: dict[str, Any]) -> dict[str, bool]:
    trace = row.get("trace", "") or ""
    problem = row.get("problem", "") or ""
    output_loss = parse_float(OUTPUT_LOSS_RE, trace)
    input_loss = parse_float(INPUT_LOSS_RE, problem)
    return {
        "has_problem": bool(problem.strip()),
        "has_trace": bool(trace.strip()),
        "has_think": bool(THINK_RE.search(trace)),
        "has_answer": bool(ANSWER_RE.search(trace)),
        "has_open_lines": bool(OPEN_LINES_RE.search(trace)),
        "loss_not_worse": (
            input_loss is not None and output_loss is not None and output_loss <= input_loss
        ),
    }


def quality_bucket(row: dict[str, Any]) -> str:
    source = row.get("source", "unknown")
    if source == "stage2":
        return "stage2"
    if source == "stage3":
        iou = float(row.get("iou", 0.0) or 0.0)
        if iou >= 1.0:
            return "stage3_exact"
        if iou >= 0.5:
            return "stage3_iou_ge_0.5"
        if iou >= 0.3:
            return "stage3_iou_0.3_0.5"
        return "stage3_iou_lt_0.3"
    return str(source)


def source_weight(row: dict[str, Any], args: argparse.Namespace) -> float:
    if row.get("source") == "stage2":
        return args.stage2_weight
    if row.get("source") == "stage3":
        return args.stage3_weight
    return 1.0


def quality_weight(row: dict[str, Any], args: argparse.Namespace) -> float:
    if row.get("source") != "stage3":
        return 1.0
    iou = float(row.get("iou", 0.0) or 0.0)
    if iou < args.stage3_very_low_iou:
        return args.stage3_very_low_iou_mult
    if iou < args.stage3_low_iou:
        return args.stage3_low_iou_mult
    return 1.0


def compute_bus_weights(rows: list[dict[str, Any]], strength: float) -> dict[int, float]:
    counts = Counter(extract_bus(row) for row in rows if extract_bus(row) is not None)
    if not counts or strength <= 0:
        return {bus: 1.0 for bus in counts}
    target = sum(counts.values()) / len(counts)
    return {bus: (target / count) ** strength for bus, count in counts.items()}


def annotate(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bus_weights = compute_bus_weights(rows, args.bus_balance_strength)
    annotated: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for row in rows:
        rec = {k: v for k, v in row.items() if k != "_line_no"}
        bus = extract_bus(row)
        fmt = format_status(row)
        graph = (
            {"checked": False, "valid": True}
            if args.skip_graph_check
            else graph_status(row.get("problem", "") or "", row.get("trace", "") or "", bus)
        )
        hard_ok = (
            fmt["has_problem"]
            and fmt["has_trace"]
            and fmt["has_think"]
            and fmt["has_answer"]
            and fmt["has_open_lines"]
            and fmt["loss_not_worse"]
            and graph["valid"]
        )
        qbucket = quality_bucket(row)
        w = (
            bus_weights.get(bus, 1.0)
            * source_weight(row, args)
            * quality_weight(row, args)
        )
        if not hard_ok:
            w = 0.0

        rec["bus"] = bus
        rec["quality_bucket"] = qbucket
        rec["format_status"] = fmt
        rec["graph_status"] = graph
        rec["sample_weight"] = round(w, 6)

        if hard_ok or args.keep_hard_bad:
            annotated.append(rec)
        else:
            dropped.append(rec)

    positive = [r["sample_weight"] for r in annotated if r["sample_weight"] > 0]
    if args.normalize_mean_weight and positive:
        mean_w = sum(positive) / len(positive)
        for rec in annotated:
            if rec["sample_weight"] > 0:
                rec["sample_weight"] = round(rec["sample_weight"] / mean_w, 6)

    summary = build_summary(rows, annotated, dropped, bus_weights)
    return annotated, summary


def build_summary(
    raw_rows: list[dict[str, Any]],
    kept_rows: list[dict[str, Any]],
    dropped_rows: list[dict[str, Any]],
    bus_weights: dict[int, float],
) -> dict[str, Any]:
    def counter_dict(items) -> dict[str, int]:
        return {str(k): v for k, v in sorted(Counter(items).items(), key=lambda kv: str(kv[0]))}

    weights = [r["sample_weight"] for r in kept_rows]
    pos_weights = [w for w in weights if w > 0]
    by_source_bus = Counter((r.get("source"), r.get("bus")) for r in kept_rows)
    return {
        "raw_records": len(raw_rows),
        "kept_records": len(kept_rows),
        "dropped_records": len(dropped_rows),
        "kept_by_bus": counter_dict(r.get("bus") for r in kept_rows),
        "kept_by_source": counter_dict(r.get("source") for r in kept_rows),
        "kept_by_quality_bucket": counter_dict(r.get("quality_bucket") for r in kept_rows),
        "kept_by_source_bus": {
            f"{source}:{bus}": count
            for (source, bus), count in sorted(by_source_bus.items(), key=lambda kv: str(kv[0]))
        },
        "bus_weight_raw": {str(k): round(v, 6) for k, v in sorted(bus_weights.items())},
        "sample_weight": {
            "min": min(weights) if weights else None,
            "max": max(weights) if weights else None,
            "mean": sum(weights) / len(weights) if weights else None,
            "positive_mean": sum(pos_weights) / len(pos_weights) if pos_weights else None,
            "positive_median": stats.median(pos_weights) if pos_weights else None,
        },
        "dropped_examples": [
            {
                "idx": r.get("idx"),
                "bus": r.get("bus"),
                "source": r.get("source"),
                "format_status": r.get("format_status"),
                "graph_status": r.get("graph_status"),
            }
            for r in dropped_rows[:10]
        ],
    }


def prompt_response_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": row.get("problem", ""),
        "response": row.get("trace", ""),
        "sample_weight": row.get("sample_weight", 1.0),
        "metadata": {
            "idx": row.get("idx"),
            "bus": row.get("bus"),
            "source": row.get("source"),
            "task_type": row.get("task_type"),
            "iou": row.get("iou"),
            "reasoning_len": row.get("reasoning_len"),
            "quality_bucket": row.get("quality_bucket"),
        },
    }


def messages_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_NOTE},
            {"role": "user", "content": row.get("problem", "")},
            {"role": "assistant", "content": row.get("trace", "")},
        ],
        "sample_weight": row.get("sample_weight", 1.0),
        "metadata": {
            "idx": row.get("idx"),
            "bus": row.get("bus"),
            "source": row.get("source"),
            "task_type": row.get("task_type"),
            "iou": row.get("iou"),
            "reasoning_len": row.get("reasoning_len"),
            "quality_bucket": row.get("quality_bucket"),
        },
    }


def weighted_resample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    candidates = [r for r in rows if r.get("sample_weight", 0) > 0]
    if not candidates:
        return []
    rng = random.Random(seed)
    weights = [float(r["sample_weight"]) for r in candidates]
    return rng.choices(candidates, weights=weights, k=size)


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not in_path.is_file():
        raise SystemExit(f"[FATAL] input not found: {in_path}")

    rows = read_jsonl(in_path)
    annotated, summary = annotate(rows, args)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "weighted" in args.formats:
        write_jsonl(out_dir / "sft_mix_weighted.jsonl", annotated)
    if "prompt_response" in args.formats:
        write_jsonl(out_dir / "sft_train_prompt_response.jsonl", [prompt_response_view(r) for r in annotated])
    if "messages" in args.formats:
        write_jsonl(out_dir / "sft_train_messages.jsonl", [messages_view(r) for r in annotated])

    if args.resample_size > 0:
        sampled = weighted_resample(annotated, args.resample_size, args.seed)
        write_jsonl(
            out_dir / f"sft_train_prompt_response_resampled_{args.resample_size}_seed{args.seed}.jsonl",
            [prompt_response_view(r) for r in sampled],
        )
        write_jsonl(
            out_dir / f"sft_train_messages_resampled_{args.resample_size}_seed{args.seed}.jsonl",
            [messages_view(r) for r in sampled],
        )
        summary["resampled_records"] = args.resample_size

    stats_path = out_dir / "stage4_stats.json"
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[done] input: {in_path}")
    print(f"[done] output dir: {out_dir}")
    print(f"[done] raw={summary['raw_records']} kept={summary['kept_records']} dropped={summary['dropped_records']}")
    print(f"[done] kept_by_bus={summary['kept_by_bus']}")
    print(f"[done] kept_by_source={summary['kept_by_source']}")
    print(f"[done] stats: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
