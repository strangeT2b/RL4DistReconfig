#!/usr/bin/env python3
"""Evaluate closed-source models via OpenAI-compatible API."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.config_utils import load_project_env  # noqa: E402
from utils.dataset_utils import prepare_train_data  # noqa: E402
from utils.metrics_utils import (  # noqa: E402
    compute_gt_match,
    graph_penalties_from_open_lines,
    parse_available_lines,
    parse_open_lines_full_xml,
    parse_open_lines_xml,
    prep_csv,
    write_to_csv,
    write_to_txt,
)
from utils.prompt_format_utils import format_prompt  # noqa: E402


COLUMNS = [
    "dataset_index",
    "prompt",
    "gen_open_lines",
    "gt_open_lines",
    "is_valid",
    "gt_exact_match",
    "gt_iou",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate via OpenAI-compatible API.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_name", default=os.environ.get("API_MODEL", "gpt-4o"),
                        help="Model name for the API.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--prompt_format", choices=["legacy", "qwen_chat", "llama3_chat"], default="llama3_chat")
    parser.add_argument("--output_xml_format", choices=["open_lines", "full_xml"], default="full_xml")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--save_hard_samples", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    load_project_env(REPO_ROOT / ".env")

    api_base = os.environ.get("API_BASE", "")
    api_key = os.environ.get("API_KEY", "")

    if not api_base or not api_key:
        raise SystemExit("Set API_BASE and API_KEY in .env")

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Install openai: pip install openai")

    client = OpenAI(base_url=api_base, api_key=api_key)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.model_name.replace("/", "_")
    filename_txt = str(output_dir / f"{name}_metrics.txt")
    filename_csv = str(output_dir / f"{name}_metrics.csv")

    _, _, test_ds = prepare_train_data(args.data_path)
    dataset = test_ds
    total = len(dataset)

    if args.num_samples == -1:
        indices = list(range(total))
    else:
        indices = list(range(min(args.num_samples, total)))

    parse_output_open_lines = (
        parse_open_lines_full_xml
        if args.output_xml_format == "full_xml"
        else parse_open_lines_xml
    )

    rows = []
    counts = {"total": len(indices), "improper": 0, "proper": 0, "valid": 0, "invalid": 0, "exact": 0}
    sums = {"cycles": 0.0, "invalid_edges": 0.0, "subgraphs": 0.0, "inference_time": 0.0, "iou": 0.0}

    for i in indices:
        raw_prompt = dataset[int(i)]["prompt"]
        gt_output = dataset[int(i)]["output"]
        prompt = format_prompt(raw_prompt, args.prompt_format)

        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
            )
            gen_text = resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"[{i}] API error: {exc}")
            counts["improper"] += 1
            continue

        elapsed = time.perf_counter() - t0
        sums["inference_time"] += elapsed

        gen_open = parse_output_open_lines(gen_text)
        gt_open = parse_output_open_lines(gt_output)

        if not gen_open or not gt_open:
            counts["improper"] += 1
            continue

        counts["proper"] += 1
        parts = graph_penalties_from_open_lines(raw_prompt, gen_open)
        cycles = float(parts["cycles"])
        invalid = float(parts["invalid_edges"])
        subgraphs = float(parts["subgraphs"])
        sums["cycles"] += cycles
        sums["invalid_edges"] += invalid
        sums["subgraphs"] += subgraphs

        is_valid = int(cycles == 0 and invalid == 0 and subgraphs == 0)
        if is_valid:
            counts["valid"] += 1
        else:
            counts["invalid"] += 1

        exact, iou = compute_gt_match(gen_open, gt_open)
        if exact:
            counts["exact"] += 1
        sums["iou"] += iou

        rows.append({
            "dataset_index": i,
            "prompt": prompt,
            "gen_open_lines": gen_open,
            "gt_open_lines": gt_open,
            "is_valid": is_valid,
            "gt_exact_match": int(exact),
            "gt_iou": f"{iou:.6f}",
        })

    prep_csv(filename_csv, COLUMNS)
    write_to_csv(filename_csv, rows, COLUMNS)

    proper = max(counts["proper"], 1)
    avg_inference = sums["inference_time"] / max(counts["total"], 1)
    valid_rate = counts["valid"] / proper

    valid_rows = [r for r in rows if r["is_valid"] == 1]
    valid_n = max(len(valid_rows), 1)
    valid_exact = sum(r["gt_exact_match"] for r in valid_rows)
    valid_iou = sum(float(r["gt_iou"]) for r in valid_rows) / valid_n

    sections = [
        ("Run", [
            ("Split", args.split),
            ("Model", args.model_name),
            ("Total samples", str(counts["total"])),
            ("Improper (XML parse failed)", str(counts["improper"])),
            ("Proper XML", str(counts["proper"])),
            ("Avg inference time (s)", f"{avg_inference:.4f}"),
        ]),
        ("Graph Validity", [
            ("Valid", f"{counts['valid']}  ({valid_rate:.1%} of proper)"),
            ("Invalid", str(counts["invalid"])),
            ("Avg cycles", f"{sums['cycles']/proper:.4f}"),
            ("Avg invalid edges", f"{sums['invalid_edges']/proper:.4f}"),
            ("Avg subgraphs", f"{sums['subgraphs']/proper:.4f}"),
        ]),
        ("GT Match (undirected)", [
            ("Exact match (all proper)",
             f"{counts['exact']} / {counts['proper']}  ({counts['exact']/proper:.1%})"),
            ("Mean IoU (all proper)", f"{sums['iou']/proper:.4f}"),
            ("Exact match (valid only)",
             f"{valid_exact} / {valid_n}  ({valid_exact/valid_n:.1%})"),
            ("Mean IoU (valid only)", f"{valid_iou:.4f}"),
        ]),
    ]

    write_to_txt(filename_txt, sections)
    print("---- Evaluation Metrics ----")
    for sec, items in sections:
        print(f"\n[{sec}]")
        for k, v in items:
            print(f"  {k}: {v}")
    print(f"\nWrote metrics to {filename_txt}")
    print(f"Wrote CSV     to {filename_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
