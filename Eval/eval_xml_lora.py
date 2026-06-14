#!/usr/bin/env python3
"""Evaluate XML-only Open Lines outputs with vLLM batch inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.dataset_utils import prepare_train_data  # noqa: E402
from utils.metrics_utils import (  # noqa: E402
    compute_gt_match,
    get_number_of_nodes,
    graph_penalties_from_open_lines,
    parse_open_lines_full_xml,
    parse_open_lines_xml,
    prep_csv,
    write_to_csv,
    write_to_txt,
)
from utils.prompt_format_utils import format_prompt  # noqa: E402


COLUMNS: list[str] = [
    "dataset_index",
    "prompt",
    "num_nodes",
    "available_lines",
    "gen_open_lines",
    "gt_open_lines",
    "is_valid",
    "gt_exact_match",
    "gt_iou",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate XML-only Open Lines outputs with grouped metrics."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", default="",
                        help="LoRA adapter path. Leave empty to evaluate the base model.")
    parser.add_argument("--adapter_name", default="adapter")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--prompt_format", choices=["legacy", "qwen_chat", "llama3_chat"], default="legacy")
    parser.add_argument("--output_xml_format", choices=["open_lines", "full_xml"], default="open_lines",
                        help="Expected XML answer shape to parse from model/GT outputs.")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of samples (-1 = all).")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--save_hard_samples", type=int, default=1,
                        help="Save hard samples (is_valid=0 or gen != GT) as a separate CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if Path.cwd().resolve() != REPO_ROOT:
        raise SystemExit(f"Please run from repo root: {REPO_ROOT}")

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise SystemExit("Failed to import vLLM. Install vllm in this environment.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_txt = str(output_dir / f"{args.adapter_name}_metrics.txt")
    filename_csv = str(output_dir / f"{args.adapter_name}_metrics.csv")

    train_ds, val_ds, test_ds = prepare_train_data(args.data_path)
    split_map = {"train": train_ds, "validation": val_ds, "test": test_ds}
    dataset = split_map[args.split]
    total_available = len(dataset)

    if args.num_samples == -1:
        sample_indices = list(range(total_available))
    else:
        sample_indices = list(range(min(args.num_samples, total_available)))

    prompts = [format_prompt(dataset[int(i)]["prompt"], args.prompt_format)
               for i in sample_indices]
    raw_prompts = [dataset[int(i)]["prompt"] for i in sample_indices]
    correct_outputs = [dataset[int(i)]["output"] for i in sample_indices]
    parse_output_open_lines = (
        parse_open_lines_full_xml
        if args.output_xml_format == "full_xml"
        else parse_open_lines_xml
    )

    use_lora = bool(args.adapter_path)
    llm = LLM(
        model=args.base_model,
        enable_lora=use_lora,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        seed=args.seed,
    )
    stop_tokens = ["</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        stop=stop_tokens,
    )
    lora_request = LoRARequest(args.adapter_name, 1, args.adapter_path) if use_lora else None

    rows: list[dict] = []
    counts = {
        "total": len(sample_indices),
        "improper": 0,
        "proper": 0,
        "valid": 0,
        "invalid": 0,
        "gt_exact_match": 0,
    }
    sums = {
        "cycles": 0.0,
        "invalid_edges": 0.0,
        "subgraphs": 0.0,
        "inference_time": 0.0,
        "gt_iou": 0.0,
    }

    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start: start + args.batch_size]
        batch_correct = correct_outputs[start: start + args.batch_size]

        t0 = perf_counter()
        gen_kwargs = {"lora_request": lora_request} if lora_request else {}
        outputs = llm.generate(batch_prompts, sampling_params, **gen_kwargs)
        batch_time = perf_counter() - t0
        per_sample_time = batch_time / max(len(outputs), 1)

        for batch_offset, (prompt, correct_output, output) in enumerate(
            zip(batch_prompts, batch_correct, outputs)
        ):
            dataset_index = sample_indices[start + batch_offset]
            raw_prompt = raw_prompts[start + batch_offset]
            generated_text = output.outputs[0].text
            gen_open_lines = parse_output_open_lines(generated_text)
            gt_open = parse_output_open_lines(correct_output)
            sums["inference_time"] += per_sample_time

            if not gen_open_lines or not gt_open:
                counts["improper"] += 1
                continue

            counts["proper"] += 1
            available_lines = parse_available_lines(raw_prompt)
            graph_parts = graph_penalties_from_open_lines(raw_prompt, gen_open_lines)
            cycles_loss = float(graph_parts["cycles"])
            invalid_edges_loss = float(graph_parts["invalid_edges"])
            subgraphs_loss = float(graph_parts["subgraphs"])
            sums["cycles"] += cycles_loss
            sums["invalid_edges"] += invalid_edges_loss
            sums["subgraphs"] += subgraphs_loss

            num_nodes = get_number_of_nodes(available_lines)
            is_valid = int(
                invalid_edges_loss == 0.0
                and cycles_loss == 0.0
                and subgraphs_loss == 0.0
            )
            if is_valid:
                counts["valid"] += 1
            else:
                counts["invalid"] += 1

            gt_exact, gt_iou = compute_gt_match(gen_open_lines, gt_open)
            row: dict = {
                "dataset_index": dataset_index,
                "prompt": prompt,
                "num_nodes": num_nodes,
                "available_lines": available_lines,
                "gen_open_lines": gen_open_lines,
                "gt_open_lines": gt_open,
                "is_valid": is_valid,
                "gt_exact_match": int(gt_exact),
                "gt_iou": f"{gt_iou:.6f}",
            }
            if gt_exact == 1.0:
                counts["gt_exact_match"] += 1
            sums["gt_iou"] += gt_iou

            rows.append(row)

    prep_csv(filename_csv, COLUMNS)
    write_to_csv(filename_csv, rows, COLUMNS)

    proper = max(counts["proper"], 1)
    avg_cycles = sums["cycles"] / proper
    avg_invalid = sums["invalid_edges"] / proper
    avg_subgraphs = sums["subgraphs"] / proper
    avg_inference = sums["inference_time"] / max(counts["total"], 1)
    valid_rate = counts["valid"] / proper

    valid_rows = [r for r in rows if r.get("is_valid") == 1]
    valid_n = max(len(valid_rows), 1)
    valid_exact = sum(1 for r in valid_rows if r.get("gt_exact_match") == 1)
    valid_iou = sum(float(r.get("gt_iou") or 0.0) for r in valid_rows) / valid_n

    sections: list[tuple[str, list[tuple[str, str]]]] = []
    sections.append(("Run", [
        ("Split", args.split),
        ("Total samples", str(counts["total"])),
        ("Improper (XML parse failed)", str(counts["improper"])),
        ("Proper XML", str(counts["proper"])),
        ("Avg inference time (s)", f"{avg_inference:.4f}"),
    ]))
    sections.append(("Graph Validity", [
        ("Valid", f"{counts['valid']}  ({valid_rate:.1%} of proper)"),
        ("Invalid", str(counts["invalid"])),
        ("Avg cycles", f"{avg_cycles:.4f}"),
        ("Avg invalid edges", f"{avg_invalid:.4f}"),
        ("Avg subgraphs", f"{avg_subgraphs:.4f}"),
    ]))

    hard_count = counts["proper"] - counts["gt_exact_match"]
    sections.append(("GT Match (undirected)", [
        ("Exact match (all proper)",
         f"{counts['gt_exact_match']} / {counts['proper']}  "
         f"({counts['gt_exact_match']/proper:.1%})"),
        ("Mean IoU (all proper)", f"{sums['gt_iou']/proper:.4f}"),
        ("Exact match (valid only)",
         f"{valid_exact} / {len(valid_rows)}  ({valid_exact/valid_n:.1%})"),
        ("Mean IoU (valid only)", f"{valid_iou:.4f}"),
        ("Hard samples (gen != GT)",
         f"{hard_count} / {counts['proper']}  ({hard_count/proper:.1%})"),
    ]))

    write_to_txt(filename_txt, sections)

    print("---- Evaluation Metrics ----")
    for section_name, items in sections:
        print(f"\n[{section_name}]")
        for label, value in items:
            print(f"  {label}: {value}")
    print(f"\nWrote metrics to {filename_txt}")
    print(f"Wrote CSV     to {filename_csv}")

    hard_rows = [r for r in rows if r.get("is_valid") == 0 or r.get("gt_exact_match") == 0]
    if args.save_hard_samples and hard_rows:
        import csv as _csv
        hard_csv = str(output_dir / f"{args.adapter_name}_hard_samples.csv")
        with open(hard_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=["prompt", "output"])
            writer.writeheader()
            for r in hard_rows:
                idx = r["dataset_index"]
                writer.writerow({
                    "prompt": raw_prompts[sample_indices.index(idx)].strip(),
                    "output": correct_outputs[sample_indices.index(idx)].strip(),
                })
        print(f"Hard samples: {hard_csv}  ({len(hard_rows)} samples, XML-ready for RL)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
