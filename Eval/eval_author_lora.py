#!/usr/bin/env python3
"""Evaluate a LoRA adapter with vLLM batch inference.

Per-sample metrics, grouped:
    [identification] dataset_index, prompt, num_nodes, available_lines
    [model output]   gen_open_lines, gen_node_voltages, gen_system_loss
    [ground truth]   gt_open_lines, gt_node_voltages, gt_system_loss_dataset
    [graph validity] is_valid
    [model sim]      model_sim_*
    [gt sim]         gt_sim_*
    [comparison]     gt_exact_match, gt_iou, model_vs_gt_pct
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.dataset_utils import prepare_train_data  # noqa: E402
from utils.generation_utils import extract_output_data  # noqa: E402
from utils.metrics_utils import (  # noqa: E402
    compute_gt_match,
    get_number_of_nodes,
    graph_penalties_from_open_lines,
    parse_available_lines,
    prep_csv,
    write_to_csv,
    write_to_txt,
)
from utils.prompt_format_utils import format_prompt  # noqa: E402


COLUMNS: list[str] = [
    # identification
    "dataset_index",
    "prompt",
    "num_nodes",
    "available_lines",
    # model output (parsed from generation)
    "gen_open_lines",
    "gen_node_voltages",
    "gen_system_loss",
    # ground truth (parsed from dataset output field)
    "gt_open_lines",
    "gt_node_voltages",
    "gt_system_loss_dataset",
    # graph validity (computed on generated output)
    "is_valid",
    # pandapower simulation of model's answer
    "model_sim_converged",
    "model_sim_original_loss_mw",
    "model_sim_new_loss_mw",
    "model_sim_improvement_pct",
    "model_sim_failure_reason",
    # pandapower simulation of GT answer
    "gt_sim_converged",
    "gt_sim_loss_mw",
    "gt_sim_failure_reason",
    # model vs GT comparison (undirected)
    "gt_exact_match",
    "gt_iou",
    "model_vs_gt_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a vLLM LoRA adapter with grouped metrics."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", default="",
                        help="LoRA adapter path. Leave empty to evaluate the base model.")
    parser.add_argument("--adapter_name", default="adapter")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--prompt_format", choices=["legacy", "qwen_chat", "llama3_chat"], default="legacy")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of samples (-1 = all).")
    parser.add_argument("--sample_mode", choices=["sequential", "author_random"],
                        default="sequential",
                        help=("Sample selection mode. sequential preserves the old evaluator "
                              "behavior; author_random matches the author's torch.randint "
                              "sampling with replacement."))
    parser.add_argument("--sample_seed", type=int, default=None,
                        help=("Optional seed for author_random sampling. Leave unset to match "
                              "the author's unseeded torch.randint behavior."))
    parser.add_argument("--max_new_tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling. Use 5 to match the author's default generation config.")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--save_hard_samples", type=int, default=1,
                        help="Save hard samples (is_valid=0) as a separate CSV for RL training.")
    parser.add_argument("--sim_loss", type=int, default=1,
                        help="Run pandapower simulator on valid model outputs.")
    parser.add_argument("--gt_sim", type=int, default=1,
                        help="Re-simulate GT open lines so GT loss is comparable to model loss.")
    return parser.parse_args()


def _select_sample_indices(
    total_available: int,
    num_samples: int,
    sample_mode: str,
    sample_seed: int | None,
) -> list[int]:
    if total_available <= 0:
        return []

    if sample_mode == "author_random":
        if num_samples > total_available:
            raise ValueError("num_samples cannot exceed dataset size")
        draw_count = total_available if num_samples == -1 else num_samples
        import torch

        if sample_seed is None:
            return torch.randint(0, total_available, (draw_count,)).tolist()

        generator = torch.Generator()
        generator.manual_seed(sample_seed)
        return torch.randint(
            0,
            total_available,
            (draw_count,),
            generator=generator,
        ).tolist()

    if num_samples == -1:
        return list(range(total_available))
    return list(range(min(num_samples, total_available)))


def _run_sim(prompt: str, open_lines):
    """Wrapper around evaluate_reconfig that returns (ok, result_or_reason)."""
    try:
        from RL.simulator.grid_simulator import evaluate_reconfig
        result = evaluate_reconfig(prompt, list(open_lines))
        return True, result
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _fmt_pct(x):
    return f"{x:+.2f}%" if x is not None else "n/a"


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

    sample_indices = _select_sample_indices(
        total_available,
        args.num_samples,
        args.sample_mode,
        args.sample_seed,
    )

    prompts = [format_prompt(dataset[int(i)]["prompt"], args.prompt_format)
               for i in sample_indices]
    correct_outputs = [dataset[int(i)]["output"] for i in sample_indices]

    use_lora = bool(args.adapter_path)
    llm = LLM(
        model=args.base_model,
        enable_lora=use_lora,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        seed=args.seed,
    )
    eos = "<" + "/s>"
    endoftext = "<" + "|" + "endoftext" + "|" + ">"
    eot_id = "<" + "|" + "eot_id" + "|" + ">"
    stop_tokens = [eos, "<|im_end|>", endoftext, eot_id]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        stop=stop_tokens,
    )
    lora_request = LoRARequest(args.adapter_name, 1, args.adapter_path) if use_lora else None

    rows: list[dict] = []

    counts = {
        "total": len(sample_indices),
        "no_response": 0,
        "proper": 0,
        "valid": 0,
        "invalid": 0,
        "model_sim_converged": 0,
        "model_sim_improved": 0,
        "model_sim_unchanged": 0,
        "model_sim_worsened": 0,
        "model_sim_failed": 0,
        "gt_sim_converged": 0,
        "gt_sim_failed": 0,
        "both_converged": 0,
        "better_than_gt": 0,
        "matches_gt_loss": 0,
        "worse_than_gt": 0,
        "gt_exact_match": 0,
    }
    sums = {
        "cycles": 0.0,
        "invalid_edges": 0.0,
        "subgraphs": 0.0,
        "inference_time": 0.0,
        "model_sim_original_loss": 0.0,
        "model_sim_new_loss": 0.0,
        "model_sim_improvement_pct": 0.0,
        "gt_sim_loss": 0.0,
        "model_vs_gt_pct": 0.0,
        "gt_iou": 0.0,
        # both-converged subset, for directly comparable averages
        "both_orig_loss": 0.0,
        "both_model_loss": 0.0,
        "both_gt_loss": 0.0,
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
            generated_text = output.outputs[0].text
            reformatted = extract_output_data(generated_text)
            correct_data = extract_output_data(correct_output)

            sums["inference_time"] += per_sample_time

            if reformatted == "No output data found in the response." or correct_data is None:
                counts["no_response"] += 1
                continue

            counts["proper"] += 1

            gen_open_lines = reformatted["Open Lines"]
            gen_voltages = reformatted["Node Voltages"]
            gen_sys_loss = reformatted["System Loss"]

            predicted_lines = gen_open_lines
            if predicted_lines and not all(
                isinstance(line, (tuple, list)) and len(line) == 2 for line in predicted_lines
            ):
                predicted_lines = []
            available_lines = parse_available_lines(prompt)

            graph_parts = graph_penalties_from_open_lines(prompt, predicted_lines)
            cycles_loss = float(graph_parts["cycles"])
            invalid_edges_loss = float(graph_parts["invalid_edges"])
            subgraphs_loss = float(graph_parts["subgraphs"])
            sums["cycles"] += cycles_loss
            sums["invalid_edges"] += invalid_edges_loss
            sums["subgraphs"] += subgraphs_loss

            num_nodes = get_number_of_nodes(available_lines)
            gt_open = correct_data["Open Lines"]
            gt_voltages = correct_data["Node Voltages"]
            gt_sys_loss_dataset = correct_data["System Loss"]

            is_valid = int(
                invalid_edges_loss == 0.0
                and cycles_loss == 0.0
                and subgraphs_loss == 0.0
            )
            if is_valid:
                counts["valid"] += 1
            else:
                counts["invalid"] += 1

            row: dict = {
                "dataset_index": dataset_index,
                "prompt": prompt,
                "_correct_output": correct_output.strip(),
                "num_nodes": num_nodes,
                "available_lines": available_lines,
                "gen_open_lines": gen_open_lines,
                "gen_node_voltages": gen_voltages,
                "gen_system_loss": gen_sys_loss,
                "gt_open_lines": gt_open,
                "gt_node_voltages": gt_voltages,
                "gt_system_loss_dataset": gt_sys_loss_dataset,
                "is_valid": is_valid,
                "model_sim_failure_reason": "not_run",
                "gt_sim_failure_reason": "not_run",
            }

            # ---- model sim ----
            model_sim_new_loss = None
            model_sim_orig_loss = None
            if args.sim_loss:
                if not is_valid:
                    row["model_sim_failure_reason"] = "not_run_graph_invalid"
                elif not gen_open_lines:
                    row["model_sim_failure_reason"] = "not_run_no_open_lines"
                else:
                    ok, res = _run_sim(prompt, gen_open_lines)
                    if not ok:
                        counts["model_sim_failed"] += 1
                        row["model_sim_converged"] = 0
                        row["model_sim_failure_reason"] = res
                    else:
                        conv = bool(res.get("converged", False))
                        orig_raw = res.get("original_loss_mw")
                        new_raw = res.get("system_loss")
                        orig_loss = float(orig_raw) if orig_raw is not None else -1.0
                        new_loss = float(new_raw) if new_raw is not None else -1.0
                        row["model_sim_converged"] = int(conv)
                        if conv and orig_loss > 0:
                            counts["model_sim_converged"] += 1
                            impr_pct = (orig_loss - new_loss) / orig_loss * 100
                            row["model_sim_original_loss_mw"] = f"{orig_loss:.10g}"
                            row["model_sim_new_loss_mw"] = f"{new_loss:.10g}"
                            row["model_sim_improvement_pct"] = f"{impr_pct:.6f}"
                            row["model_sim_failure_reason"] = ""
                            sums["model_sim_original_loss"] += orig_loss
                            sums["model_sim_new_loss"] += new_loss
                            sums["model_sim_improvement_pct"] += impr_pct
                            model_sim_new_loss = new_loss
                            model_sim_orig_loss = orig_loss
                            if impr_pct > 0:
                                counts["model_sim_improved"] += 1
                            elif impr_pct < 0:
                                counts["model_sim_worsened"] += 1
                            else:
                                counts["model_sim_unchanged"] += 1
                        else:
                            counts["model_sim_failed"] += 1
                            row["model_sim_failure_reason"] = (
                                "not_converged" if not conv else "missing_original_loss"
                            )

            # ---- GT sim (no cache: rerun for every sample) ----
            gt_sim_new_loss = None
            if args.gt_sim:
                if not gt_open:
                    row["gt_sim_failure_reason"] = "not_run_no_gt_open_lines"
                else:
                    ok, res = _run_sim(prompt, gt_open)
                    if not ok:
                        counts["gt_sim_failed"] += 1
                        row["gt_sim_converged"] = 0
                        row["gt_sim_failure_reason"] = res
                    else:
                        conv = bool(res.get("converged", False))
                        new_raw = res.get("system_loss")
                        new_loss = float(new_raw) if new_raw is not None else -1.0
                        row["gt_sim_converged"] = int(conv)
                        if conv and new_loss >= 0:
                            row["gt_sim_loss_mw"] = f"{new_loss:.10g}"
                            row["gt_sim_failure_reason"] = ""
                            counts["gt_sim_converged"] += 1
                            sums["gt_sim_loss"] += new_loss
                            gt_sim_new_loss = new_loss
                        else:
                            counts["gt_sim_failed"] += 1
                            row["gt_sim_failure_reason"] = "not_converged"

            # ---- comparison: undirected exact-match, IoU, model_vs_gt_pct ----
            gt_exact, gt_iou = compute_gt_match(gen_open_lines, gt_open)
            row["gt_exact_match"] = int(gt_exact)
            row["gt_iou"] = f"{gt_iou:.6f}"
            if gt_exact == 1.0:
                counts["gt_exact_match"] += 1
            sums["gt_iou"] += gt_iou

            if model_sim_new_loss is not None and gt_sim_new_loss is not None and gt_sim_new_loss > 0:
                model_vs_gt = (gt_sim_new_loss - model_sim_new_loss) / gt_sim_new_loss * 100
                row["model_vs_gt_pct"] = f"{model_vs_gt:.6f}"
                counts["both_converged"] += 1
                sums["model_vs_gt_pct"] += model_vs_gt
                if model_sim_orig_loss is not None:
                    sums["both_orig_loss"] += model_sim_orig_loss
                sums["both_model_loss"] += model_sim_new_loss
                sums["both_gt_loss"] += gt_sim_new_loss
                eps = 1e-6
                if model_vs_gt > eps:
                    counts["better_than_gt"] += 1
                elif model_vs_gt < -eps:
                    counts["worse_than_gt"] += 1
                else:
                    counts["matches_gt_loss"] += 1

            rows.append(row)

    # ---- write CSV ----
    prep_csv(filename_csv, COLUMNS)
    write_to_csv(filename_csv, rows, COLUMNS)

    # ---- build summary sections ----
    proper = max(counts["proper"], 1)
    avg_cycles = sums["cycles"] / proper
    avg_invalid = sums["invalid_edges"] / proper
    avg_subgraphs = sums["subgraphs"] / proper
    avg_inference = sums["inference_time"] / max(counts["total"], 1)
    valid_rate = counts["valid"] / proper

    # Subset stats: gt match restricted to valid samples (more meaningful)
    valid_rows = [r for r in rows if r.get("is_valid") == 1]
    valid_n = max(len(valid_rows), 1)
    valid_exact = sum(1 for r in valid_rows if r.get("gt_exact_match") == 1)
    valid_iou = sum(float(r.get("gt_iou") or 0.0) for r in valid_rows) / valid_n

    sections: list[tuple[str, list[tuple[str, str]]]] = []
    sections.append(("Run", [
        ("Split", args.split),
        ("Total samples", str(counts["total"])),
        ("Improper (parse failed)", str(counts["no_response"])),
        ("Proper", str(counts["proper"])),
        ("Avg inference time (s)", f"{avg_inference:.4f}"),
    ]))
    sections.append(("Graph Validity", [
        ("Valid", f"{counts['valid']}  ({valid_rate:.1%} of proper)"),
        ("Invalid", str(counts["invalid"])),
        ("Avg cycles", f"{avg_cycles:.4f}"),
        ("Avg invalid edges", f"{avg_invalid:.4f}"),
        ("Avg subgraphs", f"{avg_subgraphs:.4f}"),
    ]))

    if args.sim_loss:
        msc = max(counts["model_sim_converged"], 1)
        sections.append(("Model Sim (on valid outputs)", [
            ("Converged", f"{counts['model_sim_converged']} / {counts['valid']}"),
            ("Loss improved vs original", str(counts["model_sim_improved"])),
            ("Loss unchanged vs original (model echoed default)",
             str(counts["model_sim_unchanged"])),
            ("Loss worsened vs original", str(counts["model_sim_worsened"])),
            ("Sim failed", str(counts["model_sim_failed"])),
            ("Avg original loss (MW)", f"{sums['model_sim_original_loss']/msc:.4f}"
             if counts["model_sim_converged"] else "n/a"),
            ("Avg model new loss (MW)", f"{sums['model_sim_new_loss']/msc:.4f}"
             if counts["model_sim_converged"] else "n/a"),
            ("Avg improvement vs original",
             _fmt_pct(sums["model_sim_improvement_pct"]/msc)
             if counts["model_sim_converged"] else "n/a"),
        ]))

    if args.gt_sim:
        gsc = max(counts["gt_sim_converged"], 1)
        sections.append(("GT Sim (re-simulated on same simulator)", [
            ("Converged", f"{counts['gt_sim_converged']} / {counts['proper']}"),
            ("Failed", str(counts["gt_sim_failed"])),
            ("Avg GT sim loss (MW)", f"{sums['gt_sim_loss']/gsc:.4f}"
             if counts["gt_sim_converged"] else "n/a"),
        ]))

    hard_count = counts["proper"] - counts["gt_exact_match"]
    sections.append(("GT Match (undirected)", [
        ("Exact match (all proper)",
         f"{counts['gt_exact_match']} / {counts['proper']}  "
         f"({counts['gt_exact_match']/proper:.1%})"),
        ("Mean IoU (all proper)", f"{sums['gt_iou']/proper:.4f}"),
        ("Exact match (valid only)",
         f"{valid_exact} / {len(valid_rows)}  "
         f"({valid_exact/valid_n:.1%})"),
        ("Mean IoU (valid only)", f"{valid_iou:.4f}"),
        ("Hard samples (gen != GT)",
         f"{hard_count} / {counts['proper']}  ({hard_count/proper:.1%})"),
    ]))

    if args.sim_loss and args.gt_sim:
        bn = max(counts["both_converged"], 1)
        has_both = counts["both_converged"] > 0
        sections.append(("Model vs GT (both sims converged)", [
            ("Both-converged samples", str(counts["both_converged"])),
            ("Model better than GT", f"{counts['better_than_gt']}  "
             f"({counts['better_than_gt']/bn:.1%})"),
            ("Model matches GT loss", str(counts["matches_gt_loss"])),
            ("Model worse than GT", f"{counts['worse_than_gt']}  "
             f"({counts['worse_than_gt']/bn:.1%})"),
            ("Avg original loss (MW)",
             f"{sums['both_orig_loss']/bn:.4f}" if has_both else "n/a"),
            ("Avg model new loss (MW)",
             f"{sums['both_model_loss']/bn:.4f}" if has_both else "n/a"),
            ("Avg GT loss (MW)",
             f"{sums['both_gt_loss']/bn:.4f}" if has_both else "n/a"),
            ("Mean model_vs_gt_pct (>0 = better than GT)",
             _fmt_pct(sums["model_vs_gt_pct"]/bn) if has_both else "n/a"),
        ]))

    write_to_txt(filename_txt, sections)

    # ---- stdout summary ----
    print("---- Evaluation Metrics ----")
    for section_name, items in sections:
        print(f"\n[{section_name}]")
        for label, value in items:
            print(f"  {label}: {value}")
    print(f"\nWrote metrics to {filename_txt}")
    print(f"Wrote CSV     to {filename_csv}")

    # ---- save hard samples for RL (anything that doesn't match GT exactly) ----
    hard_rows = [r for r in rows if r.get("gt_exact_match") == 0]
    if args.save_hard_samples and hard_rows:
        import csv as _csv
        hard_csv = str(output_dir / f"{args.adapter_name}_hard_samples.csv")
        with open(hard_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=["prompt", "output"])
            writer.writeheader()
            for r in hard_rows:
                writer.writerow({
                    "prompt": r["prompt"],
                    "output": r["_correct_output"],
                })
        print(f"Hard samples: {hard_csv}  ({len(hard_rows)} samples, ready for RL)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
