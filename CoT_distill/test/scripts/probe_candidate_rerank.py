"""Probe whether multi-sampling improves long-CoT answer IoU.

This is a test-only script. It does not change the production distillation
pipeline; it samples K independent initial traces per selected problem, evaluates
each candidate with the local hidden verifier, and writes a markdown report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CoT_distill import long_CoT_distill as lcd  # noqa: E402
from utils.config_utils import load_project_env  # noqa: E402
from utils.metrics_utils import parse_open_lines_full_xml  # noqa: E402


def canon(edges):
    out = set()
    for edge in edges or []:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            a, b = int(edge[0]), int(edge[1])
            out.add((min(a, b), max(a, b)))
    return tuple(sorted(out))


def fmt_edges(edges) -> str:
    return "[" + ", ".join(f"({a},{b})" for a, b in edges) + "]"


def candidate_score(local: dict) -> float:
    if local["format"] == 0.0:
        return -100.0
    if local["graph_validity"] == 0.0:
        return -50.0 - local["invalid_edges"] - local["cycles"] - local["subgraphs"]
    return float(local["iou_hidden"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(lcd.DEFAULT_CONFIG_PATH))
    ap.add_argument("--indices", default="0,5840,11680")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--output-dir", default="CoT_distill/test/outputs/candidate_rerank_probe")
    ap.add_argument("--reason-model-name", default="gpt-5.5")
    args = ap.parse_args()

    load_project_env(REPO_ROOT / ".env")
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = lcd.load_yaml_config(config_path)
    cfg.setdefault("models", {}).setdefault("reason", {})["name"] = args.reason_model_name
    # No eval/reflexion calls in this probe, but build_agents still needs eval.
    cfg.setdefault("models", {}).setdefault("eval", {})["name"] = args.reason_model_name
    pipeline_cfg = cfg.get("pipeline") or {}
    timeout_cfg = lcd.parse_timeout_cfg(pipeline_cfg)

    qa_path = lcd.resolve_path((cfg.get("data") or {})["qa_data_path"], base_dir=config_dir)
    qa_data = lcd.load_jsonl_records(str(qa_path), source_name="qa_data_path")
    indices = [int(x) for x in args.indices.split(",") if x.strip()]

    few_shot_cfg = lcd.parse_few_shot_config(cfg, config_dir=config_dir)
    few_shot_pool = lcd.load_few_shot_examples(few_shot_cfg.path)
    reason_agent, evaluate_agent, reflexion_agent = lcd.build_agents(
        cfg,
        config_dir,
        timeout_cfg=timeout_cfg,
    )
    user_templates = lcd.load_user_templates(cfg.get("prompts") or {}, config_dir=config_dir)
    pipeline = lcd.ReconfigCoTPipeline(
        reason_agent=reason_agent,
        evaluate_agent=evaluate_agent,
        reflexion_agent=reflexion_agent,
        problems=[],
        max_iterations=1,
        score_threshold=float(pipeline_cfg.get("score_threshold", 0.75)),
        output_path=None,
        batch_size=1,
        max_workers=1,
        few_shot_pool=few_shot_pool,
        few_shot_num_samples=few_shot_cfg.num_samples,
        few_shot_seed=few_shot_cfg.seed,
        quality_cfg=pipeline_cfg,
        timeout_cfg=timeout_cfg,
        user_templates=user_templates,
    )

    results = []
    for source_index in indices:
        sample = qa_data[source_index]
        problem = lcd.extract_question(sample)
        solution = lcd.extract_solution(sample)
        meta = sample.get("meta") or {}
        problem_obj = {"problem": problem, "solution": solution}

        lcd._few_shot_ctx.examples_str = pipeline._sample_few_shot_str(problem_obj)
        lcd._few_shot_ctx.solution = solution

        candidates = []
        for cidx in range(args.k):
            trace = pipeline.generate_reasoning_trace(problem)
            local = pipeline._compute_verifier_metrics(problem, trace)
            pred = canon(parse_open_lines_full_xml(trace))
            gt = canon(parse_open_lines_full_xml(solution))
            hit = len(set(pred) & set(gt))
            candidates.append(
                {
                    "candidate": cidx + 1,
                    "score": candidate_score(local),
                    "format": local["format"],
                    "valid": local["graph_validity"],
                    "iou": local["iou_hidden"],
                    "exact": local["gt_exact_match"],
                    "invalid_edges": local["invalid_edges"],
                    "cycles": local["cycles"],
                    "subgraphs": local["subgraphs"],
                    "pred_edges": pred,
                    "gt_edges": gt,
                    "hit": hit,
                    "trace": trace,
                }
            )
            print(
                f"idx={source_index} cand={cidx + 1}/{args.k} "
                f"valid={local['graph_validity']:.0f} iou={local['iou_hidden']:.4f}",
                flush=True,
            )

        best = max(candidates, key=lambda x: x["score"])
        first = candidates[0]
        results.append(
            {
                "source_index": source_index,
                "sample_id": meta.get("sample_id"),
                "bus": meta.get("bus"),
                "first_iou": first["iou"],
                "best_iou": best["iou"],
                "first_exact": first["exact"],
                "best_exact": best["exact"],
                "candidates": candidates,
            }
        )

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "candidate_rerank_probe_raw.json"
    report_path = out_dir / "candidate_rerank_probe_report.md"
    raw_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    first_mean = sum(r["first_iou"] for r in results) / len(results)
    best_mean = sum(r["best_iou"] for r in results) / len(results)
    lines = [
        "# Candidate Rerank Probe",
        "",
        f"- K: `{args.k}`",
        f"- indices: `{indices}`",
        f"- Raw: `{raw_path.relative_to(REPO_ROOT)}`",
        "",
        "## Summary",
        "",
        "| n | first mean IoU | best-of-K mean IoU | delta | first exact | best exact |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {len(results)} | {first_mean:.4f} | {best_mean:.4f} | "
            f"{best_mean - first_mean:+.4f} | "
            f"{sum(r['first_exact'] for r in results)}/{len(results)} | "
            f"{sum(r['best_exact'] for r in results)}/{len(results)} |"
        ),
        "",
        "## Per Sample",
        "",
        "| sample | bus | first IoU | best IoU | delta | candidate IoUs |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        ious = ", ".join(f"{c['iou']:.4f}" for c in r["candidates"])
        lines.append(
            f"| {r['sample_id']} | {r['bus']} | {r['first_iou']:.4f} | "
            f"{r['best_iou']:.4f} | {r['best_iou'] - r['first_iou']:+.4f} | {ious} |"
        )
    lines.append("")
    lines.append("## Candidates")
    for r in results:
        lines.append(f"\n### {r['sample_id']} ({r['bus']}-bus)\n")
        for c in r["candidates"]:
            lines.append(
                f"#### Candidate {c['candidate']} | valid={int(c['valid'])} "
                f"IoU={c['iou']:.4f} exact={int(c['exact'])}"
            )
            lines.append(f"- Pred: `{fmt_edges(c['pred_edges'])}`")
            lines.append(f"- GT: `{fmt_edges(c['gt_edges'])}`")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
