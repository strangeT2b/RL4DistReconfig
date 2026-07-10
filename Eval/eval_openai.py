#!/usr/bin/env python3
"""Evaluate closed-source models via OpenAI-compatible API."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.config_utils import load_project_env  # noqa: E402
from utils.dataset_utils import prepare_train_data  # noqa: E402
from utils.metrics_utils import (  # noqa: E402
    compute_edge_precision_recall,
    compute_gt_match,
    graph_penalties_from_open_lines,
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
    "gt_precision",
    "gt_recall",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate via OpenAI-compatible API.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_name", default="gpt-4o")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--prompt_format", default="llama3_chat")
    parser.add_argument("--output_xml_format", choices=["open_lines", "full_xml", "legacy"], default="full_xml",
                        help="legacy = old Open Lines=[...] format")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--sample_mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--save_hard_samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--reasoning_effort", choices=["none", "low", "medium", "high"], default="none",
                        help="Set reasoning_effort for reasoning-capable models. 'none' omits the param.")
    parser.add_argument("--stream", type=int, default=1,
                        help="Use SSE streaming to bypass gateway idle timeouts (e.g. pjlab 60s 504). 1=on, 0=off.")
    parser.add_argument("--few_shot", default=None,
                        help="Optional JSONL of few-shot demos (bus, prompt, output). Each demo's input+GT "
                             "answer is prepended to every test prompt as an in-context example.")
    parser.add_argument("--api_env", choices=["API", "PJ_API", "MY"], default="PJ_API",
                        help="Which .env gateway: API, PJ_API, or MY. MY reads MY_BASE_URL/MY_API_KEY.")
    return parser.parse_args()


def _collect_stream(resp):
    """Parse an SSE OpenAI chat stream and return (content, reasoning, finish_reason).

    `content` is the model's final answer channel; `reasoning` collects the
    thinking channel (reasoning_content / reasoning). Both are kept separately so
    callers can persist the full reasoning chain alongside the final answer.
    """
    import json as _json
    content_parts = []
    rc_parts = []
    rsn_parts = []
    finish = None
    for line in resp.iter_lines():
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            chunk = _json.loads(data)
        except (ValueError, SyntaxError):
            continue
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta", {}) or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("reasoning_content"):
            rc_parts.append(delta["reasoning_content"])
        if delta.get("reasoning"):
            rsn_parts.append(delta["reasoning"])
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
    content = "".join(content_parts)
    reasoning = "".join(rc_parts) or "".join(rsn_parts)
    return content, reasoning, finish


def _call_one(api_url, api_key, model, prompt, temp, top_p, max_tok, idx, timeout,
              reasoning_effort="none", stream=True):
    """Return (idx, content, reasoning, finish_reason, error)."""
    import random
    import httpx
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "top_p": top_p,
        "max_tokens": max_tok,
    }
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning_effort"] = reasoning_effort
    if stream:
        payload["stream"] = True

    # Rate-limit (429) aware retry: more attempts + longer backoff with jitter so
    # concurrent workers don't all retry at once and re-trigger the limit.
    MAX_ATTEMPTS = 8
    for attempt in range(MAX_ATTEMPTS):
        try:
            if stream:
                # Streaming keeps the connection alive with incremental bytes,
                # bypassing gateway (e.g. pjlab nginx) idle read timeouts that
                # otherwise 504 long-running generations.
                with httpx.stream("POST", api_url, json=payload, headers=headers, timeout=timeout) as resp:
                    if resp.status_code == 429:
                        # Rate limited: back off and retry. Respect Retry-After if present.
                        wait = float(resp.headers.get("Retry-After", 0)) or (5 * (2 ** attempt) + random.uniform(0, 3))
                        time.sleep(min(wait, 120))
                        continue
                    if resp.status_code != 200:
                        err = f"HTTP {resp.status_code}: {resp.read().decode(errors='replace')[:200]}"
                    else:
                        content, reasoning, finish = _collect_stream(resp)
                        return idx, content, reasoning, finish, None
            else:
                resp = httpx.post(api_url, json=payload, headers=headers, timeout=timeout)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", 0)) or (5 * (2 ** attempt) + random.uniform(0, 3))
                    time.sleep(min(wait, 120))
                    continue
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    msg = choice.get("message", {})
                    # Some gateways return the answer in the reasoning field and leave
                    # `content` empty/None when reasoning is on. Field name varies by
                    # gateway (pjlab: "reasoning"; some OpenAI proxies: "reasoning_content").
                    content = msg.get("content") or ""
                    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                    return idx, content, reasoning, choice.get("finish_reason"), None
                err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            err = str(exc)
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(2 ** attempt + random.uniform(0, 1))
    return idx, "", "", None, err


def main():
    args = parse_args()
    load_project_env(REPO_ROOT / ".env")

    # Choose gateway: --api_env API/PJ_API read <PREFIX>_BASE/<PREFIX>_KEY.
    # MY is kept compatible with the local .env names MY_BASE_URL/MY_API_KEY.
    base_key = args.api_env
    if base_key == "MY":
        api_base = (os.environ.get("MY_BASE_URL") or os.environ.get("MY_BASE") or "").rstrip("/")
        api_key = os.environ.get("MY_API_KEY") or os.environ.get("MY_KEY") or ""
        expected = "MY_BASE_URL and MY_API_KEY"
    else:
        api_base = os.environ.get(f"{base_key}_BASE", "").rstrip("/")
        api_key = os.environ.get(f"{base_key}_KEY", "")
        expected = f"{base_key}_BASE and {base_key}_KEY"

    if not api_base or not api_key:
        raise SystemExit(f"Set {expected} in .env")

    api_url = f"{api_base}/chat/completions"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.model_name.replace("/", "_")
    filename_txt = str(output_dir / f"{name}_metrics.txt")
    filename_csv = str(output_dir / f"{name}_metrics.csv")
    filename_jsonl = str(output_dir / f"{name}_generations.jsonl")

    # Load ONLY the requested split to avoid OOM on multi-million-row CSVs.
    # (prepare_train_data loads all splits + builds a `text` column via pandas.)
    # Read just prompt/output/split in chunks and keep the target split only.
    import pandas as _pd
    split = args.split
    parts = []
    for chunk in _pd.read_csv(args.data_path, usecols=["prompt", "output", "split"],
                              chunksize=200_000):
        parts.append(chunk[chunk["split"] == split][["prompt", "output"]])
    df = _pd.concat(parts, ignore_index=True) if parts else _pd.DataFrame(columns=["prompt", "output"])
    dataset = df.to_dict("records")
    total = len(dataset)
    print(f"Loaded {total} samples (split={split})")

    if args.num_samples == -1:
        indices = list(range(total))
    elif args.sample_mode == "random":
        n = min(args.num_samples, total)
        indices = random.Random(args.seed).sample(range(total), n)
    else:
        indices = list(range(min(args.num_samples, total)))

    if args.output_xml_format == "full_xml":
        parse_output = parse_open_lines_full_xml
    elif args.output_xml_format == "open_lines":
        parse_output = parse_open_lines_xml
    else:
        from utils.metrics_utils import parse_open_lines
        parse_output = parse_open_lines

    # Few-shot demos: each demo is prepended as "input -> <answer>GT</answer>"
    # to every test prompt, as in-context examples. Demos come ONLY from train
    # split (no leakage into test).
    few_shot_block = ""
    if args.few_shot:
        import json as _fsjson
        demos = []
        with open(args.few_shot, encoding="utf-8") as _fsf:
            for line in _fsf:
                line = line.strip()
                if line:
                    demos.append(_fsjson.loads(line))
        # Build the few-shot prefix: each demo shows a task input followed by
        # its GT <answer>...</answer>, so the model sees correct reconfigurations.
        parts = []
        for d in demos:
            parts.append(d["prompt"].strip() + "\n" + d["output"].strip())
        few_shot_block = "\n\n".join(parts) + "\n\n"
        print(f"Few-shot: {len(demos)} demos loaded from {args.few_shot}")

    prompts_data = []
    for i in indices:
        raw_prompt = dataset[int(i)]["prompt"]
        # In few-shot mode, prepend the demo block before the test input.
        full_prompt = few_shot_block + raw_prompt if few_shot_block else raw_prompt
        prompts_data.append((i, raw_prompt, dataset[int(i)]["output"],
                             format_prompt(full_prompt, args.prompt_format)))

    # Resume support: skip indices already present in the generations JSONL.
    # Re-running with the same --output_dir continues from where it stopped.
    import json as _json
    done_idx = set()
    if Path(filename_jsonl).exists():
        with open(filename_jsonl, encoding="utf-8") as _jf:
            for line in _jf:
                try:
                    done_idx.add(_json.loads(line)["dataset_index"])
                except (ValueError, KeyError):
                    pass
        if done_idx:
            print(f"Resuming: {len(done_idx)} samples already in {filename_jsonl}, skipping them.")
    prompts_data = [p for p in prompts_data if p[0] not in done_idx]

    results = {}
    t_start = time.perf_counter()
    print(f"Sending {len(prompts_data)} requests with concurrency={args.concurrency}...")

    counts = {"total": len(indices), "improper": 0, "proper": 0, "valid": 0, "invalid": 0, "exact": 0}
    sums = {
        "cycles": 0.0,
        "invalid_edges": 0.0,
        "subgraphs": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    rows = []

    # Precompute gt_open per idx so we can score each sample the moment it returns.
    gt_open_by_idx = {i: parse_output(dataset[int(i)]["output"]) for i in indices}
    prompt_by_idx = {p[0]: p for p in prompts_data}

    # Incremental JSONL writer: append each finished record immediately so a crash
    # or interrupt never loses completed work, and re-runs skip already-done idx.
    jsonl_fh = open(filename_jsonl, "a", encoding="utf-8")

    def _record(idx, content, reasoning, finish, error):
        """Score one finished sample and append its full record to the JSONL."""
        raw_prompt = prompt_by_idx[idx][1]
        gen_text = content if content else reasoning
        gt_open = gt_open_by_idx.get(idx, [])

        improper_reason = None
        if error:
            improper_reason = f"error: {error}"
        elif not gen_text:
            improper_reason = "empty_text"
        else:
            gen_open = parse_output(gen_text)
            if not gen_open:
                improper_reason = "parse_failed"
            elif not gt_open:
                improper_reason = "gt_parse_failed"

        if improper_reason:
            counts["improper"] += 1
            rec = {
                "dataset_index": idx,
                "content": content,
                "reasoning": reasoning,
                "finish_reason": finish,
                "gen_open_lines": [],
                "gt_open_lines": gt_open,
                "is_valid": 0,
                "gt_exact_match": 0,
                "gt_iou": 0.0,
                "gt_precision": 0.0,
                "gt_recall": 0.0,
                "improper_reason": improper_reason,
            }
            jsonl_fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            jsonl_fh.flush()
            return

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
        precision, recall = compute_edge_precision_recall(gen_open, gt_open)
        sums["iou"] += iou
        sums["precision"] += precision
        sums["recall"] += recall

        rows.append({
            "dataset_index": idx,
            "prompt": format_prompt(raw_prompt, args.prompt_format),
            "gen_open_lines": gen_open,
            "gt_open_lines": gt_open,
            "is_valid": is_valid,
            "gt_exact_match": int(exact),
            "gt_iou": f"{iou:.6f}",
            "gt_precision": f"{precision:.6f}",
            "gt_recall": f"{recall:.6f}",
        })
        rec = {
            "dataset_index": idx,
            "content": content,
            "reasoning": reasoning,
            "finish_reason": finish,
            "gen_open_lines": gen_open,
            "gt_open_lines": gt_open,
            "is_valid": is_valid,
            "gt_exact_match": int(exact),
            "gt_iou": iou,
            "gt_precision": precision,
            "gt_recall": recall,
            "improper_reason": None,
        }
        jsonl_fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        jsonl_fh.flush()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for idx, _, _, prompt in prompts_data:
            f = pool.submit(_call_one, api_url, api_key, args.model_name,
                            prompt, args.temperature, args.top_p,
                            args.max_new_tokens, idx, args.timeout, args.reasoning_effort,
                            bool(args.stream))
            futures[f] = idx

        from tqdm import tqdm
        done_count = 0
        ok_count = 0
        err_count = 0
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc="eval", unit="sample", dynamic_ncols=True)
        for f in pbar:
            idx, content, reasoning, finish, error = f.result()
            _record(idx, content, reasoning, finish, error)
            done_count += 1
            if error or not (content if content else reasoning):
                err_count += 1
            else:
                ok_count += 1
            pbar.set_postfix(ok=ok_count, err=err_count,
                             rate=f"{done_count / max(time.perf_counter() - t_start, 1e-9):.2f}/s")
        pbar.close()

    jsonl_fh.close()

    t_total = time.perf_counter() - t_start
    print(f"Done: {ok_count} OK, {err_count} errors in {t_total:.1f}s")

    # Recompute final aggregates from the COMPLETE jsonl (includes resumed runs)
    # so the reported metrics are accurate regardless of resume/incremental writes.
    all_records = []
    if Path(filename_jsonl).exists():
        with open(filename_jsonl, encoding="utf-8") as _jf:
            for line in _jf:
                try:
                    all_records.append(_json.loads(line))
                except ValueError:
                    pass
    final = {"total": len(all_records), "improper": 0, "proper": 0, "valid": 0,
             "invalid": 0, "exact": 0}
    fsums = {
        "cycles": 0.0,
        "invalid_edges": 0.0,
        "subgraphs": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    csv_rows = []
    for rec in all_records:
        if rec.get("improper_reason"):
            final["improper"] += 1
            continue
        final["proper"] += 1
        is_valid = rec.get("is_valid", 0)
        if is_valid:
            final["valid"] += 1
        else:
            final["invalid"] += 1
        if rec.get("gt_exact_match"):
            final["exact"] += 1
        gt_iou = float(rec.get("gt_iou", 0.0))
        if "gt_precision" in rec and "gt_recall" in rec:
            gt_precision = float(rec.get("gt_precision", 0.0))
            gt_recall = float(rec.get("gt_recall", 0.0))
        else:
            gt_precision, gt_recall = compute_edge_precision_recall(
                rec.get("gen_open_lines", []),
                rec.get("gt_open_lines", []),
            )
        fsums["iou"] += gt_iou
        fsums["precision"] += gt_precision
        fsums["recall"] += gt_recall
        csv_rows.append({
            "dataset_index": rec["dataset_index"],
            "prompt": "",
            "gen_open_lines": rec.get("gen_open_lines", []),
            "gt_open_lines": rec.get("gt_open_lines", []),
            "is_valid": is_valid,
            "gt_exact_match": int(rec.get("gt_exact_match", 0)),
            "gt_iou": f"{gt_iou:.6f}",
            "gt_precision": f"{gt_precision:.6f}",
            "gt_recall": f"{gt_recall:.6f}",
        })
    # Note: cycles/invalid_edges/subgraphs sums are not stored per-record in the
    # JSONL (only is_valid), so we cannot recompute their averages from disk. They
    # are reported only for the samples scored in THIS run (may be partial on resume).
    prep_csv(filename_csv, COLUMNS)
    write_to_csv(filename_csv, csv_rows, COLUMNS)

    proper = max(final["proper"], 1)
    valid_rate = final["valid"] / proper

    valid_rows = [r for r in csv_rows if r["is_valid"] == 1]
    valid_n = max(len(valid_rows), 1)
    valid_exact = sum(r["gt_exact_match"] for r in valid_rows)
    valid_iou = sum(float(r["gt_iou"]) for r in valid_rows) / valid_n
    valid_precision = sum(float(r["gt_precision"]) for r in valid_rows) / valid_n
    valid_recall = sum(float(r["gt_recall"]) for r in valid_rows) / valid_n

    sections = [
        ("Run", [
            ("Split", args.split),
            ("Model", args.model_name),
            ("Total samples", str(final["total"])),
            ("Improper (XML parse failed)", str(final["improper"])),
            ("Proper XML", str(final["proper"])),
            ("Wall time (s)", f"{t_total:.1f}"),
            ("Concurrency", str(args.concurrency)),
        ]),
        ("Graph Validity", [
            ("Valid", f"{final['valid']}  ({valid_rate:.1%} of proper)"),
            ("Invalid", str(final["invalid"])),
            ("Avg cycles (this run only)", f"{sums['cycles']/max(counts['proper'],1):.4f}"),
            ("Avg invalid edges (this run only)", f"{sums['invalid_edges']/max(counts['proper'],1):.4f}"),
            ("Avg subgraphs (this run only)", f"{sums['subgraphs']/max(counts['proper'],1):.4f}"),
        ]),
        ("GT Match (undirected)", [
            ("Exact match (all proper)",
             f"{final['exact']} / {final['proper']}  ({final['exact']/proper:.1%})"),
            ("Mean IoU (all proper)", f"{fsums['iou']/proper:.4f}"),
            ("Mean precision (all proper)", f"{fsums['precision']/proper:.4f}"),
            ("Mean recall (all proper)", f"{fsums['recall']/proper:.4f}"),
            ("Exact match (valid only)",
             f"{valid_exact} / {valid_n}  ({valid_exact/valid_n:.1%})"),
            ("Mean IoU (valid only)", f"{valid_iou:.4f}"),
            ("Mean precision (valid only)", f"{valid_precision:.4f}"),
            ("Mean recall (valid only)", f"{valid_recall:.4f}"),
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
