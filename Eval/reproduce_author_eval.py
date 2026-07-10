#!/usr/bin/env python3
"""Run the author's original HF-generation evaluation path locally.

This is intentionally *not* the corrected/vLLM evaluator. It mirrors
``LLM4DistReconfig/Model-Notebooks/llama3-notebooks/generate-metrics.py``:

- load data with the author's ``prepare_train_data``;
- load the base model/tokenizer with the author's utilities;
- merge a PEFT adapter with the author's ``peft_merge_unload`` unless
  ``--untrained True`` is passed;
- generate one sampled response per selected item with the author's
  ``generate_response`` defaults;
- compute metrics with the author's ``generate_metrics`` implementation.

That means it also preserves author quirks such as random sampling indices and
the original parsing/graph-validity behavior.
"""

from __future__ import annotations

import argparse
from functools import partial
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_UTILS_DIR = REPO_ROOT / "LLM4DistReconfig" / "Dataset-Notebooks" / "utils"


def ensure_upstream_utils_on_path() -> None:
    utils_path = str(UPSTREAM_UTILS_DIR)
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the author's original generate-metrics evaluation."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument(
        "--model_for_generation_path",
        required=True,
        help="PEFT adapter/checkpoint path used by the author's merge step.",
    )
    parser.add_argument("--filename_txt", required=True)
    parser.add_argument("--filename_csv", required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument(
        "--untrained",
        default="False",
        help="Match author script: pass 'True' to skip adapter merge.",
    )
    return parser.parse_args()


def main() -> int:
    if Path.cwd().resolve() != REPO_ROOT:
        raise SystemExit(f"Please run from repo root: {REPO_ROOT}")

    args = parse_args()
    ensure_upstream_utils_on_path()

    # Imports are intentionally from LLM4DistReconfig/Dataset-Notebooks/utils.
    from dataset_utils import prepare_train_data  # pylint: disable=import-error
    from generation_utils import (  # pylint: disable=import-error
        generate_response,
        peft_merge_unload,
    )
    from metrics_utils import generate_metrics  # pylint: disable=import-error
    from model_utils import get_model, get_tokenizer  # pylint: disable=import-error

    Path(args.filename_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.filename_csv).parent.mkdir(parents=True, exist_ok=True)

    print("Evaluation configurations:\n", args)
    print("Using upstream utils:", UPSTREAM_UTILS_DIR)

    _, _, test_dataset = prepare_train_data(args.data_path)

    model = get_model(args.model_id)
    tokenizer = get_tokenizer(args.model_id)

    if args.untrained == "False":
        model = peft_merge_unload(args.model_id, args.model_for_generation_path)

    response_fn = partial(
        generate_response,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )
    generate_metrics(
        test_dataset,
        response_fn,
        num_samples=args.num_samples,
        filename_txt=args.filename_txt,
        filename_csv=args.filename_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
