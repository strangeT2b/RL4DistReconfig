#!/usr/bin/env python3
"""Convert RL4DistReconfig CSV data into veRL parquet files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.config_utils import load_project_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare veRL parquet data.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="Dataset/verl/train_33_69_84_nodes")
    parser.add_argument("--data_source", default="rl4dist")
    parser.add_argument("--ability", default="grid_reconfiguration")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    return parser.parse_args()


def _limit_dataset(dataset, max_samples: int):
    if max_samples > 0:
        return dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def _to_verl_rows(dataset, split: str, args: argparse.Namespace) -> list[dict]:
    rows = []
    for idx, example in enumerate(dataset):
        raw_prompt = example["prompt"]
        gt_output = example["output"]
        rows.append({
            "data_source": args.data_source,
            "prompt": [{"role": "user", "content": raw_prompt}],
            "ability": args.ability,
            "reward_model": {
                "style": "rule",
                "ground_truth": gt_output,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "raw_prompt": raw_prompt,
                "gt_output": gt_output,
            },
        })
    return rows


def main() -> int:
    load_project_env()

    from datasets import Dataset  # pylint: disable=import-error
    from utils.dataset_utils import prepare_train_data  # noqa: WPS433

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, validation_dataset, _ = prepare_train_data(args.data_path)
    train_dataset = _limit_dataset(train_dataset, args.max_train_samples)
    validation_dataset = _limit_dataset(validation_dataset, args.max_val_samples)

    splits = {
        "train": _to_verl_rows(train_dataset, "train", args),
        "validation": _to_verl_rows(validation_dataset, "validation", args),
    }

    for split, rows in splits.items():
        path = output_dir / f"{split}.parquet"
        Dataset.from_list(rows).to_parquet(str(path))
        print(f"Wrote {len(rows)} rows to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
