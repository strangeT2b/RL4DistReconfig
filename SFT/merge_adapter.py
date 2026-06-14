#!/usr/bin/env python3
"""Merge a PEFT/LoRA adapter into its base model and save a full model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.config_utils import load_project_env  # noqa: E402
from utils.model_utils import peft_merge_unload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base_model", required=True,
                        help="Base model path or HF id, e.g. ../models/Qwen/Qwen3-8B.")
    parser.add_argument("--adapter_path", required=True,
                        help="PEFT adapter checkpoint/final_adapter path.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for the merged full model.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", type=int, default=1)
    parser.add_argument("--safe_serialization", type=int, default=1)
    parser.add_argument("--max_shard_size", default="5GB")
    return parser.parse_args()


def main() -> int:
    load_project_env()

    import torch
    from transformers import AutoTokenizer

    args = parse_args()
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Merging adapter:")
    print("  base_model:  ", args.base_model)
    print("  adapter_path:", args.adapter_path)
    print("  output_dir:  ", output_dir)
    print("  dtype:       ", args.dtype)

    merged_model = peft_merge_unload(
        model_id=args.base_model,
        model_path=args.adapter_path,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
        trust_remote_code=bool(args.trust_remote_code),
    )
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=bool(args.safe_serialization),
        max_shard_size=args.max_shard_size,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.adapter_path,
            trust_remote_code=bool(args.trust_remote_code),
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=bool(args.trust_remote_code),
        )
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved merged model/tokenizer to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
