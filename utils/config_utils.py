"""Configuration and environment helpers for training scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_project_env(env_path: Path = REPO_ROOT / ".env") -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("'\"")


def validate_precision_args(args: argparse.Namespace) -> None:
    if args.fp16 and args.bf16:
        raise ValueError("Choose only one mixed precision mode: set either --fp16 1 or --bf16 1.")


def report_to_value(report_to: str):
    parts = [part.strip() for part in report_to.split(",") if part.strip()]
    if not parts or parts == ["none"]:
        return "none"
    return parts


def uses_wandb(report_to: str) -> bool:
    parsed = report_to_value(report_to)
    return parsed != "none" and "wandb" in parsed


def configure_wandb(args: argparse.Namespace) -> None:
    if not uses_wandb(args.report_to):
        return

    for env_key, arg_value in {
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_RUN_GROUP": args.wandb_run_group,
    }.items():
        if arg_value:
            os.environ.setdefault(env_key, arg_value)

    tags = ",".join(tag.strip() for tag in args.wandb_tags.split(",") if tag.strip())
    if tags:
        os.environ.setdefault("WANDB_TAGS", tags)
    elif not os.environ.get("WANDB_TAGS", "").strip():
        os.environ.pop("WANDB_TAGS", None)
