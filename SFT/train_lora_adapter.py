#!/usr/bin/env python3
"""Train one LoRA adapter for the distribution reconfiguration task.

The script is intentionally single-run: choose ``--loss_type ce`` for normal
SFT/causal-LM loss or ``--loss_type custom`` to add the author's graph penalties.
Outputs are written under ``output_root/run_name``.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.config_utils import (  # noqa: E402
    configure_wandb,
    load_project_env,
    report_to_value,
    uses_wandb,
    validate_precision_args,
)
from utils.dataset_utils import format_dataset_text, maybe_limit_dataset  # noqa: E402
from utils.dataset_utils import prepare_train_data  # noqa: E402
from utils.metrics_utils import graph_penalties  # noqa: E402
from utils.model_utils import (  # noqa: E402
    import_training_deps,
    load_model_and_tokenizer,
    peft_merge_unload,
    set_all_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one CE or custom-loss LoRA adapter.")

    parser.add_argument("--data_path", required=True, help="Prepared CSV dataset.")
    parser.add_argument("--model_id", required=True, help="Base model id or local path.")
    parser.add_argument("--output_root", required=True, help="Root directory for this experiment.")
    parser.add_argument("--run_name", default="", help="Subdirectory name. Defaults to loss_type.")
    parser.add_argument("--loss_type", choices=["ce", "custom"], default="ce")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument(
        "--prompt_format",
        choices=["dataset_text", "legacy", "qwen_chat", "llama3_chat"],
        default="legacy",
        help="Prompt template. legacy matches the author's original source format.",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--optim", default="paged_adamw_32bit")
    parser.add_argument("--fp16", type=int, default=1)
    parser.add_argument("--bf16", type=int, default=0)
    parser.add_argument("--load_in_4bit", type=int, default=0,
                        help="4-bit NF4 QLoRA. Default 0 (bf16 全精度) 对齐 RL 端。"
                             "想用 4-bit 显式传 --load_in_4bit 1。")
    parser.add_argument("--gradient_checkpointing", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_strategy", choices=["no", "steps", "epoch"], default="no")
    parser.add_argument("--eval_steps", type=int, default=500)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--custom_loss_config", default="IEL,SUL,CYL")
    parser.add_argument("--cycles_loss_scaling_factor", type=float, default=1.0,
                        help="Scaling factor for cycles penalty (matching author's naming).")
    parser.add_argument("--run_order", default="", help=argparse.SUPPRESS)

    parser.add_argument("--gradient_log_every", type=int, default=10)
    parser.add_argument("--log_per_parameter", type=int, default=0)
    parser.add_argument("--save_raw_gradients_every", type=int, default=0)

    parser.add_argument("--report_to", default="none", help="Use 'wandb' to enable W&B.")
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_run_group", default="")
    parser.add_argument("--wandb_tags", default="", help="Comma-separated non-empty tags.")
    parser.add_argument("--wandb_log_gradients", type=int, default=1)

    parser.add_argument("--merge_after_train", type=int, default=0)
    return parser.parse_args()


def load_splits(data_path: str, deps: dict[str, Any], prompt_format: str):
    train_dataset, validation_dataset, test_dataset = prepare_train_data(data_path)
    train_dataset = format_dataset_text(train_dataset, prompt_format)
    validation_dataset = format_dataset_text(validation_dataset, prompt_format)
    test_dataset = format_dataset_text(test_dataset, prompt_format)
    train_dataset = keep_text_only(train_dataset)
    validation_dataset = keep_text_only(validation_dataset)
    return train_dataset, validation_dataset, test_dataset


def keep_text_only(dataset):
    remove_columns = [column for column in dataset.column_names if column != "text"]
    return dataset.remove_columns(remove_columns) if remove_columns else dataset


def process_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    return process_rank() == 0


def trainer_accepts_legacy_args(sft_trainer_cls) -> bool:
    return "dataset_text_field" in inspect.signature(sft_trainer_cls.__init__).parameters


def training_args(args: argparse.Namespace, deps: dict[str, Any], run_dir: Path):
    common = {
        "output_dir": str(run_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optim": args.optim,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "num_train_epochs": args.num_train_epochs,
        "fp16": bool(args.fp16),
        "bf16": bool(args.bf16),
        "report_to": report_to_value(args.report_to),
        "run_name": f"{Path(args.output_root).name}/{args.run_name}",
        "logging_dir": str(run_dir / "logs"),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "seed": args.seed,
        "data_seed": args.seed,
    }

    legacy_trainer = trainer_accepts_legacy_args(deps["SFTTrainer"])
    if not legacy_trainer and deps["SFTConfig"] is None:
        raise RuntimeError("This TRL version requires SFTConfig, but it is unavailable.")
    args_cls = deps["TrainingArguments"] if legacy_trainer else deps["SFTConfig"]
    args_params = inspect.signature(args_cls.__init__).parameters
    if "eval_strategy" in args_params:
        common["eval_strategy"] = args.eval_strategy
    elif "evaluation_strategy" in args_params:
        common["evaluation_strategy"] = args.eval_strategy
    if "eval_steps" in args_params:
        common["eval_steps"] = args.eval_steps

    if not legacy_trainer:
        return deps["SFTConfig"](
            **common,
            dataset_text_field="text",
            packing=False,
            max_length=args.max_seq_length,
        )
    return deps["TrainingArguments"](**common)


def make_trainer_class(args: argparse.Namespace, deps: dict[str, Any]):
    torch = deps["torch"]
    base_cls = deps["SFTTrainer"]

    class ReconfigurationTrainer(base_cls):
        def __init__(
            self,
            *trainer_args,
            gradient_log_path: Path,
            raw_gradient_dir: Path,
            **trainer_kwargs,
        ):
            self.gradient_log_path = gradient_log_path
            self.raw_gradient_dir = raw_gradient_dir
            self.gradient_micro_step = 0
            if is_main_process():
                self.gradient_log_path.parent.mkdir(parents=True, exist_ok=True)
                self.gradient_log_path.write_text("", encoding="utf-8")
                if args.save_raw_gradients_every > 0:
                    self.raw_gradient_dir.mkdir(parents=True, exist_ok=True)
            super().__init__(*trainer_args, **trainer_kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # noqa: ANN001
            outputs = model(**inputs)
            loss = outputs.loss
            if args.loss_type == "custom":
                ce_loss_value = loss.detach().float().item()

                tokenizer = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
                input_text = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
                output_text = tokenizer.batch_decode(
                    outputs.logits.argmax(dim=-1), skip_special_tokens=True
                )
                penalties = graph_penalties(input_text[0], output_text[0])
                selected = {part.strip() for part in args.custom_loss_config.split(",")}
                penalty = 0.0
                if "IEL" in selected:
                    penalty += penalties["invalid_edges"]
                if "CYL" in selected:
                    penalty += args.cycles_loss_scaling_factor * penalties["cycles"]
                if "SUL" in selected:
                    penalty += penalties["subgraphs"]
                loss = loss + torch.tensor(penalty, device=loss.device, dtype=loss.dtype)

                self.log({
                    "train/ce_loss": ce_loss_value,
                    "train/total_loss": loss.detach().float().item(),
                })
                if "IEL" in selected:
                    self.log({"train/invalid_edges_loss": float(penalties["invalid_edges"])})
                if "CYL" in selected:
                    self.log({"train/cycles_loss": float(penalties["cycles"])})
                if "SUL" in selected:
                    self.log({"train/subgraphs_loss": float(penalties["subgraphs"])})

            return (loss, outputs) if return_outputs else loss

        def training_step(self, model, inputs, *step_args, **step_kwargs):  # noqa: ANN001
            loss = super().training_step(model, inputs, *step_args, **step_kwargs)
            self.gradient_micro_step += 1
            self.log_gradient_summary(model, loss)
            return loss

        def log_gradient_summary(self, model, loss) -> None:  # noqa: ANN001
            if not is_main_process():
                return
            if args.gradient_log_every <= 0:
                return
            if self.gradient_micro_step % args.gradient_log_every != 0:
                return

            total_sq = 0.0
            max_abs = 0.0
            trainable = 0
            per_parameter = {}
            raw_grads = {}
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                trainable += 1
                grad = parameter.grad.detach()
                grad_float = grad.float()
                l2_norm = torch.linalg.vector_norm(grad_float).item()
                grad_max = grad_float.abs().max().item()
                total_sq += l2_norm * l2_norm
                max_abs = max(max_abs, grad_max)
                if args.log_per_parameter:
                    per_parameter[name] = {
                        "l2_norm": l2_norm,
                        "max_abs": grad_max,
                        "numel": int(grad.numel()),
                    }
                if (
                    args.save_raw_gradients_every > 0
                    and self.gradient_micro_step % args.save_raw_gradients_every == 0
                ):
                    raw_grads[name] = grad.cpu().clone()

            record = {
                "loss_type": args.loss_type,
                "micro_step": self.gradient_micro_step,
                "trainer_global_step": int(getattr(self.state, "global_step", 0)),
                "loss_returned_by_training_step": float(loss.detach().cpu()),
                "trainable_tensors_with_grad": trainable,
                "grad_global_l2_norm": total_sq**0.5,
                "grad_max_abs": max_abs,
            }
            if args.log_per_parameter:
                record["parameters"] = per_parameter
            with self.gradient_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            if uses_wandb(args.report_to) and args.wandb_log_gradients:
                try:
                    import wandb

                    if wandb.run is not None:
                        prefix = f"{args.run_name}/gradient"
                        wandb.log(
                            {
                                f"{prefix}/loss_returned_by_training_step": record[
                                    "loss_returned_by_training_step"
                                ],
                                f"{prefix}/global_l2_norm": record["grad_global_l2_norm"],
                                f"{prefix}/max_abs": record["grad_max_abs"],
                                f"{prefix}/trainable_tensors_with_grad": trainable,
                            },
                            step=self.gradient_micro_step,
                        )
                except Exception as exc:
                    print(f"W&B gradient logging skipped: {exc}", file=sys.stderr)

            if raw_grads:
                raw_path = self.raw_gradient_dir / f"micro_step_{self.gradient_micro_step:08d}.pt"
                torch.save(raw_grads, raw_path)

    return ReconfigurationTrainer


def build_trainer(args: argparse.Namespace, deps: dict[str, Any], train_dataset, eval_dataset):
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(deps, args.seed)
    model, tokenizer = load_model_and_tokenizer(args, deps)
    peft_config = deps["LoraConfig"](
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset if args.eval_strategy != "no" else None,
        "peft_config": peft_config,
        "args": training_args(args, deps, run_dir),
        "gradient_log_path": run_dir / "gradient_log.jsonl",
        "raw_gradient_dir": run_dir / "raw_gradients",
    }
    if trainer_accepts_legacy_args(deps["SFTTrainer"]):
        trainer_kwargs.update(
            {
                "dataset_text_field": "text",
                "tokenizer": tokenizer,
                "packing": False,
                "max_seq_length": args.max_seq_length,
            }
        )
    else:
        trainer_kwargs["processing_class"] = tokenizer

    return make_trainer_class(args, deps)(**trainer_kwargs), tokenizer, run_dir


def save_outputs(args: argparse.Namespace, deps: dict[str, Any], trainer, tokenizer, run_dir: Path) -> None:
    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    if is_main_process():
        tokenizer.save_pretrained(str(final_dir))
        print(f"Saved adapter/tokenizer to {final_dir}")

        if args.merge_after_train:
            merged_dir = run_dir / "merged_model"
            merged_model = peft_merge_unload(args.model_id, str(final_dir))
            merged_model.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            print(f"Saved merged model to {merged_dir}")

    del trainer
    gc.collect()
    if deps["torch"].cuda.is_available():
        deps["torch"].cuda.empty_cache()


def main() -> int:
    load_project_env()
    args = parse_args()
    if args.run_order:
        requested = [part.strip() for part in args.run_order.split(",") if part.strip()]
        if len(requested) != 1 or requested[0] not in {"ce", "custom"}:
            raise ValueError("This script now trains one run. Use --loss_type ce or --loss_type custom.")
        args.loss_type = requested[0]
    if args.cycles_loss_scaling_factor is not None:
        args.cycles_weight = args.cycles_loss_scaling_factor
    args.run_name = args.run_name or args.loss_type
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    configure_wandb(args)
    validate_precision_args(args)
    if args.loss_type == "custom":
        print(
            "Warning: custom graph penalties are computed from argmax decoded text, "
            "so they are not differentiable. Gradients still mainly come from the CE loss."
        )

    deps = import_training_deps()
    set_all_seeds(deps, args.seed)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_dataset, validation_dataset, test_dataset = load_splits(
        args.data_path, deps, args.prompt_format
    )
    train_dataset = maybe_limit_dataset(train_dataset, args.max_train_samples)

    print(f"Loss type: {args.loss_type}")
    print(f"Run directory: {run_dir}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    trainer, tokenizer, run_dir = build_trainer(args, deps, train_dataset, validation_dataset)
    trainer.train()
    save_outputs(args, deps, trainer, tokenizer, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
