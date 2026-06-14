#!/usr/bin/env python3
"""Reproduce the author's Llama QLoRA training recipe with local paths.

This script intentionally stays close to
``LLM4DistReconfig/Model-Notebooks/*/grid-reconfiguration.py``:

- upstream dataset/model utility functions are used directly;
- QLoRA uses 4-bit NF4 with float16 compute from the author's model utils;
- TrainingArguments keep fp16=True, lr=2e-4, cosine scheduler, epoch saves;
- LoRA uses r=8, alpha=16, dropout=0.05.

Local-only changes: no hard-coded Hugging Face token, no push_to_hub, and no
absolute upstream path.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_utils import load_project_env  # noqa: E402
from utils.prompt_format_utils import format_sft_text  # noqa: E402

UPSTREAM_UTILS_DIR = REPO_ROOT / "utils"


def ensure_upstream_utils_on_path() -> None:
    utils_path = str(UPSTREAM_UTILS_DIR)
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the author's Llama QLoRA recipe.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--output_root", required=True, help="Root directory for this experiment, e.g. runs/llama31_8b.")
    parser.add_argument("--run_name", required=True, help="Subdirectory name under output_root, e.g. sft_ce_v1.")
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--model_name_hf", required=True)
    parser.add_argument("--tokenizer_name_hf", required=True)
    parser.add_argument("--custom_loss", type=int, choices=[0, 1], required=True)
    parser.add_argument("--custom_loss_config", required=True)
    parser.add_argument("--cycles_loss_scaling_factor", type=float, required=True)
    parser.add_argument("--model_for_generation_path", required=True)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--prompt_format", choices=["legacy", "qwen_chat", "llama3_chat"], default="legacy")
    parser.add_argument("--save_strategy", choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_run_group", default="")
    parser.add_argument("--wandb_tags", default="")
    return parser.parse_args(argv)


def report_to_value(report_to: str):
    cleaned = report_to.strip()
    if not cleaned or cleaned.lower() == "none":
        return "none"
    return [part.strip() for part in cleaned.split(",") if part.strip()]


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


def filter_predicted_lines(predicted_lines):
    return predicted_lines if all(
        isinstance(line, (tuple, list)) and len(line) == 2 for line in predicted_lines
    ) else []


def _apply_prompt_format(dataset, prompt_format: str):
    """Re-format the ``text`` column when it differs from the default legacy format."""
    if prompt_format == "legacy":
        return dataset

    def reformat_row(row):
        row["text"] = format_sft_text(row.get("prompt", ""), row.get("output", ""), prompt_format)
        return row

    return dataset.map(reformat_row)


def main() -> int:
    load_project_env(REPO_ROOT / ".env")
    ensure_upstream_utils_on_path()
    from peft import LoraConfig
    from utils.dataset_utils import prepare_train_data  # pylint: disable=import-error
    from utils.model_utils import get_model_and_tokenizer_qlora
    from utils.metrics_utils import (  # pylint: disable=import-error
        compute_cycles_loss,
        compute_invalid_edges_loss,
        compute_subgraphs_loss,
        get_output_graph_edges,
        parse_available_lines,
        parse_open_lines,
    )
    from trl import SFTConfig, SFTTrainer

    args = parse_args()
    configure_wandb(args)
    output_model = Path(args.output_root) / args.run_name
    output_model.mkdir(parents=True, exist_ok=True)

    print("Training configurations:\n", args)
    print("Using upstream utils (from utils/):", UPSTREAM_UTILS_DIR)
    print("Local reproduction mode: skipping login, merge, generation, and push_to_hub.")

    train_dataset, validation_dataset, test_dataset = prepare_train_data(args.data_path)
    train_dataset = _apply_prompt_format(train_dataset, args.prompt_format)
    validation_dataset = _apply_prompt_format(validation_dataset, args.prompt_format)
    test_dataset = _apply_prompt_format(test_dataset, args.prompt_format)
    print(train_dataset)
    print(validation_dataset)
    print(test_dataset)

    model, tokenizer = get_model_and_tokenizer_qlora(args.model_id)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = SFTConfig(
        completion_only_loss=False,
        remove_unused_columns=False,
        output_dir=str(output_model),
        run_name=f"{Path(args.output_root).name}/{args.run_name}",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=10,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        fp16=True,
        report_to=report_to_value(args.report_to),
        gradient_checkpointing=True,
    )

    if args.custom_loss == 0:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            formatting_func=lambda x: x["text"],
            args=training_arguments,
            processing_class=tokenizer,
        )
    else:

        class CustomTrainer(SFTTrainer):
            def compute_loss(self, model, inputs, return_outputs=False):  # noqa: ANN001
                outputs = model(**inputs)
                loss = outputs.loss

                input_text = self.tokenizer.batch_decode(
                    inputs["input_ids"], skip_special_tokens=True
                )
                output_text = self.tokenizer.batch_decode(
                    outputs.logits.argmax(dim=-1), skip_special_tokens=True
                )

                available_lines = parse_available_lines(input_text[0])
                predicted_lines = parse_open_lines(output_text[0])
                # Local robustness patch: the public source assumes every parsed item is
                # an edge pair; malformed parses are treated as invalid outputs, matching
                # the paper's max-penalty fallback instead of crashing in reverseTuple().
                predicted_lines = filter_predicted_lines(predicted_lines)
                graph_edges = get_output_graph_edges(predicted_lines, available_lines)
                print(predicted_lines)

                if predicted_lines != []:
                    if "IEL" in args.custom_loss_config:
                        invalid_edges_loss = compute_invalid_edges_loss(
                            predicted_lines, available_lines
                        ) / len(predicted_lines)
                    if "CYL" in args.custom_loss_config:
                        cycles_loss = compute_cycles_loss(graph_edges) / len(available_lines)
                    if "SUL" in args.custom_loss_config:
                        subgraphs_loss = compute_subgraphs_loss(graph_edges) / len(
                            predicted_lines
                        )
                else:
                    if "IEL" in args.custom_loss_config:
                        invalid_edges_loss = 1
                    if "CYL" in args.custom_loss_config:
                        cycles_loss = 1
                    if "SUL" in args.custom_loss_config:
                        subgraphs_loss = 1

                ce_loss_value = loss.detach().float().item()

                total_loss = loss
                if "IEL" in args.custom_loss_config:
                    total_loss += invalid_edges_loss
                if "CYL" in args.custom_loss_config:
                    total_loss += cycles_loss * args.cycles_loss_scaling_factor
                if "SUL" in args.custom_loss_config:
                    total_loss += subgraphs_loss

                self.log({
                    "train/ce_loss": ce_loss_value,
                    "train/total_loss": total_loss.detach().float().item(),
                })
                if "IEL" in args.custom_loss_config:
                    self.log({"train/invalid_edges_loss": float(invalid_edges_loss)})
                if "CYL" in args.custom_loss_config:
                    self.log({"train/cycles_loss": float(cycles_loss)})
                if "SUL" in args.custom_loss_config:
                    self.log({"train/subgraphs_loss": float(subgraphs_loss)})

                if "IEL" in args.custom_loss_config:
                    print('Invalid edges loss: ', invalid_edges_loss)
                if "CYL" in args.custom_loss_config:
                    print('Cycles Loss: ', cycles_loss)
                if "SUL" in args.custom_loss_config:
                    print('Subgraphs loss: ', subgraphs_loss)
                print('Total loss: ', total_loss.item())

                return (total_loss, outputs) if return_outputs else total_loss

        trainer = CustomTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            formatting_func=lambda x: x["text"],
            args=training_arguments,
            processing_class=tokenizer,
        )

    print("Device: ", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "default")
    trainer.train()
    trainer.save_model(str(output_model / "final_adapter"))
    tokenizer.save_pretrained(str(output_model / "final_adapter"))
    print("Saved final adapter/tokenizer to", output_model / "final_adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
