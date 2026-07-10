#!/usr/bin/env python3
"""GRPO LoRA training for grid reconfiguration using trl's GRPOTrainer.

Compared to the handwritten version (train_grpo_adapter_handwritten.py):
- Uses trl.GRPOTrainer which handles generation, log-probs, KL, advantage, loss
- Reference model is created automatically by trl (frozen copy of initial policy)
- Reward function is passed via reward_funcs, not called manually
- W&B logging is handled by trl's built-in integration
- Checkpoint save/load uses trl's built-in mechanism
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training  # noqa: E402

from utils.config_utils import (  # noqa: E402
    load_project_env,
    report_to_value,
    validate_precision_args,
)
from utils.dataset_utils import prepare_train_data  # noqa: E402
from utils.generation_utils import configure_generation  # noqa: E402
from utils.model_utils import (  # noqa: E402
    import_training_deps,
    load_model_and_tokenizer as load_base_model_and_tokenizer,
    set_all_seeds,
)
from utils.prompt_format_utils import format_prompt  # noqa: E402
from RL.reward import compute_reward_iou as _compute_reward_iou  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO LoRA training with trl GRPOTrainer.")

    # Data / model
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="train",
                        help="Which data split to train on.")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--run_name", default="grpo")
    parser.add_argument("--init_adapter", default="",
                        help="SFT LoRA adapter to start from (also used as reference).")

    # Training budget
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--prompts_per_step", type=int, default=4,
                        help="Prompts per device per optimizer step (= trl per_device_train_batch_size).")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Responses sampled per prompt (>=2 for group-relative).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Generation
    parser.add_argument("--prompt_format", choices=["legacy", "qwen_chat", "llama3_chat"], default="legacy")
    parser.add_argument("--max_prompt_length", type=int, default=3072)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    # Optimizer / scheduler
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", choices=["cosine", "linear", "constant"],
                        default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="LR warmup steps. Default 0 = 3%% of max_steps.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.0,
                        help="Min LR as fraction of learning_rate (trl cosine schedule bottoms out at 0 by default).")

    # Precision
    parser.add_argument("--fp16", type=int, default=0)
    parser.add_argument("--bf16", type=int, default=1)
    parser.add_argument("--load_in_4bit", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=int, default=1)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # KL
    parser.add_argument("--kl_beta", type=float, default=0.04,
                        help="KL penalty weight (maps to trl GRPOConfig.beta).")

    # Reward weights
    parser.add_argument("--invalid_edges_weight", type=float, default=1.0)
    parser.add_argument("--cycles_weight", type=float, default=1.0)
    parser.add_argument("--subgraphs_weight", type=float, default=1.0)
    parser.add_argument("--format_penalty_weight", type=float, default=0.0)
    parser.add_argument("--iou_weight", type=float, default=10.0,
                        help="Weight applied to IoU(gen_open_lines, gt_open_lines) on the valid branch. "
                             "IoU is in [0,1], so reward = valid_base + iou_weight * iou.")
    parser.add_argument("--valid_base", type=float, default=1.0,
                        help="Fixed bonus added to every valid (no invalid_edges/cycles/subgraphs) sample.")
    parser.add_argument("--invalid_penalty_scale", type=float, default=10.0,
                        help="Multiplier on graph-penalty sum for invalid samples (keeps invalid reward < valid).")
    parser.add_argument("--use_vllm", type=int, default=0,
                        help="Use vLLM for faster generation (1=on).")
    parser.add_argument("--vllm_mode", choices=["server", "colocate"], default="server",
                        help="TRL vLLM integration mode when --use_vllm 1.")
    parser.add_argument("--vllm_server_host", default="0.0.0.0",
                        help="vLLM server host for TRL server mode.")
    parser.add_argument("--vllm_server_port", type=int, default=8000,
                        help="vLLM server port for TRL server mode.")
    parser.add_argument("--vllm_server_timeout", type=float, default=240.0,
                        help="Seconds to wait for vLLM server in TRL server mode.")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1,
                        help="Tensor-parallel GPUs for TRL/vLLM colocate mode when supported.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3,
                        help="GPU memory fraction for TRL/vLLM colocate mode when supported.")
    parser.add_argument("--vllm_max_model_len", type=int, default=1024,
                        help="vLLM max_model_len.  Must be >= prompt + max_new_tokens.")

    # Logging / saving
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--log_responses", type=int, default=0)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_run_group", default="",
                        help="W&B run group (set via WANDB_RUN_GROUP env var).")
    parser.add_argument("--wandb_tags", default="")

    # Validation
    parser.add_argument("--eval_steps", type=int, default=0,
                        help="Run evaluation every N steps (0 = disabled).")
    parser.add_argument("--eval_samples", type=int, default=50)

    # Resume
    parser.add_argument("--resume_from", default="",
                        help="Path to a checkpoint dir to resume from.")

    # GRPO-specific (trl)
    parser.add_argument("--loss_type", choices=["grpo", "dr_grpo", "sapo", "dapo"], default="grpo",
                        help="trl GRPO loss type.")
    parser.add_argument("--scale_rewards", choices=["group", "batch", "none"], default="group",
                        help="Reward normalization: group=(r-mean)/std per prompt, batch=global std, none=raw advantage.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reward function (trl-compatible wrapper)
# ---------------------------------------------------------------------------

def make_reward_func(invalid_edges_weight, cycles_weight, subgraphs_weight,
                     format_penalty_weight, iou_weight=10.0,
                     valid_base=1.0, invalid_penalty_scale=10.0):
    """Build a layered reward function (graph validity + IoU vs GT) for trl.

    Layering guarantees that any valid sample's reward is strictly greater
    than any invalid sample's reward, so the model cannot game the system
    by deliberately producing invalid configs.

    Reward ranges (with defaults iou_weight=10, valid_base=1, scale=10):
      invalid sample : -graph_penalty_sum * 10        ≈ [-30, 0]
      valid sample   : 1 + iou * 10                   ∈ [1, 11]

    GT response is pulled from kwargs["output"] (trl forwards dataset columns
    matching the reward func's extra kwargs).
    """
    def reward_func(prompts, completions, completion_ids, **kwargs):
        rewards = []
        parts_accum = {"invalid_edges": 0.0, "cycles": 0.0,
                       "subgraphs": 0.0, "format_penalty": 0.0,
                       "is_valid": 0.0, "iou": 0.0,
                       "gt_exact_match": 0.0}

        gt_responses = kwargs.get("output") or [""] * len(prompts)

        for prompt, response, gt_response in zip(prompts, completions, gt_responses):
            reward, parts = _compute_reward_iou(
                prompt, response, gt_response,
                invalid_edges_weight=invalid_edges_weight,
                cycles_weight=cycles_weight,
                subgraphs_weight=subgraphs_weight,
                format_penalty_weight=format_penalty_weight,
                iou_weight=iou_weight,
                valid_base=valid_base,
                invalid_penalty_scale=invalid_penalty_scale,
            )
            for k in parts_accum:
                parts_accum[k] += parts.get(k, 0.0)
            rewards.append(reward)

        n = max(len(rewards), 1)
        try:
            import wandb
            wandb.log({
                "train/invalid_edges_loss": parts_accum["invalid_edges"] / n,
                "train/cycles_loss": parts_accum["cycles"] / n,
                "train/subgraphs_loss": parts_accum["subgraphs"] / n,
                "reward/format_penalty": parts_accum["format_penalty"] / n,
                "reward/valid_rate": parts_accum["is_valid"] / n,
                "reward/iou_mean": parts_accum["iou"] / n,
                "reward/gt_exact_match_rate": parts_accum["gt_exact_match"] / n,
                "reward/mean": sum(rewards) / n,
            }, commit=False)
        except Exception:
            pass

        return rewards

    return reward_func


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    load_project_env()
    args = parse_args()
    validate_precision_args(args)

    # Set W&B env vars that TrainingArguments doesn't directly expose
    # Set W&B env vars before GRPOConfig (trl reads them from environment)
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.wandb_run_group:
        os.environ.setdefault("WANDB_RUN_GROUP", args.wandb_run_group)
    if args.wandb_tags:
        os.environ.setdefault("WANDB_TAGS", args.wandb_tags)

    if args.num_generations < 2:
        raise ValueError("--num_generations must be at least 2 for group-relative RL.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_all_seeds(import_training_deps(), args.seed)

    # --- directories ---
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8",
    )

    # --- data ---
    train_dataset, validation_dataset, test_dataset = prepare_train_data(args.data_path)
    if args.split == "validation":
        train_dataset = validation_dataset
        print("Using validation split for RL training.")
    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(
            range(min(args.max_train_samples, len(train_dataset)))
        )

    # --- pre-format prompts with chat template ---
    # trl passes the "prompt" column directly to the tokenizer, so we need the
    # chat template applied before tokenization.
    def _apply_format(example):
        example["prompt"] = format_prompt(example["prompt"], args.prompt_format)
        return example

    train_dataset = train_dataset.map(_apply_format)
    # Keep prompt + output: trl forwards extra columns into reward_func **kwargs,
    # and the IoU reward needs the GT response.
    keep_cols = {"prompt", "output"}
    cols_to_remove = [c for c in train_dataset.column_names if c not in keep_cols]
    train_dataset = train_dataset.remove_columns(cols_to_remove)

    eval_dataset = None
    if args.eval_steps > 0 and len(validation_dataset) > 0:
        eval_dataset = validation_dataset.map(_apply_format)
        eval_dataset = eval_dataset.remove_columns(
            [c for c in eval_dataset.column_names if c not in keep_cols]
        )

    print(f"Train samples: {len(train_dataset)}  "
          f"Val samples: {len(validation_dataset)}  "
          f"Test samples: {len(test_dataset)}")

    # --- model ---
    deps = import_training_deps()
    base_model, tokenizer = load_base_model_and_tokenizer(args, deps)
    if args.load_in_4bit:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=bool(args.gradient_checkpointing),
        )

    # Set tokenizer padding for generation. Keep tokenizer.eos_token_id as the
    # tokenizer defined it; HF generate can receive an EOS list separately, but
    # pad_token_id must remain a single integer.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
            if isinstance(tokenizer.eos_token_id, int)
            else list(tokenizer.eos_token_id or [])[0]
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_prompt_length

    pad_token_id = None
    eos_token_ids = None

    # Build the PeftModel or peft_config depending on --init_adapter
    if args.init_adapter:
        # Disable Qwen3 thinking mode BEFORE wrapping with PeftModel,
        # otherwise the adapter will be applied on top of the thinking path.
        if hasattr(base_model.config, "enable_thinking"):
            base_model.config.enable_thinking = False
        # Reset generation defaults that would override trl sampling params.
        if hasattr(base_model, "generation_config"):
            gcfg = base_model.generation_config
            gcfg.enable_thinking = False
            gcfg.do_sample = False
            gcfg.temperature = None
            gcfg.top_p = None
            gcfg.top_k = 0

        model = PeftModel.from_pretrained(
            base_model, args.init_adapter, is_trainable=True,
        )
        pad_token_id, eos_token_ids = configure_generation(
            model, tokenizer, args.prompt_format,
        )
        peft_config = None  # Already has adapter, don't create new one
        print(f"Loaded init adapter from {args.init_adapter}")
    else:
        model = base_model
        pad_token_id, eos_token_ids = configure_generation(
            model, tokenizer, args.prompt_format,
        )
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print("No init_adapter — starting with fresh LoRA.")

    # --- training config ---
    # Map current-style args to GRPOConfig / TrainingArguments fields
    lr_scheduler_kwargs = {}
    if args.min_lr_ratio > 0:
        lr_scheduler_kwargs["min_lr"] = args.min_lr_ratio * args.learning_rate

    # Do not mutate tokenizer.eos_token_id: recent Transformers tokenizers
    # reject non-string EOS assignments.  The model generation_config was
    # already updated by configure_generation(), and newer TRL versions accept
    # generation_kwargs for an explicit EOS list.
    grpo_extra_kwargs = {}
    grpo_params = inspect.signature(GRPOConfig.__init__).parameters
    if "vllm_mode" in grpo_params:
        grpo_extra_kwargs["vllm_mode"] = args.vllm_mode
    if "vllm_server_host" in grpo_params:
        grpo_extra_kwargs["vllm_server_host"] = args.vllm_server_host
    if "vllm_server_port" in grpo_params:
        grpo_extra_kwargs["vllm_server_port"] = args.vllm_server_port
    if "vllm_server_timeout" in grpo_params:
        grpo_extra_kwargs["vllm_server_timeout"] = args.vllm_server_timeout
    if "generation_kwargs" in grpo_params:
        grpo_extra_kwargs["generation_kwargs"] = {
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_ids,
        }
    if args.vllm_mode == "colocate" and "vllm_tensor_parallel_size" in grpo_params:
        grpo_extra_kwargs["vllm_tensor_parallel_size"] = args.vllm_tensor_parallel_size
    if args.vllm_mode == "colocate" and "vllm_max_model_length" in grpo_params:
        grpo_extra_kwargs["vllm_max_model_length"] = args.max_prompt_length + args.max_new_tokens
    if args.vllm_mode == "colocate" and "vllm_gpu_memory_utilization" in grpo_params:
        grpo_extra_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization

    training_args = GRPOConfig(
        output_dir=str(run_dir),
        run_name=args.run_name,
        # Training budget
        max_steps=args.max_steps,
        per_device_train_batch_size=args.prompts_per_step,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Generation
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        # Optimizer
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=(
            args.warmup_steps
            if args.warmup_steps > 0
            else int(args.max_steps * 0.03)
        ),
        lr_scheduler_kwargs=lr_scheduler_kwargs or None,
        # Precision
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # KL
        beta=args.kl_beta,
        # GRPO-specific
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        use_vllm=bool(args.use_vllm),
        **grpo_extra_kwargs,
        # Logging / saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        log_completions=bool(args.log_responses),
        report_to=report_to_value(args.report_to),
        # Misc
        seed=args.seed,
        save_total_limit=3,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # --- trainer ---
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=make_reward_func(
            args.invalid_edges_weight,
            args.cycles_weight,
            args.subgraphs_weight,
            args.format_penalty_weight,
            iou_weight=args.iou_weight,
            valid_base=args.valid_base,
            invalid_penalty_scale=args.invalid_penalty_scale,
        ),
        peft_config=peft_config,
    )

    # --- train ---
    resume_from_checkpoint = args.resume_from or None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # --- final save ---
    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved final adapter: {final_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
