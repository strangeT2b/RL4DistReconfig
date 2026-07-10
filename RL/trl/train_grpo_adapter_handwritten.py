#!/usr/bin/env python3
"""GRPO LoRA training for grid reconfiguration with KL constraint.

One step = sample several prompts, generate N responses each, compute
group-relative advantages, apply KL penalty, and update via policy gradient.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.config_utils import (  # noqa: E402
    configure_wandb,
    load_project_env,
    report_to_value,
    uses_wandb,
    validate_precision_args,
)
from utils.dataset_utils import prepare_train_data  # noqa: E402
from utils.generation_utils import (  # noqa: E402
    build_stopping_criteria,
    configure_generation,
    truncate_on_stop,
)
from utils.model_utils import (  # noqa: E402
    import_training_deps,
    load_model_and_tokenizer as load_base_model_and_tokenizer,
    set_all_seeds,
)
from utils.prompt_format_utils import format_prompt  # noqa: E402
from RL.reward import compute_reward, group_advantages  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO LoRA training with KL constraint.")

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
                        help="Number of distinct prompts per optimizer step.")
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
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)

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
                        help="KL penalty weight applied to the reward.")

    # Reward weights
    parser.add_argument("--invalid_edges_weight", type=float, default=1.0)
    parser.add_argument("--cycles_weight", type=float, default=1.0)
    parser.add_argument("--subgraphs_weight", type=float, default=1.0)
    parser.add_argument("--format_penalty_weight", type=float, default=0.0)

    # Logging / saving
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--log_responses", type=int, default=0)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_run_group", default="")
    parser.add_argument("--wandb_tags", default="")

    # Validation
    parser.add_argument("--eval_steps", type=int, default=0,
                        help="Run validation every N steps (0 = disabled).")
    parser.add_argument("--eval_samples", type=int, default=50)

    # Resume
    parser.add_argument("--resume_from", default="",
                        help="Path to a checkpoint dir to resume from.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_policy_model(args: argparse.Namespace):
    """Load the trainable policy model (base + LoRA)."""
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if not args.load_in_4bit:
        raise ValueError("This script requires --load_in_4bit 1.")

    deps = import_training_deps()
    model, tokenizer = load_base_model_and_tokenizer(args, deps)

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(args.gradient_checkpointing),
    )

    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    return model, tokenizer


def load_reference_model(args: argparse.Namespace, tokenizer):
    """Load a frozen reference model: same base + initial LoRA, not trainable."""
    from peft import PeftModel

    deps = import_training_deps()
    ref_model, _ = load_base_model_and_tokenizer(args, deps)

    if args.init_adapter:
        ref_model = PeftModel.from_pretrained(ref_model, args.init_adapter, is_trainable=False)
    else:
        # No adapter → reference is the raw base model (no LoRA needed)
        pass

    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    return ref_model


# ---------------------------------------------------------------------------
# Log-probability helpers
# ---------------------------------------------------------------------------

def compute_model_logprobs(model, input_ids, prompt_length: int):
    """Per-sequence sum of log-probabilities over generated tokens.

    Returns a 1-D tensor of shape (batch_size,) where each element is the
    **sum** of log-probs over the generated (non-prompt) token positions.
    """
    import torch
    import torch.nn.functional as F

    # Ensure input_ids is on the same device as model
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)

    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    log_probs = F.log_softmax(outputs.logits[:, :-1], dim=-1)
    target_ids = input_ids[:, 1:]
    label_mask = labels[:, 1:] != -100
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * label_mask).sum(dim=1)


def per_token_kl(log_pi_theta, log_pi_ref):
    """Unbiased per-token KL estimator (DeepSeek-R1).

    KL_est = exp(log_ref - log_theta) - (log_ref - log_theta) - 1
    """
    log_ratio = log_pi_ref - log_pi_theta
    return log_ratio.exp() - log_ratio - 1


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def run_validation(model, ref_model, tokenizer, dataset, args, device):
    """Run reward + KL evaluation on a subset of the validation split."""
    import torch

    model.eval()
    indices = random.sample(range(len(dataset)), min(args.eval_samples, len(dataset)))
    rewards = []
    kl_divs = []

    with torch.no_grad():
        for index in indices:
            prompt = dataset[int(index)]["prompt"]
            prompt_text = format_prompt(prompt, args.prompt_format)
            prompt_inputs = tokenizer(
                prompt_text, return_tensors="pt", truncation=True,
                max_length=args.max_prompt_length,
            )
            prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
            prompt_length = prompt_inputs["input_ids"].shape[1]
            pad_token_id, eos_token_ids = configure_generation(
                model, tokenizer, args.prompt_format,
            )

            generated_ids = model.generate(
                **prompt_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_generations,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_ids,
                stopping_criteria=build_stopping_criteria(tokenizer, prompt_length),
            )
            generated_ids = generated_ids.to(device)
            responses = tokenizer.batch_decode(
                generated_ids[:, prompt_length:], skip_special_tokens=True,
            )
            responses = [truncate_on_stop(response) for response in responses]

            log_pi_theta = compute_model_logprobs(model, generated_ids, prompt_length)
            log_pi_ref = compute_model_logprobs(ref_model, generated_ids, prompt_length)
            kl = (log_pi_theta - log_pi_ref).abs()  # Consistent with training

            for i, response in enumerate(responses):
                reward, _ = compute_reward(
                    prompt, response,
                    args.invalid_edges_weight, args.cycles_weight,
                    args.subgraphs_weight, args.format_penalty_weight,
                )
                adjusted = reward - args.kl_beta * kl[i].item()
                rewards.append(adjusted)
                kl_divs.append(kl[i].item())

    model.train()
    return {
        "eval/mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "eval/best_reward": max(rewards) if rewards else 0.0,
        "eval/worst_reward": min(rewards) if rewards else 0.0,
        "eval/mean_kl": sum(kl_divs) / len(kl_divs) if kl_divs else 0.0,
    }


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_training_state(run_dir, model, optimizer, scheduler, step, rng_state, name):
    import torch

    checkpoint_dir = run_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))

    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    (checkpoint_dir / "training_state.json").write_text(
        json.dumps({"step": step, "rng_state": rng_state}, indent=2),
        encoding="utf-8",
    )
    return checkpoint_dir


def load_training_state(checkpoint_dir, model, optimizer, scheduler):
    import torch

    state = json.loads((checkpoint_dir / "training_state.json").read_text("utf-8"))
    opt_path = checkpoint_dir / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
    sched_path = checkpoint_dir / "scheduler.pt"
    if scheduler is not None and sched_path.exists():
        scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
    return state["step"], state.get("rng_state", random.getstate())


# ---------------------------------------------------------------------------
# LR scheduler builders
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, args):
    import torch

    warmup = args.warmup_steps if args.warmup_steps > 0 else int(args.max_steps * 0.03)

    if args.lr_scheduler_type == "constant":
        return None  # no scheduler
    if args.lr_scheduler_type == "linear":
        def linear_lr(t: int) -> float:
            if t < warmup:
                return float(t) / max(warmup, 1)
            progress = (t - warmup) / max(args.max_steps - warmup, 1)
            return max(1.0 - progress * (1.0 - args.min_lr_ratio), args.min_lr_ratio)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, linear_lr)
    # cosine
    def cosine_lr(t: int) -> float:
        if t < warmup:
            return float(t) / max(warmup, 1)
        progress = (t - warmup) / max(args.max_steps - warmup, 1)
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)


# ---------------------------------------------------------------------------
# Single-prompt generation + reward + logprob step
# ---------------------------------------------------------------------------

def process_prompt(model, ref_model, tokenizer, prompt, device, args):
    """Generate N responses for one prompt and return everything needed for the loss."""
    import torch

    prompt_text = format_prompt(prompt, args.prompt_format)
    prompt_inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True,
        max_length=args.max_prompt_length,
    )
    prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
    prompt_length = prompt_inputs["input_ids"].shape[1]
    pad_token_id, eos_token_ids = configure_generation(
        model, tokenizer, args.prompt_format,
    )

    # --- generate (policy model, no grad) ---
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **prompt_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_generations,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_ids,
            stopping_criteria=build_stopping_criteria(tokenizer, prompt_length),
        )
    # Ensure generated_ids is on the same device
    generated_ids = generated_ids.to(device)
    model.train()

    responses = tokenizer.batch_decode(
        generated_ids[:, prompt_length:], skip_special_tokens=True,
    )
    responses = [truncate_on_stop(response) for response in responses]

    # --- rewards ---
    rewards = []
    reward_parts_list = []
    for response in responses:
        reward, parts = compute_reward(
            prompt, response,
            args.invalid_edges_weight, args.cycles_weight,
            args.subgraphs_weight, args.format_penalty_weight,
        )
        rewards.append(reward)
        reward_parts_list.append(parts)

    # --- policy log-probs (with grad) ---
    log_pi_theta = compute_model_logprobs(model, generated_ids, prompt_length)

    # --- reference log-probs (no grad) ---
    with torch.no_grad():
        log_pi_ref = compute_model_logprobs(ref_model, generated_ids, prompt_length)
        # KL penalty: penalize any deviation from reference (use abs to keep non-negative)
        kl_per_seq = (log_pi_theta - log_pi_ref.detach()).abs()

    # --- KL-adjusted rewards ---
    adjusted_rewards = [
        rewards[i] - args.kl_beta * kl_per_seq[i].item()
        for i in range(len(rewards))
    ]

    # --- group advantages ---
    advantages = group_advantages(adjusted_rewards)
    advantages_tensor = torch.tensor(advantages, device=device, dtype=torch.float32)

    # --- GRPO loss for this prompt ---
    prompt_loss = -(advantages_tensor * log_pi_theta).mean()

    # --- response lengths ---
    response_lengths = [
        (generated_ids.shape[1] - prompt_length) for _ in range(len(responses))
    ]

    return prompt_loss, {
        "rewards": rewards,
        "adjusted_rewards": adjusted_rewards,
        "kl_mean": kl_per_seq.mean().item(),
        "advantages": advantages,
        "reward_parts": reward_parts_list,
        "responses": responses,
        "response_lengths": response_lengths,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    load_project_env()
    args = parse_args()
    validate_precision_args(args)

    import torch
    if args.num_generations < 2:
        raise ValueError("--num_generations must be at least 2 for group-relative RL.")
    if args.prompts_per_step < 1:
        raise ValueError("--prompts_per_step must be at least 1.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    deps = import_training_deps()
    set_all_seeds(deps, args.seed)
    rng = random.Random(args.seed)

    # --- directories ---
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    reward_log_path = run_dir / "reward_log.jsonl"
    if not args.resume_from:
        reward_log_path.write_text("", encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # --- W&B ---
    configure_wandb(args)
    _wandb_available = uses_wandb(args.report_to)
    if _wandb_available:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "RL4DistReconfig"),
            entity=os.environ.get("WANDB_ENTITY", None),
            group=os.environ.get("WANDB_RUN_GROUP", None),
            tags=[t.strip() for t in os.environ.get("WANDB_TAGS", "").split(",") if t.strip()],
            name=args.run_name,
            config=vars(args),
            dir=str(run_dir),
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
    print(f"Train samples: {len(train_dataset)}  "
          f"Val samples: {len(validation_dataset)}  "
          f"Test samples: {len(test_dataset)}")

    # --- models ---
    model, tokenizer = load_policy_model(args)
    device = next(model.parameters()).device
    ref_model = load_reference_model(args, tokenizer)
    # Note: ref_model and model can be on different devices
    # compute_model_logprobs handles device placement automatically

    # --- optimizer ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args)

    # --- resume ---
    start_step = 0
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        print(f"Resuming from {resume_dir}")
        from peft import PeftModel
        # Reload adapter weights
        PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)
        start_step, saved_rng = load_training_state(
            resume_dir, model, optimizer, scheduler,
        )
        rng.setstate(saved_rng)
        print(f"  Resumed at step {start_step}")

    if args.warmup_steps == 0 and args.lr_scheduler_type != "constant":
        args.warmup_steps = int(args.max_steps * 0.03)

    # --- training loop ---
    accumulation_steps = max(args.gradient_accumulation_steps, 1)
    effective_max_steps = args.max_steps * accumulation_steps

    print(f"Run directory: {run_dir}")
    print(f"Max steps: {args.max_steps}  "
          f"Accumulation: {accumulation_steps}  "
          f"Prompts/step: {args.prompts_per_step}  "
          f"Generations/prompt: {args.num_generations}")

    for micro_step in range(start_step + 1, effective_max_steps + 1):
        step = (micro_step - 1) // accumulation_steps + 1
        if step > args.max_steps:
            break

        # --- sample prompts ---
        indices = [rng.randrange(len(train_dataset)) for _ in range(args.prompts_per_step)]
        total_loss = torch.tensor(0.0, device=device)
        all_rewards = []
        all_adjusted = []
        all_kl = []
        all_advantages = []
        all_parts = []
        all_responses = []
        all_lengths = []

        for index in indices:
            prompt = train_dataset[int(index)]["prompt"]
            prompt_loss, info = process_prompt(model, ref_model, tokenizer, prompt, device, args)
            total_loss = total_loss + prompt_loss
            all_rewards.extend(info["rewards"])
            all_adjusted.extend(info["adjusted_rewards"])
            all_kl.append(info["kl_mean"])
            all_advantages.extend(info["advantages"])
            all_parts.extend(info["reward_parts"])
            all_responses.extend(info["responses"])
            all_lengths.extend(info["response_lengths"])

        # --- scale for accumulation ---
        loss = total_loss / (args.prompts_per_step * accumulation_steps)
        loss.backward()

        # --- optimizer step ---
        if micro_step % accumulation_steps == 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # --- logging ---
        if args.logging_steps > 0 and step % args.logging_steps == 0 and micro_step % accumulation_steps == 0:
            record = {
                "step": step,
                "micro_step": micro_step,
                "loss": float(total_loss.detach().cpu() / args.prompts_per_step),
                "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
                "mean_adjusted_reward": sum(all_adjusted) / len(all_adjusted) if all_adjusted else 0.0,
                "best_reward": max(all_rewards) if all_rewards else 0.0,
                "worst_reward": min(all_rewards) if all_rewards else 0.0,
                "mean_kl": sum(all_kl) / len(all_kl) if all_kl else 0.0,
                "mean_advantage": sum(all_advantages) / len(all_advantages) if all_advantages else 0.0,
                "grad_norm": _compute_grad_norm(model),
                "lr": optimizer.param_groups[0]["lr"],
                "reward_parts": _mean_dicts(all_parts),
                "mean_response_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0,
            }
            if args.log_responses:
                record["responses"] = all_responses

            with reward_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            print(
                f"step={step:4d} loss={record['loss']:.4f} "
                f"mean_reward={record['mean_reward']:.3f} "
                f"mean_adj_reward={record['mean_adjusted_reward']:.3f} "
                f"mean_kl={record['mean_kl']:.4f}"
            )

            if _wandb_available:
                wandb.log({
                    "rl/loss": record["loss"],
                    "rl/mean_reward": record["mean_reward"],
                    "rl/mean_adjusted_reward": record["mean_adjusted_reward"],
                    "rl/best_reward": record["best_reward"],
                    "rl/worst_reward": record["worst_reward"],
                    "rl/mean_kl": record["mean_kl"],
                    "rl/mean_advantage": record["mean_advantage"],
                    "rl/grad_norm": record["grad_norm"],
                    "rl/lr": record["lr"],
                    "rl/mean_response_length": record["mean_response_length"],
                    "rl/reward_invalid_edges": record["reward_parts"].get("invalid_edges", 0),
                    "rl/reward_cycles": record["reward_parts"].get("cycles", 0),
                    "rl/reward_subgraphs": record["reward_parts"].get("subgraphs", 0),
                    "rl/reward_format_penalty": record["reward_parts"].get("format_penalty", 0),
                }, step=step)

        # --- validation ---
        if (args.eval_steps > 0 and step % args.eval_steps == 0
                and micro_step % accumulation_steps == 0
                and len(validation_dataset) > 0):
            eval_metrics = run_validation(
                model, ref_model, tokenizer, validation_dataset, args, device,
            )
            print(f"  eval step={step} {eval_metrics}")
            with reward_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, **eval_metrics}) + "\n")
            if _wandb_available:
                wandb.log(eval_metrics, step=step)

        # --- save ---
        if args.save_steps > 0 and step % args.save_steps == 0 and micro_step % accumulation_steps == 0:
            checkpoint_dir = save_training_state(
                run_dir, model, optimizer, scheduler, step,
                rng.getstate(), f"checkpoint-{step}",
            )
            print(f"  Saved: {checkpoint_dir}")

    # --- final save ---
    final_dir = run_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    save_training_state(run_dir, model, optimizer, scheduler, args.max_steps,
                        rng.getstate(), "final_adapter")
    print(f"Saved final adapter: {final_dir}")

    if _wandb_available:
        wandb.finish()

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_grad_norm(model) -> float:
    import torch
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _mean_dicts(dicts: list[dict]) -> dict[str, float]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: sum(d.get(k, 0.0) for d in dicts) / len(dicts) for k in keys}


if __name__ == "__main__":
    raise SystemExit(main())
