"""Model loading helpers (training + inference)."""

from __future__ import annotations

import os
import random
from typing import Any


def import_training_deps():
    try:
        import numpy as np
        import torch
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
            set_seed,
        )
        from trl import SFTTrainer

        try:
            from trl import SFTConfig
        except ImportError:
            SFTConfig = None
    except Exception as exc:
        print("Could not import training dependencies.")
        print("Activate the training environment with torch/transformers/peft/trl.")
        print(f"Import error: {exc}")
        raise

    return locals()


def set_all_seeds(deps: dict[str, Any], seed: int) -> None:
    random.seed(seed)
    deps["np"].random.seed(seed)
    deps["torch"].manual_seed(seed)
    if deps["torch"].cuda.is_available():
        deps["torch"].cuda.manual_seed_all(seed)
    if "set_seed" in deps:
        deps["set_seed"](seed)


def model_device_map():
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return "auto"
    return {"": int(local_rank)}


def load_model_and_tokenizer(args, deps: dict[str, Any]):
    tokenizer = deps["AutoTokenizer"].from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = deps["torch"].bfloat16 if args.bf16 else deps["torch"].float16

    use_4bit = getattr(args, "load_in_4bit", False)
    if use_4bit:
        bnb_config = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = deps["AutoModelForCausalLM"].from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map=model_device_map(),
        )
    else:
        model = deps["AutoModelForCausalLM"].from_pretrained(
            args.model_id,
            torch_dtype=compute_dtype,
            device_map=model_device_map(),
        )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def get_model(model_id):
    """Standalone model loader (matches upstream get_model)."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def get_tokenizer(model_id):
    """Standalone tokenizer loader (matches upstream get_tokenizer)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer




def get_model_and_tokenizer_qlora(model_id):
    """Author's QLoRA 4-bit model loading, preserved for SFT reproduction."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer

def peft_merge_unload(
    model_id,
    model_path,
    torch_dtype=None,
    load_in_8bit=False,
    device_map="auto",
    trust_remote_code=True,
    from_transformers=True,
):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    import torch

    torch_dtype = torch_dtype or torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch_dtype,
        load_in_8bit=load_in_8bit,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(
        model=model,
        model_id=model_path,
        from_transformers=from_transformers,
        device_map=device_map,
    )
    return peft_model.merge_and_unload()
