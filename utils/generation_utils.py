"""Generation config and output parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re

DEFAULT_STOP_STRINGS = ("</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>")


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------

def eos_token_ids_for_generation(tokenizer, prompt_format: str) -> list[int]:
    """Return EOS ids without mutating ``tokenizer.eos_token_id``."""
    eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        eos_ids = [eos]
    else:
        eos_ids = [int(token_id) for token_id in (eos or [])]

    if prompt_format == "legacy":
        stop_strings = ("</s>",) + tuple(
            stop for stop in DEFAULT_STOP_STRINGS if stop != "</s>"
        )
    elif prompt_format == "llama3_chat":
        stop_strings = ("<|eot_id|>",) + tuple(
            stop for stop in DEFAULT_STOP_STRINGS if stop != "<|eot_id|>"
        )
    elif prompt_format == "qwen_chat":
        stop_strings = ("<|im_end|>",) + tuple(
            stop for stop in DEFAULT_STOP_STRINGS if stop != "<|im_end|>"
        )
    else:
        stop_strings = DEFAULT_STOP_STRINGS

    for stop in stop_strings:
        ids = tokenizer.encode(stop, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in eos_ids:
            eos_ids.append(int(ids[0]))
    return eos_ids


def pad_token_id_for_generation(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if isinstance(tokenizer.eos_token_id, int):
        return int(tokenizer.eos_token_id)
    eos_ids = list(tokenizer.eos_token_id or [])
    if eos_ids:
        return int(eos_ids[0])
    raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")


def configure_generation(model, tokenizer, prompt_format: str):
    """Apply generation defaults and return ``(pad_id, eos_ids)``."""
    pad_id = pad_token_id_for_generation(tokenizer)
    eos_ids = eos_token_ids_for_generation(tokenizer, prompt_format)

    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = pad_id
        model.generation_config.eos_token_id = eos_ids if len(eos_ids) > 1 else eos_ids[0]
        if hasattr(model.generation_config, "enable_thinking"):
            model.generation_config.enable_thinking = False

    if hasattr(model, "config") and hasattr(model.config, "enable_thinking"):
        model.config.enable_thinking = False

    if hasattr(model, "set_adapter"):
        try:
            model.set_adapter("default")
        except Exception:
            pass
    if hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()

    return pad_id, eos_ids


def truncate_on_stop(text: str, stop_strings=DEFAULT_STOP_STRINGS) -> str:
    first = None
    for stop in stop_strings:
        index = text.find(stop)
        if index >= 0:
            first = index if first is None else min(first, index)
    return text if first is None else text[:first]


@dataclass
class StopStringCriteria:
    """Per-row HF stopping criterion for string stops."""

    tokenizer: object
    prompt_length: int
    stop_strings: tuple[str, ...] = DEFAULT_STOP_STRINGS
    tail_tokens: int = 16

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        done = []
        for row in input_ids:
            generated = row[self.prompt_length :]
            if generated.numel() == 0:
                done.append(False)
                continue
            tail = generated[-self.tail_tokens :]
            text = self.tokenizer.decode(tail, skip_special_tokens=False)
            done.append(any(stop in text for stop in self.stop_strings))
        return torch.tensor(done, device=input_ids.device, dtype=torch.bool)


def build_stopping_criteria(tokenizer, prompt_length: int):
    from transformers import StoppingCriteriaList

    return StoppingCriteriaList(
        [StopStringCriteria(tokenizer=tokenizer, prompt_length=prompt_length)]
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def extract_output_data(response):
    output_pattern = re.compile(
        r"Open Lines=\[(.*?)\],\s*(?:Node Voltages|Network Variables: NodeVoltages)=\[(.*?)\],\s*System Loss=(\d+\.\d+)"
    )

    match = output_pattern.search(response)

    if match:
        open_lines = match.group(1)
        node_voltages = match.group(2)
        system_loss = match.group(3)

        if not open_lines.startswith("("):
            open_lines = "(" + open_lines
        if not open_lines.endswith(")"):
            open_lines = open_lines + ")"

        try:
            open_lines_list = [
                tuple(map(int, line.split(",")))
                for line in re.findall(r"\((\d+,\s*\d+)\)", open_lines)
            ]
        except ValueError:
            open_lines_list = []

        try:
            node_voltages_list = [
                float(voltage.strip()) for voltage in node_voltages.split(",")
            ]
        except ValueError:
            node_voltages_list = []

        return {
            "Open Lines": open_lines_list,
            "Node Voltages": node_voltages_list,
            "System Loss": float(system_loss),
        }
    else:
        return "No output data found in the response."

