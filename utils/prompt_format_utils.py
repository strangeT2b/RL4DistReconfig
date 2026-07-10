"""Prompt formatting helpers shared by SFT, RL, and evaluation."""

def format_prompt(prompt: str, prompt_format: str) -> str:
    """Format a prompt up to the assistant turn for generation/RL."""
    if prompt_format in ("raw", "plain", "none"):
        return prompt
    if prompt_format == "qwen_chat":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if prompt_format == "llama3_chat":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return f"<|user|>\n{prompt}</s>\n<|assistant|>\n"


def format_sft_text(prompt: str, output: str, prompt_format: str) -> str:
    """Format a full prompt + answer sequence for SFT."""
    if prompt_format in ("raw", "plain", "none"):
        return prompt + output
    if prompt_format == "qwen_chat":
        return format_prompt(prompt, prompt_format) + f"{output}<|im_end|>"
    if prompt_format == "llama3_chat":
        return format_prompt(prompt, prompt_format) + f"{output}<|eot_id|>"
    return format_prompt(prompt, prompt_format) + f"{output}</s>"


def formatted_prompt(question: str) -> str:
    """Upstream-compatible legacy Llama prompt format."""
    return format_prompt(question, "legacy")
