#!/usr/bin/env python3
"""Compare CE-loss gradients against CE + graph-penalty gradients.

This is a diagnostic for the LLM4DistReconfig custom-loss implementation.
It intentionally reproduces the important part of the training path:

    logits -> argmax -> batch_decode -> parse graph -> NetworkX penalties

Those operations are discrete/Python-side, so the graph penalties should be
constants with respect to model parameters. If that is true, gradients from
CE loss and CE + custom penalties will be identical for the same batch.

Run from the repository root:

    python SFT/check_custom_loss_gradients.py --assert-equal

Use the conda environment that has torch and networkx installed.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether LLM4DistReconfig graph penalties change gradients."
    )
    parser.add_argument("--seed", type=int, default=1234, help="Torch random seed.")
    parser.add_argument("--vocab-size", type=int, default=64, help="Toy vocabulary size.")
    parser.add_argument("--seq-len", type=int, default=32, help="Toy sequence length.")
    parser.add_argument("--hidden-size", type=int, default=16, help="Toy hidden size.")
    parser.add_argument("--atol", type=float, default=1e-10, help="Absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-10, help="Relative tolerance.")
    parser.add_argument(
        "--assert-equal",
        action="store_true",
        help="Exit with non-zero status if gradients differ beyond tolerances.",
    )
    return parser.parse_args()


class FixedTextTokenizer:
    """Tiny tokenizer stub that mimics the training script's decode boundary.

    The real code decodes both input ids and argmax output ids to strings.
    For this gradient check, the exact ids do not matter after argmax because
    parsing happens outside the torch graph. We still accept the ids so the
    call shape matches the original code.
    """

    def __init__(self) -> None:
        self.input_text = (
            "Power Distribution Network: Busses=4, "
            "Lines=[(1, 2), (2, 3), (3, 4), (4, 1), (2, 4)], "
            "Line Impedances=[0.1, 0.1, 0.1, 0.1, 0.1], "
            "Open Lines=[(2, 4)]\n"
            "Network Variables: NodeVoltages=[1.0, 0.99, 0.98, 0.97], "
            "System Loss=1.0, System Load=[0j, (0.1+0.02j), "
            "(0.1+0.02j), (0.1+0.02j)]\n"
        )
        self.output_text = (
            "Output: Open Lines=[(1, 99)], "
            "Node Voltages=[1.0, 0.99, 0.98, 0.97], System Loss=0.9\n"
        )

    def batch_decode(self, ids, skip_special_tokens: bool = True):  # noqa: ANN001
        batch_size = int(ids.shape[0]) if hasattr(ids, "shape") else 1
        if getattr(self, "_decode_inputs_next", False):
            self._decode_inputs_next = False
            return [self.input_text for _ in range(batch_size)]
        return [self.output_text for _ in range(batch_size)]

    def decode_inputs_next(self) -> None:
        self._decode_inputs_next = True


def build_toy_batch(torch, *, vocab_size: int, seq_len: int) -> dict:
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0) % vocab_size
    labels = (input_ids + 1) % vocab_size
    return {"input_ids": input_ids, "labels": labels}


def build_toy_model(torch, nn, *, vocab_size: int, hidden_size: int):
    class ToyCausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, labels=None):  # noqa: ANN001
            hidden = self.embed(input_ids)
            logits = self.lm_head(hidden)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                )
            return SimpleNamespace(logits=logits, loss=loss)

    return ToyCausalLM()


def compute_author_style_penalties(torch, tokenizer, inputs, outputs):
    from utils.metrics_utils import (  # pylint: disable=import-outside-toplevel
        compute_cycles_loss,
        compute_invalid_edges_loss,
        compute_subgraphs_loss,
        get_output_graph_edges,
        parse_available_lines,
        parse_open_lines,
    )

    tokenizer.decode_inputs_next()
    input_text = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
    argmax_ids = outputs.logits.argmax(dim=-1)
    output_text = tokenizer.batch_decode(argmax_ids, skip_special_tokens=True)

    available_lines = parse_available_lines(input_text[0])
    predicted_lines = parse_open_lines(output_text[0])
    graph_edges = get_output_graph_edges(predicted_lines, available_lines)

    if predicted_lines:
        invalid_edges_loss = compute_invalid_edges_loss(predicted_lines, available_lines) / len(
            predicted_lines
        )
        cycles_loss = compute_cycles_loss(graph_edges) / len(available_lines)
        subgraphs_loss = compute_subgraphs_loss(graph_edges) / len(predicted_lines)
    else:
        invalid_edges_loss = torch.tensor(1.0)
        cycles_loss = torch.tensor(1.0)
        subgraphs_loss = torch.tensor(1.0)

    return {
        "argmax_ids_requires_grad": getattr(argmax_ids, "requires_grad", None),
        "predicted_lines": predicted_lines,
        "available_lines": available_lines,
        "invalid_edges_loss": invalid_edges_loss,
        "cycles_loss": cycles_loss,
        "subgraphs_loss": subgraphs_loss,
        "total_penalty": invalid_edges_loss + cycles_loss + subgraphs_loss,
    }


def collect_gradients(model, torch, batch, tokenizer, *, use_custom_loss: bool):
    model.zero_grad(set_to_none=True)
    outputs = model(**batch)
    ce_loss = outputs.loss
    penalty_info = None

    if use_custom_loss:
        penalty_info = compute_author_style_penalties(torch, tokenizer, batch, outputs)
        total_loss = ce_loss + penalty_info["total_penalty"]
    else:
        total_loss = ce_loss

    total_loss.backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    return ce_loss.detach(), total_loss.detach(), penalty_info, grads


def compare_gradients(torch, ce_grads: dict, custom_grads: dict, *, atol: float, rtol: float):
    names = sorted(set(ce_grads) | set(custom_grads))
    rows = []
    max_abs_diff = 0.0
    allclose = True

    for name in names:
        if name not in ce_grads or name not in custom_grads:
            rows.append((name, float("inf"), False, "missing"))
            allclose = False
            continue

        diff = (ce_grads[name] - custom_grads[name]).abs().max().item()
        close = torch.allclose(ce_grads[name], custom_grads[name], atol=atol, rtol=rtol)
        rows.append((name, diff, bool(close), "ok"))
        max_abs_diff = max(max_abs_diff, diff)
        allclose = allclose and bool(close)

    return allclose, max_abs_diff, rows


def main() -> int:
    args = parse_args()

    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on local env
        print("Failed to import torch. Activate an environment with torch installed.", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    base_model = build_toy_model(
        torch, nn, vocab_size=args.vocab_size, hidden_size=args.hidden_size
    )
    batch = build_toy_batch(torch, vocab_size=args.vocab_size, seq_len=args.seq_len)

    ce_model = copy.deepcopy(base_model)
    custom_model = copy.deepcopy(base_model)

    ce_loss, ce_total, _, ce_grads = collect_gradients(
        ce_model,
        torch,
        batch,
        FixedTextTokenizer(),
        use_custom_loss=False,
    )
    custom_ce_loss, custom_total, penalty_info, custom_grads = collect_gradients(
        custom_model,
        torch,
        batch,
        FixedTextTokenizer(),
        use_custom_loss=True,
    )

    allclose, max_abs_diff, rows = compare_gradients(
        torch, ce_grads, custom_grads, atol=args.atol, rtol=args.rtol
    )

    print("=== Loss values ===")
    print(f"CE-only CE loss:        {ce_loss.item():.12f}")
    print(f"CE-only total loss:     {ce_total.item():.12f}")
    print(f"Custom path CE loss:    {custom_ce_loss.item():.12f}")
    print(f"Custom path total loss: {custom_total.item():.12f}")

    print("\n=== Author-style penalty terms ===")
    assert penalty_info is not None
    print(f"argmax_ids.requires_grad: {penalty_info['argmax_ids_requires_grad']}")
    print(f"predicted_lines:          {penalty_info['predicted_lines']}")
    for key in ("invalid_edges_loss", "cycles_loss", "subgraphs_loss", "total_penalty"):
        value = penalty_info[key]
        print(
            f"{key}: value={float(value):.12f}, "
            f"requires_grad={value.requires_grad}, grad_fn={value.grad_fn}"
        )

    print("\n=== Gradient comparison ===")
    print(f"trainable tensors compared: {len(rows)}")
    print(f"max_abs_grad_diff:          {max_abs_diff:.12g}")
    print(f"allclose(atol={args.atol}, rtol={args.rtol}): {allclose}")
    for name, diff, close, status in rows:
        print(f"{name}: max_abs_diff={diff:.12g}, allclose={close}, status={status}")

    if args.assert_equal and not allclose:
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
