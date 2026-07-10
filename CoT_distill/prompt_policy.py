"""Prompt-safety policy constants and helpers for CoT distillation."""

from __future__ import annotations

import re
from typing import TypedDict


class ForbiddenTermHit(TypedDict):
    term: str
    matched: str
    start: int
    end: int


MODEL_FACING_FORBIDDEN_TERMS = (
    "ground truth",
    "gt",
    "reference",
    "hidden reference",
    "verifier",
    "evaluator",
    "external feedback",
    "benchmark",
    "oracle",
    "label",
    "iou",
    "recall",
    "precision",
    "automatic check",
    "automatic checks",
    "numeric metric",
    "numeric metrics",
    "system tool",
    "system tools",
    "process detail",
    "process details",
    "scoring",
    "tooling",
)


def _term_pattern(term: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(part) for part in term.split())
    return re.compile(
        rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )


_MODEL_FACING_FORBIDDEN_PATTERNS = tuple(
    (term, _term_pattern(term)) for term in MODEL_FACING_FORBIDDEN_TERMS
)


def find_model_facing_forbidden_terms(text: str) -> list[ForbiddenTermHit]:
    """Return boundary-aware forbidden-term hits in model-facing text."""
    if not text:
        return []
    hits: list[ForbiddenTermHit] = []
    for term, pattern in _MODEL_FACING_FORBIDDEN_PATTERNS:
        for match in pattern.finditer(text):
            hits.append(
                {
                    "term": term,
                    "matched": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return hits


def has_model_facing_forbidden_terms(text: str) -> bool:
    """Return True when text contains a boundary-aware forbidden-term hit."""
    return bool(find_model_facing_forbidden_terms(text))
