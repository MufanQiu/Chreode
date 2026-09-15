"""Numerical transforms applied AFTER a caller's vocabulary mapping.

The foundation catalog and external ortholog exporters have different existing
mapping policies. This module deliberately does not change either policy. It
shares the exact mapped-count normalization used during VAE training.
"""
from __future__ import annotations

import numpy as np

INPUT_TRANSFORM_MODES = ("match-pretraining", "legacy-as-is")
FOUNDATION_TARGET_SUM = 1e4


def normalize_mapped_expression(x: np.ndarray, *, target_sum: float = FOUNDATION_TARGET_SUM) -> np.ndarray:
    """Original FoundationExpressionDataset arithmetic, without added clipping.

    Nonpositive row sums retain the original zero-scale behavior. Negative
    entries in a positive-sum row can still produce NaN under log1p, exactly as
    in the original contract; callers must not mistake this for count repair.
    """
    sums = x.sum(axis=1, keepdims=True)
    scale = np.divide(target_sum, sums, out=np.zeros_like(sums), where=sums > 0)
    return np.log1p(x * scale).astype(np.float32, copy=False)


def input_transform_spec(mode: str) -> dict:
    if mode not in INPUT_TRANSFORM_MODES:
        raise ValueError(f"input_transform={mode!r}; expected one of {INPUT_TRANSFORM_MODES}")
    return {
        "mode": mode,
        "scope": "after the caller's existing vocabulary mapping",
        "target_sum": FOUNDATION_TARGET_SUM if mode == "match-pretraining" else None,
        "operations": ["normalize_total", "log1p"] if mode == "match-pretraining" else [],
        "training_helper": "cellworldmodel.foundation.encoder_input.normalize_mapped_expression",
        "gene_mapping_changed": False,
    }


def transform_encoder_input(x: np.ndarray, *, mode: str) -> np.ndarray:
    """Explicit selection: corrected training-scale input or exact legacy replay."""
    input_transform_spec(mode)
    if mode == "legacy-as-is":
        return x
    validate_linear_count_values(x)
    return normalize_mapped_expression(x)


def validate_linear_count_values(values: np.ndarray) -> None:
    """Reject invalid new export inputs; do not clip or change training math."""
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("match-pretraining requires finite, nonnegative mapped linear expression")
