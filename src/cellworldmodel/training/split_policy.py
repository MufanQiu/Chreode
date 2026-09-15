"""Split policies for timepoint-indexed population adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def get(self, split: str) -> np.ndarray:
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        if split == "test":
            return self.test
        if split == "all":
            return np.concatenate([self.train, self.val, self.test])
        raise ValueError(f"Unknown split={split!r}; expected train/val/test/all")


def split_indices(n: int, seed: int, ratios=(0.7, 0.1, 0.2)) -> SplitIndices:
    """Create deterministic train/val/test indices for one population."""
    if n <= 0:
        empty = np.array([], dtype=np.int64)
        return SplitIndices(empty, empty, empty)
    train_ratio, val_ratio, test_ratio = ratios
    total = float(train_ratio + val_ratio + test_ratio)
    if total <= 0:
        raise ValueError(f"Invalid split ratios={ratios}")
    train_ratio /= total
    val_ratio /= total
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).astype(np.int64)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), n - n_train)
    train = perm[:n_train]
    val = perm[n_train:n_train + n_val]
    test = perm[n_train + n_val:]
    if len(test) == 0 and n > 1:
        test = train[-1:]
        train = train[:-1]
    return SplitIndices(train=train, val=val, test=test)


def build_timepoint_splits(coords_by_t: Mapping[float, np.ndarray], seed: int,
                           ratios=(0.7, 0.1, 0.2)) -> dict[float, SplitIndices]:
    """Build deterministic per-timepoint splits.

    The seed is offset by timepoint order so each timepoint receives an
    independent permutation while remaining reproducible from one split_seed.
    """
    splits: dict[float, SplitIndices] = {}
    for offset, t in enumerate(sorted(coords_by_t.keys())):
        splits[float(t)] = split_indices(coords_by_t[t].shape[0], seed + offset * 1009, ratios)
    return splits
