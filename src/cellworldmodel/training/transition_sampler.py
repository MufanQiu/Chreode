"""Transition samplers shared by benchmark and ZESTA training loops."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np
import torch

from cellworldmodel.training.split_policy import SplitIndices, build_timepoint_splits


@dataclass
class TransitionBatch:
    source: torch.Tensor
    target: torch.Tensor
    delta: float
    source_t: float
    target_t: float
    source_weight: torch.Tensor | None = None
    target_weight: torch.Tensor | None = None
    target_mass_ratio: float | None = None


class TimepointTransitionSampler:
    """Sample transition batches from adapters exposing `coords_by_t`.

    This class is intentionally thin over existing adapters. It does not own
    data loading; it only centralizes transition-pair sampling and per-timepoint
    train/val/test splits.
    """

    def __init__(
        self,
        adapter,
        split_seed: int,
        pairs: Sequence[tuple[float, float]] | None = None,
        endpoint_prob: float | None = None,
        endpoint_pair: tuple[float, float] | None = None,
        split_ratios=(0.7, 0.1, 0.2),
        reference_target_times: Sequence[float] | None = None,
        growth_scores_by_t: dict[float, np.ndarray] | None = None,
        growth_weight_scale: float = 1.0,
        population_mass_by_t: dict[float, float] | None = None,
    ) -> None:
        if not hasattr(adapter, "coords_by_t"):
            raise TypeError(f"{type(adapter).__name__} does not expose coords_by_t")
        self.adapter = adapter
        self.dim = int(adapter.dim)
        self.timepoints = [float(t) for t in adapter.timepoints]
        if hasattr(adapter, "splits_by_t"):
            self.splits = {float(t): split for t, split in adapter.splits_by_t.items()}
        else:
            self.splits = build_timepoint_splits(adapter.coords_by_t, split_seed, split_ratios)
        if pairs is None:
            pairs = [(s, t) for i, s in enumerate(self.timepoints) for t in self.timepoints[i + 1:]]
        self.pairs = [(float(s), float(t)) for s, t in pairs]
        if not self.pairs:
            raise ValueError("At least one transition pair is required")
        self.endpoint_pair = (
            (float(endpoint_pair[0]), float(endpoint_pair[1]))
            if endpoint_pair is not None else (self.pairs[0][0], max(t for _, t in self.pairs))
        )
        self.pair_probs = self._build_pair_probs(endpoint_prob)
        if reference_target_times is None:
            reference_target_times = sorted({t for _, t in self.pairs})
        self.reference_target_times = [float(t) for t in reference_target_times]
        self.growth_scores_by_t = {
            float(t): np.asarray(scores, dtype=np.float32)
            for t, scores in (growth_scores_by_t or {}).items()
        }
        for t, scores in self.growth_scores_by_t.items():
            if t not in self.adapter.coords_by_t:
                raise ValueError(f"growth_scores timepoint={t} not found in adapter")
            if scores.shape[0] != self.adapter.coords_by_t[t].shape[0]:
                raise ValueError(
                    f"growth_scores length mismatch for t={t}: "
                    f"{scores.shape[0]} vs {self.adapter.coords_by_t[t].shape[0]}"
                )
        self.growth_weight_scale = float(growth_weight_scale)
        if population_mass_by_t is None and hasattr(adapter, "population_mass_by_t"):
            population_mass_by_t = adapter.population_mass_by_t
        self.population_mass_by_t = {
            float(t): float(mass)
            for t, mass in (population_mass_by_t or {}).items()
        }
        for t, mass in self.population_mass_by_t.items():
            if t not in self.adapter.coords_by_t:
                raise ValueError(f"population mass timepoint={t} not found in adapter")
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError(f"population mass must be finite and positive at t={t}")

    def _build_pair_probs(self, endpoint_prob: float | None):
        if endpoint_prob is None:
            return None
        endpoint_prob = float(endpoint_prob)
        if not (0.0 < endpoint_prob < 1.0):
            raise ValueError(f"endpoint_prob must be in (0,1), got {endpoint_prob}")
        endpoint_pair = self.endpoint_pair
        if endpoint_pair not in self.pairs:
            raise ValueError(f"Endpoint pair {endpoint_pair} not in pairs={self.pairs}")
        probs = np.full(len(self.pairs), (1.0 - endpoint_prob) / (len(self.pairs) - 1))
        probs[self.pairs.index(endpoint_pair)] = endpoint_prob
        return probs

    def split_for(self, t: float) -> SplitIndices:
        return self.splits[float(t)]

    def subsample_training(self, fraction: float, *, benchmark: str, seed: int) -> dict:
        """Restrict both sides of every training transition, preserving held-out pools."""
        if not 0 < fraction <= 1:
            raise ValueError("Training fraction must be in (0, 1]")
        key = json.dumps([benchmark, float(fraction), int(seed)], separators=(",", ":"))
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
        counts = {}
        for t in sorted(self.splits):
            split = self.splits[t]
            if not len(split.train):
                raise ValueError(f"Empty training pool at timepoint={t}")
            indices = split.train
            if fraction < 1:
                n = max(1, int(np.floor(len(indices) * fraction)))
                positions = np.sort(rng.choice(len(indices), n, replace=False))
                indices = indices[positions]
                self.splits[t] = SplitIndices(indices, split.val, split.test)
            counts[str(t)] = {
                "available": len(split.train), "selected": len(indices),
                "indices": indices.tolist(),
                "sha256": hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest(),
            }
        return {"fraction": fraction, "rng_key": key, "timepoints": counts}

    def get_population(self, t: float, split: str = "all") -> torch.Tensor:
        t = float(t)
        arr = self.adapter.coords_by_t[t]
        idx = self.splits[t].get(split)
        return torch.from_numpy(arr[idx])

    def sample_indices(self, t: float, split: str, batch_size: int,
                       rng: np.random.Generator) -> np.ndarray:
        t = float(t)
        idx_pool = self.splits[t].get(split)
        if len(idx_pool) == 0:
            raise ValueError(f"No cells for timepoint={t} split={split}")
        return rng.choice(idx_pool, size=batch_size, replace=batch_size > len(idx_pool))

    def sample_population(self, t: float, split: str, batch_size: int,
                          rng: np.random.Generator) -> torch.Tensor:
        t = float(t)
        arr = self.adapter.coords_by_t[t]
        idx = self.sample_indices(t, split, batch_size, rng)
        return torch.from_numpy(arr[idx])

    def source_growth_weight(self, t: float, idx: np.ndarray, delta: float) -> torch.Tensor | None:
        t = float(t)
        if t not in self.growth_scores_by_t:
            return None
        score = self.growth_scores_by_t[t][idx].astype(np.float32)
        weight = np.exp(float(delta) * float(self.growth_weight_scale) * score)
        return torch.from_numpy(weight.astype(np.float32))

    def target_mass_ratio(self, source_t: float, target_t: float) -> float | None:
        source_t = float(source_t)
        target_t = float(target_t)
        if source_t not in self.population_mass_by_t or target_t not in self.population_mass_by_t:
            return None
        return self.population_mass_by_t[target_t] / self.population_mass_by_t[source_t]

    def sample_train_batch(self, batch_size: int, rng: np.random.Generator) -> TransitionBatch:
        if self.pair_probs is None:
            pair_idx = int(rng.integers(len(self.pairs)))
        else:
            pair_idx = int(rng.choice(len(self.pairs), p=self.pair_probs))
        source_t, target_t = self.pairs[pair_idx]
        source_idx = self.sample_indices(source_t, "train", batch_size, rng)
        target_idx = self.sample_indices(target_t, "train", batch_size, rng)
        delta = float(target_t - source_t)
        return TransitionBatch(
            source=torch.from_numpy(self.adapter.coords_by_t[source_t][source_idx]),
            target=torch.from_numpy(self.adapter.coords_by_t[target_t][target_idx]),
            delta=delta,
            source_t=float(source_t),
            target_t=float(target_t),
            source_weight=self.source_growth_weight(source_t, source_idx, delta),
            target_mass_ratio=self.target_mass_ratio(source_t, target_t),
        )

    def reference_target(self, split: str = "train") -> torch.Tensor:
        chunks = [self.get_population(t, split=split) for t in self.reference_target_times]
        return torch.cat(chunks, dim=0)


def all_ordered_pairs(timepoints: Sequence[float]) -> list[tuple[float, float]]:
    tps = [float(t) for t in timepoints]
    return [(s, t) for i, s in enumerate(tps) for t in tps[i + 1:]]
