"""Latent datasets for foundation static and temporal DiT pretraining."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cellworldmodel.foundation.latent_cache import LatentCacheDataset
from cellworldmodel.foundation.transition_index import FoundationTransitionIdSampler


@dataclass(frozen=True)
class StaticLatentBatch:
    cell_ids: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class LatentTransitionBatch:
    source_ids: np.ndarray
    target_ids: np.ndarray
    source: np.ndarray
    target: np.ndarray
    delta: float
    source_t: float
    target_t: float
    leaf_dataset: str
    transition_id: int


class FoundationStaticLatentDataset:
    """Sample frozen VAE latents for the static-DiT reconstruction control."""

    def __init__(self, latent_cache: LatentCacheDataset, split: str = "train") -> None:
        self.latent_cache = latent_cache
        rows = latent_cache.index[latent_cache.index["foundation_split"] == split]
        if rows.empty:
            raise ValueError(f"No latent rows for split={split!r}")
        self.cell_ids = rows["global_cell_id"].to_numpy(dtype=np.int64)

    def sample(self, batch_size: int, rng: np.random.Generator) -> StaticLatentBatch:
        ids = rng.choice(self.cell_ids, size=int(batch_size), replace=int(batch_size) > len(self.cell_ids))
        return StaticLatentBatch(cell_ids=ids.astype(np.int64), z=self.latent_cache.load_ids(ids))


class FoundationLatentTransitionDataset:
    """Sample source/target frozen VAE latents for temporal dynamics pretraining."""

    def __init__(
        self,
        latent_cache: LatentCacheDataset,
        catalog_dir: str,
        transition_index: pd.DataFrame,
        *,
        split: str = "train",
        leaf_sampling_alpha: float = 0.5,
    ) -> None:
        self.latent_cache = latent_cache
        self.id_sampler = FoundationTransitionIdSampler(
            catalog_dir,
            transition_index,
            split=split,
            leaf_sampling_alpha=leaf_sampling_alpha,
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> LatentTransitionBatch:
        ids = self.id_sampler.sample(batch_size, rng)
        return LatentTransitionBatch(
            source_ids=ids.source_ids,
            target_ids=ids.target_ids,
            source=self.latent_cache.load_ids(ids.source_ids),
            target=self.latent_cache.load_ids(ids.target_ids),
            delta=ids.delta,
            source_t=ids.source_t,
            target_t=ids.target_t,
            leaf_dataset=ids.leaf_dataset,
            transition_id=ids.transition_id,
        )
