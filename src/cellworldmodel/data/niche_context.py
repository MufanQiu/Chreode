"""Frozen source-only neighborhood summaries for spatial conditioning."""
from __future__ import annotations

import numpy as np


def neighborhood_composition(coordinates: np.ndarray, labels: np.ndarray,
                             categories: list[str], *, neighbors: int,
                             block_size: int = 256) -> np.ndarray:
    """One composition per source cell, excluding itself from nearest neighbors.

    Call separately for each source time and split. ``categories`` must be fit
    on training source labels; the final column explicitly represents unknown
    labels. Distances use only relative coordinates, with row index breaking
    exact ties deterministically. No target observations are accepted here.
    """
    xy = np.asarray(coordinates, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    if xy.ndim != 2 or xy.shape[1] not in (2, 3):
        raise ValueError("coordinates must be [N,2] or [N,3]")
    if labels.shape != (len(xy),) or len(xy) < 2:
        raise ValueError("need at least two source cells and one label per cell")
    if not np.isfinite(xy).all():
        raise ValueError("coordinates must be finite")
    if neighbors < 1 or block_size < 1:
        raise ValueError("neighbors and block_size must be positive")
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("categories must be nonempty and unique")
    vocabulary = {name: i for i, name in enumerate(categories)}
    codes = np.asarray([vocabulary.get(label, len(categories)) for label in labels])
    k = min(neighbors, len(xy) - 1)
    result = np.zeros((len(xy), len(categories) + 1), dtype=np.float32)
    for start in range(0, len(xy), block_size):
        stop = min(start + block_size, len(xy))
        distance = ((xy[start:stop, None] - xy[None]) ** 2).sum(-1)
        distance[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest = np.argsort(distance, axis=1, kind="stable")[:, :k]
        for local_row, ids in enumerate(nearest):
            result[start + local_row] = np.bincount(
                codes[ids], minlength=len(categories) + 1,
            ) / k
    return result


def shared_composition(training_source_labels: np.ndarray,
                       categories: list[str]) -> np.ndarray:
    """Global source composition for the equal-capacity spatially blind arm."""
    vocabulary = {name: i for i, name in enumerate(categories)}
    labels = np.asarray(training_source_labels).astype(str)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("training source labels must be a nonempty vector")
    codes = [vocabulary.get(label, len(categories)) for label in labels]
    return (np.bincount(codes, minlength=len(categories) + 1) / len(labels)).astype(np.float32)


class NicheTransitionData:
    """One adjacent transition from an existing audited representation export.

    Unlike the original benchmark sampler, this retains cell row identities so
    source states and their neighborhood vectors cannot get out of alignment.
    The export's train/val/test labels are reused, never reconstructed.
    """

    def __init__(self, root, *, source_time: float, target_time: float,
                 label_column: str, neighbors: int):
        import json
        from pathlib import Path

        import pandas as pd

        self.root = Path(root)
        self.meta = pd.read_csv(self.root / "metadata.tsv", sep="\t")
        required = {"time", "split", "cell_id", "slice", "spatial_x", "spatial_y", label_column}
        missing = required - set(self.meta.columns)
        if missing:
            raise ValueError(f"missing metadata columns: {sorted(missing)}")
        if self.meta["cell_id"].duplicated().any():
            raise ValueError("cell_id must be unique")
        if self.meta[list(required)].isna().any().any():
            raise ValueError("required metadata contains missing values")
        with np.load(self.root / "representations.npz") as rep:
            self.z = rep["scvi128"].astype(np.float32)
        if self.z.ndim != 2 or len(self.z) != len(self.meta) or not np.isfinite(self.z).all():
            raise ValueError("invalid or misaligned representations")
        self.manifest = json.loads((self.root / "split_manifest.json").read_text())
        encoder = self.manifest.get("encoder") or {}
        if not encoder.get("vae_sha256") or not encoder.get("gene_vocab_sha256"):
            raise ValueError("audited export must identify its encoder and vocabulary")
        time = self.meta["time"].to_numpy(dtype=float)
        if not np.isfinite(time).all():
            raise ValueError("time must be finite")
        split = self.meta["split"].astype(str).to_numpy()
        if set(split) - {"train", "val", "test"}:
            raise ValueError("metadata must use explicit train/val/test splits")
        timepoints = sorted(set(time))
        if source_time not in timepoints or target_time not in timepoints:
            raise ValueError("requested times are not present in this export")
        if timepoints.index(target_time) != timepoints.index(source_time) + 1:
            raise ValueError("first niche canary requires one adjacent forward transition")
        if self.meta.loc[time == source_time, "slice"].nunique() != 1:
            raise ValueError("first niche canary requires one source coordinate frame (slice)")
        self.source_time = float(source_time)
        self.target_time = float(target_time)
        self.delta = self.target_time - self.source_time
        self.labels = self.meta[label_column].astype(str).to_numpy()
        self.source_ids = {sp: np.flatnonzero((time == source_time) & (split == sp))
                           for sp in ("train", "val", "test")}
        self.target_ids = {sp: np.flatnonzero((time == target_time) & (split == sp))
                           for sp in ("train", "val", "test")}
        if any(len(ids) < 2 for ids in [*self.source_ids.values(), *self.target_ids.values()]):
            raise ValueError("each source and target split needs at least two cells")
        training_labels = self.labels[self.source_ids["train"]]
        self.categories = sorted(set(training_labels))
        if len(self.categories) < 2:
            raise ValueError("a one-category source cannot test niche composition")
        self.shared = shared_composition(training_labels, self.categories)
        self.context = np.zeros((len(self.z), len(self.categories) + 1), dtype=np.float32)
        self.effective_neighbors = {}
        for sp, ids in self.source_ids.items():
            coordinates = self.meta.iloc[ids][["spatial_x", "spatial_y"]].to_numpy()
            self.context[ids] = neighborhood_composition(
                coordinates, self.labels[ids], self.categories, neighbors=neighbors,
            )
            self.effective_neighbors[sp] = min(neighbors, len(ids) - 1)
        # Reuse the established train-only representation statistics. Verify the
        # existing file rather than silently overwriting a stale export.
        train_z = self.z[split == "train"]
        self.mu = train_z.mean(0).astype(np.float32)
        self.sd = np.maximum(train_z.std(0), 1e-6).astype(np.float32)
        with np.load(self.root / "metric_standardization.npz") as stats:
            if not np.allclose(stats["mean"], self.mu, rtol=1e-5, atol=1e-6):
                raise ValueError("export mean is not the current train-only mean")
            if not np.allclose(stats["std"], self.sd, rtol=1e-5, atol=1e-6):
                raise ValueError("export standard deviation is not train-only")

    def context_for(self, ids: np.ndarray, mode: str) -> np.ndarray:
        if mode == "neighborhood":
            return self.context[ids]
        if mode == "shared":
            return np.broadcast_to(self.shared, (len(ids), len(self.shared))).copy()
        raise ValueError(f"unknown context mode: {mode}")

    def standardize(self, z: np.ndarray) -> np.ndarray:
        return (z - self.mu) / self.sd
