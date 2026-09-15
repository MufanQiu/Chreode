"""Streaming expression batches from foundation catalog h5ad files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from cellworldmodel.foundation.encoder_input import normalize_mapped_expression
from cellworldmodel.foundation.h5ad_meta import read_var_names


@dataclass(frozen=True)
class ExpressionBatch:
    x: np.ndarray
    cell_ids: np.ndarray
    leaf_dataset: np.ndarray
    timepoint: np.ndarray
    foundation_split: np.ndarray


class FoundationExpressionDataset:
    """Read ortholog log1p-normalized expression batches from catalog artifacts."""

    def __init__(
        self,
        catalog_dir: str | Path,
        *,
        target_sum: float = 1e4,
    ) -> None:
        self.catalog_dir = Path(catalog_dir)
        self.cell_index = pd.read_parquet(self.catalog_dir / "cell_index.parquet")
        self.cell_index_by_id = self.cell_index.set_index("global_cell_id", drop=False)
        self.gene_vocab = pd.read_parquet(self.catalog_dir / "gene_vocab.parquet")
        self.target_sum = float(target_sum)
        self._source_to_vocab_cache: dict[str, np.ndarray] = {}
        self._split_leaf_pool_cache: dict[tuple[str, str], np.ndarray] = {}
        self.n_genes = len(self.gene_vocab)

    def cell_ids_for_split(self, split: str, leaf_dataset: str | None = None) -> np.ndarray:
        mask = self.cell_index["foundation_split"] == split
        if leaf_dataset is not None:
            mask &= self.cell_index["leaf_dataset"] == leaf_dataset
        return self.cell_index.loc[mask, "global_cell_id"].to_numpy(dtype=np.int64)

    def sample_cell_ids(
        self,
        split: str,
        n: int,
        rng: np.random.Generator,
        *,
        leaf_dataset: str | None = None,
    ) -> np.ndarray:
        pool = self.cell_ids_for_split(split, leaf_dataset=leaf_dataset)
        if len(pool) == 0:
            raise ValueError(f"No cells for split={split!r} leaf_dataset={leaf_dataset!r}")
        return rng.choice(pool, size=int(n), replace=int(n) > len(pool))

    def _split_leaf_pool(self, split: str, leaf_dataset: str) -> np.ndarray:
        key = (split, leaf_dataset)
        cached = self._split_leaf_pool_cache.get(key)
        if cached is not None:
            return cached
        mask = (
            (self.cell_index["foundation_split"] == split)
            & (self.cell_index["leaf_dataset"] == leaf_dataset)
        )
        pool = self.cell_index.loc[mask, "global_cell_id"].to_numpy(dtype=np.int64)
        self._split_leaf_pool_cache[key] = pool
        return pool

    def sample_cell_ids_balanced_by_leaf(
        self,
        split: str,
        n: int,
        rng: np.random.Generator,
        *,
        alpha: float = 0.5,
    ) -> np.ndarray:
        rows = self.cell_index[self.cell_index["foundation_split"] == split]
        if rows.empty:
            raise ValueError(f"No cells for split={split!r}")
        counts = rows["leaf_dataset"].astype(str).value_counts().sort_index()
        weights = counts.to_numpy(dtype=float) ** float(alpha)
        weights = weights / weights.sum()
        leaves = counts.index.to_numpy(dtype=object)
        sampled_leaves = rng.choice(leaves, size=int(n), replace=True, p=weights)
        out = np.empty(int(n), dtype=np.int64)
        for leaf in np.unique(sampled_leaves):
            positions = np.flatnonzero(sampled_leaves == leaf)
            pool = self._split_leaf_pool(split, str(leaf))
            if len(pool) == 0:
                raise ValueError(f"No cells for split={split!r} leaf_dataset={leaf!r}")
            out[positions] = rng.choice(pool, size=len(positions), replace=len(positions) > len(pool))
        rng.shuffle(out)
        return out

    def _source_to_vocab(self, h5ad_path: str) -> np.ndarray:
        cached = self._source_to_vocab_cache.get(h5ad_path)
        if cached is not None:
            return cached
        with h5py.File(h5ad_path, "r") as handle:
            genes = read_var_names(handle)
        first_pos: dict[str, int] = {}
        for i, gene in enumerate(genes):
            first_pos.setdefault(gene, i)
        source_to_vocab = np.full(len(genes), -1, dtype=np.int32)
        for vocab_idx, gene in enumerate(self.gene_vocab["canonical_gene"].astype(str)):
            pos = first_pos.get(gene)
            if pos is not None:
                source_to_vocab[pos] = int(vocab_idx)
        self._source_to_vocab_cache[h5ad_path] = source_to_vocab
        return source_to_vocab

    def _read_file_rows(self, h5ad_path: str, local_indices: np.ndarray) -> np.ndarray:
        source_to_vocab = self._source_to_vocab(h5ad_path)
        x_out = np.zeros((len(local_indices), self.n_genes), dtype=np.float32)
        with h5py.File(h5ad_path, "r") as handle:
            x = handle["X"]
            if isinstance(x, h5py.Group) and {"data", "indices", "indptr"}.issubset(x.keys()):
                data = x["data"]
                indices = x["indices"]
                indptr = x["indptr"]
                for out_i, row_i in enumerate(local_indices):
                    start = int(indptr[int(row_i)])
                    end = int(indptr[int(row_i) + 1])
                    row_indices = np.asarray(indices[start:end], dtype=np.int64)
                    row_values = np.asarray(data[start:end], dtype=np.float32)
                    mapped = source_to_vocab[row_indices]
                    keep = mapped >= 0
                    if np.any(keep):
                        x_out[out_i, mapped[keep]] = row_values[keep]
            else:
                dense = np.asarray(x[local_indices], dtype=np.float32)
                keep = source_to_vocab >= 0
                x_out[:, source_to_vocab[keep]] = dense[:, keep]
        return x_out

    def load_cells(self, cell_ids: np.ndarray | list[int], *, normalize_log1p: bool = True) -> ExpressionBatch:
        ids = np.asarray(cell_ids, dtype=np.int64)
        rows = self.cell_index_by_id.loc[ids].reset_index(drop=True)
        x = np.zeros((len(rows), self.n_genes), dtype=np.float32)
        for h5ad_path, group in rows.groupby("h5ad_path", sort=False):
            group_positions = group.index.to_numpy(dtype=np.int64)
            local_indices = group["local_cell_index"].to_numpy(dtype=np.int64)
            x[group_positions] = self._read_file_rows(str(h5ad_path), local_indices)
        if normalize_log1p:
            x = normalize_mapped_expression(x, target_sum=self.target_sum)
        return ExpressionBatch(
            x=x,
            cell_ids=rows["global_cell_id"].to_numpy(dtype=np.int64),
            leaf_dataset=rows["leaf_dataset"].astype(str).to_numpy(),
            timepoint=rows["timepoint"].to_numpy(dtype=np.float32),
            foundation_split=rows["foundation_split"].astype(str).to_numpy(),
        )
