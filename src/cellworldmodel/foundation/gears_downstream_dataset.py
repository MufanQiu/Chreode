"""GEARS Norman dataset adapter for shared-vocabulary downstream training."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cellworldmodel.foundation.gene_space import (
    adata_gene_names,
    build_gene_to_id,
    build_source_to_vocab,
    foundation_gene_names_from_vocab,
)
from cellworldmodel.foundation.io_utils import as_dense_array
from cellworldmodel.foundation.perturbation_dataset import normalize_norman_condition


def load_gears_split(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open("rb") as handle:
        obj = pickle.load(handle)
    return {key: [str(x) for x in value] for key, value in obj.items()}


def load_gears_subgroup_conditions(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    with Path(path).open("rb") as handle:
        obj = pickle.load(handle)
    test_subgroup = obj.get("test_subgroup", {}) if isinstance(obj, dict) else {}
    out: list[str] = []
    for key in ("combo_seen0", "combo_seen1", "combo_seen2", "unseen_single"):
        out.extend(str(x) for x in test_subgroup.get(key, []))
    return sorted(dict.fromkeys(out))


@dataclass(frozen=True)
class GearsDownstreamDataOptions:
    gears_adata: str | Path
    split: str | Path
    subgroup: str | Path | None
    gene_vocab: str | Path
    top_k: int = 20
    condition_col: str = "condition"
    control_label: str = "ctrl"


class GearsDownstreamDataset:
    def __init__(self, options: GearsDownstreamDataOptions) -> None:
        try:
            import anndata as ad
        except ImportError as exc:  # pragma: no cover
            raise ImportError("anndata is required for GEARS downstream training") from exc
        self.options = options
        self.adata = ad.read_h5ad(str(options.gears_adata))
        self.gene_vocab = pd.read_parquet(options.gene_vocab)
        self.gene_names = foundation_gene_names_from_vocab(options.gene_vocab)
        self.n_genes = len(self.gene_names)
        self.conditions_raw = self.adata.obs[options.condition_col].astype(str).to_numpy()
        self.control_idx = np.flatnonzero(self.conditions_raw == options.control_label)
        if len(self.control_idx) == 0:
            raise ValueError(f"No controls found with label={options.control_label!r}")
        self.split = load_gears_split(options.split)
        self.train_conditions = [c for c in self.split.get("train", []) if c != options.control_label]
        self.val_conditions = [c for c in self.split.get("val", []) if c != options.control_label]
        self.test_conditions = load_gears_subgroup_conditions(options.subgroup) or [
            c for c in self.split.get("test", []) if c != options.control_label
        ]
        self.condition_to_idx = {
            c: np.flatnonzero(self.conditions_raw == c)
            for c in sorted(set(self.train_conditions + self.val_conditions + self.test_conditions))
        }
        self.source_to_vocab = build_source_to_vocab(adata_gene_names(self.adata), self.gene_names)
        self.gene_to_id = build_gene_to_id(self.gene_vocab)
        self.control_mean = self.load_rows(self.control_idx).mean(axis=0)
        self.de_idx = self._build_de_idx()

    def load_rows(self, rows: np.ndarray) -> np.ndarray:
        x = as_dense_array(self.adata.X[np.asarray(rows, dtype=np.int64)])
        x = np.asarray(x, dtype=np.float32)
        out = np.zeros((len(rows), self.n_genes), dtype=np.float32)
        keep = self.source_to_vocab >= 0
        out[:, self.source_to_vocab[keep]] = x[:, keep]
        return out

    def condition_gene_arrays(self, condition: str, max_genes: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        normalized = normalize_norman_condition(condition)
        genes = [] if normalized == "ctrl" else [g for g in normalized.split("+") if g]
        gene_ids = np.full(max_genes, self.n_genes, dtype=np.int64)
        signs = np.zeros(max_genes, dtype=np.float32)
        modality_ids = np.zeros(max_genes, dtype=np.int64)
        strengths = np.ones(max_genes, dtype=np.float32)
        mask = np.zeros(max_genes, dtype=bool)
        for i, gene in enumerate(genes[:max_genes]):
            gene_ids[i] = self.gene_to_id.get(gene.upper(), self.n_genes)
            signs[i] = 1.0
            modality_ids[i] = 0
            strengths[i] = 1.0
            mask[i] = True
        return gene_ids, signs, modality_ids, strengths, mask

    def _build_de_idx(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for condition, rows in self.condition_to_idx.items():
            if len(rows) == 0:
                continue
            mean = self.load_rows(rows).mean(axis=0)
            effect = np.abs(mean - self.control_mean)
            k = min(int(self.options.top_k), len(effect))
            out[condition] = np.argsort(effect)[-k:][::-1].astype(np.int64)
        return out

    def sample_batch(self, condition: str, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        target_idx = self.condition_to_idx[condition]
        if len(target_idx) == 0:
            raise ValueError(f"No cells for condition={condition!r}")
        control_rows = rng.choice(self.control_idx, size=batch_size, replace=batch_size > len(self.control_idx))
        target_rows = rng.choice(target_idx, size=batch_size, replace=batch_size > len(target_idx))
        return self.load_rows(control_rows), self.load_rows(target_rows)

    def sample_set_pair(
        self,
        condition: str,
        set_size: int,
        rng: np.random.Generator,
        target_idx: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample unpaired control/target populations for one perturbation condition."""
        if target_idx is None:
            return self.sample_batch(condition, set_size, rng)
        if len(target_idx) == 0:
            raise ValueError(f"No cells for condition={condition!r}")
        control_rows = rng.choice(self.control_idx, size=set_size, replace=set_size > len(self.control_idx))
        target_rows = rng.choice(target_idx, size=set_size, replace=set_size > len(target_idx))
        return self.load_rows(control_rows), self.load_rows(target_rows)

    def condition_mean_arrays(self, condition: str, rng: np.random.Generator, max_cells: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        rows = self.condition_to_idx[condition]
        if max_cells is not None and len(rows) > int(max_cells):
            rows = rng.choice(rows, size=int(max_cells), replace=False)
        n = len(rows)
        control_rows = rng.choice(self.control_idx, size=n, replace=n > len(self.control_idx))
        return self.load_rows(control_rows), self.load_rows(rows), int(n)
