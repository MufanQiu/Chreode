"""Norman perturbation dataset adapter for foundation VAE latent fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_norman_condition(value: str) -> str:
    value = str(value)
    if value in {"ctrl", "control", "non-targeting", "NT"}:
        return "ctrl"
    if "__" in value:
        value = value.split("__", 1)[0]
    value = value.replace("_", "+")
    parts = [part for part in value.split("+") if part and part.lower() not in {"ctrl", "control"}]
    if not parts:
        return "ctrl"
    return "+".join(sorted(dict.fromkeys(parts)))


@dataclass(frozen=True)
class PerturbationBatch:
    control_x: np.ndarray
    target_x: np.ndarray
    gene_ids: np.ndarray
    signs: np.ndarray
    modality_ids: np.ndarray
    strengths: np.ndarray
    mask: np.ndarray
    condition: str
    split: str


class NormanPerturbationDataset:
    """Read Norman h5ad and expose control -> perturbed population batches.

    The first implementation uses condition-level categorical action ids. This
    is useful for testing the A0/A1/A2 initialization pipeline, but it is not a
    final unseen-gene action encoder.
    """

    def __init__(
        self,
        data_path: str | Path,
        gene_vocab_path: str | Path,
        *,
        condition_col: str | None = None,
        control_label: str = "ctrl",
        split_method: str = "additive",
        split_seed: int = 42,
        target_sum: float = 1e4,
    ) -> None:
        try:
            import anndata as ad
        except ImportError as exc:  # pragma: no cover
            raise ImportError("anndata is required for NormanPerturbationDataset") from exc
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Norman h5ad not found: {self.data_path}")
        self.adata = ad.read_h5ad(str(self.data_path))
        self.gene_vocab = pd.read_parquet(gene_vocab_path)
        self.target_sum = float(target_sum)
        x_max = self.adata.X.max()
        self.needs_normalize_log1p = float(x_max.toarray()[0, 0] if hasattr(x_max, "toarray") else x_max) > 50.0
        if condition_col is None:
            if "condition" in self.adata.obs:
                condition_col = "condition"
            elif "guide_identity" in self.adata.obs:
                condition_col = "guide_identity"
            else:
                raise ValueError(f"Could not infer Norman condition column from {list(self.adata.obs.columns)}")
        self.condition_col = condition_col
        raw_conditions = self.adata.obs[condition_col].astype(str).map(normalize_norman_condition)
        self.conditions = raw_conditions.to_numpy(dtype=object)
        self.control_label = normalize_norman_condition(control_label)
        self.control_idx = np.flatnonzero(self.conditions == self.control_label)
        self.perturbed_conditions = sorted(c for c in np.unique(self.conditions) if c != self.control_label)
        self.single_conditions = [c for c in self.perturbed_conditions if "+" not in c]
        self.double_conditions = [c for c in self.perturbed_conditions if "+" in c]
        self.train_conditions, self.test_conditions = self._split_conditions(split_method, split_seed)
        self.gene_to_id = self._build_vocab_gene_to_id()
        self._condition_idx_cache: dict[str, np.ndarray] = {}
        self.source_to_vocab = self._build_source_to_vocab()

    @property
    def n_actions(self) -> int:
        return self.n_genes + 1

    @property
    def n_genes(self) -> int:
        return len(self.gene_vocab)

    def _build_vocab_gene_to_id(self) -> dict[str, int]:
        out = {}
        for idx, row in self.gene_vocab.reset_index(drop=True).iterrows():
            for key in ("human_symbol", "canonical_gene"):
                value = str(row.get(key, "")).upper()
                if value and value != "NAN":
                    out.setdefault(value, int(idx))
        return out

    def _split_conditions(self, split_method: str, split_seed: int) -> tuple[list[str], list[str]]:
        rng = np.random.default_rng(split_seed)
        if split_method == "additive":
            shuffled = rng.permutation(len(self.double_conditions))
            n_test = int(len(self.double_conditions) * 0.3)
            test = [self.double_conditions[i] for i in shuffled[:n_test]]
            train = self.single_conditions + [self.double_conditions[i] for i in shuffled[n_test:]]
            return sorted(train), sorted(test)
        if split_method == "holdout":
            shuffled = rng.permutation(len(self.single_conditions))
            n_test_single = int(len(self.single_conditions) * 0.3)
            held = {self.single_conditions[i] for i in shuffled[:n_test_single]}
            test = list(held) + [c for c in self.double_conditions if any(g in held for g in c.split("+"))]
            train = [c for c in self.perturbed_conditions if c not in set(test)]
            return sorted(train), sorted(test)
        raise ValueError(f"Unknown split_method={split_method!r}")

    def _source_gene_names(self) -> list[str]:
        var = self.adata.var
        for key in ("gene_name", "gene_symbols", "symbol", "feature_name"):
            if key in var:
                return [str(x) for x in var[key].to_numpy()]
        return [str(x) for x in var.index.to_numpy()]

    def _build_source_to_vocab(self) -> np.ndarray:
        genes = self._source_gene_names()
        first_pos = {gene.upper(): i for i, gene in enumerate(genes)}
        mapping = np.full(len(genes), -1, dtype=np.int32)
        for vocab_idx, row in self.gene_vocab.reset_index(drop=True).iterrows():
            candidates = [
                str(row.get("human_symbol", "")).upper(),
                str(row.get("canonical_gene", "")).upper(),
            ]
            for candidate in candidates:
                pos = first_pos.get(candidate)
                if pos is not None:
                    mapping[pos] = int(vocab_idx)
                    break
        return mapping

    def _cells_for_condition(self, condition: str) -> np.ndarray:
        cached = self._condition_idx_cache.get(condition)
        if cached is not None:
            return cached
        idx = np.flatnonzero(self.conditions == condition)
        if len(idx) == 0:
            raise ValueError(f"No Norman cells for condition={condition!r}")
        self._condition_idx_cache[condition] = idx
        return idx

    def condition_gene_ids(self, condition: str, max_genes: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        genes = [] if condition == self.control_label else [g for g in condition.split("+") if g]
        gene_ids = np.full(max_genes, self.n_genes, dtype=np.int64)
        signs = np.zeros(max_genes, dtype=np.float32)
        modality_ids = np.zeros(max_genes, dtype=np.int64)
        strengths = np.ones(max_genes, dtype=np.float32)
        mask = np.zeros(max_genes, dtype=bool)
        for i, gene in enumerate(genes[:max_genes]):
            gene_ids[i] = self.gene_to_id.get(gene.upper(), self.n_genes)
            signs[i] = 1.0
            modality_ids[i] = 0  # Norman CRISPRa / over-expression.
            strengths[i] = 1.0
            mask[i] = True
        return gene_ids, signs, modality_ids, strengths, mask

    def _load_rows(self, rows: np.ndarray) -> np.ndarray:
        x = self.adata.X[rows]
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)
        out = np.zeros((len(rows), self.n_genes), dtype=np.float32)
        keep = self.source_to_vocab >= 0
        out[:, self.source_to_vocab[keep]] = x[:, keep]
        if not self.needs_normalize_log1p:
            return out
        sums = out.sum(axis=1, keepdims=True)
        scale = np.divide(self.target_sum, sums, out=np.zeros_like(sums), where=sums > 0)
        return np.log1p(out * scale).astype(np.float32, copy=False)

    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        split: str = "train",
        condition: str | None = None,
    ) -> PerturbationBatch:
        conditions = self.train_conditions if split == "train" else self.test_conditions
        if condition is None:
            if not conditions:
                raise ValueError(f"No Norman conditions for split={split!r}")
            condition = str(rng.choice(conditions))
        target_idx = self._cells_for_condition(condition)
        control_rows = rng.choice(self.control_idx, size=int(batch_size), replace=int(batch_size) > len(self.control_idx))
        target_rows = rng.choice(target_idx, size=int(batch_size), replace=int(batch_size) > len(target_idx))
        gene_ids, signs, modality_ids, strengths, mask = self.condition_gene_ids(condition)
        return PerturbationBatch(
            control_x=self._load_rows(control_rows),
            target_x=self._load_rows(target_rows),
            gene_ids=np.repeat(gene_ids[None, :], int(batch_size), axis=0),
            signs=np.repeat(signs[None, :], int(batch_size), axis=0),
            modality_ids=np.repeat(modality_ids[None, :], int(batch_size), axis=0),
            strengths=np.repeat(strengths[None, :], int(batch_size), axis=0),
            mask=np.repeat(mask[None, :], int(batch_size), axis=0),
            condition=condition,
            split=split,
        )
