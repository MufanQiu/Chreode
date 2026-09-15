"""Shared-vocabulary GEARS-style perturbation evaluation.

This module intentionally does not import the GEARS package.  It consumes a
GEARS processed AnnData object plus prediction arrays and computes metrics on
the intersection of the GEARS gene vocabulary and the model output vocabulary.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from cellworldmodel.foundation.gene_space import adata_gene_names, gene_key, load_gene_names, shared_gene_indexes
from cellworldmodel.foundation.io_utils import write_json


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    value = pearsonr(x, y)[0]
    return 0.0 if np.isnan(value) else float(value)


def _mean_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).reshape(-1)
    return np.asarray(x).mean(axis=0).reshape(-1)


def _load_prediction_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = {"pred", "truth", "conditions", "gene_names"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"{path} is missing arrays: {missing}")
    return {key: data[key] for key in data.files}


def _load_gears_test_res(path: str | Path, gene_names: list[str]) -> dict[str, np.ndarray]:
    with Path(path).open("rb") as handle:
        test_res = pickle.load(handle)
    return {
        "pred": np.asarray(test_res["pred"], dtype=np.float32),
        "truth": np.asarray(test_res["truth"], dtype=np.float32),
        "conditions": np.asarray(test_res["pert_cat"]).astype(str),
        "gene_names": np.asarray(gene_names).astype(str),
    }


def _condition_full_id_map(adata) -> dict[str, str]:
    if "condition_name" not in adata.obs:
        return {}
    return dict(adata.obs[["condition", "condition_name"]].drop_duplicates().values)


@dataclass(frozen=True)
class SharedVocabularyEvalOptions:
    gears_adata: str | Path
    prediction: str | Path | None
    output_dir: str | Path
    ours_genes: str | Path
    subgroup: str | Path | None = None
    gears_test_res: str | Path | None = None
    top_k: int = 20
    condition_col: str = "condition"
    control_label: str = "ctrl"


class SharedVocabularyDEEvaluator:
    def __init__(self, options: SharedVocabularyEvalOptions) -> None:
        try:
            import anndata as ad
        except ImportError as exc:  # pragma: no cover
            raise ImportError("anndata is required for shared-vocabulary GEARS evaluation") from exc
        self.options = options
        self.output_dir = Path(options.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adata = ad.read_h5ad(str(options.gears_adata))
        self.gears_gene_names = adata_gene_names(self.adata)
        if options.prediction is not None:
            self.prediction = _load_prediction_npz(options.prediction)
        elif options.gears_test_res is not None:
            self.prediction = _load_gears_test_res(options.gears_test_res, self.gears_gene_names)
        else:
            raise ValueError("Provide either prediction or gears_test_res")
        self.ours_gene_names = load_gene_names(options.ours_genes)
        self.subgroup = self._load_subgroup(options.subgroup)

    @staticmethod
    def _load_subgroup(path: str | Path | None) -> dict:
        if path is None:
            return {}
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        return obj if isinstance(obj, dict) else {}

    def _shared_indexes(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        return shared_gene_indexes(self.gears_gene_names, [str(x) for x in self.prediction["gene_names"]], self.ours_gene_names)

    def _control_mean(self, adata_idx: np.ndarray) -> np.ndarray:
        obs = self.adata.obs
        control_mask = obs[self.options.condition_col].astype(str).to_numpy() == self.options.control_label
        if not np.any(control_mask):
            raise ValueError(f"No control cells found with label={self.options.control_label!r}")
        return _mean_dense(self.adata.X[control_mask][:, adata_idx]).astype(np.float64)

    def _official_de20_coverage(
        self,
        condition: str,
        shared_keys: set[str],
    ) -> tuple[float | None, int | None]:
        full_map = _condition_full_id_map(self.adata)
        full_id = full_map.get(condition)
        if full_id is None:
            return None, None
        de_map = self.adata.uns.get("rank_genes_groups_cov_all")
        if de_map is None or full_id not in de_map:
            return None, None
        top = [str(x) for x in list(de_map[full_id])[: self.options.top_k]]
        if not top:
            return None, 0
        var_to_gene = {
            gene_key(var_id): self.gears_gene_names[i]
            for i, var_id in enumerate(self.adata.var_names.to_numpy())
        }
        top_keys = {gene_key(var_to_gene.get(gene_key(g), g)) for g in top}
        overlap = len(top_keys.intersection(shared_keys))
        return float(overlap / len(top)), int(overlap)

    def evaluate(self) -> dict[str, Any]:
        pred = np.asarray(self.prediction["pred"], dtype=np.float64)
        truth = np.asarray(self.prediction["truth"], dtype=np.float64)
        conditions = np.asarray(self.prediction["conditions"]).astype(str)
        if pred.shape != truth.shape:
            raise ValueError(f"pred/truth shape mismatch: {pred.shape} vs {truth.shape}")
        if pred.shape[0] != len(conditions):
            raise ValueError("conditions length does not match prediction rows")

        shared_genes, pred_idx, adata_idx = self._shared_indexes()
        shared_keys = {gene_key(g) for g in shared_genes}
        ctrl_mean = self._control_mean(adata_idx)
        pred_shared = pred[:, pred_idx]
        truth_shared = truth[:, pred_idx]

        condition_rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        for condition in sorted(np.unique(conditions)):
            if condition == self.options.control_label:
                continue
            mask = conditions == condition
            pred_mean = pred_shared[mask].mean(axis=0)
            truth_mean = truth_shared[mask].mean(axis=0)
            effect = np.abs(truth_mean - ctrl_mean)
            k = min(int(self.options.top_k), len(effect))
            de_local = np.argsort(effect)[-k:][::-1]
            row = {
                "condition": condition,
                "n_cells": int(mask.sum()),
                "n_shared_genes": int(len(shared_genes)),
                "de_k": int(k),
                "shared_mse": float(mean_squared_error(truth_mean, pred_mean)),
                "shared_pearson": _safe_pearson(pred_mean, truth_mean),
                "shared_delta_pearson": _safe_pearson(pred_mean - ctrl_mean, truth_mean - ctrl_mean),
                "shared_de_mse": float(mean_squared_error(truth_mean[de_local], pred_mean[de_local])),
                "shared_de_pearson": _safe_pearson(pred_mean[de_local], truth_mean[de_local]),
                "shared_de_delta_pearson": _safe_pearson(
                    pred_mean[de_local] - ctrl_mean[de_local],
                    truth_mean[de_local] - ctrl_mean[de_local],
                ),
                "shared_de_opposite_direction": float(np.mean(
                    np.sign(pred_mean[de_local] - ctrl_mean[de_local])
                    != np.sign(truth_mean[de_local] - ctrl_mean[de_local])
                )),
                "shared_de_genes": ",".join(shared_genes[i] for i in de_local),
            }
            coverage, overlap = self._official_de20_coverage(condition, shared_keys)
            row["official_de20_coverage"] = coverage
            row["official_de20_overlap"] = overlap
            condition_rows.append(row)
            coverage_rows.append({
                "condition": condition,
                "official_de20_coverage": coverage,
                "official_de20_overlap": overlap,
                "n_shared_genes": int(len(shared_genes)),
            })

        condition_df = pd.DataFrame(condition_rows)
        condition_df.to_csv(self.output_dir / "shared_vocab_conditions.tsv", sep="\t", index=False)
        pd.DataFrame(coverage_rows).to_csv(self.output_dir / "shared_vocab_gene_coverage.tsv", sep="\t", index=False)

        metric_cols = [
            "shared_mse",
            "shared_pearson",
            "shared_delta_pearson",
            "shared_de_mse",
            "shared_de_pearson",
            "shared_de_delta_pearson",
            "shared_de_opposite_direction",
            "official_de20_coverage",
        ]
        overall = {
            col: float(condition_df[col].dropna().mean())
            for col in metric_cols
            if col in condition_df
        }
        subgroup_summary: dict[str, Any] = {}
        for subgroup_name, subgroup_conditions in self.subgroup.get("test_subgroup", {}).items():
            subset = condition_df[condition_df["condition"].isin(list(subgroup_conditions))]
            subgroup_summary[subgroup_name] = {"n_conditions": int(len(subset))}
            for col in metric_cols:
                if col in subset:
                    subgroup_summary[subgroup_name][col] = float(subset[col].dropna().mean())

        summary = {
            "n_prediction_rows": int(pred.shape[0]),
            "n_gears_genes": int(len(self.gears_gene_names)),
            "n_ours_genes_raw": int(len(self.ours_gene_names)),
            "n_shared_genes": int(len(shared_genes)),
            "top_k": int(self.options.top_k),
            "overall": overall,
            "subgroup": subgroup_summary,
            "outputs": {
                "conditions": str(self.output_dir / "shared_vocab_conditions.tsv"),
                "coverage": str(self.output_dir / "shared_vocab_gene_coverage.tsv"),
            },
        }
        write_json(self.output_dir / "shared_vocab_summary.json", summary)
        with (self.output_dir / "shared_vocab_genes.txt").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(shared_genes) + "\n")
        return summary


def run_shared_vocab_eval(options: SharedVocabularyEvalOptions) -> dict[str, Any]:
    return SharedVocabularyDEEvaluator(options).evaluate()
