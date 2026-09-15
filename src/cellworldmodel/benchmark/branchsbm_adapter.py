"""Adapter to load BranchSBM datasets as source/target population tensors.

Provides a lightweight interface (no PyTorch Lightning dependency) that reads
the BranchSBM CSV data files and returns source/target populations for our
population-level training loop.

Supported datasets (must have CSV in 3rdparty/BranchSBM/data/):
  - mouse_hematopoiesis.csv     (2D, 3 timepoints, 2 terminal branches)
  - Veres_alltime.csv           (30D PCA, 8 timepoints)
  - pca_and_leiden_labels.csv   (Clonidine, 50D+ PCA, 2 branches)
  - Trametinib_5.0uM_pca_and_leidenumap_labels.csv (50D PCA, 3 branches)

For perturbation datasets (Clonidine, Trametinib), source = DMSO control cells,
target = drug-perturbed cells. For developmental datasets (Mouse, Veres),
source = t=0, target = final timepoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch


DEFAULT_DATA_ROOT = Path(__file__).parent.parent.parent.parent / "3rdparty" / "BranchSBM" / "data"


@dataclass
class PopulationBatch:
    """A source + target population pair for one transition."""

    source: torch.Tensor  # (N_s, dim)
    target: torch.Tensor  # (N_t, dim)
    delta: float  # time interval (Δ)
    source_time: float  # absolute source time (for optional t conditioning)
    target_time: float
    meta: dict  # dataset-specific metadata (branch labels, etc.)


class TimePointAdapter:
    """Base class for datasets with per-timepoint coordinates + train/test source split.

    Subclass responsibilities:
      - populate `self.coords_by_t: dict[timepoint_key, np.ndarray]` in __init__
      - populate `self.dim: int`, `self.timepoints: list`
      - then call `self._init_split(split_ratio, seed)` to set train/test source idx

    Provides: get_transition, sample_source_batch, sample_target_batch.
    Timepoint keys can be int or float; subclass controls the type.
    """

    coords_by_t: dict
    dim: int
    timepoints: list
    train_src_idx: np.ndarray
    test_src_idx: np.ndarray
    seed: int
    _meta_name: str = "timepoint_adapter"

    def _init_split(self, split_ratio: float, seed: int) -> None:
        n0 = self.coords_by_t[self.timepoints[0]].shape[0]
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n0)
        n_train = int(n0 * split_ratio)
        self.train_src_idx = perm[:n_train]
        self.test_src_idx = perm[n_train:]
        self.seed = seed

    def get_transition(
        self,
        split: Literal["train", "test"] = "train",
        source_t=None,
        target_t=None,
    ) -> PopulationBatch:
        """Get source/target populations for a given transition.

        If source_t/target_t not specified, uses earliest → latest timepoints.
        """
        if source_t is None:
            source_t = self.timepoints[0]
        if target_t is None:
            target_t = self.timepoints[-1]

        src_all = self.coords_by_t[source_t]
        tgt_all = self.coords_by_t[target_t]

        if source_t == self.timepoints[0]:
            idx = self.train_src_idx if split == "train" else self.test_src_idx
            src = src_all[idx]
        else:
            src = src_all

        return PopulationBatch(
            source=torch.from_numpy(src),
            target=torch.from_numpy(tgt_all),
            delta=float(target_t - source_t),
            source_time=float(source_t),
            target_time=float(target_t),
            meta={"dataset": self._meta_name},
        )

    def sample_source_batch(
        self,
        batch_size: int,
        split: Literal["train", "test"] = "train",
        source_t=None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()
        if source_t is None:
            source_t = self.timepoints[0]
        src_all = self.coords_by_t[source_t]
        if source_t == self.timepoints[0]:
            avail = self.train_src_idx if split == "train" else self.test_src_idx
        else:
            avail = np.arange(src_all.shape[0])
        idx = rng.choice(avail, size=batch_size, replace=batch_size > len(avail))
        return torch.from_numpy(src_all[idx])

    def sample_target_batch(
        self,
        batch_size: int,
        target_t=None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()
        if target_t is None:
            target_t = self.timepoints[-1]
        tgt_all = self.coords_by_t[target_t]
        idx = rng.choice(tgt_all.shape[0], size=batch_size, replace=batch_size > tgt_all.shape[0])
        return torch.from_numpy(tgt_all[idx])


class MouseHematopoiesisAdapter(TimePointAdapter):
    """Mouse hematopoiesis 2D branching dataset (1429 source cells, 3 timepoints).

    Loads mouse_hematopoiesis.csv with columns [samples, x1, x2].
    Timepoints: 0 (progenitors), 1 (intermediate), 2 (2 terminal branches).

    Source = t=0, target = t=2 (all cells, no branch filtering).
    Optionally provides KMeans-based branch labels for eval.
    """

    _meta_name = "mouse_hematopoiesis"

    def __init__(
        self,
        data_path: Optional[Path | str] = None,
        split_ratio: float = 0.8,
        seed: int = 42,
    ):
        path = Path(data_path) if data_path else DEFAULT_DATA_ROOT / "mouse_hematopoiesis.csv"
        df = pd.read_csv(path)
        self.df = df

        self.coords_by_t = {
            t: df[df["samples"] == t][["x1", "x2"]].values.astype(np.float32)
            for t in sorted(df["samples"].unique())
        }
        self.dim = 2
        self.timepoints = sorted(self.coords_by_t.keys())
        self._init_split(split_ratio, seed)


class PerturbationAdapter:
    """Base adapter for perturbation datasets (Clonidine, Trametinib).

    These CSVs have columns: BARCODE_SUB_LIB_ID, PC1...PCn, leiden_<drug>_<dose>uM

    Source = DMSO control (leiden_DMSO_TF_0.0uM populated), target = drug-perturbed.
    """

    def __init__(
        self,
        data_path: Path | str,
        pcs: int,
        drug_name: str,  # e.g., 'Clonidine (hydrochloride)_5.0uM'
        split_ratio: float = 0.8,
        seed: int = 42,
    ):
        df = pd.read_csv(data_path).iloc[:, 1:]  # drop BARCODE column
        df = df.replace("", pd.NA)

        self.pc_cols = [f"PC{i + 1}" for i in range(pcs)]
        self.dim = pcs

        dmso_col = "leiden_DMSO_TF_0.0uM"
        pert_col = f"leiden_{drug_name}"

        self.control_df = df[df[dmso_col].notna()].reset_index(drop=True)
        self.perturbed_df = df[df[pert_col].notna()].reset_index(drop=True)
        self.dmso_col = dmso_col
        self.pert_col = pert_col

        # Split control cells (source) into train/test
        n_ctrl = len(self.control_df)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_ctrl)
        n_train = int(n_ctrl * split_ratio)
        self.train_src_idx = perm[:n_train]
        self.test_src_idx = perm[n_train:]
        self.seed = seed

    def _get_coords(self, df: pd.DataFrame, idx: Optional[np.ndarray] = None) -> np.ndarray:
        if idx is not None:
            df = df.iloc[idx]
        return df[self.pc_cols].values.astype(np.float32)

    def get_transition(self, split: Literal["train", "test"] = "train") -> PopulationBatch:
        idx = self.train_src_idx if split == "train" else self.test_src_idx
        src = self._get_coords(self.control_df, idx)
        tgt = self._get_coords(self.perturbed_df)

        return PopulationBatch(
            source=torch.from_numpy(src),
            target=torch.from_numpy(tgt),
            delta=1.0,  # control → perturbed is a single "step"
            source_time=0.0,
            target_time=1.0,
            meta={
                "dataset": "perturbation",
                "drug": self.pert_col,
                "pcs": self.dim,
            },
        )

    def sample_source_batch(
        self, batch_size: int, split: Literal["train", "test"] = "train",
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()
        avail = self.train_src_idx if split == "train" else self.test_src_idx
        sel = rng.choice(avail, size=batch_size, replace=batch_size > len(avail))
        return torch.from_numpy(self._get_coords(self.control_df, sel))

    def sample_target_batch(
        self, batch_size: int, rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()
        n = len(self.perturbed_df)
        sel = rng.choice(n, size=batch_size, replace=batch_size > n)
        return torch.from_numpy(self._get_coords(self.perturbed_df, sel))

    def get_target_cluster_labels(self) -> np.ndarray:
        """Return leiden cluster labels for target (perturbed) cells.

        Useful for per-branch metrics (Hungarian matching).
        """
        return self.perturbed_df[self.pert_col].astype(float).astype(int).values


class ClonidineAdapter(PerturbationAdapter):
    """Clonidine perturbation, 50D/100D/150D PCA, 2 major branches."""

    def __init__(self, pcs: int = 50, data_path: Optional[Path | str] = None, **kwargs):
        path = Path(data_path) if data_path else DEFAULT_DATA_ROOT / "pca_and_leiden_labels.csv"
        super().__init__(
            data_path=path,
            pcs=pcs,
            drug_name="Clonidine (hydrochloride)_5.0uM",
            **kwargs,
        )


class TrametinibAdapter(PerturbationAdapter):
    """Trametinib perturbation, 50D PCA, 3 branches."""

    def __init__(self, pcs: int = 50, data_path: Optional[Path | str] = None, **kwargs):
        path = Path(data_path) if data_path else DEFAULT_DATA_ROOT / "Trametinib_5.0uM_pca_and_leidenumap_labels.csv"
        super().__init__(
            data_path=path,
            pcs=pcs,
            drug_name="Trametinib_5.0uM",
            **kwargs,
        )


class VeresAdapter(TimePointAdapter):
    """Veres pancreatic β-cell differentiation (30D PCA, 8 timepoints, ~51K cells).

    Loads Veres_alltime.csv with columns [samples, x1..x30]. Timepoints 0..7,
    with t=7 as the final differentiated state (11 Leiden clusters per paper).

    Source = t=0 (pluripotent stem cells), target = t=7 (β-like cells).
    Intermediate timepoints (t=1..6) are accessible via get_intermediate(t).

    Branch labels at t=7 are computed by KMeans (default 11 clusters, matching
    paper's post-Leiden merge). Optional leiden via scanpy if available.
    """

    _meta_name = "veres"

    def __init__(
        self,
        data_path: Optional[Path | str] = None,
        split_ratio: float = 0.8,
        seed: int = 42,
        dim: int = 30,
    ):
        path = Path(data_path) if data_path else DEFAULT_DATA_ROOT / "Veres_alltime.csv"
        df = pd.read_csv(path)
        self.df = df

        self.pc_cols = [f"x{i + 1}" for i in range(dim)]
        self.dim = dim

        self.coords_by_t = {
            int(t): df[df["samples"] == t][self.pc_cols].values.astype(np.float32)
            for t in sorted(df["samples"].unique())
        }
        self.timepoints = sorted(self.coords_by_t.keys())
        self._init_split(split_ratio, seed)
        # Lazy cache for target cluster labels
        self._target_labels_cache: Optional[np.ndarray] = None

    def get_intermediate(self, t: int) -> torch.Tensor:
        """Get cells at an intermediate timepoint (for t=1..6 eval)."""
        assert t in self.coords_by_t, f"timepoint {t} not in {self.timepoints}"
        return torch.from_numpy(self.coords_by_t[t])

    def get_target_cluster_labels(
        self, n_clusters: int = 11, target_t: Optional[int] = None, seed: int = 42,
    ) -> np.ndarray:
        """Post-hoc KMeans clustering on target cells to produce branch labels.

        Paper uses Leiden clustering → merge small clusters → 11 branches.
        We default to KMeans(n=11) for reproducibility without graph libraries.
        Returns (N_target,) integer labels.
        """
        if target_t is None:
            target_t = self.timepoints[-1]
        if self._target_labels_cache is not None:
            return self._target_labels_cache
        from sklearn.cluster import KMeans
        tgt = self.coords_by_t[target_t]
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(tgt)
        self._target_labels_cache = km.labels_.astype(np.int64)
        return self._target_labels_cache


def _weinberger_guide_to_condition(guide_identity: str) -> str:
    """Convert Norman Weinberger-labeled guide_identity to scDFM-style condition.

    Weinberger format: "GENE1_GENE2__GENE1_GENE2" (duplicated for CRISPRi double guides).
    NegCtrl* = non-targeting control guide.

    Returns:
      - "ctrl" if both guides are NegCtrl*
      - "GENE" if one guide is NegCtrl* (single KO)
      - "GENE1+GENE2" if both are non-NegCtrl (double KO)
    """
    # Strip the `__A_B` duplicate suffix
    s = str(guide_identity).split("__")[0]
    parts = s.split("_")
    # parts may be like ["NegCtrl10", "NegCtrl0"] or ["CEBPE", "RUNX1T1"]
    # or ["NegCtrl0", "ETS2"] etc. Length is typically 2.
    if len(parts) != 2:
        return str(guide_identity)  # unexpected format, keep raw
    g1, g2 = parts
    g1_ctrl = g1.startswith("NegCtrl")
    g2_ctrl = g2.startswith("NegCtrl")
    if g1_ctrl and g2_ctrl:
        return "ctrl"
    if g1_ctrl:
        return g2
    if g2_ctrl:
        return g1
    return f"{g1}+{g2}"


class NormanAdapter:
    """Norman 2019 Perturb-seq CRISPR perturbation dataset (scDFM benchmark D5/D6).

    Loads norman_2019_adata.h5ad. Two supported versions:
      (A) Weinberger labeled version (Figshare): obs['guide_identity'] like
          "GENE1_GENE2__GENE1_GENE2", 2000 HVGs pre-selected.
      (B) scDFM Google Drive preprocessed: obs['condition'] like 'ctrl'/'GENE'/'GENE1+GENE2',
          5000 HVGs + splits.

    The adapter normalizes both to scDFM-style conditions and provides per-condition
    control → perturbed transitions.

    Split modes:
      - 'additive': all singles + 70% doubles in train, 30% doubles held out for test
      - 'holdout': 30% singles held out + all doubles involving them
    """

    def __init__(
        self,
        data_path: Optional[Path | str] = None,
        n_top_genes: int = 5000,
        split_method: Literal["additive", "holdout"] = "additive",
        split_seed: int = 42,
        control_label: str = "ctrl",
        condition_col: Optional[str] = None,  # auto-detect if None
        use_hvg_filter: bool = True,
        precomputed_pca_dim: Optional[int] = None,
    ):
        try:
            import scanpy as sc
            import anndata as ad
        except ImportError as e:
            raise ImportError("scanpy + anndata required for NormanAdapter") from e

        path = Path(data_path) if data_path else (
            Path(__file__).parent.parent.parent.parent
            / "3rdparty" / "scDFM" / "data" / "norman_2019_adata.h5ad"
        )
        if not path.exists():
            raise FileNotFoundError(f"Norman h5ad not found: {path}. Download from "
                                    f"https://ndownloader.figshare.com/files/43390776")
        print(f"[NormanAdapter] loading {path} ...")
        adata = ad.read_h5ad(str(path))
        print(f"[NormanAdapter] raw shape: {adata.shape}")

        # Auto-detect version: scDFM uses 'condition', Weinberger uses 'guide_identity'
        if condition_col is None:
            if "condition" in adata.obs.columns:
                condition_col = "condition"
                print(f"[NormanAdapter] detected scDFM version (obs.condition exists)")
            elif "guide_identity" in adata.obs.columns:
                condition_col = "_condition"  # derived column
                adata.obs[condition_col] = adata.obs["guide_identity"].apply(
                    _weinberger_guide_to_condition
                )
                print(f"[NormanAdapter] detected Weinberger version (derived condition from guide_identity)")
            else:
                raise ValueError(
                    f"Neither 'condition' nor 'guide_identity' in obs. Found: "
                    f"{list(adata.obs.columns)}"
                )
        elif condition_col not in adata.obs.columns:
            raise ValueError(f"obs['{condition_col}'] not found")

        # HVG filter (skip if data already HVG-selected, e.g. Weinberger = 2000 HVGs)
        if use_hvg_filter and adata.shape[1] > n_top_genes * 1.2:
            if "log1p" not in adata.uns:
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            adata = adata[:, adata.var["highly_variable"]].copy()
            print(f"[NormanAdapter] after HVG({n_top_genes}): {adata.shape}")
        else:
            print(f"[NormanAdapter] keeping all {adata.shape[1]} features (no extra HVG)")
            # Still ensure log-normalization
            if adata.X.max() > 50:  # likely raw counts
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                print(f"[NormanAdapter] applied normalize_total + log1p (raw counts detected)")

        # Store conditions
        self.adata = adata
        self.condition_col = condition_col
        self.control_label = control_label
        self.dim = adata.shape[1]

        cond_values = adata.obs[condition_col].astype(str)
        all_conds = cond_values.unique().tolist()

        # Identify control vs perturbed
        self.control_mask = (cond_values == control_label).values
        self.perturbed_conds = [c for c in all_conds if c != control_label]
        print(f"[NormanAdapter] control cells: {self.control_mask.sum()}, "
              f"n perturbation conds: {len(self.perturbed_conds)}")

        # Singles vs doubles (for Norman: "GENE" or "GENE1+GENE2")
        self.single_conds = [c for c in self.perturbed_conds if "+" not in c]
        self.double_conds = [c for c in self.perturbed_conds if "+" in c]

        # Compute splits
        rng = np.random.RandomState(split_seed)
        self.split_method = split_method
        if split_method == "additive":
            # 70% of doubles in train, 30% in test
            shuffled = rng.permutation(len(self.double_conds))
            n_test = int(len(self.double_conds) * 0.3)
            self.test_conds = [self.double_conds[i] for i in shuffled[:n_test]]
            self.train_conds = self.single_conds + [
                self.double_conds[i] for i in shuffled[n_test:]
            ]
        elif split_method == "holdout":
            # Remove 30% of singles + all doubles involving them
            shuffled = rng.permutation(len(self.single_conds))
            n_test_single = int(len(self.single_conds) * 0.3)
            held_singles = set(self.single_conds[i] for i in shuffled[:n_test_single])
            self.test_conds = list(held_singles) + [
                c for c in self.double_conds
                if any(g in held_singles for g in c.split("+"))
            ]
            self.train_conds = [c for c in self.single_conds if c not in held_singles] + [
                c for c in self.double_conds if c not in self.test_conds
            ]
        else:
            raise ValueError(f"Unknown split_method: {split_method}")

        print(f"[NormanAdapter] split={split_method}: train={len(self.train_conds)} "
              f"conds, test={len(self.test_conds)} conds")

        # Optional PCA projection (to make MMD/W2 tractable for high-D gene expression)
        self.pca_dim = precomputed_pca_dim
        self._pca_state = None
        if precomputed_pca_dim is not None:
            from sklearn.decomposition import PCA
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            self._pca_state = PCA(n_components=precomputed_pca_dim, random_state=split_seed).fit(X)
            self._X_pca = self._pca_state.transform(X).astype(np.float32)
            self.dim = precomputed_pca_dim
            print(f"[NormanAdapter] fitted PCA({precomputed_pca_dim}), EVR={self._pca_state.explained_variance_ratio_.sum():.3f}")
        else:
            # Cache X as numpy (may be sparse)
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            self._X_pca = np.asarray(X, dtype=np.float32)

        self.seed = split_seed

    def _cells_for_condition(self, cond: str) -> np.ndarray:
        """Return indices of cells under a given condition."""
        mask = (self.adata.obs[self.condition_col].astype(str) == cond).values
        return np.where(mask)[0]

    def get_control_cells(self) -> torch.Tensor:
        idx = np.where(self.control_mask)[0]
        return torch.from_numpy(self._X_pca[idx])

    def get_transition(
        self,
        condition: Optional[str] = None,
        split: Literal["train", "test"] = "test",
    ) -> PopulationBatch:
        """Return control → perturbed batch for a specific condition.

        If condition is None, picks the first test condition.
        """
        if condition is None:
            conds = self.test_conds if split == "test" else self.train_conds
            if not conds:
                raise ValueError(f"No {split} conditions")
            condition = conds[0]

        src_idx = np.where(self.control_mask)[0]
        tgt_idx = self._cells_for_condition(condition)
        if len(tgt_idx) == 0:
            raise ValueError(f"No cells with condition={condition}")

        return PopulationBatch(
            source=torch.from_numpy(self._X_pca[src_idx]),
            target=torch.from_numpy(self._X_pca[tgt_idx]),
            delta=1.0,
            source_time=0.0,
            target_time=1.0,
            meta={"dataset": "norman", "condition": condition, "split": split},
        )

    def sample_source_batch(
        self, batch_size: int, split: Literal["train", "test"] = "train",
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        """Sample control cells (source)."""
        if rng is None:
            rng = np.random.default_rng()
        src_idx = np.where(self.control_mask)[0]
        sel = rng.choice(src_idx, size=batch_size, replace=batch_size > len(src_idx))
        return torch.from_numpy(self._X_pca[sel])

    def sample_target_batch(
        self, batch_size: int, condition: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        """Sample perturbed cells for a condition."""
        if rng is None:
            rng = np.random.default_rng()
        if condition is None:
            # Default: sample from training perturbation pool (union of all train conds)
            cond_mask = self.adata.obs[self.condition_col].astype(str).isin(self.train_conds).values
            tgt_idx = np.where(cond_mask)[0]
        else:
            tgt_idx = self._cells_for_condition(condition)
        sel = rng.choice(tgt_idx, size=batch_size, replace=batch_size > len(tgt_idx))
        return torch.from_numpy(self._X_pca[sel])

    def get_test_conditions(self) -> list[str]:
        """Return list of held-out perturbation conditions for evaluation."""
        return list(self.test_conds)


def get_adapter(dataset_name: str, **kwargs):
    """Factory function for adapters."""
    name = dataset_name.lower()
    if name == "mouse":
        return MouseHematopoiesisAdapter(**kwargs)
    elif name.startswith("clonidine"):
        pcs = 50
        if "100" in name:
            pcs = 100
        elif "150" in name:
            pcs = 150
        return ClonidineAdapter(pcs=pcs, **kwargs)
    elif name.startswith("trametinib"):
        return TrametinibAdapter(**kwargs)
    elif name == "veres":
        return VeresAdapter(**kwargs)
    elif name == "norman":
        return NormanAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
