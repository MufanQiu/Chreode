"""Veres 2019 Stage 5 scVI-latent adapter (8 timepoints, CellWeek 0..7).

Input: output/scvi/v1_veres/{latent_z.npy, latent_metadata.parquet}
       (built by scripts/veres/build_veres_ortholog_h5ad.py + train_weinreb_scvi.py)

Source: t=0 (D0, hES progenitors), target: t=7 (D20+, β-like cells).
Intermediate timepoints t=1..6 accessible via get_intermediate(t).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from cellworldmodel.benchmark.branchsbm_adapter import TimePointAdapter


DEFAULT_SCVI_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "output" / "scvi" / "v1_veres"
)


class VeresScVIAdapter(TimePointAdapter):
    _meta_name = "veres_scvi"

    def __init__(
        self,
        scvi_dir: Optional[Path | str] = None,
        split_ratio: float = 0.8,
        seed: int = 42,
    ):
        root = Path(scvi_dir) if scvi_dir else DEFAULT_SCVI_DIR
        latent = np.load(root / "latent_z.npy").astype(np.float32)
        meta = pd.read_parquet(root / "latent_metadata.parquet")
        assert len(latent) == len(meta)

        self._all_celltypes = meta["cell_type"].values if "cell_type" in meta.columns else None

        self.coords_by_t = {}
        for t in sorted(meta["Time_point"].unique()):
            mask = (meta["Time_point"] == t).values
            self.coords_by_t[float(t)] = latent[mask]
        self.dim = latent.shape[1]
        self.timepoints = [float(t) for t in sorted(self.coords_by_t.keys())]

        # final timepoint cell types
        mask_last = (meta["Time_point"] == self.timepoints[-1]).values
        self._d_final_celltypes = meta["cell_type"].values[mask_last] if self._all_celltypes is not None else None

        self._init_split(split_ratio, seed)

    def get_intermediate(self, t) -> torch.Tensor:
        t = float(t)
        assert t in self.coords_by_t, f"timepoint {t} not in {self.timepoints}"
        return torch.from_numpy(self.coords_by_t[t])

    def get_target_cluster_labels(self, n_clusters: int = 11, seed: int = 42) -> np.ndarray:
        """Return cell_type label indices for final timepoint cells."""
        if self._d_final_celltypes is None:
            from sklearn.cluster import KMeans
            tgt = self.coords_by_t[self.timepoints[-1]]
            return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(tgt).labels_.astype(np.int64)
        unique = sorted(set(self._d_final_celltypes))
        type_to_idx = {t: i for i, t in enumerate(unique)}
        return np.array([type_to_idx[ct] for ct in self._d_final_celltypes], dtype=np.int64)


if __name__ == "__main__":
    a = VeresScVIAdapter()
    print(f"dim={a.dim}, timepoints={a.timepoints}")
    for t, arr in a.coords_by_t.items():
        print(f"  t={t}: {arr.shape}")
    b = a.get_transition()
    print(f"source {b.source.shape}, target {b.target.shape}, delta {b.delta}")
    labels = a.get_target_cluster_labels()
    print(f"labels shape {labels.shape}, unique {np.unique(labels)}")
