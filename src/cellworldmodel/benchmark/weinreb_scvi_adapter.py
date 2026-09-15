"""Weinreb 2020 adapter using scVI latent (from output/scvi/v1_weinreb/).

Replaces WeinrebHVGAdapter (which uses PRESCIENT's PCA-50). This version uses
scVI 64-dim latent, hoping to reduce the d2 -> d6 mean-shift magnitude from
12.68 units (PCA-50) down to 1-3 units (Gaussian-prior scVI latent), which
our one-step residual `ẑ = z + α(Δ)·R_θ` can actually cover.

Data:
  - Input: output/scvi/v1_weinreb/latent_z.npy (130887, 64)
  - Metadata: output/scvi/v1_weinreb/latent_metadata.parquet (Time_point, cell_type, Library)
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
    / "output" / "scvi" / "v1_weinreb"
)


class WeinrebScVIAdapter(TimePointAdapter):
    """Weinreb in scVI-64 latent space, d2/d4/d6 timepoints.

    Source: d2 latent (28,249 cells; 80% train / 20% test split by split_seed=42).
    Target: d6 latent (54,140 cells).
    Intermediate: d4 latent (48,498 cells) accessible via `get_intermediate(4)`.

    split_seed kept separate from model seed (GPT review fix).
    """

    _meta_name = "weinreb_scvi"

    def __init__(
        self,
        scvi_dir: Optional[Path | str] = None,
        split_ratio: float = 0.8,
        seed: int = 42,  # this is now SPLIT seed, not model seed
    ):
        root = Path(scvi_dir) if scvi_dir else DEFAULT_SCVI_DIR
        latent_path = root / "latent_z.npy"
        meta_path = root / "latent_metadata.parquet"
        if not latent_path.exists():
            raise FileNotFoundError(f"scVI latent not found: {latent_path}. Run scripts/scvi/train_weinreb_scvi.py first.")
        if not meta_path.exists():
            raise FileNotFoundError(f"scVI metadata not found: {meta_path}")

        z = np.load(latent_path).astype(np.float32)  # (N, 64)
        meta = pd.read_parquet(meta_path)
        assert len(z) == len(meta), f"latent shape {z.shape} does not match meta {len(meta)}"

        # Cell type labels (for per-branch eval)
        self._all_celltypes = meta["cell_type"].values

        # Build coords_by_t from Time_point column (values are float 2.0/4.0/6.0)
        self.coords_by_t = {}
        for t in sorted(meta["Time_point"].unique()):
            mask = (meta["Time_point"] == t).values
            self.coords_by_t[float(t)] = z[mask]
        self.dim = z.shape[1]
        self.timepoints = [float(t) for t in sorted(self.coords_by_t.keys())]

        # Save d6 cell type labels for get_target_cluster_labels
        mask_d6 = (meta["Time_point"] == self.timepoints[-1]).values
        self._d6_celltypes = self._all_celltypes[mask_d6]

        self._init_split(split_ratio, seed)

    def get_intermediate(self, t: float) -> torch.Tensor:
        """Return latent cells at intermediate timepoint `t`."""
        assert float(t) in self.coords_by_t, f"timepoint {t} not in {self.timepoints}"
        return torch.from_numpy(self.coords_by_t[float(t)])

    def get_target_cluster_labels(self, n_clusters: int = 11, seed: int = 42) -> np.ndarray:
        """Return integer branch labels for d6 cells via Weinreb's cell_type annotation."""
        unique = sorted(set(self._d6_celltypes))
        type_to_idx = {t: i for i, t in enumerate(unique)}
        return np.array([type_to_idx[ct] for ct in self._d6_celltypes], dtype=np.int64)

    def compute_shift_magnitude_diagnostic(self) -> dict:
        """Compute the d2 -> d6 mean shift norm in scVI latent space.

        Returns:
            dict with 'shift_norm', 'd2_cells', 'd6_cells', 'd2_d6_center_dist',
            'd4_center_dist', 'target_std_mean'.
            Key metric: shift_norm should ideally be 1-3 (within α(Δ)·R_θ's
            reachable range).
        """
        d2 = self.coords_by_t[2.0]
        d4 = self.coords_by_t[4.0]
        d6 = self.coords_by_t[6.0]
        shift_d2_d6 = d6.mean(0) - d2.mean(0)
        shift_d2_d4 = d4.mean(0) - d2.mean(0)
        return {
            "shift_d2_d6_norm": float(np.linalg.norm(shift_d2_d6)),
            "shift_d2_d4_norm": float(np.linalg.norm(shift_d2_d4)),
            "n_d2": int(d2.shape[0]),
            "n_d4": int(d4.shape[0]),
            "n_d6": int(d6.shape[0]),
            "d6_target_std_mean": float(d6.std(0).mean()),
            "d2_std_mean": float(d2.std(0).mean()),
        }


def _main_smoke():
    a = WeinrebScVIAdapter()
    print(f"dim={a.dim}, timepoints={a.timepoints}")
    for t, arr in a.coords_by_t.items():
        print(f"  t={t}: {arr.shape}")
    diag = a.compute_shift_magnitude_diagnostic()
    print(f"\nShift diagnostic: {diag}")
    batch_train = a.get_transition(split="train")
    batch_test = a.get_transition(split="test")
    print(f"train src: {batch_train.source.shape}, target: {batch_train.target.shape}, delta: {batch_train.delta}")
    print(f"test src: {batch_test.source.shape}")
    labels = a.get_target_cluster_labels()
    print(f"branch labels: {labels.shape}, unique: {np.unique(labels)}")


if __name__ == "__main__":
    _main_smoke()
