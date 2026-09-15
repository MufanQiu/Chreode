"""Synthetic branching dynamics with analytic per-cell growth targets."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from cellworldmodel.benchmark.branchsbm_adapter import TimePointAdapter


ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "synthetic_growth" / "synthetic_growth.npz"
DEFAULT_STVCR_RECTANGLE_PATH = (
    ROOT / "3rdparty" / "stVCR" / "datasets" / "sim_data_rectangle"
    / "sim_data_rectangle.h5ad"
)


class SyntheticGrowthAdapter(TimePointAdapter):
    """Load synthetic snapshots with valid absolute population-mass semantics."""

    _meta_name = "synthetic_growth"

    def __init__(
        self,
        data_path: str | Path | None = None,
        split_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Synthetic growth data not found: {path}. "
                "Run scripts/spatial_temporal/make_synthetic_growth.py first."
            )
        payload = np.load(path)
        coords = payload["coords"].astype(np.float32)
        time = payload["time"].astype(np.float32)
        growth = payload["growth_true"].astype(np.float32)
        if len(coords) != len(time) or len(coords) != len(growth):
            raise ValueError("coords, time, and growth_true must have equal lengths")

        self.coords_by_t: dict[float, np.ndarray] = {}
        self.growth_truth_by_t: dict[float, np.ndarray] = {}
        self.population_mass_by_t: dict[float, float] = {}
        for t in sorted(np.unique(time)):
            mask = np.isclose(time, t)
            key = float(t)
            self.coords_by_t[key] = coords[mask]
            self.growth_truth_by_t[key] = growth[mask]
            self.population_mass_by_t[key] = float(mask.sum())

        self.dim = int(coords.shape[1])
        self.timepoints = sorted(self.coords_by_t)
        self._init_split(split_ratio, seed)


class StVCRRectangleGeneAdapter(TimePointAdapter):
    """Gene-only view of the official stVCR rectangle simulation."""

    _meta_name = "stvcr_rectangle_gene"

    def __init__(
        self,
        data_path: str | Path | None = None,
        split_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        path = Path(data_path) if data_path is not None else DEFAULT_STVCR_RECTANGLE_PATH
        if not path.exists():
            raise FileNotFoundError(f"stVCR rectangle data not found: {path}")
        adata = ad.read_h5ad(path)
        if "X_input" not in adata.obsm or "time" not in adata.obs:
            raise ValueError("stVCR rectangle data requires obsm['X_input'] and obs['time']")
        expression = np.asarray(adata.obsm["X_input"], dtype=np.float32)
        time = adata.obs["time"].to_numpy(dtype=np.float32)

        self.coords_by_t = {}
        self.population_mass_by_t = {}
        for t in sorted(np.unique(time)):
            mask = np.isclose(time, t)
            key = float(t)
            self.coords_by_t[key] = expression[mask]
            self.population_mass_by_t[key] = float(mask.sum())
        self.dim = int(expression.shape[1])
        self.timepoints = sorted(self.coords_by_t)
        self._init_split(split_ratio, seed)
