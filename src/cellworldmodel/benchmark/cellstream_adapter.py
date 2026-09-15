"""Adapters for local CellStream benchmark datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cellworldmodel.benchmark.branchsbm_adapter import TimePointAdapter
from cellworldmodel.training.split_policy import SplitIndices, build_timepoint_splits


ROOT = Path(__file__).parent.parent.parent.parent
CELLSTREAM_ROOT = ROOT / "baseline" / "CellStream"


@dataclass
class CellStreamDataset:
    name: str
    values: np.ndarray
    labels: np.ndarray
    coords_by_t: dict[float, np.ndarray]
    splits_by_t: dict[float, SplitIndices]
    real_v: np.ndarray | None = None
    real_g: np.ndarray | None = None
    raw_values: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return int(self.values.shape[1])

    @property
    def timepoints(self) -> list[float]:
        return [float(t) for t in sorted(self.coords_by_t)]

    def get_intermediate(self, t: float) -> torch.Tensor:
        return torch.from_numpy(self.coords_by_t[float(t)])


def _cellstream_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    m = values.min(axis=0, keepdims=True)
    M = values.max(axis=0, keepdims=True)
    span = M - m
    span[span == 0] = 1.0
    return (0.05 + 0.9 * (values - m) / span).astype(np.float32)


def _build_dataset(name: str, values: np.ndarray, labels: np.ndarray,
                   *, seed: int, normalize: bool,
                   real_v: np.ndarray | None = None,
                   real_g: np.ndarray | None = None) -> CellStreamDataset:
    raw = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    values = _cellstream_minmax(raw) if normalize else raw.astype(np.float32)
    coords_by_t = {
        float(t): values[np.isclose(labels, float(t))].copy()
        for t in sorted(np.unique(labels.astype(float)))
    }
    splits_by_t = build_timepoint_splits(coords_by_t, seed=seed, ratios=(0.7, 0.1, 0.2))
    return CellStreamDataset(
        name=name,
        values=values,
        labels=labels,
        coords_by_t=coords_by_t,
        splits_by_t=splits_by_t,
        real_v=None if real_v is None else np.asarray(real_v, dtype=np.float32),
        real_g=None if real_g is None else np.asarray(real_g, dtype=np.float32),
        raw_values=raw,
    )


def load_cellstream_dataset(name: str, *, seed: int = 42) -> CellStreamDataset:
    """Load one of the local CellStream datasets.

    For SimData, we mirror the notebook normalization exactly: global min-max on
    coordinates and velocity rescaling by the same scalar range.
    """
    key = name.lower()
    if key in {"sim", "simdata", "simdata2d_example"}:
        points, labels, real_v, real_g = torch.load(
            CELLSTREAM_ROOT / "data" / "Sim" / "SimData2D_example.pth",
            map_location="cpu",
            weights_only=False,
        )
        points_np = points.numpy().astype(np.float32)
        scale = float(points_np.max() - points_np.min())
        values = ((points_np - points_np.min()) / max(scale, 1e-6)).astype(np.float32)
        real_v_np = (real_v.numpy().astype(np.float32) / max(scale, 1e-6))
        return _build_dataset(
            "SimData2D_example",
            values,
            labels.numpy().astype(np.float32),
            seed=seed,
            normalize=False,
            real_v=real_v_np,
            real_g=real_g.numpy().astype(np.float32),
        )
    if key == "emt":
        df = pd.read_csv(CELLSTREAM_ROOT / "data" / "EMT" / "emt_normalized.csv")
        return _build_dataset("EMT", df.iloc[:, 1:].to_numpy(), df.iloc[:, 0].to_numpy(), seed=seed, normalize=True)
    if key == "ipsc":
        df = pd.read_csv(CELLSTREAM_ROOT / "data" / "iPSC" / "ipsc.csv")
        return _build_dataset("iPSC", df.iloc[:, 1:].to_numpy(), df.iloc[:, 0].to_numpy(), seed=seed, normalize=True)
    if key == "mosta":
        df = pd.read_csv(CELLSTREAM_ROOT / "data" / "MOSTA" / "mosta.csv")
        return _build_dataset("MOSTA", df.iloc[:, 1:].to_numpy(), df.iloc[:, 0].to_numpy(), seed=seed, normalize=True)
    raise ValueError(f"Unknown CellStream dataset={name!r}")


class CellStreamTimepointAdapter:
    """Minimal TimePointAdapter-compatible wrapper for CWM training."""

    def __init__(self, dataset: CellStreamDataset):
        self.dataset = dataset
        self.coords_by_t = dataset.coords_by_t
        self.splits_by_t = dataset.splits_by_t
        self.timepoints = dataset.timepoints
        self.dim = dataset.dim
        self.seed = 42

    def get_intermediate(self, t: float) -> torch.Tensor:
        return self.dataset.get_intermediate(t)


class CellStreamSimGrowthAdapter(TimePointAdapter):
    """CellStream SimData adapter with validated counts and per-cell GT growth."""

    _meta_name = "cellstream_sim_growth"

    def __init__(self, split_ratio: float = 0.8, seed: int = 42):
        dataset = load_cellstream_dataset("sim", seed=seed)
        if dataset.real_g is None:
            raise ValueError("CellStream SimData is missing ground-truth growth")

        self.dataset = dataset
        self.coords_by_t = dataset.coords_by_t
        self.timepoints = dataset.timepoints
        self.dim = dataset.dim
        self.population_mass_by_t = {
            float(t): float(len(coords))
            for t, coords in self.coords_by_t.items()
        }
        self.growth_truth_by_t = {
            float(t): dataset.real_g[np.isclose(dataset.labels, float(t))].copy()
            for t in self.timepoints
        }
        self._init_split(split_ratio, seed)
