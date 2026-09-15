"""Adapters over paper-benchmark exported foundation scVI representations."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cellworldmodel.benchmark.branchsbm_adapter import TimePointAdapter
from cellworldmodel.training.split_policy import SplitIndices
from cellworldmodel.provenance.writer import fingerprint_file, ProvenanceUnavailable


ROOT = Path(__file__).parent.parent.parent.parent

REPRESENTATION_ROOT_ENV = "CWM_PAPER_BENCH_REPRESENTATION_ROOT"


def read_encoder_provenance(root: Path) -> dict:
    """Which Stage-1 encoder produced this representation directory.

    Two representation trees of the same benchmark can differ only in the
    encoder that wrote them, and mixing them silently makes arms incomparable
    (2026-09-02 audit). Exports written by the current exporters record the
    encoder under ``split_manifest.json``; older ones record nothing, and we say
    so explicitly rather than guessing.
    """
    manifest_path = root / "split_manifest.json"
    if not manifest_path.exists():
        return {"encoder": None, "reason": f"no split_manifest.json in {root}"}
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"encoder": None, "reason": f"unreadable split_manifest.json: {exc}"}
    encoder = manifest.get("encoder")
    if not isinstance(encoder, dict):
        if manifest.get("representation_is_native_space"):
            # A benchmark shipping its own coordinates has no Stage-1 encoder,
            # so there is nothing to record and nothing to compare.
            return {"encoder": None, "reason": "native coordinates, no Stage-1 encoder"}
        return {"encoder": None, "reason": "split_manifest.json carries no encoder block"}
    return {"encoder": encoder, "reason": None}


class PaperBenchScVI128Adapter(TimePointAdapter):
    """Timepoint adapter over an exported shared-representation directory.

    The representation root must be given explicitly through
    ``CWM_PAPER_BENCH_REPRESENTATION_ROOT``. There is deliberately no default:
    the repository holds several exports of the same benchmark that differ only
    in the Stage-1 encoder, and a default silently picked one of them for a
    whole batch of runs while the baselines used another (2026-09-02 audit).
    """

    def __init__(self, dataset: str, split_ratio: float = 0.8, seed: int = 42):
        self.dataset = dataset
        root_env = os.environ.get(REPRESENTATION_ROOT_ENV)
        if not root_env:
            raise RuntimeError(
                f"{REPRESENTATION_ROOT_ENV} is unset. Point it at the representation "
                "tree that matches the encoder your checkpoints were pretrained "
                "with; this adapter no longer falls back to a default path."
            )
        representation_root = Path(root_env)
        root = representation_root / dataset
        reps_path = root / "representations.npz"
        meta_path = root / "metadata.tsv"
        if not reps_path.exists():
            raise FileNotFoundError(f"paper-bench representations not found: {reps_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"paper-bench metadata not found: {meta_path}")
        self.representation_root = representation_root
        self.representation_dir = root
        manifest_path = root / "split_manifest.json"
        if not manifest_path.is_file():
            raise ProvenanceUnavailable(f"{root}: required representation manifest is missing")
        self.input_fingerprints = {
            "representations": fingerprint_file(reps_path),
            "metadata": fingerprint_file(meta_path),
            "representation_manifest": fingerprint_file(manifest_path),
        }
        provenance = read_encoder_provenance(root)
        self.encoder_provenance = provenance["encoder"]
        if self.encoder_provenance is None:
            if not json.loads(manifest_path.read_text()).get("representation_is_native_space"):
                raise ProvenanceUnavailable(f"{root}: {provenance['reason']}")
        else:
            print(f"[paper-bench] {root}: encoder {self.encoder_provenance}")
        reps = np.load(reps_path)
        meta = pd.read_csv(meta_path, sep="\t")
        z = reps["scvi128"].astype(np.float32)
        if len(z) != len(meta):
            raise ValueError(f"representation/meta length mismatch: {len(z)} vs {len(meta)}")

        self.meta = meta
        self.standardize_metrics = os.environ.get(
            "CWM_PAPER_BENCH_STANDARDIZE_METRICS",
            "0",
        ).lower() in {"1", "true", "yes"}
        train_mask = meta["split"].astype(str).eq("train").to_numpy()
        self.metric_mean = z[train_mask].mean(axis=0, keepdims=True).astype(np.float32)
        self.metric_std = np.maximum(
            z[train_mask].std(axis=0, keepdims=True),
            1e-6,
        ).astype(np.float32)
        self.coords_by_t = {}
        self.splits_by_t: dict[float, SplitIndices] = {}
        for t in sorted(meta["time"].astype(float).unique()):
            mask = np.isclose(meta["time"].astype(float).to_numpy(), float(t))
            self.coords_by_t[float(t)] = z[mask]
            local_split = meta.loc[mask, "split"].astype(str).to_numpy()
            self.splits_by_t[float(t)] = SplitIndices(
                train=np.where(local_split == "train")[0].astype(np.int64),
                val=np.where(local_split == "val")[0].astype(np.int64),
                test=np.where(local_split == "test")[0].astype(np.int64),
            )
        self.dim = int(z.shape[1])
        self.timepoints = [float(t) for t in sorted(self.coords_by_t)]
        self._meta_name = f"paper_{dataset}_scvi128"

        self._final_celltypes = None
        if "cell_type" in meta.columns:
            final_t = self.timepoints[-1]
            mask_final = np.isclose(meta["time"].astype(float).to_numpy(), final_t)
            self._final_celltypes = meta.loc[mask_final, "cell_type"].astype(str).to_numpy()

        # Preserve the exported paper-bench split instead of creating a new one.
        source_split = self.splits_by_t[self.timepoints[0]]
        self.train_src_idx = source_split.train
        self.test_src_idx = source_split.test
        self.seed = seed

    def transform_for_metrics(self, values):
        if not self.standardize_metrics:
            return values
        if torch.is_tensor(values):
            mean = torch.from_numpy(self.metric_mean).to(
                device=values.device,
                dtype=values.dtype,
            )
            std = torch.from_numpy(self.metric_std).to(
                device=values.device,
                dtype=values.dtype,
            )
            return (values - mean) / std
        return (np.asarray(values) - self.metric_mean) / self.metric_std

    def get_intermediate(self, t: float) -> torch.Tensor:
        t = float(t)
        if t not in self.coords_by_t:
            raise ValueError(f"timepoint {t} not in {self.timepoints}")
        return torch.from_numpy(self.coords_by_t[t])

    def get_target_cluster_labels(self, n_clusters: int = 11, seed: int = 42) -> np.ndarray:
        if self._final_celltypes is None:
            from sklearn.cluster import KMeans
            target = self.coords_by_t[self.timepoints[-1]]
            return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(target).labels_.astype(np.int64)
        unique = sorted(set(self._final_celltypes))
        lut = {name: i for i, name in enumerate(unique)}
        return np.asarray([lut[name] for name in self._final_celltypes], dtype=np.int64)
