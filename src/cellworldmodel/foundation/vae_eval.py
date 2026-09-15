"""Reusable embedding QC for trained foundation VAE checkpoints."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from cellworldmodel.foundation.expression_dataset import FoundationExpressionDataset
from cellworldmodel.foundation.vae_registry import build_foundation_vae


@dataclass(frozen=True)
class EmbeddingEvalConfig:
    checkpoint: str
    catalog_dir: str
    output_dir: str
    split: str = "val"
    sample_size: int = 10000
    batch_size: int = 512
    seed: int = 0
    device: str = "cpu"
    save_embeddings: bool = True
    silhouette_max_samples: int = 2000
    knn_k: int = 15
    make_plots: bool = True
    leiden_resolution: float = 1.0


@dataclass(frozen=True)
class LoadedVAE:
    model: torch.nn.Module
    leaf_to_id: dict[str, int]
    config: dict


def load_vae_checkpoint(checkpoint: str | Path, device: torch.device) -> LoadedVAE:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    leaf_to_id = cfg.get("leaf_to_id", {})
    model = build_foundation_vae(
        cfg["architecture"],
        n_genes=int(cfg["n_genes"]),
        latent_dim=int(cfg["latent_dim"]),
        n_batches=len(leaf_to_id),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return LoadedVAE(model=model, leaf_to_id=leaf_to_id, config=cfg)


def batch_codes(leaves: np.ndarray, leaf_to_id: dict[str, int]) -> torch.Tensor | None:
    if not leaf_to_id:
        return None
    return torch.tensor([leaf_to_id[str(x)] for x in leaves], dtype=torch.long)


def encoder_uses_batch(loaded: LoadedVAE) -> bool:
    return bool(loaded.config.get("encoder_uses_batch", bool(loaded.leaf_to_id)))


class EmbeddingExtractor:
    def __init__(self, dataset: FoundationExpressionDataset, loaded: LoadedVAE, device: torch.device):
        self.dataset = dataset
        self.loaded = loaded
        self.device = device

    def sample_ids(self, cfg: EmbeddingEvalConfig) -> np.ndarray:
        rng = np.random.default_rng(cfg.seed)
        return self.dataset.sample_cell_ids_balanced_by_leaf(cfg.split, cfg.sample_size, rng, alpha=0.0)

    def encode_ids(self, ids: np.ndarray, batch_size: int) -> tuple[np.ndarray, pd.DataFrame]:
        z_chunks = []
        meta_chunks = []
        for start in range(0, len(ids), batch_size):
            chunk_ids = ids[start:start + batch_size]
            batch = self.dataset.load_cells(chunk_ids)
            x = torch.from_numpy(batch.x).to(self.device)
            codes = batch_codes(batch.leaf_dataset, self.loaded.leaf_to_id) if encoder_uses_batch(self.loaded) else None
            if codes is not None:
                codes = codes.to(self.device)
            with torch.no_grad():
                mu, _ = self.loaded.model.encode(x, codes)
            z_chunks.append(mu.detach().cpu().numpy().astype(np.float32))
            rows = self.dataset.cell_index_by_id.loc[batch.cell_ids]
            meta_chunks.append(pd.DataFrame({
                "global_cell_id": batch.cell_ids,
                "leaf_dataset": batch.leaf_dataset,
                "top_dataset": rows["top_dataset"].astype(str).to_numpy(),
                "timepoint": batch.timepoint,
                "cell_type": rows["cell_type"].astype(str).to_numpy(),
                "foundation_split": batch.foundation_split,
                "input_row_sum": batch.x.sum(axis=1),
                "input_nonzero_fraction": (batch.x > 0).mean(axis=1),
            }))
        return np.concatenate(z_chunks, axis=0), pd.concat(meta_chunks, ignore_index=True)


def safe_silhouette(z: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> float | None:
    labels = labels.astype(str)
    valid = labels != ""
    z = z[valid]
    labels = labels[valid]
    unique = np.unique(labels)
    if len(unique) < 2 or len(labels) <= len(unique):
        return None
    if len(labels) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(len(labels)), size=max_samples, replace=False)
        z = z[idx]
        labels = labels[idx]
        if len(np.unique(labels)) < 2:
            return None
    return float(silhouette_score(z, labels, metric="euclidean"))


def kmeans_label_metrics(z: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float | int | None]:
    labels = labels.astype(str)
    valid = labels != ""
    z = z[valid]
    labels = labels[valid]
    classes = np.unique(labels)
    if len(classes) < 2 or len(labels) <= len(classes):
        return {"n_classes": int(len(classes)), "kmeans_ari": None, "kmeans_nmi": None}
    pred = KMeans(n_clusters=len(classes), random_state=seed, n_init=10).fit_predict(z)
    return {
        "n_classes": int(len(classes)),
        "kmeans_ari": float(adjusted_rand_score(labels, pred)),
        "kmeans_nmi": float(normalized_mutual_info_score(labels, pred)),
    }


def knn_purity(z: np.ndarray, labels: np.ndarray, k: int) -> float | None:
    labels = labels.astype(str)
    valid = labels != ""
    z = z[valid]
    labels = labels[valid]
    if len(np.unique(labels)) < 2 or len(labels) <= k + 1:
        return None
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(z)
    idx = nbrs.kneighbors(return_distance=False)[:, 1:]
    return float((labels[idx] == labels[:, None]).mean())


def compute_leiden_labels(
    z: np.ndarray,
    *,
    n_neighbors: int,
    resolution: float,
    seed: int,
) -> np.ndarray:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Leiden clustering requires igraph and leidenalg") from exc
    nbrs = NearestNeighbors(n_neighbors=int(n_neighbors) + 1).fit(z)
    distances, indices = nbrs.kneighbors(z)
    edges = set()
    weights = []
    for i in range(indices.shape[0]):
        for j, dist in zip(indices[i, 1:], distances[i, 1:]):
            a, b = sorted((int(i), int(j)))
            if a == b or (a, b) in edges:
                continue
            edges.add((a, b))
            weights.append(float(1.0 / (1.0 + dist)))
    graph = ig.Graph(n=z.shape[0], edges=list(edges), directed=False)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=float(resolution),
        seed=int(seed),
    )
    labels = np.asarray(partition.membership, dtype=int).astype(str)
    return labels


class EmbeddingMetricSuite:
    label_columns = ("leaf_dataset", "top_dataset", "timepoint", "cell_type", "leiden")

    def __init__(self, cfg: EmbeddingEvalConfig):
        self.cfg = cfg

    def basic_metrics(self, z: np.ndarray) -> dict[str, float | int]:
        return {
            "sample_size": int(len(z)),
            "latent_dim": int(z.shape[1]),
            "embedding_mean": float(z.mean()),
            "embedding_std": float(z.std()),
            "embedding_dim_std_mean": float(z.std(axis=0).mean()),
            "embedding_dim_std_min": float(z.std(axis=0).min()),
        }

    def label_metrics(self, z: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for i, column in enumerate(self.label_columns):
            labels = meta[column].astype(str).to_numpy()
            row = {
                "label": column,
                "silhouette": safe_silhouette(z, labels, self.cfg.silhouette_max_samples, self.cfg.seed + i),
                "knn_purity": knn_purity(z, labels, self.cfg.knn_k),
            }
            row.update(kmeans_label_metrics(z, labels, self.cfg.seed + i))
            rows.append(row)
        return pd.DataFrame(rows)

    def centroid_metrics(self, z: np.ndarray, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | None]]:
        rows = []
        for (leaf, t), idx in meta.groupby(["leaf_dataset", "timepoint"]).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            center = z[idx_arr].mean(axis=0)
            rows.append({
                "leaf_dataset": leaf,
                "timepoint": float(t),
                "n": int(len(idx_arr)),
                "centroid_norm": float(np.linalg.norm(center)),
            })
        centroids = pd.DataFrame(rows).sort_values(["leaf_dataset", "timepoint"])
        shifts = []
        for leaf, group in meta.groupby("leaf_dataset"):
            means = []
            for t, idx in group.groupby("timepoint").groups.items():
                means.append((float(t), z[np.asarray(list(idx), dtype=int)].mean(axis=0)))
            means.sort(key=lambda x: x[0])
            for (_, a), (_, b) in zip(means, means[1:]):
                shifts.append(float(np.linalg.norm(b - a)))
        summary = {
            "adjacent_time_centroid_shift_mean": float(np.mean(shifts)) if shifts else None,
            "adjacent_time_centroid_shift_min": float(np.min(shifts)) if shifts else None,
            "adjacent_time_centroid_shift_max": float(np.max(shifts)) if shifts else None,
        }
        return centroids, summary


def evaluate_vae_embedding(config: EmbeddingEvalConfig) -> dict:
    t0 = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    dataset = FoundationExpressionDataset(config.catalog_dir)
    loaded = load_vae_checkpoint(config.checkpoint, device)
    extractor = EmbeddingExtractor(dataset, loaded, device)
    ids = extractor.sample_ids(config)
    z, meta = extractor.encode_ids(ids, config.batch_size)
    meta["leiden"] = compute_leiden_labels(
        z,
        n_neighbors=config.knn_k,
        resolution=config.leiden_resolution,
        seed=config.seed,
    )

    if config.save_embeddings:
        np.savez_compressed(output_dir / "embedding_sample.npz", z=z, **{c: meta[c].to_numpy() for c in meta.columns})
    meta.to_csv(output_dir / "embedding_metadata.tsv", sep="\t", index=False)

    metric_suite = EmbeddingMetricSuite(config)
    label_df = metric_suite.label_metrics(z, meta)
    label_df.to_csv(output_dir / "label_metrics.tsv", sep="\t", index=False)
    centroid_df, centroid_summary = metric_suite.centroid_metrics(z, meta)
    centroid_df.to_csv(output_dir / "centroid_metrics.tsv", sep="\t", index=False)
    plot_outputs = {}
    if config.make_plots:
        from cellworldmodel.foundation.vae_plot import (
            plot_centroid_trajectory,
            plot_metric_summary,
            plot_pca_panels,
            plot_umap_panels,
        )

        plot_dir = output_dir / "plots"
        plot_outputs.update(plot_umap_panels(z, meta, plot_dir, seed=config.seed))
        plot_outputs.update(plot_pca_panels(z, meta, plot_dir))
        plot_outputs.update(plot_metric_summary(label_df, plot_dir))
        plot_outputs.update(plot_centroid_trajectory(centroid_df, plot_dir))

    metrics: dict[str, object] = {
        "checkpoint": str(config.checkpoint),
        "architecture": loaded.config.get("architecture"),
        "split": config.split,
        "device": config.device,
        "eval_time_s": float(time.time() - t0),
        **metric_suite.basic_metrics(z),
        **{f"centroid/{k}": v for k, v in centroid_summary.items()},
    }
    for row in label_df.to_dict(orient="records"):
        label = row.pop("label")
        for key, value in row.items():
            metrics[f"{label}/{key}"] = value
    manifest = {"config": asdict(config), "checkpoint_config": loaded.config, "outputs": {
        "metrics": str(output_dir / "metrics.json"),
        "label_metrics": str(output_dir / "label_metrics.tsv"),
        "centroid_metrics": str(output_dir / "centroid_metrics.tsv"),
        "metadata": str(output_dir / "embedding_metadata.tsv"),
        "embeddings": str(output_dir / "embedding_sample.npz") if config.save_embeddings else None,
        "plots": plot_outputs,
    }}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return metrics
