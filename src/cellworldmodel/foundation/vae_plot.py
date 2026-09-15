"""Plot registry for foundation VAE embedding evaluation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import umap


def _pca2(z: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=0).fit_transform(z)


def _umap2(z: np.ndarray, seed: int = 0) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.3,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(z)


def _scatter_panel(
    coords: np.ndarray,
    meta: pd.DataFrame,
    label: str,
    output: Path,
    *,
    title_prefix: str,
    max_legend: int = 20,
) -> None:
    values = meta[label].astype(str)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    n_unique = values.nunique()
    if n_unique <= max_legend:
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=values, s=7, linewidth=0, ax=ax)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6, markerscale=2)
    else:
        codes = pd.Categorical(values).codes
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=codes, s=7, linewidths=0, cmap="tab20")
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label=f"{label} code")
    ax.set_title(f"{title_prefix} embedding colored by {label}")
    ax.set_xlabel(f"{title_prefix}1")
    ax.set_ylabel(f"{title_prefix}2")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_pca_panels(z: np.ndarray, meta: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    coords = _pca2(z)
    outputs = {}
    for label in ("leaf_dataset", "top_dataset", "timepoint", "cell_type"):
        path = output_dir / f"pca_by_{label}.png"
        _scatter_panel(coords, meta, label, path, title_prefix="PCA")
        outputs[f"pca_by_{label}"] = str(path)
    return outputs


def plot_umap_panels(z: np.ndarray, meta: pd.DataFrame, output_dir: Path, seed: int = 0) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    coords = _umap2(z, seed=seed)
    np.save(output_dir / "umap_coords.npy", coords.astype(np.float32))
    outputs = {"umap_coords": str(output_dir / "umap_coords.npy")}
    for label in ("leaf_dataset", "top_dataset", "timepoint", "cell_type", "leiden"):
        if label not in meta:
            continue
        path = output_dir / f"umap_by_{label}.png"
        _scatter_panel(coords, meta, label, path, title_prefix="UMAP")
        outputs[f"umap_by_{label}"] = str(path)
    return outputs


def plot_metric_summary(label_metrics: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    long = label_metrics.melt(
        id_vars=["label"],
        value_vars=["silhouette", "knn_purity", "kmeans_ari", "kmeans_nmi"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    sns.barplot(data=long, x="label", y="value", hue="metric", ax=ax)
    ax.set_title("Embedding semantic QC metrics")
    ax.set_ylim(min(-0.25, float(long["value"].min(skipna=True)) - 0.05), 1.0)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = output_dir / "metric_summary.png"
    fig.savefig(path)
    plt.close(fig)
    outputs["metric_summary"] = str(path)
    return outputs


def plot_centroid_trajectory(centroids: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if centroids.empty:
        return outputs
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    sns.lineplot(
        data=centroids.sort_values(["leaf_dataset", "timepoint"]),
        x="timepoint",
        y="centroid_norm",
        hue="leaf_dataset",
        marker="o",
        ax=ax,
        linewidth=1.2,
    )
    ax.set_title("Latent centroid norm over time")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6)
    fig.tight_layout()
    path = output_dir / "centroid_norm_by_time.png"
    fig.savefig(path)
    plt.close(fig)
    outputs["centroid_norm_by_time"] = str(path)
    return outputs


PLOTTERS = {
    "pca_panels": plot_pca_panels,
    "umap_panels": plot_umap_panels,
    "metric_summary": plot_metric_summary,
    "centroid_trajectory": plot_centroid_trajectory,
}
