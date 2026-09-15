"""Smoke trainer for the local foundation VAE."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score

from cellworldmodel.foundation.config import FoundationConfig
from cellworldmodel.foundation.expression_dataset import FoundationExpressionDataset
from cellworldmodel.foundation.vae_model import Log1pGaussianVAE
from cellworldmodel.foundation.vae_registry import build_foundation_vae
from cellworldmodel.script.wandb_utils import flatten_numeric, wandb_run_info, wandb_summary_update


@dataclass(frozen=True)
class VaeTrainOptions:
    catalog_dir: str | Path
    output_dir: str | Path
    latent_dim: int
    batch_strategy: str
    max_steps: int
    batch_size: int
    seed: int
    architecture: str = "mlp512"
    beta_kl: float = 1e-3
    lr: float = 1e-3
    device: str | None = None
    qc_sample_size: int = 2048
    silhouette_max_samples: int = 2000
    checkpoint_every_steps: int | None = None
    checkpoint_prefix: str = "checkpoint"
    run_label: str = "smoke"


def leaf_vocab(dataset: FoundationExpressionDataset, split: str = "train") -> dict[str, int]:
    rows = dataset.cell_index[dataset.cell_index["foundation_split"] == split]
    leaves = sorted(rows["leaf_dataset"].astype(str).unique().tolist())
    return {leaf: i for i, leaf in enumerate(leaves)}


def batch_codes(leaves: np.ndarray, vocab: dict[str, int]) -> torch.Tensor:
    return torch.tensor([vocab[str(x)] for x in leaves], dtype=torch.long)


def parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"n_params": int(total), "n_trainable_params": int(trainable)}


def input_stats(x: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_std": float(x.std()),
        f"{prefix}_min": float(x.min()),
        f"{prefix}_max": float(x.max()),
        f"{prefix}_nonzero_fraction": float(np.count_nonzero(x) / x.size) if x.size else 0.0,
        f"{prefix}_row_sum_mean": float(x.sum(axis=1).mean()) if x.size else 0.0,
        f"{prefix}_row_sum_std": float(x.sum(axis=1).std()) if x.size else 0.0,
    }


def mean_row_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x0 = x - x.mean(dim=1, keepdim=True)
    y0 = y - y.mean(dim=1, keepdim=True)
    denom = torch.linalg.norm(x0, dim=1) * torch.linalg.norm(y0, dim=1)
    corr = (x0 * y0).sum(dim=1) / denom.clamp_min(1e-8)
    return float(corr.mean().item())


def evaluate_vae(
    model: Log1pGaussianVAE,
    dataset: FoundationExpressionDataset,
    rng: np.random.Generator,
    device: torch.device,
    *,
    split: str,
    batch_size: int,
    leaf_to_id: dict[str, int],
    beta_kl: float,
) -> dict[str, float]:
    ids = dataset.sample_cell_ids(split, batch_size, rng)
    batch = dataset.load_cells(ids)
    x = torch.from_numpy(batch.x).to(device)
    codes = batch_codes(batch.leaf_dataset, leaf_to_id).to(device) if model.n_batches > 0 else None
    model.eval()
    with torch.no_grad():
        out = model(x, codes)
        loss = out["recon_loss"] + beta_kl * out["kl"]
        latent_std = out["mu"].std(dim=0)
        recon_pearson = mean_row_pearson(x, out["recon"])
    stats = input_stats(batch.x, f"{split}_input")
    return {
        f"{split}_loss": float(loss.item()),
        f"{split}_recon_loss": float(out["recon_loss"].item()),
        f"{split}_recon_pearson": recon_pearson,
        f"{split}_kl": float(out["kl"].item()),
        f"{split}_latent_std_mean": float(latent_std.mean().item()),
        f"{split}_latent_std_min": float(latent_std.min().item()),
        **stats,
    }


def evaluate_null_decode_gap(
    model: Log1pGaussianVAE,
    dataset: FoundationExpressionDataset,
    rng: np.random.Generator,
    device: torch.device,
    *,
    split: str,
    batch_size: int,
    leaf_to_id: dict[str, int],
) -> dict[str, float]:
    if not bool(getattr(model, "supports_null_decode", False)) or not leaf_to_id:
        return {}
    ids = dataset.sample_cell_ids(split, batch_size, rng)
    batch = dataset.load_cells(ids)
    x = torch.from_numpy(batch.x).to(device)
    codes = batch_codes(batch.leaf_dataset, leaf_to_id).to(device)
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x, None)
        recon_seen = model.decode(mu, codes)
        recon_null = model.decode(mu, None)
        seen_pearson = mean_row_pearson(x, recon_seen)
        null_pearson = mean_row_pearson(x, recon_null)
    return {
        f"{split}_recon_pearson_seen": seen_pearson,
        f"{split}_recon_pearson_null": null_pearson,
        f"{split}_null_decode_gap": float(seen_pearson - null_pearson),
        f"{split}_recon_mse_seen": float(torch.mean((recon_seen - x) ** 2).item()),
        f"{split}_recon_mse_null": float(torch.mean((recon_null - x) ** 2).item()),
    }


def _safe_silhouette(z: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> float | None:
    labels = labels.astype(str)
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


def latent_qc(
    model: Log1pGaussianVAE,
    dataset: FoundationExpressionDataset,
    rng: np.random.Generator,
    device: torch.device,
    *,
    split: str,
    sample_size: int,
    leaf_to_id: dict[str, int],
    output_dir: Path,
    seed: int,
    silhouette_max_samples: int = 2000,
) -> dict:
    ids = dataset.sample_cell_ids_balanced_by_leaf(split, sample_size, rng, alpha=0.0)
    batch = dataset.load_cells(ids)
    x = torch.from_numpy(batch.x).to(device)
    codes = batch_codes(batch.leaf_dataset, leaf_to_id).to(device) if model.n_batches > 0 else None
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x, codes)
    z = mu.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(
        output_dir / "latent_qc_sample.npz",
        z=z,
        cell_ids=batch.cell_ids,
        leaf_dataset=batch.leaf_dataset,
        timepoint=batch.timepoint,
        split=batch.foundation_split,
    )
    meta = pd.DataFrame({
        "cell_id": batch.cell_ids,
        "leaf_dataset": batch.leaf_dataset,
        "timepoint": batch.timepoint,
        "split": batch.foundation_split,
        "input_row_sum": batch.x.sum(axis=1),
        "input_nonzero_fraction": (batch.x > 0).mean(axis=1),
    })
    meta.to_csv(output_dir / "latent_qc_sample.tsv", sep="\t", index=False)
    by_leaf = meta.groupby("leaf_dataset").agg(
        n=("cell_id", "count"),
        input_row_sum_mean=("input_row_sum", "mean"),
        input_nonzero_fraction_mean=("input_nonzero_fraction", "mean"),
    ).reset_index()
    by_leaf.to_csv(output_dir / "qc_by_leaf.tsv", sep="\t", index=False)
    by_time = meta.groupby("timepoint").agg(
        n=("cell_id", "count"),
        input_row_sum_mean=("input_row_sum", "mean"),
        input_nonzero_fraction_mean=("input_nonzero_fraction", "mean"),
    ).reset_index()
    by_time.to_csv(output_dir / "qc_by_timepoint.tsv", sep="\t", index=False)

    centroid_rows = []
    for (leaf, t), idx in meta.groupby(["leaf_dataset", "timepoint"]).groups.items():
        z_mean = z[np.asarray(list(idx), dtype=int)].mean(axis=0)
        centroid_rows.append({
            "leaf_dataset": leaf,
            "timepoint": float(t),
            "n": int(len(idx)),
            "centroid_norm": float(np.linalg.norm(z_mean)),
        })
    centroids = pd.DataFrame(centroid_rows)
    centroids.to_csv(output_dir / "latent_centroids.tsv", sep="\t", index=False)
    adjacent_shifts = []
    for leaf, group in centroids.groupby("leaf_dataset"):
        group = group.sort_values("timepoint")
        times = group["timepoint"].to_numpy()
        if len(times) < 2:
            continue
        leaf_indices = []
        for t in times:
            mask = (meta["leaf_dataset"] == leaf) & (meta["timepoint"] == t)
            leaf_indices.append(np.flatnonzero(mask.to_numpy()))
        means = [z[idx].mean(axis=0) for idx in leaf_indices if len(idx)]
        for a, b in zip(means, means[1:]):
            adjacent_shifts.append(float(np.linalg.norm(b - a)))

    qc = {
        "latent_qc_n": int(len(z)),
        "latent_qc_leaf_silhouette": _safe_silhouette(z, batch.leaf_dataset, silhouette_max_samples, seed),
        "latent_qc_timepoint_silhouette": _safe_silhouette(
            z, batch.timepoint.astype(str), silhouette_max_samples, seed + 1
        ),
        "latent_qc_mu_mean": float(z.mean()),
        "latent_qc_mu_std": float(z.std()),
        "latent_qc_adjacent_time_centroid_shift_mean": float(np.mean(adjacent_shifts)) if adjacent_shifts else None,
        "latent_qc_adjacent_time_centroid_shift_min": float(np.min(adjacent_shifts)) if adjacent_shifts else None,
        "latent_qc_adjacent_time_centroid_shift_max": float(np.max(adjacent_shifts)) if adjacent_shifts else None,
        "latent_qc_leaf_input_nonzero_fraction_mean": float(by_leaf["input_nonzero_fraction_mean"].mean()),
        "latent_qc_time_input_nonzero_fraction_mean": float(by_time["input_nonzero_fraction_mean"].mean()),
    }
    return qc


class VaeTrainer:
    def __init__(self, cfg: FoundationConfig, options: VaeTrainOptions, wandb_run=None) -> None:
        self.cfg = cfg
        self.options = options
        self.wandb_run = wandb_run
        self.rng = np.random.default_rng(options.seed)
        torch.manual_seed(options.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(options.seed)
        self.device = torch.device(options.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dataset = FoundationExpressionDataset(options.catalog_dir)
        self.use_batch = options.batch_strategy in {
            "b1_leaf_dataset",
            "b2_leaf_covariate",
            "b2_encoder_nobatch_decoder_residual",
        }
        self.leaf_to_id = leaf_vocab(self.dataset, split="train") if self.use_batch else {}
        self.model = build_foundation_vae(
            options.architecture,
            n_genes=self.dataset.n_genes,
            latent_dim=int(options.latent_dim),
            n_batches=len(self.leaf_to_id) if self.use_batch else 0,
        ).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(options.lr), weight_decay=1e-4)
        self.output_dir = Path(options.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []
        self.first_batch_stats = None
        self.model_stats = parameter_summary(self.model)

    @property
    def checkpoint_config(self) -> dict:
        return {
            "n_genes": self.dataset.n_genes,
            "latent_dim": int(self.options.latent_dim),
            "architecture": self.options.architecture,
            "batch_strategy": self.options.batch_strategy,
            "leaf_to_id": self.leaf_to_id,
            "run_label": self.options.run_label,
            "encoder_uses_batch": bool(getattr(self.model, "encoder_uses_batch", bool(self.leaf_to_id))),
            "supports_null_decode": bool(getattr(self.model, "supports_null_decode", False)),
        }

    def sample_train_batch(self):
        ids = self.dataset.sample_cell_ids_balanced_by_leaf(
            "train",
            int(self.options.batch_size),
            self.rng,
            alpha=float(self.cfg.vae.leaf_sampling_alpha),
        )
        return self.dataset.load_cells(ids)

    def train_step(self, step: int) -> dict:
        step_t0 = time.time()
        batch = self.sample_train_batch()
        if self.first_batch_stats is None:
            self.first_batch_stats = input_stats(batch.x, "first_train_input")
        x = torch.from_numpy(batch.x).to(self.device)
        codes = batch_codes(batch.leaf_dataset, self.leaf_to_id).to(self.device) if self.use_batch else None
        self.model.train()
        out = self.model(x, codes)
        loss = out["recon_loss"] + float(self.options.beta_kl) * out["kl"]
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        elapsed = time.time() - step_t0
        metrics = {
            "step": step + 1,
            "loss": float(loss.item()),
            "recon_loss": float(out["recon_loss"].item()),
            "recon_pearson": mean_row_pearson(x.detach(), out["recon"].detach()),
            "kl": float(out["kl"].item()),
            "cells_per_s": float(self.options.batch_size / elapsed) if elapsed > 0 else None,
        }
        if self.wandb_run is not None:
            self.wandb_run.log(flatten_numeric({f"train/{k}": v for k, v in metrics.items()}), step=step + 1)
        return metrics

    def fit(self) -> dict:
        t0 = time.time()
        for step in range(int(self.options.max_steps)):
            metrics = self.train_step(step)
            if step == 0 or (step + 1) % max(1, min(50, self.options.max_steps)) == 0:
                self.history.append(metrics)
            if self.options.checkpoint_every_steps:
                if (step + 1) % int(self.options.checkpoint_every_steps) == 0:
                    epoch_idx = (step + 1) // int(self.options.checkpoint_every_steps)
                    self.save_checkpoint(
                        self.output_dir / f"{self.options.checkpoint_prefix}_{epoch_idx}.pt",
                        step=step + 1,
                        elapsed_s=time.time() - t0,
                    )
        return self.finalize(time.time() - t0)

    def save_checkpoint(self, path: Path, *, step: int, elapsed_s: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "config": self.checkpoint_config,
            "trainer": {
                "step": int(step),
                "elapsed_s": float(elapsed_s),
                "max_steps": int(self.options.max_steps),
                "batch_size": int(self.options.batch_size),
                "seed": int(self.options.seed),
                "lr": float(self.options.lr),
                "beta_kl": float(self.options.beta_kl),
                "checkpoint_every_steps": self.options.checkpoint_every_steps,
            },
            "model_stats": self.model_stats,
        }, path)

    def finalize(self, elapsed: float) -> dict:
        qc = {
            "latent_dim": int(self.options.latent_dim),
            "batch_strategy": self.options.batch_strategy,
            "architecture": self.options.architecture,
            "run_label": self.options.run_label,
            "max_steps": int(self.options.max_steps),
            "batch_size": int(self.options.batch_size),
            "seed": int(self.options.seed),
            "device": str(self.device),
            "train_time_s": float(elapsed),
            "steps_per_s": float(self.options.max_steps / elapsed) if elapsed > 0 else None,
            "n_genes": int(self.dataset.n_genes),
            "n_train_leaves": int(len(self.leaf_to_id)) if self.use_batch else 0,
            **self.model_stats,
            "model_architecture": {
                "architecture": self.options.architecture,
                "latent_dim": int(self.options.latent_dim),
                "batch_conditioning": self.options.batch_strategy if self.use_batch else "none",
                "encoder_uses_batch": bool(getattr(self.model, "encoder_uses_batch", bool(self.leaf_to_id))),
                "supports_null_decode": bool(getattr(self.model, "supports_null_decode", False)),
            },
            **(self.first_batch_stats or {}),
            "history": self.history,
        }
        qc.update(evaluate_vae(
            self.model,
            self.dataset,
            self.rng,
            self.device,
            split="val",
            batch_size=min(self.options.batch_size, 512),
            leaf_to_id=self.leaf_to_id,
            beta_kl=float(self.options.beta_kl),
        ))
        qc.update(evaluate_null_decode_gap(
            self.model,
            self.dataset,
            self.rng,
            self.device,
            split="val",
            batch_size=min(self.options.batch_size, 512),
            leaf_to_id=self.leaf_to_id,
        ))
        qc.update(latent_qc(
            self.model,
            self.dataset,
            self.rng,
            self.device,
            split="val",
            sample_size=int(self.options.qc_sample_size),
            leaf_to_id=self.leaf_to_id,
            output_dir=self.output_dir,
            seed=int(self.options.seed),
            silhouette_max_samples=int(self.options.silhouette_max_samples),
        ))
        if self.wandb_run is not None:
            wandb_summary_update(self.wandb_run, qc)
            qc["wandb"] = wandb_run_info(self.wandb_run)
        self.save_checkpoint(self.output_dir / "model.pt", step=int(self.options.max_steps), elapsed_s=elapsed)
        with (self.output_dir / "qc.json").open("w", encoding="utf-8") as handle:
            json.dump(qc, handle, indent=2, sort_keys=True)
        pd.DataFrame(self.history).to_csv(self.output_dir / "history.tsv", sep="\t", index=False)
        return qc


def train_vae_smoke(cfg: FoundationConfig, options: VaeTrainOptions, wandb_run=None) -> dict:
    return VaeTrainer(cfg, options, wandb_run=wandb_run).fit()


def steps_per_epoch(catalog_dir: str | Path, batch_size: int, split: str = "train") -> int:
    dataset = FoundationExpressionDataset(catalog_dir)
    n_cells = int((dataset.cell_index["foundation_split"] == split).sum())
    if n_cells <= 0:
        raise ValueError(f"No cells for split={split!r}")
    return int(np.ceil(n_cells / int(batch_size)))
