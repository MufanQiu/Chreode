"""Lightweight perturbation baselines in the foundation VAE latent space."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from cellworldmodel.benchmark.common_metrics import mmd2_unbiased_multi_sigma, sinkhorn_w2
from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.foundation.perturbation_dataset import NormanPerturbationDataset
from cellworldmodel.foundation.vae_eval import load_vae_checkpoint
from cellworldmodel.foundation.vae_train import mean_row_pearson


@dataclass(frozen=True)
class PerturbationBaselineOptions:
    catalog_dir: str | Path
    output_dir: str | Path
    norman_path: str | Path
    vae_checkpoint: str | Path
    dynamics_checkpoint: str | Path | None = None
    split_method: str = "additive"
    seed: int = 0
    max_cells_per_condition: int = 512
    eval_cells_per_condition: int = 256
    z_score: float = 5.0
    ridge_alpha: float = 1.0
    device: str | None = None


def _as_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


class PerturbationBaselineEvaluator:
    def __init__(self, options: PerturbationBaselineOptions) -> None:
        self.options = options
        self.rng = np.random.default_rng(options.seed)
        self.output_dir = Path(options.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(options.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vae = load_vae_checkpoint(options.vae_checkpoint, self.device)
        for param in self.vae.model.parameters():
            param.requires_grad_(False)
        if bool(self.vae.config.get("encoder_uses_batch", bool(self.vae.leaf_to_id))):
            raise ValueError("Perturbation baselines require a strict/no-batch encoder VAE")
        self.dataset = NormanPerturbationDataset(
            options.norman_path,
            Path(options.catalog_dir) / "gene_vocab.parquet",
            split_method=options.split_method,
            split_seed=options.seed,
        )
        self.dynamics_model = self._load_dynamics_model(options.dynamics_checkpoint)
        self.control_mean = None
        self.control_std = None

    def _load_dynamics_model(self, checkpoint: str | Path | None):
        if checkpoint is None:
            return None
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        train_cfg = dict(cfg["train_cfg"])
        train_cfg["action_dim"] = 0
        model = build_model(cfg["method"], int(cfg["latent_dim"]), train_cfg, tau_init=1.0)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        return model

    def _encode_x(self, x: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            z, _ = self.vae.model.encode(_as_tensor(x, self.device), None)
        return z

    def _sample_condition_x(self, condition: str, n: int) -> np.ndarray:
        idx = self.dataset._cells_for_condition(condition)
        rows = self.rng.choice(idx, size=min(int(n), len(idx)), replace=int(n) > len(idx))
        return self.dataset._load_rows(rows)

    def _sample_control_x(self, n: int) -> np.ndarray:
        rows = self.rng.choice(
            self.dataset.control_idx,
            size=min(int(n), len(self.dataset.control_idx)),
            replace=int(n) > len(self.dataset.control_idx),
        )
        return self.dataset._load_rows(rows)

    def _condition_feature(self, condition: str) -> np.ndarray:
        gene_ids, signs, _, _, mask = self.dataset.condition_gene_ids(condition)
        feat = np.zeros(self.dataset.n_genes, dtype=np.float32)
        for gene_id, sign, keep in zip(gene_ids, signs, mask):
            if keep and 0 <= int(gene_id) < self.dataset.n_genes:
                feat[int(gene_id)] += float(sign)
        return feat

    def _fit_train_centroids(self):
        control_x = self._sample_control_x(self.options.max_cells_per_condition)
        self.control_mean = control_x.mean(axis=0)
        self.control_std = control_x.std(axis=0) + 1e-6
        control_z = self._encode_x(control_x)
        control_center = control_z.mean(dim=0).detach().cpu().numpy()
        rows = []
        single_delta = {}
        features = []
        deltas = []
        for condition in self.dataset.train_conditions:
            target_x = self._sample_condition_x(condition, self.options.max_cells_per_condition)
            target_z = self._encode_x(target_x)
            delta = target_z.mean(dim=0).detach().cpu().numpy() - control_center
            features.append(self._condition_feature(condition))
            deltas.append(delta)
            genes = condition.split("+")
            if len(genes) == 1:
                single_delta[genes[0].upper()] = delta
            rows.append({"condition": condition, "n_genes": len(genes)})
        features_arr = np.stack(features)
        deltas_arr = np.stack(deltas)
        ridge = Ridge(alpha=float(self.options.ridge_alpha), fit_intercept=True)
        ridge.fit(features_arr, deltas_arr)
        return {
            "control_center": control_center,
            "global_delta": deltas_arr.mean(axis=0),
            "single_delta": single_delta,
            "ridge": ridge,
            "train_condition_table": pd.DataFrame(rows),
        }

    def _prescient_like_initial_z(self, control_x: np.ndarray, condition: str) -> torch.Tensor:
        x = np.array(control_x, dtype=np.float32, copy=True)
        gene_ids, signs, _, _, mask = self.dataset.condition_gene_ids(condition)
        for gene_id, sign, keep in zip(gene_ids, signs, mask):
            if keep and 0 <= int(gene_id) < self.dataset.n_genes:
                x[:, int(gene_id)] = self.control_mean[int(gene_id)] + float(sign) * float(self.options.z_score) * self.control_std[int(gene_id)]
        return self._encode_x(x)

    def _predict(self, baseline: str, src_z: torch.Tensor, condition: str, fit: dict) -> torch.Tensor:
        if baseline == "identity":
            return src_z
        if baseline == "mean_delta":
            return src_z + _as_tensor(fit["global_delta"][None, :], self.device)
        if baseline == "additive_single":
            delta = np.zeros_like(fit["global_delta"])
            found = False
            for gene in condition.split("+"):
                gene_delta = fit["single_delta"].get(gene.upper())
                if gene_delta is not None:
                    delta = delta + gene_delta
                    found = True
            if not found:
                delta = fit["global_delta"]
            return src_z + _as_tensor(delta[None, :], self.device)
        if baseline == "ridge":
            delta = fit["ridge"].predict(self._condition_feature(condition)[None, :])[0]
            return src_z + _as_tensor(delta[None, :], self.device)
        raise ValueError(f"Unknown baseline={baseline}")

    def _metrics(self, pred: torch.Tensor, src: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        pred_flat = pred.reshape(-1, pred.shape[-1])
        return {
            "latent_mse": float(torch.mean((pred_flat[: target.shape[0]] - target) ** 2).item()),
            "latent_delta_pearson": mean_row_pearson(
                (pred_flat[: target.shape[0]] - src[: target.shape[0]]).detach(),
                (target - src[: target.shape[0]]).detach(),
            ),
            "latent_mmd": float(mmd2_unbiased_multi_sigma(pred_flat, target).item()),
            "latent_w2": float(sinkhorn_w2(pred_flat, target, epsilon=0.1, num_iters=50).item()),
        }

    def evaluate(self) -> dict:
        fit = self._fit_train_centroids()
        fit["train_condition_table"].to_csv(self.output_dir / "train_conditions.tsv", sep="\t", index=False)
        baselines = ["identity", "mean_delta", "additive_single", "ridge", "prescient_init"]
        if self.dynamics_model is not None:
            baselines.append("prescient_init_dynamics")
        rows = []
        for condition in self.dataset.test_conditions:
            control_x = self._sample_control_x(self.options.eval_cells_per_condition)
            target_x = self._sample_condition_x(condition, self.options.eval_cells_per_condition)
            src_z = self._encode_x(control_x)
            target_z = self._encode_x(target_x)
            for baseline in baselines:
                if baseline == "prescient_init":
                    pred = self._prescient_like_initial_z(control_x, condition)
                elif baseline == "prescient_init_dynamics":
                    z0 = self._prescient_like_initial_z(control_x, condition)
                    delta = torch.ones(z0.shape[0], device=self.device, dtype=z0.dtype)
                    with torch.no_grad():
                        pred = self.dynamics_model.predict_mean(z0, delta, n_mc=8)
                else:
                    pred = self._predict(baseline, src_z, condition, fit)
                row = {
                    "condition": condition,
                    "baseline": baseline,
                    "n_genes": int(self.dataset.condition_gene_ids(condition)[-1].sum()),
                }
                row.update(self._metrics(pred, src_z, target_z))
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "baseline_conditions.tsv", sep="\t", index=False)
        summary_df = df.groupby("baseline").agg(
            n=("condition", "nunique"),
            latent_delta_pearson=("latent_delta_pearson", "mean"),
            latent_mse=("latent_mse", "mean"),
            latent_mmd=("latent_mmd", "mean"),
            latent_w2=("latent_w2", "mean"),
        ).reset_index()
        summary_df.to_csv(self.output_dir / "baseline_summary.tsv", sep="\t", index=False)
        summary = {
            "split_method": self.options.split_method,
            "n_train_conditions": int(len(self.dataset.train_conditions)),
            "n_test_conditions": int(len(self.dataset.test_conditions)),
            "baselines": summary_df.to_dict(orient="records"),
            "prescient_protocol_note": (
                "PRESCIENT perturbation modifies initial gene expression z-scores and simulates forward "
                "with unchanged dynamics. prescient_init mirrors the initial-state modification in VAE "
                "latent space; prescient_init_dynamics additionally rolls the perturbed initial state "
                "through the pretrained temporal W-DiT."
            ),
        }
        with (self.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return summary


def run_perturbation_baselines(options: PerturbationBaselineOptions) -> dict:
    return PerturbationBaselineEvaluator(options).evaluate()
