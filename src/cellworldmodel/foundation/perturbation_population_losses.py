"""Population-level losses for unpaired perturbation prediction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cellworldmodel.benchmark.common_metrics import mmd2_unbiased_multi_sigma, sinkhorn_w2


@dataclass(frozen=True)
class PopulationLossWeights:
    latent_mmd: float = 1.0
    latent_w2: float = 0.1
    expr_bulk: float = 1.0
    delta_bulk: float = 0.0
    delta_cosine: float = 0.2
    de_bulk: float = 2.0
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 50


class PopulationPerturbationLoss:
    def __init__(self, weights: PopulationLossWeights) -> None:
        self.weights = weights

    def __call__(
        self,
        *,
        pred_z: torch.Tensor,
        target_z: torch.Tensor,
        pred_x: torch.Tensor,
        target_x: torch.Tensor,
        control_mean: torch.Tensor,
        de_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        latent_mmd = (
            mmd2_unbiased_multi_sigma(pred_z, target_z)
            if float(self.weights.latent_mmd) != 0.0 else pred_z.new_tensor(0.0)
        )
        latent_w2 = (
            sinkhorn_w2(
                pred_z,
                target_z,
                epsilon=float(self.weights.sinkhorn_eps),
                num_iters=int(self.weights.sinkhorn_iters),
            )
            if float(self.weights.latent_w2) != 0.0 else pred_z.new_tensor(0.0)
        )
        pred_bulk = pred_x.mean(dim=0)
        target_bulk = target_x.mean(dim=0)
        expr_bulk = F.mse_loss(pred_bulk, target_bulk)
        pred_delta_full = pred_bulk - control_mean
        true_delta_full = target_bulk - control_mean
        delta_bulk = F.mse_loss(pred_delta_full, true_delta_full)
        de_bulk = F.mse_loss(pred_delta_full[de_idx], true_delta_full[de_idx])
        pred_delta = pred_delta_full[de_idx]
        true_delta = true_delta_full[de_idx]
        delta_cosine = 1.0 - F.cosine_similarity(pred_delta[None, :], true_delta[None, :]).mean()

        total = (
            float(self.weights.latent_mmd) * latent_mmd
            + float(self.weights.latent_w2) * latent_w2
            + float(self.weights.expr_bulk) * expr_bulk
            + float(self.weights.delta_bulk) * delta_bulk
            + float(self.weights.de_bulk) * de_bulk
            + float(self.weights.delta_cosine) * delta_cosine
        )
        return total, {
            "latent_mmd": float(latent_mmd.item()),
            "latent_w2": float(latent_w2.item()),
            "expr_bulk_loss": float(expr_bulk.item()),
            "delta_bulk_loss": float(delta_bulk.item()),
            "de_bulk_loss": float(de_bulk.item()),
            "delta_cosine_loss": float(delta_cosine.item()),
        }
