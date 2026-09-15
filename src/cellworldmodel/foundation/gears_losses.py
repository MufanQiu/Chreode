"""Loss components for GEARS downstream perturbation fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GearsLossWeights:
    latent: float = 0.1
    expression: float = 1.0
    de: float = 2.0
    delta: float = 0.2
    direction: float = 0.0


class GearsLossComputer:
    def __init__(self, weights: GearsLossWeights) -> None:
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
        latent_loss = F.mse_loss(pred_z, target_z)
        expr_loss = F.mse_loss(pred_x, target_x)
        de_loss = F.mse_loss(pred_x[:, de_idx], target_x[:, de_idx])

        pred_delta = pred_x[:, de_idx].mean(dim=0) - control_mean[de_idx]
        true_delta = target_x[:, de_idx].mean(dim=0) - control_mean[de_idx]
        delta_loss = 1.0 - F.cosine_similarity(pred_delta[None, :], true_delta[None, :]).mean()

        true_sign = torch.sign(true_delta).detach()
        nonzero = true_sign.abs() > 0
        if bool(nonzero.any().item()):
            direction_loss = F.softplus(-pred_delta[nonzero] * true_sign[nonzero]).mean()
            direction_opposite_rate = (
                torch.sign(pred_delta[nonzero].detach()) != true_sign[nonzero]
            ).to(dtype=pred_x.dtype).mean()
        else:
            direction_loss = pred_delta.sum() * 0.0
            direction_opposite_rate = pred_delta.detach().sum() * 0.0

        total = (
            float(self.weights.latent) * latent_loss
            + float(self.weights.expression) * expr_loss
            + float(self.weights.de) * de_loss
            + float(self.weights.delta) * delta_loss
            + float(self.weights.direction) * direction_loss
        )
        return total, {
            "latent_loss": float(latent_loss.item()),
            "expr_loss": float(expr_loss.item()),
            "de_loss": float(de_loss.item()),
            "delta_loss": float(delta_loss.item()),
            "direction_loss": float(direction_loss.item()),
            "direction_opposite_rate": float(direction_opposite_rate.item()),
        }
