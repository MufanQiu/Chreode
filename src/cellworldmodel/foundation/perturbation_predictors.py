"""Composable perturbation predictor strategies for downstream fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PredictorOutput:
    z: torch.Tensor
    aux: dict[str, float]


class BasePerturbationPredictor(nn.Module):
    model_type: str

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> PredictorOutput:
        raise NotImplementedError


@dataclass(frozen=True)
class PerturbationPredictorSpec:
    name: str
    cls: type[BasePerturbationPredictor]
    transition_action_dim: bool
    description: str


PREDICTOR_REGISTRY: dict[str, PerturbationPredictorSpec] = {}


def register_perturbation_predictor(
    name: str,
    *,
    transition_action_dim: bool,
    description: str,
) -> Callable[[type[BasePerturbationPredictor]], type[BasePerturbationPredictor]]:
    def decorator(cls: type[BasePerturbationPredictor]) -> type[BasePerturbationPredictor]:
        if name in PREDICTOR_REGISTRY:
            raise ValueError(f"Duplicate perturbation predictor registration: {name}")
        cls.model_type = name
        PREDICTOR_REGISTRY[name] = PerturbationPredictorSpec(
            name=name,
            cls=cls,
            transition_action_dim=bool(transition_action_dim),
            description=description,
        )
        return cls

    return decorator


def available_perturbation_predictors() -> list[str]:
    return sorted(PREDICTOR_REGISTRY)


@register_perturbation_predictor(
    "direct_action",
    transition_action_dim=True,
    description="Action-conditioned transition predicts the perturbation endpoint directly.",
)
class DirectActionPredictor(BasePerturbationPredictor):
    def __init__(self, transition_model: nn.Module, latent_dim: int, action_dim: int, k_samples: int) -> None:
        super().__init__()
        del latent_dim, action_dim
        self.transition_model = transition_model
        self.k_samples = int(k_samples)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> PredictorOutput:
        delta = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        pred = self.transition_model.predict_mean(z, delta, action=action, n_mc=self.k_samples)
        return PredictorOutput(pred, {"kick_norm": 0.0, "gate_mean": 1.0})


@register_perturbation_predictor(
    "kick_only",
    transition_action_dim=False,
    description="Learn an immediate perturbation kick without rolling through dynamics.",
)
class KickOnlyPredictor(BasePerturbationPredictor):
    def __init__(self, transition_model: nn.Module, latent_dim: int, action_dim: int, k_samples: int) -> None:
        super().__init__()
        del transition_model, k_samples
        self.kick_net = build_kick_net(latent_dim, action_dim)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> PredictorOutput:
        kick = self.kick_net(torch.cat([z, action], dim=-1))
        pred = z + kick
        return PredictorOutput(pred, {"kick_norm": float(kick.norm(dim=1).mean().detach().cpu()), "gate_mean": 0.0})


@register_perturbation_predictor(
    "kick_rollout",
    transition_action_dim=False,
    description="Apply an action kick, then roll the kicked state through the pretrained dynamics.",
)
class KickRolloutPredictor(BasePerturbationPredictor):
    def __init__(self, transition_model: nn.Module, latent_dim: int, action_dim: int, k_samples: int) -> None:
        super().__init__()
        self.transition_model = transition_model
        self.k_samples = int(k_samples)
        self.kick_net = build_kick_net(latent_dim, action_dim)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> PredictorOutput:
        kick = self.kick_net(torch.cat([z, action], dim=-1))
        z_kick = z + kick
        delta = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        pred = self.transition_model.predict_mean(z_kick, delta, action=None, n_mc=self.k_samples)
        return PredictorOutput(pred, {"kick_norm": float(kick.norm(dim=1).mean().detach().cpu()), "gate_mean": 1.0})


@register_perturbation_predictor(
    "hybrid_kick_rollout",
    transition_action_dim=False,
    description="Blend immediate action kick and dynamics rollout with a learned gate.",
)
class HybridKickRolloutPredictor(BasePerturbationPredictor):
    def __init__(self, transition_model: nn.Module, latent_dim: int, action_dim: int, k_samples: int) -> None:
        super().__init__()
        self.transition_model = transition_model
        self.k_samples = int(k_samples)
        self.kick_net = build_kick_net(latent_dim, action_dim)
        self.gate_net = build_gate_net(latent_dim, action_dim)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> PredictorOutput:
        h = torch.cat([z, action], dim=-1)
        kick = self.kick_net(h)
        z_kick = z + kick
        delta = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        z_dyn = self.transition_model.predict_mean(z_kick, delta, action=None, n_mc=self.k_samples)
        gamma = torch.sigmoid(self.gate_net(h))
        pred = (1.0 - gamma) * z_kick + gamma * z_dyn
        return PredictorOutput(pred, {
            "kick_norm": float(kick.norm(dim=1).mean().detach().cpu()),
            "gate_mean": float(gamma.mean().detach().cpu()),
        })


def build_kick_net(latent_dim: int, action_dim: int) -> nn.Sequential:
    net = nn.Sequential(
        nn.LayerNorm(int(latent_dim) + int(action_dim)),
        nn.Linear(int(latent_dim) + int(action_dim), int(latent_dim) * 2),
        nn.SiLU(),
        nn.Linear(int(latent_dim) * 2, int(latent_dim)),
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


def build_gate_net(latent_dim: int, action_dim: int) -> nn.Sequential:
    net = nn.Sequential(
        nn.LayerNorm(int(latent_dim) + int(action_dim)),
        nn.Linear(int(latent_dim) + int(action_dim), int(latent_dim)),
        nn.SiLU(),
        nn.Linear(int(latent_dim), 1),
    )
    nn.init.constant_(net[-1].bias, 0.0)
    return net


def transition_uses_action(model_type: str) -> bool:
    try:
        return PREDICTOR_REGISTRY[str(model_type)].transition_action_dim
    except KeyError as exc:
        known = ", ".join(available_perturbation_predictors())
        raise ValueError(f"Unknown perturbation predictor model_type={model_type!r}. Known: {known}") from exc


def build_perturbation_predictor(
    *,
    model_type: str,
    transition_model: nn.Module,
    latent_dim: int,
    action_dim: int,
    k_samples: int,
) -> BasePerturbationPredictor:
    try:
        spec = PREDICTOR_REGISTRY[str(model_type)]
    except KeyError as exc:
        known = ", ".join(available_perturbation_predictors())
        raise ValueError(f"Unknown perturbation predictor model_type={model_type!r}. Known: {known}") from exc
    return spec.cls(transition_model, latent_dim, action_dim, k_samples)
