"""Composable loss-balancing strategies for benchmark training.

The training loop owns the model forward pass and component loss construction.
This module owns only the scalarization rule:

    {name: loss_i, base_weight_i} -> total_loss, logging_info

Keeping the API narrow makes it easy to add more multi-objective methods later
without scattering method-specific code through the benchmark loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class LossComponent:
    name: str
    loss: torch.Tensor
    base_weight: float


class LossBalancer(nn.Module):
    requires_model_params: bool = False

    def combine(
        self,
        components: Sequence[LossComponent],
        *,
        step: int,
        model_params: Sequence[torch.nn.Parameter] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    @staticmethod
    def _weighted_sum(components: Sequence[LossComponent], weights: dict[str, torch.Tensor]):
        total = None
        for component in components:
            term = weights[component.name] * component.loss
            total = term if total is None else total + term
        if total is None:
            raise ValueError("No loss components were provided")
        return total

    @staticmethod
    def _info(weights: dict[str, torch.Tensor], prefix: str = "loss_weight") -> dict[str, float]:
        return {f"{prefix}/{name}": float(weight.detach().item()) for name, weight in weights.items()}


class FixedLossBalancer(LossBalancer):
    def combine(self, components, *, step, model_params=None):
        del step, model_params
        weights = {
            component.name: component.loss.new_tensor(float(component.base_weight))
            for component in components
        }
        return self._weighted_sum(components, weights), self._info(weights)


class UncertaintyLossBalancer(LossBalancer):
    """Homoscedastic uncertainty weighting with per-component log variance."""

    def __init__(self, names: Iterable[str], init_log_var: float = 0.0, clamp: float = 5.0):
        super().__init__()
        self.names = list(names)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(float(init_log_var))) for name in self.names
        })
        self.clamp = float(clamp)

    def combine(self, components, *, step, model_params=None):
        del step, model_params
        total = None
        weights: dict[str, torch.Tensor] = {}
        info: dict[str, float] = {}
        for component in components:
            s = torch.clamp(self.log_vars[component.name], -self.clamp, self.clamp)
            base = component.loss.new_tensor(float(component.base_weight))
            weight = base * torch.exp(-s)
            term = weight * component.loss + 0.5 * s
            total = term if total is None else total + term
            weights[component.name] = weight
            info[f"loss_log_var/{component.name}"] = float(s.detach().item())
        if total is None:
            raise ValueError("No loss components were provided")
        info.update(self._info(weights))
        return total, info


class DwaLossBalancer(LossBalancer):
    """Dynamic Weight Average from task loss descent rates."""

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = float(temperature)
        self.history: dict[str, list[float]] = {}

    def combine(self, components, *, step, model_params=None):
        del model_params
        raw: dict[str, float] = {}
        for component in components:
            hist = self.history.get(component.name, [])
            if len(hist) < 2:
                raw[component.name] = 1.0
            else:
                raw[component.name] = hist[-1] / max(hist[-2], 1e-12)
        names = [component.name for component in components]
        raw_t = components[0].loss.new_tensor([raw[name] / self.temperature for name in names])
        multipliers = torch.softmax(raw_t, dim=0) * len(names)
        weights = {
            component.name: component.loss.new_tensor(float(component.base_weight)) * multipliers[i]
            for i, component in enumerate(components)
        }
        total = self._weighted_sum(components, weights)
        for component in components:
            self.history.setdefault(component.name, []).append(float(component.loss.detach().item()))
        info = self._info(weights)
        info.update({f"loss_dwa_ratio/{name}": float(raw[name]) for name in names})
        return total, info


class ReLoBRaLoLossBalancer(LossBalancer):
    """Relative loss balancing with random lookback.

    This is a lightweight ReLoBRaLo-style implementation: weights are based on
    relative progress versus either the initial loss or a randomly sampled
    historical lookback. Slower-improving losses receive larger weights.
    """

    def __init__(self, temperature: float = 1.0, lookback_prob: float = 0.9, seed: int = 0):
        super().__init__()
        self.temperature = float(temperature)
        self.lookback_prob = float(lookback_prob)
        self.rng = np.random.default_rng(seed)
        self.history: dict[str, list[float]] = {}

    def combine(self, components, *, step, model_params=None):
        del step, model_params
        names = [component.name for component in components]
        rel = []
        info: dict[str, float] = {}
        for component in components:
            hist = self.history.get(component.name, [])
            cur = float(component.loss.detach().item())
            if not hist:
                ref = cur
            elif self.rng.random() < self.lookback_prob:
                ref = hist[int(self.rng.integers(len(hist)))]
            else:
                ref = hist[0]
            ratio = cur / max(ref, 1e-12)
            rel.append(ratio)
            info[f"loss_relobralo_ratio/{component.name}"] = float(ratio)
        logits = components[0].loss.new_tensor(rel) / self.temperature
        multipliers = torch.softmax(logits, dim=0) * len(names)
        weights = {
            component.name: component.loss.new_tensor(float(component.base_weight)) * multipliers[i]
            for i, component in enumerate(components)
        }
        total = self._weighted_sum(components, weights)
        for component in components:
            self.history.setdefault(component.name, []).append(float(component.loss.detach().item()))
        info.update(self._info(weights))
        return total, info


class RandomLossWeightBalancer(LossBalancer):
    """Random Loss Weighting baseline using a Dirichlet distribution."""

    def __init__(self, alpha: float = 1.0, seed: int = 0):
        super().__init__()
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(seed)

    def combine(self, components, *, step, model_params=None):
        del step, model_params
        names = [component.name for component in components]
        sample = self.rng.dirichlet(np.full(len(names), self.alpha)) * len(names)
        weights = {
            component.name: component.loss.new_tensor(float(component.base_weight) * float(sample[i]))
            for i, component in enumerate(components)
        }
        return self._weighted_sum(components, weights), self._info(weights)


class GradNormLiteLossBalancer(LossBalancer):
    """Gradient-norm inverse weighting on a small shared parameter subset."""

    requires_model_params = True

    def __init__(self, eps: float = 1e-8, max_multiplier: float = 5.0):
        super().__init__()
        self.eps = float(eps)
        self.max_multiplier = float(max_multiplier)

    def combine(self, components, *, step, model_params=None):
        del step
        if not model_params:
            weights = {
                component.name: component.loss.new_tensor(float(component.base_weight))
                for component in components
            }
            total = self._weighted_sum(components, weights)
            info = self._info(weights)
            info["loss_balancer/gradnorm_fallback"] = 1.0
            return total, info
        params = [p for p in model_params if p.requires_grad]
        norms = []
        for component in components:
            grads = torch.autograd.grad(
                component.loss,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            norm_sq = component.loss.new_tensor(0.0)
            for grad in grads:
                if grad is not None:
                    norm_sq = norm_sq + grad.detach().pow(2).sum()
            norms.append(torch.sqrt(norm_sq + self.eps))
        norm_t = torch.stack(norms)
        inv = norm_t.mean() / torch.clamp(norm_t, min=self.eps)
        inv = torch.clamp(inv, max=self.max_multiplier)
        inv = inv / torch.clamp(inv.mean(), min=self.eps)
        weights = {
            component.name: component.loss.new_tensor(float(component.base_weight)) * inv[i]
            for i, component in enumerate(components)
        }
        total = self._weighted_sum(components, weights)
        info = self._info(weights)
        for component, norm in zip(components, norms):
            info[f"loss_grad_norm/{component.name}"] = float(norm.item())
        return total, info


def build_loss_balancer(cfg: dict, component_names: Sequence[str], seed: int = 0) -> LossBalancer:
    name = str(cfg.get("loss_balancer", "fixed"))
    if name == "fixed":
        return FixedLossBalancer()
    if name == "uncertainty":
        return UncertaintyLossBalancer(
            component_names,
            init_log_var=float(cfg.get("loss_balancer_init_log_var", 0.0)),
            clamp=float(cfg.get("loss_balancer_log_var_clamp", 5.0)),
        )
    if name == "dwa":
        return DwaLossBalancer(temperature=float(cfg.get("loss_balancer_temperature", 2.0)))
    if name == "relobralo":
        return ReLoBRaLoLossBalancer(
            temperature=float(cfg.get("loss_balancer_temperature", 1.0)),
            lookback_prob=float(cfg.get("loss_balancer_lookback_prob", 0.9)),
            seed=seed,
        )
    if name == "rlw":
        return RandomLossWeightBalancer(alpha=float(cfg.get("loss_balancer_alpha", 1.0)), seed=seed)
    if name == "gradnorm_lite":
        return GradNormLiteLossBalancer(
            max_multiplier=float(cfg.get("loss_balancer_max_multiplier", 5.0))
        )
    raise ValueError(f"Unknown loss_balancer={name!r}")


def select_gradnorm_params(model: nn.Module, max_tensors: int = 8) -> list[nn.Parameter]:
    """Pick a small, shared-ish parameter subset for GradNorm-lite probes."""
    preferred = ("final_modulation", "final_norm", "U_head", "A_head")
    selected: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if any(key in name for key in preferred):
            selected.append(param)
        if len(selected) >= max_tensors:
            break
    if selected:
        return selected
    return [param for _, param in list(model.named_parameters())[-max_tensors:]]
