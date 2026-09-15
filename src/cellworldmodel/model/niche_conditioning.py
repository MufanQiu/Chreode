"""Source-niche conditioning without changing the pretrained state space.

The low-rank potential correction is the continuous-context counterpart of
``FieldSurgeryAdapter.potential_correction`` in the cytokine runner. Spatial
coordinates only define the external neighborhood summary; neither coordinates
nor additional state tokens enter the base model.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class NichePotentialCorrection(nn.Module):
    """Return -grad_z sum_k c_k(n, Delta) B_k(z), with zero initial output."""

    def __init__(self, dim: int, context_dim: int, *, delta_scale: float,
                 programs: int = 8, hidden_dim: int = 64):
        super().__init__()
        if min(dim, context_dim, programs, hidden_dim) < 1:
            raise ValueError("all dimensions must be positive")
        if not math.isfinite(delta_scale) or delta_scale <= 0:
            raise ValueError("delta_scale must be finite and positive")
        self.dim = dim
        self.context_dim = context_dim
        self.delta_scale = float(delta_scale)
        self.coefficients = nn.Sequential(
            nn.Linear(context_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, programs),
        )
        # A single zero-initialized stage allows immediate coefficient gradients.
        # Zeroing both a gate and this layer would prevent learning entirely.
        nn.init.zeros_(self.coefficients[-1].weight)
        nn.init.zeros_(self.coefficients[-1].bias)
        self.basis = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, 1))
            for _ in range(programs)
        )

    def forward(self, z: torch.Tensor, delta: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError("z has the wrong shape")
        if context.shape != (z.shape[0], self.context_dim):
            raise ValueError("context must have one vector per source cell")
        if delta.shape != (z.shape[0],):
            raise ValueError("delta must have one value per source cell")
        if not torch.isfinite(context).all() or not torch.isfinite(delta).all():
            raise ValueError("context and delta must be finite")
        # Like the existing potential head, this works under torch.no_grad(),
        # but not inference_mode(), which prohibits the required local derivative.
        with torch.enable_grad():
            zin = z if z.requires_grad else z.detach().requires_grad_(True)
            inputs = torch.cat((context, delta[:, None] / self.delta_scale), dim=-1)
            coeff = self.coefficients(inputs)
            fields = torch.cat([basis(zin) for basis in self.basis], dim=-1)
            potential = (coeff * fields).sum()
            correction = -torch.autograd.grad(
                potential, zin, create_graph=self.training,
            )[0]
        return correction if self.training else correction.detach()


class NicheConditionedTransition(nn.Module):
    """Frozen Waddington base plus an optional external potential correction.

    Disabled or missing context calls the original model directly. The adapter
    is not evaluated and cannot affect either base parameters or random draws.
    ``shared`` and ``neighborhood`` controls use this identical architecture;
    the runner changes only the supplied context vectors.
    """

    def __init__(self, base: nn.Module, correction: NichePotentialCorrection,
                 *, enabled: bool = True):
        super().__init__()
        if getattr(base, "dim", None) != correction.dim:
            raise ValueError("base and correction must have the same state dimension")
        if getattr(base, "action_dim", 0) != 0:
            raise ValueError("this adapter requires the unconditioned base checkpoint")
        self.base = base
        self.correction = correction
        self.enabled = enabled
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, z: torch.Tensor, delta: torch.Tensor,
                epsilon: torch.Tensor, context: torch.Tensor | None = None):
        base_output = self.base(z, delta, epsilon)
        if not self.enabled or context is None:
            return base_output
        correction = self.correction(z, delta, context)
        return base_output + self.base.alpha_gate(delta)[:, None, None] * correction[:, None, :]
