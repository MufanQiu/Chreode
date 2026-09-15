"""Optional position and log-mass updates outside the gene-state backbone."""
from __future__ import annotations

import math

import torch
from torch import nn


def _dimension(name: str, value: int, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} integer")


def _scale(name: str, value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _mlp(input_dim: int, hidden_dim: int, seed: int, *, output_bias: bool) -> nn.Sequential:
    _dimension("init_seed", seed, allow_zero=True)
    if seed >= 2**63:
        raise ValueError("init_seed must be less than 2**63")
    # Construct on CPU even when the caller changed PyTorch's default device.
    # Seed only this fork's CPU generator; CUDA RNG states are never touched.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        options = {"device": "cpu", "dtype": torch.float32}
        network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, **options), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, **options), nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=output_bias, **options),
        )
        nn.init.zeros_(network[-1].weight)
        if network[-1].bias is not None:
            nn.init.zeros_(network[-1].bias)
    return network


def _tensor(name: str, value: torch.Tensor, shape: tuple[int, ...],
            reference: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not value.is_floating_point() or value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must match the floating dtype and device of z")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite, including masked padding")


def _shared(z: torch.Tensor, delta: torch.Tensor, alpha: torch.Tensor,
            gene_dim: int, parameter: torch.Tensor) -> int:
    if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.shape[1] != gene_dim:
        raise ValueError(f"z must have shape (batch, {gene_dim})")
    batch = z.shape[0]
    _tensor("z", z, (batch, gene_dim), z)
    if parameter.device != z.device or parameter.dtype != z.dtype:
        raise ValueError("head parameters must match the dtype and device of z; move the module explicitly")
    _tensor("delta", delta, (batch,), z)
    _tensor("alpha", alpha, (batch,), z)
    if torch.any((delta == 0) & (alpha != 0)):
        raise ValueError("alpha must be zero wherever delta is zero")
    return batch


def _finite_update(name: str, original: torch.Tensor, update: torch.Tensor,
                   alpha: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(update).all():
        raise ValueError(f"{name} update is nonfinite")
    if original.ndim == 2:
        alpha = alpha[:, None]
    result = torch.where(alpha == 0, original, original + alpha * update)
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} result is nonfinite")
    return result


class EquivariantPositionHead(nn.Module):
    """Update 2D coordinates with scalar messages times relative vectors.

    ``relative_x[b,j]`` is the source neighbor's position minus the source
    query's position. The scalar network sees only ``z_i, z_j, ||relative_x|| /
    distance_scale, delta / delta_scale``. Its null message replaces ``z_j``
    with zero using the same network, removing additive terms independent of
    the neighbor gene input (not guaranteeing zero for homogeneous neighbors).
    Neighbor messages are summed (not averaged). All padding must be finite.
    """

    def __init__(self, gene_dim: int, *, delta_scale: float, distance_scale: float,
                 hidden_dim: int = 64, init_seed: int = 0):
        super().__init__()
        _dimension("gene_dim", gene_dim)
        _dimension("hidden_dim", hidden_dim)
        self.gene_dim = gene_dim
        self.delta_scale = _scale("delta_scale", delta_scale)
        self.distance_scale = _scale("distance_scale", distance_scale)
        # A final bias would cancel identically between the real and null paths.
        self.messages = _mlp(2 * gene_dim + 2, hidden_dim, init_seed, output_bias=False)

    def forward(self, z: torch.Tensor, x: torch.Tensor, neighbor_z: torch.Tensor,
                relative_x: torch.Tensor, delta: torch.Tensor, alpha: torch.Tensor,
                neighbor_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch = _shared(z, delta, alpha, self.gene_dim, self.messages[0].weight)
        _tensor("x", x, (batch, 2), z)
        if not isinstance(neighbor_z, torch.Tensor) or neighbor_z.ndim != 3:
            raise ValueError("neighbor_z must have shape (batch, neighbors, gene_dim)")
        neighbors = neighbor_z.shape[1]
        _tensor("neighbor_z", neighbor_z, (batch, neighbors, self.gene_dim), z)
        _tensor("relative_x", relative_x, (batch, neighbors, 2), z)
        if neighbor_mask is not None:
            if (not isinstance(neighbor_mask, torch.Tensor)
                    or neighbor_mask.shape != (batch, neighbors)
                    or neighbor_mask.dtype != torch.bool or neighbor_mask.device != z.device):
                raise ValueError("neighbor_mask must be a boolean (batch, neighbors) tensor on z.device")
        if neighbors == 0 or batch == 0:
            return x
        query = z[:, None, :].expand(-1, neighbors, -1)
        radius = torch.linalg.vector_norm(relative_x / self.distance_scale, dim=-1, keepdim=True)
        duration = (delta / self.delta_scale)[:, None, None].expand(-1, neighbors, -1)
        inputs = torch.cat((query, neighbor_z, radius, duration), dim=-1)
        null_inputs = torch.cat((query, torch.zeros_like(neighbor_z), radius, duration), dim=-1)
        _tensor("scaled message inputs", inputs, inputs.shape, z)
        observed = self.messages(inputs)
        null = self.messages(null_inputs)
        # Masking nonfinite messages can hide invalid values but leave NaN gradients.
        if not torch.isfinite(observed).all() or not torch.isfinite(null).all():
            raise ValueError("position messages are nonfinite before masking")
        scalar = observed - null
        if not torch.isfinite(scalar).all():
            raise ValueError("position message difference is nonfinite before masking")
        if neighbor_mask is not None:
            scalar = scalar.masked_fill(~neighbor_mask[:, :, None], 0)
        displacement = (scalar * relative_x).sum(dim=1)
        return _finite_update("position", x, displacement, alpha)


class LogMassHead(nn.Module):
    """Add ``alpha * growth(z, source_context, delta / delta_scale)`` to log mass.

    ``context_dim`` is a caller-selected source summary dimension, not a fixed
    cell-type vocabulary. Missing context is a zero vector; missing mass is
    handled by ``SpatialStateHeads`` before this branch is called. No target
    information or log mass itself is a network input.
    """

    def __init__(self, gene_dim: int, context_dim: int = 0, *, delta_scale: float,
                 hidden_dim: int = 64, init_seed: int = 1):
        super().__init__()
        _dimension("gene_dim", gene_dim)
        _dimension("context_dim", context_dim, allow_zero=True)
        _dimension("hidden_dim", hidden_dim)
        self.gene_dim = gene_dim
        self.context_dim = context_dim
        self.delta_scale = _scale("delta_scale", delta_scale)
        self.growth = _mlp(gene_dim + context_dim + 1, hidden_dim, init_seed, output_bias=True)

    def forward(self, z: torch.Tensor, log_mass: torch.Tensor, delta: torch.Tensor,
                alpha: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        batch = _shared(z, delta, alpha, self.gene_dim, self.growth[0].weight)
        _tensor("log_mass", log_mass, (batch,), z)
        if context is None:
            context = z.new_zeros(batch, self.context_dim)
        _tensor("context", context, (batch, self.context_dim), z)
        inputs = torch.cat((z, context, (delta / self.delta_scale)[:, None]), dim=-1)
        _tensor("scaled growth inputs", inputs, inputs.shape, z)
        growth = self.growth(inputs).squeeze(-1)
        return _finite_update("log_mass", log_mass, growth, alpha)


class SpatialStateHeads(nn.Module):
    """Independently optional external position and log-mass branches.

    Return ``(x_updated, log_mass_updated)``; never modify or return a gene
    state. Supply ``alpha`` from the unchanged base ``alpha_gate(delta)``.
    Disabled or absent modalities return the exact original objects without
    validating or evaluating that branch. Present coordinates require both
    neighbor tensors, which may have zero neighbors. Parameters initialize in
    CPU float32 with private seeds and must be moved explicitly with ``.to``.
    """

    def __init__(self, gene_dim: int, context_dim: int = 0, *, delta_scale: float,
                 distance_scale: float, hidden_dim: int = 64, init_seed: int = 0,
                 enabled: bool = True, position_enabled: bool = True, mass_enabled: bool = True):
        super().__init__()
        _dimension("init_seed", init_seed, allow_zero=True)
        self.enabled = enabled
        self.position_enabled = position_enabled
        self.mass_enabled = mass_enabled
        self.position_head = EquivariantPositionHead(
            gene_dim, delta_scale=delta_scale, distance_scale=distance_scale,
            hidden_dim=hidden_dim, init_seed=init_seed,
        )
        self.mass_head = LogMassHead(
            gene_dim, context_dim, delta_scale=delta_scale,
            hidden_dim=hidden_dim, init_seed=init_seed + 1,
        )

    def forward(self, z: torch.Tensor, delta: torch.Tensor, alpha: torch.Tensor, *,
                x: torch.Tensor | None = None, neighbor_z: torch.Tensor | None = None,
                relative_x: torch.Tensor | None = None, log_mass: torch.Tensor | None = None,
                context: torch.Tensor | None = None,
                neighbor_mask: torch.Tensor | None = None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.enabled:
            return x, log_mass
        if self.position_enabled and x is not None:
            x = self.position_head(z, x, neighbor_z, relative_x, delta, alpha, neighbor_mask)
        if self.mass_enabled and log_mass is not None:
            log_mass = self.mass_head(z, log_mass, delta, alpha, context)
        return x, log_mass
