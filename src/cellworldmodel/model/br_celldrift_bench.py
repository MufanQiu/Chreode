"""BR-CellDrift-MLP: Barycentric-Residual parameterization for benchmark experiments.

Architecture (M1 and M7 share this):
    ẑ = z_0 + α(Δ) · R_θ(z_0, Δ, ε, a)
    R_θ = b_θ(z, Δ, a) + G_θ(z, Δ, a) · h̃

where:
    - b_θ: MLP → R^d, deterministic mean drift (shared developmental program)
    - G_θ: MLP → R^{d × h}, stochastic basis matrix (branching tensor)
    - h̃ = h_ψ(ε) - mean_r h_ψ(ε_r): zero-mean centered noise (from batch-within-source)
    - α(Δ) = 1 - exp(-Δ / τ_0): time gate, ensures identity at Δ=0

Used as:
    - M1: trained with MMD + W2 only
    - M7: trained with MMD + W2 + drifting field V (stopgrad loss)

This is a benchmark-version backbone (pure MLP) — does NOT use chunk tokenization
or DiT blocks. For full CellDriftDiT, see model/cell_drift_dit.py.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierTimeEmbedding(nn.Module):
    """Fourier features for time interval Δ, followed by MLP projection.

    Gives the model a smooth, high-frequency-aware representation of time
    intervals without over-committing to a specific parameterization.
    """

    def __init__(self, out_dim: int, n_freqs: int = 32):
        super().__init__()
        self.n_freqs = n_freqs
        # Log-spaced frequencies spanning roughly [1, 1000]
        freqs = torch.exp(torch.linspace(0.0, torch.log(torch.tensor(1000.0)).item(), n_freqs))
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(2 * n_freqs, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta: (B,) time intervals

        Returns:
            (B, out_dim) time embedding
        """
        args = delta[:, None] * self.freqs[None, :]  # (B, n_freqs)
        fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*n_freqs)
        return self.proj(fourier)


class AlphaGate(nn.Module):
    """Time gate α(Δ) = 1 - exp(-Δ / τ_0) with learnable τ_0.

    - α(0) = 0, so predicted ẑ = z_0 when Δ = 0 (identity preserved)
    - α(Δ) → 1 as Δ → ∞, bounded residual scaling
    - τ_0 initialized so α(Δ_median) ≈ 0.5 → τ_0_init = Δ_median / ln(2)
    """

    def __init__(self, tau_init: float = 1.0):
        super().__init__()
        # Store log(τ_0) for positivity via softplus-like parameterization
        self._log_tau = nn.Parameter(torch.log(torch.tensor(float(tau_init))))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self._log_tau)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.exp(-delta / self.tau)


class BRCellDriftMLP(nn.Module):
    """Barycentric-Residual Cell Drift model (MLP backbone).

    Outputs K stochastic predictions per source cell by sampling K noise vectors
    and centering them to zero mean within each source group.

    Args:
        dim: cell state dimension (e.g., 2 for Mouse toy, 50 for Clonidine PCA, 128 for full)
        hidden_dim: MLP hidden layer width
        n_layers: depth of b_θ and G_θ MLPs (number of hidden layers)
        noise_dim: dimensionality h of stochastic basis (h̃ ∈ R^h, G ∈ R^{d×h})
        time_emb_dim: dimension of Fourier time embedding
        action_dim: dimension of action embedding (0 = no action)
        tau_init: initial τ_0 for time gate (set to median Δ / ln 2)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        noise_dim: int = 32,
        time_emb_dim: int = 64,
        action_dim: int = 0,
        tau_init: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.noise_dim = noise_dim
        self.action_dim = action_dim

        self.time_embed = FourierTimeEmbedding(time_emb_dim)
        self.alpha_gate = AlphaGate(tau_init=tau_init)

        cond_dim = time_emb_dim + action_dim  # condition on time + action

        # b_θ: mean drift MLP
        self.b_net = self._build_mlp(
            in_dim=dim + cond_dim, out_dim=dim, hidden_dim=hidden_dim, n_layers=n_layers
        )

        # G_θ: stochastic basis matrix flattened as (d * h) output
        self.G_net = self._build_mlp(
            in_dim=dim + cond_dim,
            out_dim=dim * noise_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        # h_ψ: maps noise ε ∈ R^dim → h̃ ∈ R^noise_dim.
        # Paper-aligned: h is state-independent; G_θ already z-conditioned, so
        # adding z to h is redundant double non-linearity. Simplified form matches
        # Kaiming He's Drifting Model (arXiv:2602.04770) Sec 3.2.
        self.h_net = self._build_mlp(
            in_dim=dim, out_dim=noise_dim, hidden_dim=hidden_dim, n_layers=n_layers,
        )

    def _build_mlp(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def _make_cond(
        self,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build condition vector from time + optional action."""
        t_emb = self.time_embed(delta)  # (B, time_emb_dim)
        if self.action_dim > 0:
            if action is None:
                raise ValueError(f"action_dim={self.action_dim} but action=None")
            return torch.cat([t_emb, action], dim=-1)
        return t_emb

    def forward(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        epsilon: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with K noise samples per source.

        Args:
            z: (B, dim) source cells
            delta: (B,) time intervals
            epsilon: (B, K, dim) noise samples, K predictions per source
            action: (B, action_dim) optional action embedding

        Returns:
            ẑ: (B, K, dim) predicted future states
        """
        B, K, D = epsilon.shape
        assert z.shape == (B, D), f"z shape {z.shape} != ({B}, {D})"
        assert delta.shape == (B,), f"delta shape {delta.shape} != ({B},)"

        cond = self._make_cond(delta, action)  # (B, cond_dim)

        # Compute b_θ once per source
        zc = torch.cat([z, cond], dim=-1)  # (B, dim + cond_dim)
        b = self.b_net(zc)  # (B, dim)

        # Compute G_θ once per source
        G_flat = self.G_net(zc)  # (B, dim * noise_dim)
        G = G_flat.view(B, self.dim, self.noise_dim)  # (B, dim, noise_dim)

        # Compute h̃ for each noise sample with zero-mean centering.
        # h is state-independent: h_net(ε) → (B, K, noise_dim).
        h_raw = self.h_net(epsilon.reshape(B * K, D)).view(B, K, self.noise_dim)

        # Zero-mean center within each source group
        h_tilde = h_raw - h_raw.mean(dim=1, keepdim=True)  # (B, K, h)

        # Stochastic residual: G · h̃  =>  (B, K, dim)
        # For each (i, r), dispersion = G[i] @ h_tilde[i, r]
        # Using einsum: G[i, d, j] * h[i, r, j] -> (i, r, d)
        dispersion = torch.einsum("bdj,brj->brd", G, h_tilde)  # (B, K, dim)

        # Residual R_θ = b + G·h̃
        R = b[:, None, :] + dispersion  # (B, K, dim)

        # Apply time gate and add source
        alpha = self.alpha_gate(delta)  # (B,)
        z_hat = z[:, None, :] + alpha[:, None, None] * R  # (B, K, dim)

        return z_hat

    def predict_mean(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Deterministic mean prediction: ẑ_mean = z + α(Δ) · b_θ(z, Δ, a).

        Since h̃ is zero-mean centered, the K-sample mean of forward() equals this.
        Useful for point estimates and for L_down in PC variant.
        """
        cond = self._make_cond(delta, action)
        zc = torch.cat([z, cond], dim=-1)
        b = self.b_net(zc)
        alpha = self.alpha_gate(delta)
        return z + alpha[:, None] * b
