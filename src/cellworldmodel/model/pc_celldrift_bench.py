"""PC-CellDrift-MLP: Potential-Curl parameterization for benchmark experiments (M2, M8).

Architecture:
    ẑ = z_0 + α(Δ) · R_θ(z_0, Δ, ε, a)
    R_θ = −∇_z U_θ(z, Δ, a) + (A_θ(z, Δ, a) − A_θ(z, Δ, a)^T) z + σ_θ ⊙ ε

where:
    - U_θ: MLP → ℝ  (Waddington landscape, scalar potential)
    - A_θ: MLP → ℝ^(d×d) via low-rank parameterization (A = U V^T with U, V ∈ ℝ^(d×k))
      then S = A − A^T giving antisymmetric rotation generator
    - σ_θ: global learnable ∈ ℝ^d_+ via softplus, per-dim noise amplitude

Biology story:
    - −∇U: "downhill" — Waddington landscape gradient descent
    - (A−A^T)z: "circulation" — rotational / cell-cycle dynamics (Helmholtz decomposition)
    - σ⊙ε: "stochastic" — fate branching / intrinsic cell variability

Design decisions (see m2-m7-m8-decisions.typ):
    D1 A_θ: **low-rank** UV^T − VU^T with k=16 (cheap, sufficient for cell-cycle-like rotations)
    D2 σ_θ: **global** learnable vector (stable, state-independent)
    D5 V decomposition (V^∥ / V^⊥ routing): TODO for Phase C, M8 uses undecomposed V

Used as:
    - M2: PC + MMD/W2 only
    - M8: PC + MMD/W2 + drift loss V + L_down (U(ẑ) ≤ U(z))
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from cellworldmodel.model.br_celldrift_bench import AlphaGate, FourierTimeEmbedding


class PCCellDriftMLP(nn.Module):
    """Potential-Curl Cell Drift (MLP backbone).

    Args:
        dim: cell state dimension
        hidden_dim: MLP hidden width
        n_layers: MLP depth (number of hidden layers)
        curl_rank: k for low-rank A = UV^T (A-A^T gives antisymmetric rotation).
                   Default 16. Lower = cheaper but less expressive rotation.
        time_emb_dim: Fourier time embedding dim
        action_dim: action conditioning dim (0 = no action, for Stage 2)
        tau_init: initial τ_0 for AlphaGate
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        curl_rank: int = 16,
        time_emb_dim: int = 64,
        action_dim: int = 0,
        tau_init: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.curl_rank = curl_rank
        self.action_dim = action_dim

        self.time_embed = FourierTimeEmbedding(time_emb_dim)
        self.alpha_gate = AlphaGate(tau_init=tau_init)

        cond_dim = time_emb_dim + action_dim

        # U_θ: scalar potential
        self.U_net = self._build_mlp(
            in_dim=dim + cond_dim, out_dim=1,
            hidden_dim=hidden_dim, n_layers=n_layers,
        )

        # A_θ = U V^T (low-rank), with U, V ∈ ℝ^(d×k)
        # We output 2 × dim × curl_rank flattened, then reshape
        self.A_net = self._build_mlp(
            in_dim=dim + cond_dim,
            out_dim=2 * dim * curl_rank,
            hidden_dim=hidden_dim, n_layers=n_layers,
        )

        # σ: global learnable per-dim noise amplitude (positive via softplus)
        self._sigma_raw = nn.Parameter(torch.zeros(dim))

    def _build_mlp(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    @property
    def sigma(self) -> torch.Tensor:
        """σ ∈ ℝ^d_+ via softplus."""
        return torch.nn.functional.softplus(self._sigma_raw) + 1e-4

    def _make_cond(self, delta: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_embed(delta)
        if self.action_dim > 0:
            if action is None:
                raise ValueError(f"action_dim={self.action_dim} but action=None")
            return torch.cat([t_emb, action], dim=-1)
        return t_emb

    def compute_potential(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate U_θ(z, Δ, a) — scalar potential. Used for L_down."""
        cond = self._make_cond(delta, action)
        return self.U_net(torch.cat([z, cond], dim=-1)).squeeze(-1)  # (B,)

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
            epsilon: (B, K, dim) noise samples
            action: (B, action_dim) optional

        Returns:
            ẑ: (B, K, dim) predicted future states
        """
        B, K, D = epsilon.shape
        assert z.shape == (B, D), f"z shape {z.shape} != ({B}, {D})"
        assert delta.shape == (B,), f"delta shape {delta.shape} != ({B},)"

        cond = self._make_cond(delta, action)

        # Compute −∇_z U_θ (require_grad on z to take autograd)
        # Use enable_grad to allow this even under outer `torch.no_grad()` (eval/predict).
        # In training: create_graph=True for second-order gradients through U_net params.
        with torch.enable_grad():
            z_req = z.detach().requires_grad_(True)
            U = self.U_net(torch.cat([z_req, cond], dim=-1)).sum()
            grad_U = torch.autograd.grad(
                U, z_req, create_graph=self.training, retain_graph=True,
            )[0]  # (B, dim)
        if not self.training:
            grad_U = grad_U.detach()

        # Compute A = UV^T (low-rank), then S = A − A^T
        # A_flat has 2 × dim × curl_rank entries: first half is U, second half is V
        A_flat = self.A_net(torch.cat([z, cond], dim=-1))  # (B, 2 * dim * k)
        U_mat = A_flat[:, : self.dim * self.curl_rank].view(B, self.dim, self.curl_rank)
        V_mat = A_flat[:, self.dim * self.curl_rank:].view(B, self.dim, self.curl_rank)
        # A = U V^T, S = A − A^T = U V^T − V U^T
        # Compute S · z directly: S·z = U(V^T z) − V(U^T z)
        Vz = torch.einsum("bdk,bd->bk", V_mat, z)   # (B, k)
        Uz = torch.einsum("bdk,bd->bk", U_mat, z)   # (B, k)
        S_z = torch.einsum("bdk,bk->bd", U_mat, Vz) - torch.einsum("bdk,bk->bd", V_mat, Uz)  # (B, dim)

        # Deterministic part: −∇U + S z
        det = -grad_U + S_z  # (B, dim)

        # Stochastic part: σ ⊙ ε (broadcast over K)
        sigma = self.sigma  # (dim,)
        noise = sigma[None, None, :] * epsilon  # (B, K, dim)

        # R = det (shared across K) + noise (per-sample)
        R = det[:, None, :] + noise  # (B, K, dim)

        # Time gate
        alpha = self.alpha_gate(delta)  # (B,)
        z_hat = z[:, None, :] + alpha[:, None, None] * R  # (B, K, dim)

        return z_hat

    def predict_mean(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Deterministic mean (noise-free) prediction.

        ẑ_mean = z + α(Δ) · (−∇U + S·z)

        Used for L_down evaluation (U(ẑ_mean) ≤ U(z) rule).
        """
        cond = self._make_cond(delta, action)
        with torch.enable_grad():
            z_req = z.detach().requires_grad_(True)
            U = self.U_net(torch.cat([z_req, cond], dim=-1)).sum()
            grad_U = torch.autograd.grad(U, z_req, create_graph=False)[0].detach()

        A_flat = self.A_net(torch.cat([z, cond], dim=-1))
        B = z.shape[0]
        U_mat = A_flat[:, : self.dim * self.curl_rank].view(B, self.dim, self.curl_rank)
        V_mat = A_flat[:, self.dim * self.curl_rank:].view(B, self.dim, self.curl_rank)
        Vz = torch.einsum("bdk,bd->bk", V_mat, z)
        Uz = torch.einsum("bdk,bd->bk", U_mat, z)
        S_z = torch.einsum("bdk,bk->bd", U_mat, Vz) - torch.einsum("bdk,bk->bd", V_mat, Uz)

        det = -grad_U + S_z
        alpha = self.alpha_gate(delta)
        return z + alpha[:, None] * det


def downhill_loss(
    model: PCCellDriftMLP,
    z: torch.Tensor,
    z_hat: torch.Tensor,
    delta: torch.Tensor,
    action: Optional[torch.Tensor] = None,
    margin: float = 0.0,
) -> torch.Tensor:
    """L_down: penalize cases where U(ẑ) > U(z) − margin (prefer "downhill" motion).

    L_down = mean( max(0, U(ẑ) − U(z) + margin)² )

    Args:
        model: PCCellDriftMLP
        z: (B, dim) source
        z_hat: (B, K, dim) or (B, dim) predictions
        delta: (B,) time intervals
        action: optional
        margin: require U drop by at least `margin` (default 0 = any downhill)

    Returns:
        scalar loss
    """
    if z_hat.dim() == 3:
        # (B, K, dim) — repeat z, delta K times
        B, K, D = z_hat.shape
        z_rep = z[:, None, :].expand(B, K, D).reshape(B * K, D)
        delta_rep = delta[:, None].expand(B, K).reshape(B * K)
        action_rep = None
        if action is not None:
            action_rep = action[:, None, :].expand(B, K, action.shape[-1]).reshape(B * K, -1)
        z_hat_flat = z_hat.reshape(B * K, D)
    else:
        z_rep, delta_rep, action_rep, z_hat_flat = z, delta, action, z_hat

    U_z = model.compute_potential(z_rep, delta_rep, action_rep)
    U_z_hat = model.compute_potential(z_hat_flat, delta_rep, action_rep)
    return torch.mean(torch.clamp(U_z_hat - U_z + margin, min=0.0) ** 2)
