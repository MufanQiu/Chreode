"""Waddington-DiT residual generator with explicit potential-curl decomposition.

This keeps the biological residual formula from the 2026-04-17 slides:

    R_theta(z, Delta, epsilon) =
        -grad_z U_theta(z, Delta)
        + (A_theta(z, Delta) - A_theta(z, Delta)^T) z
        + sigma_theta * epsilon

but replaces the MLP feature extractor used by `PCCellDriftMLP` with the same
short-token DiT backbone used by M9/M10. The residual is still explicit and
interpretable; DiT only parameterizes U and low-rank A from (z, Delta).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from cellworldmodel.model.br_celldrift_bench import AlphaGate, FourierTimeEmbedding
from cellworldmodel.model.drift_dit_1d import (
    DiTBlock,
    RMSNorm,
    RotaryPositionEmbedding,
)


class BoundedLowFreqFourierTimeEmbedding(nn.Module):
    """Low-frequency Fourier features on dataset-normalized Delta.

    The legacy embedding uses raw Delta times log-spaced frequencies up to 1000.
    This variant keeps phase growth bounded for long ZESTA extrapolation by first
    normalizing Delta by the training time scale and then using period-based
    frequencies.
    """

    def __init__(
        self,
        out_dim: int,
        delta_scale: float,
        periods: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
        transform: str = "normalized",
    ) -> None:
        super().__init__()
        if transform not in {"normalized", "log1p"}:
            raise ValueError(f"transform={transform!r}; expected normalized/log1p")
        self.transform = transform
        self.register_buffer("delta_scale", torch.tensor(float(max(delta_scale, 1e-6))), persistent=False)
        freqs = torch.tensor([2.0 * math.pi / p for p in periods], dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(2 * len(periods), out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _scale_delta(self, delta: torch.Tensor) -> torch.Tensor:
        scale = self.delta_scale.to(device=delta.device, dtype=delta.dtype)
        if self.transform == "log1p":
            return torch.log1p(delta) / torch.log1p(scale)
        return delta / scale

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        x = self._scale_delta(delta)
        args = x[:, None] * self.freqs.to(device=delta.device, dtype=delta.dtype)[None, :]
        fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(fourier)


class Time2VecDeltaEmbedding(nn.Module):
    """Time2Vec-style Delta embedding with bounded low-frequency periodic terms."""

    def __init__(
        self,
        out_dim: int,
        n_periodic: int,
        delta_scale: float,
        transform: str = "normalized",
        min_period: float = 4.0,
        max_period: float = 128.0,
        max_omega: float = math.pi,
    ) -> None:
        super().__init__()
        if transform not in {"normalized", "log1p"}:
            raise ValueError(f"transform={transform!r}; expected normalized/log1p")
        self.transform = transform
        self.max_omega = float(max_omega)
        self.register_buffer("delta_scale", torch.tensor(float(max(delta_scale, 1e-6))), persistent=False)
        periods = torch.exp(torch.linspace(math.log(min_period), math.log(max_period), n_periodic))
        omega = 2.0 * math.pi / periods
        omega = torch.clamp(omega, max=0.95 * self.max_omega)
        self.raw_omega = nn.Parameter(torch.atanh(omega / self.max_omega))
        self.phase = nn.Parameter(torch.zeros(n_periodic))
        self.linear_weight = nn.Parameter(torch.ones(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))
        self.proj = nn.Sequential(
            nn.Linear(1 + n_periodic, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _scale_delta(self, delta: torch.Tensor) -> torch.Tensor:
        scale = self.delta_scale.to(device=delta.device, dtype=delta.dtype)
        if self.transform == "log1p":
            return torch.log1p(delta) / torch.log1p(scale)
        return delta / scale

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        x = self._scale_delta(delta)
        omega = self.max_omega * torch.tanh(self.raw_omega).to(device=delta.device, dtype=delta.dtype)
        phase = self.phase.to(device=delta.device, dtype=delta.dtype)
        periodic = torch.sin(x[:, None] * omega[None, :] + phase[None, :])
        linear_weight = self.linear_weight.to(device=delta.device, dtype=delta.dtype)
        linear_bias = self.linear_bias.to(device=delta.device, dtype=delta.dtype)
        linear = x[:, None] * linear_weight + linear_bias
        return self.proj(torch.cat([linear, periodic], dim=-1))


class WaddingtonDiT1D(nn.Module):
    """DiT feature extractor + explicit Waddington residual decomposition."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        time_emb_dim: int = 256,
        curl_rank: int = 16,
        action_dim: int = 0,
        tau_init: float = 1.0,
        use_rope: bool = True,
        curl_update: str = "additive",
        curl_time_mode: str = "full",
        hybrid_delta0: float = 36.0,
        hybrid_slope: float = 0.25,
        hard_delta0: float = 30.0,
        time_embedding_mode: str = "legacy_fourier",
        time_delta_transform: str = "normalized",
        time_delta_scale: float | None = None,
        curl_time_embedding_mode: str = "same",
        curl_time_delta_transform: str = "normalized",
        curl_time_delta_scale: float | None = None,
        growth_head: bool = False,
        growth_log_mass_clip: float = 8.0,
        growth_integration_steps: int = 1,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.curl_rank = curl_rank
        self.num_register_tokens = num_register_tokens
        self.action_dim = action_dim
        self.use_rope = use_rope
        valid_curl_updates = {
            "additive",
            "cayley_direct",
            "cayley_residual",
            "hybrid_delta",
            "hard_delta_cayley_residual",
        }
        if curl_update not in valid_curl_updates:
            raise ValueError(
                f"curl_update={curl_update!r}; expected one of {sorted(valid_curl_updates)}"
            )
        self.curl_update = curl_update
        if curl_time_mode not in {"full", "state_only", "separate"}:
            raise ValueError(f"curl_time_mode={curl_time_mode!r}; expected full/state_only/separate")
        self.curl_time_mode = curl_time_mode
        self.hybrid_delta0 = float(hybrid_delta0)
        self.hybrid_slope = float(hybrid_slope)
        self.hard_delta0 = float(hard_delta0)
        if time_embedding_mode not in {"legacy_fourier", "bounded_lowfreq_fourier", "time2vec"}:
            raise ValueError(
                f"time_embedding_mode={time_embedding_mode!r}; "
                "expected legacy_fourier/bounded_lowfreq_fourier/time2vec"
            )
        self.time_embedding_mode = time_embedding_mode
        self.time_delta_transform = time_delta_transform
        inferred_delta_scale = float(tau_init) * math.log(2.0)
        self.time_delta_scale = float(time_delta_scale) if time_delta_scale is not None else inferred_delta_scale
        if curl_time_embedding_mode not in {"same", "legacy_fourier", "bounded_lowfreq_fourier", "time2vec"}:
            raise ValueError(
                f"curl_time_embedding_mode={curl_time_embedding_mode!r}; "
                "expected same/legacy_fourier/bounded_lowfreq_fourier/time2vec"
            )
        if curl_time_mode == "separate" and curl_time_embedding_mode == "same":
            raise ValueError("curl_time_mode='separate' requires curl_time_embedding_mode != 'same'")
        self.curl_time_embedding_mode = curl_time_embedding_mode
        self.curl_time_delta_transform = curl_time_delta_transform
        self.curl_time_delta_scale = (
            float(curl_time_delta_scale) if curl_time_delta_scale is not None else self.time_delta_scale
        )
        self.growth_log_mass_clip = float(growth_log_mass_clip)
        if self.growth_log_mass_clip <= 0:
            raise ValueError("growth_log_mass_clip must be positive")
        self.growth_integration_steps = int(growth_integration_steps)
        if self.growth_integration_steps < 1:
            raise ValueError("growth_integration_steps must be positive")
        self.tokenizer_mode = "waddington_single"
        self.num_state_tokens = 1

        self.src_proj = nn.Linear(dim, hidden_dim)
        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_dim) * 0.02)
        self.action_proj = nn.Linear(action_dim, hidden_dim) if action_dim > 0 else None
        self.action_cond = (
            nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if action_dim > 0 else None
        )
        self.time_embed = self._make_time_embedding(
            time_embedding_mode,
            hidden_dim,
            time_emb_dim,
            self.time_delta_scale,
            time_delta_transform,
        )
        self.curl_time_embed = (
            None if curl_time_embedding_mode == "same" else self._make_time_embedding(
                curl_time_embedding_mode,
                hidden_dim,
                time_emb_dim,
                self.curl_time_delta_scale,
                curl_time_delta_transform,
            )
        )
        self.alpha_gate = AlphaGate(tau_init=tau_init)

        head_dim = hidden_dim // num_heads
        max_tokens = 1 + num_register_tokens + (1 if action_dim > 0 else 0)
        self.rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=max(max_tokens, 8))
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(hidden_dim)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )

        self.U_head = nn.Linear(hidden_dim, 1)
        self.A_head = nn.Linear(hidden_dim, 2 * dim * curl_rank)
        self.growth_head = nn.Linear(hidden_dim, 1) if growth_head else None
        self._sigma_raw = nn.Parameter(torch.zeros(dim))
        self._init_weights()

    def _make_time_embedding(
        self,
        mode: str,
        hidden_dim: int,
        time_emb_dim: int,
        delta_scale: float,
        transform: str,
    ) -> nn.Module:
        if mode == "legacy_fourier":
            return FourierTimeEmbedding(hidden_dim, n_freqs=time_emb_dim // 2)
        if mode == "bounded_lowfreq_fourier":
            return BoundedLowFreqFourierTimeEmbedding(
                hidden_dim,
                delta_scale=delta_scale,
                transform=transform,
            )
        return Time2VecDeltaEmbedding(
            hidden_dim,
            n_periodic=time_emb_dim // 2,
            delta_scale=delta_scale,
            transform=transform,
        )

    @property
    def sigma(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._sigma_raw) + 1e-4

    def _init_weights(self) -> None:
        def _basic(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.normal_(self.U_head.weight, std=0.02)
        nn.init.zeros_(self.U_head.bias)
        nn.init.normal_(self.A_head.weight, std=0.02)
        nn.init.zeros_(self.A_head.bias)
        if self.growth_head is not None:
            nn.init.normal_(self.growth_head.weight, std=1e-3)
            nn.init.zeros_(self.growth_head.bias)

    def _features(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        time_embed: nn.Module | None = None,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return DiT source-token feature h(z, Delta)."""
        B = z.shape[0]
        src = self.src_proj(z)
        src = src.unsqueeze(1)
        reg = self.register_tokens.expand(B, -1, -1)
        pieces = [src, reg]
        if self.action_proj is not None and action is not None:
            pieces.append(self.action_proj(action).unsqueeze(1))
        x = torch.cat(pieces, dim=1)
        c = (time_embed or self.time_embed)(delta)
        if self.action_cond is not None and action is not None:
            c = c + self.action_cond(action)
        if self.use_rope:
            rope_cos, rope_sin = self.rope(x.shape[1], x.device)
        else:
            rope_cos = rope_sin = None
        for block in self.blocks:
            x = block(x, c, rope_cos, rope_sin)
        shift, scale = self.final_modulation(c).chunk(2, dim=1)
        h = self.final_norm(x[:, 0, :]) * (1 + scale) + shift
        return h

    def _curl_from_features(self, h: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return S z and the low-rank factors defining S=A-A^T."""
        B = z.shape[0]
        a_flat = self.A_head(h)
        p_mat = a_flat[:, : self.dim * self.curl_rank].view(B, self.dim, self.curl_rank)
        q_mat = a_flat[:, self.dim * self.curl_rank:].view(B, self.dim, self.curl_rank)
        qz = torch.einsum("bdk,bd->bk", q_mat, z)
        pz = torch.einsum("bdk,bd->bk", p_mat, z)
        s_z = torch.einsum("bdk,bk->bd", p_mat, qz) - torch.einsum("bdk,bk->bd", q_mat, pz)
        return s_z, p_mat, q_mat

    def _curl_features(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        h_full: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.curl_time_mode == "state_only":
            return self._features(
                z,
                torch.zeros_like(delta),
                self.curl_time_embed,
                action,
            )
        if self.curl_time_mode == "separate":
            return self._features(
                z,
                delta,
                self.curl_time_embed,
                action,
            )
        return h_full

    def _cayley_rotate(
        self,
        z: torch.Tensor,
        alpha: torch.Tensor,
        p_mat: torch.Tensor,
        q_mat: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Cayley(alpha * (P Q^T - Q P^T)) to z.

        For antisymmetric S, Cayley(alpha S) is orthogonal:
            (I - alpha S / 2)^(-1) (I + alpha S / 2)
        so the curl component preserves ||z|| up to numerical solve error.
        """
        B, D = z.shape
        s_mat = torch.bmm(p_mat, q_mat.transpose(1, 2)) - torch.bmm(q_mat, p_mat.transpose(1, 2))
        scaled = 0.5 * alpha[:, None, None] * s_mat
        eye = torch.eye(D, device=z.device, dtype=z.dtype).expand(B, D, D)
        rhs = torch.bmm(eye + scaled, z.unsqueeze(-1))
        rotated = torch.linalg.solve(eye - scaled, rhs).squeeze(-1)
        return rotated

    def _deterministic_residual(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        components = self.decompose_deterministic_residual(
            z, delta, action=action, create_graph=self.training, detach_eval=not self.training
        )
        return components["det"]

    def decompose_deterministic_residual(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        create_graph: bool | None = None,
        detach_eval: bool = False,
        preserve_source_grad: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return deterministic residual components.

        This is used both by training and by diagnostics. `det` is
        `-grad_u + curl`.
        """
        if create_graph is None:
            create_graph = self.training
        with torch.enable_grad():
            z_req = z if preserve_source_grad and z.requires_grad else z.detach().requires_grad_(True)
            h = self._features(
                z_req,
                delta,
                action=action,
            )
            h_curl = self._curl_features(
                z_req,
                delta,
                h,
                action,
            )
            potential = self.U_head(h).sum()
            grad_u = torch.autograd.grad(
                potential,
                z_req,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            s_z, p_mat, q_mat = self._curl_from_features(h_curl, z_req)
            det = -grad_u + s_z
            potential_per_cell = self.U_head(h).squeeze(-1)
        out = {
            "neg_grad_u": -grad_u,
            "curl": s_z,
            "det": det,
            "potential": potential_per_cell,
            "p_mat": p_mat,
            "q_mat": q_mat,
        }
        if detach_eval:
            out = {k: v.detach() for k, v in out.items()}
        return out

    def forward(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        epsilon: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        preserve_source_grad: bool = False,
    ) -> torch.Tensor:
        if self.action_dim > 0 and action is None:
            raise ValueError("action is required when action_dim > 0")
        B, K, D = epsilon.shape
        assert z.shape == (B, D)
        assert D == self.dim
        alpha = self.alpha_gate(delta)
        comp = self.decompose_deterministic_residual(
            z, delta, action=action, create_graph=self.training, detach_eval=not self.training,
            preserve_source_grad=preserve_source_grad,
        )
        neg_grad_u = comp["neg_grad_u"]
        noise = self.sigma[None, None, :] * epsilon
        additive = z[:, None, :] + alpha[:, None, None] * (
            (neg_grad_u + comp["curl"])[:, None, :] + noise
        )
        if self.curl_update == "additive":
            return additive

        z_curl = self._cayley_rotate(z, alpha, comp["p_mat"], comp["q_mat"])
        cayley_direct = z_curl[:, None, :] + alpha[:, None, None] * (neg_grad_u[:, None, :] + noise)
        if self.curl_update == "hybrid_delta":
            beta = torch.sigmoid(self.hybrid_slope * (self.hybrid_delta0 - delta))
            return beta[:, None, None] * additive + (1.0 - beta[:, None, None]) * cayley_direct
        if self.curl_update == "cayley_direct":
            return cayley_direct

        safe_alpha = torch.clamp(alpha, min=1e-8)
        curl_residual = (z_curl - z) / safe_alpha[:, None]
        residual = (curl_residual + neg_grad_u)[:, None, :] + noise
        out = z[:, None, :] + alpha[:, None, None] * residual
        cayley_residual = torch.where(alpha[:, None, None] == 0, z[:, None, :], out)
        if self.curl_update == "hard_delta_cayley_residual":
            use_cayley = delta > self.hard_delta0
            return torch.where(use_cayley[:, None, None], cayley_residual, additive)
        return cayley_residual

    def predict_mean(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        n_mc: int = 32,
        antithetic: bool = False,
    ) -> torch.Tensor:
        del n_mc, antithetic
        if self.action_dim > 0 and action is None:
            raise ValueError("action is required when action_dim > 0")
        alpha = self.alpha_gate(delta)
        comp = self.decompose_deterministic_residual(
            z, delta, action=action, create_graph=False, detach_eval=True
        )
        if self.curl_update == "additive":
            det = comp["det"]
            return z + alpha[:, None] * det
        z_curl = self._cayley_rotate(z, alpha, comp["p_mat"], comp["q_mat"])
        cayley_direct = z_curl + alpha[:, None] * comp["neg_grad_u"]
        if self.curl_update == "hybrid_delta":
            additive = z + alpha[:, None] * comp["det"]
            beta = torch.sigmoid(self.hybrid_slope * (self.hybrid_delta0 - delta))
            return beta[:, None] * additive + (1.0 - beta[:, None]) * cayley_direct
        if self.curl_update == "cayley_direct":
            return cayley_direct
        safe_alpha = torch.clamp(alpha, min=1e-8)
        curl_residual = (z_curl - z) / safe_alpha[:, None]
        out = z + alpha[:, None] * (curl_residual + comp["neg_grad_u"])
        cayley_residual = torch.where(alpha[:, None] == 0, z, out)
        if self.curl_update == "hard_delta_cayley_residual":
            additive = z + alpha[:, None] * comp["det"]
            use_cayley = delta > self.hard_delta0
            return torch.where(use_cayley[:, None], cayley_residual, additive)
        return cayley_residual

    def compute_potential(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.action_dim > 0 and action is None:
            raise ValueError("action is required when action_dim > 0")
        h = self._features(z, delta, action=action)
        return self.U_head(h).squeeze(-1)

    def compute_growth(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        *,
        detach_features: bool = False,
    ) -> torch.Tensor:
        """Predict one scalar growth rate per source cell.

        Growth is intentionally independent of epsilon: all K stochastic
        descendants of one source cell share the same source-cell mass.
        """
        if self.growth_head is None:
            raise RuntimeError("growth_head is disabled for this model")
        if self.action_dim > 0 and action is None:
            raise ValueError("action is required when action_dim > 0")
        h = self._features(
            z,
            delta,
            action=action,
        )
        if detach_features:
            h = h.detach()
        return self.growth_head(h).squeeze(-1)

    def compute_log_mass(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        *,
        detach_features: bool = False,
    ) -> torch.Tensor:
        """Return finite-interval log mass, optionally integrated along a path."""
        if self.growth_integration_steps == 1:
            growth = self.compute_growth(
                z,
                delta,
                action=action,
                detach_features=detach_features,
            )
            log_mass = self.alpha_gate(delta) * growth
        else:
            step_delta = delta / float(self.growth_integration_steps)
            log_mass = torch.zeros_like(delta)
            for step in range(self.growth_integration_steps):
                if step == 0:
                    state = z
                else:
                    fraction = float(step) / float(self.growth_integration_steps)
                    partial_delta = delta * fraction
                    state = self.predict_mean(
                        z,
                        partial_delta,
                        action=action,
                    ).detach()
                growth = self.compute_growth(
                    state,
                    step_delta,
                    action=action,
                    detach_features=detach_features,
                )
                log_mass = log_mass + self.alpha_gate(step_delta) * growth
        return torch.clamp(
            log_mass,
            min=-self.growth_log_mass_clip,
            max=self.growth_log_mass_clip,
        )

    def predict_mass(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        *,
        detach_features: bool = False,
    ) -> torch.Tensor:
        """Return positive per-source masses for a finite transition."""
        return torch.exp(
            self.compute_log_mass(
                z,
                delta,
                action=action,
                detach_features=detach_features,
            )
        )

    def waddington_regularization(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        h = self._features(
            z,
            delta,
            action=action,
        )
        h_curl = self._curl_features(
            z,
            delta,
            h,
            action,
        )
        curl, p_mat, q_mat = self._curl_from_features(h_curl, z)
        return {
            "wdit_a_fro": p_mat.pow(2).mean() + q_mat.pow(2).mean(),
            "wdit_curl_sq": curl.pow(2).sum(dim=-1).mean(),
        }

    def growth_regularization(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        log_mass: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Return growth-energy, log-mass variance and gauge penalties."""
        if self.growth_head is None:
            raise RuntimeError("growth_head is disabled for this model")
        growth = self.compute_growth(z, delta, action=action)
        if log_mass is None:
            log_mass = self.compute_log_mass(z, delta, action=action)
        centered_log_mass = log_mass - log_mass.mean()
        return {
            "growth_energy": growth.pow(2).mean(),
            "growth_logvar": centered_log_mass.pow(2).mean(),
            "growth_gauge": log_mass.mean().pow(2),
        }


def WaddingtonDiT1D_Tiny(dim: int, **kwargs) -> WaddingtonDiT1D:
    return WaddingtonDiT1D(
        dim=dim, hidden_dim=256, depth=6, num_heads=4,
        num_register_tokens=4, mlp_ratio=4.0, **kwargs,
    )


def WaddingtonDiT1D_Small(dim: int, **kwargs) -> WaddingtonDiT1D:
    return WaddingtonDiT1D(
        dim=dim, hidden_dim=384, depth=12, num_heads=6,
        num_register_tokens=4, mlp_ratio=4.0, **kwargs,
    )


def WaddingtonDiT1D_Base(dim: int, **kwargs) -> WaddingtonDiT1D:
    return WaddingtonDiT1D(
        dim=dim, hidden_dim=512, depth=16, num_heads=8,
        num_register_tokens=8, mlp_ratio=4.0, **kwargs,
    )


WADDINGTON_DIT_1D_MODELS = {
    "tiny": WaddingtonDiT1D_Tiny,
    "small": WaddingtonDiT1D_Small,
    "base": WaddingtonDiT1D_Base,
}
