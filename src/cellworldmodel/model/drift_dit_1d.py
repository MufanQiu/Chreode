"""DriftDiT-1D: DiT backbone adapted for 1-D cell state inputs (M9 benchmark).

Architecture:
    ẑ = z + α(Δ) · R_θ(z, Δ, ε, a)

where R_θ is a DiT transformer over a short token sequence:
    sequence = [z_token, reg_0, ..., reg_{R-1}, ε_token, (action_token)]
    → DiT blocks with adaLN-Zero on c = time_emb(Δ) + action_emb(a)
    → take z_token output → Linear → Δz ∈ R^dim

Adapted from 3rdparty/drifting-model/model.py. Key differences from original
(image) DriftDiT:
  - NO PatchEmbed / unpatchify (input is a single non-spatial latent vector)
  - Tokens: source + register + noise (+ optional action), not image patches
  - Conditioning: time embedding (Δ) + optional action, NOT label/alpha/style
  - Output: residual Δz in same space as z, gated by α(Δ)
  - Output shape (B, K, dim) to match BRCellDriftMLP / PCCellDriftMLP interface

See agent/human-review/stage2-encoder-backbone-design.typ for design rationale.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellworldmodel.model.br_celldrift_bench import AlphaGate, FourierTimeEmbedding


# ---------------------------------------------------------------------------
# Building blocks (RMSNorm, RoPE, SwiGLU, Attention, DiTBlock) — copied from
# drifting-model/model.py with minimal changes. Kept self-contained so this
# module does not depend on the 3rdparty/drifting-model package (which is
# image-specific and pulls `einops`).
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 64, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(device),
            self.sin_cached[:, :, :seq_len, :].to(device),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, cos, sin):
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = False,
                 use_qk_norm: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.use_qk_norm:
            q = self.q_norm(q); k = self.k_norm(k)
        if rope_cos is not None:
            q, k = _apply_rope(q, k, rope_cos, rope_sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 use_qk_norm: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, use_qk_norm=use_qk_norm)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa), rope_cos, rope_sin,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


# ---------------------------------------------------------------------------
# DriftDiT1D: the 1-D cell-state variant
# ---------------------------------------------------------------------------


class DriftDiT1D(nn.Module):
    """DiT backbone for 1-D cell state residual generation.

    Architecture (per batch element, per noise sample):
      tokens = [src_token, reg_0, ..., reg_{R-1}, noise_token]
      c = time_emb(Δ)                  # adaLN-Zero conditioning vector
      for each block:
        x = x + gate_msa · attn(mod(norm(x)))
        x = x + gate_mlp · swiglu(mod(norm(x)))
      Δz = Linear(norm(x[src_token_position]))
      ẑ = z + α(Δ) · Δz

    Shape protocol (matches BRCellDriftMLP / PCCellDriftMLP):
      Input:
        z: (B, dim)
        delta: (B,)
        epsilon: (B, K, dim)
        action: optional (B, action_dim)
      Output:
        ẑ: (B, K, dim)

    We fold K along the batch dim internally (B*K, tokens, H) so a single
    forward call produces K predictions per source (same as the MLP variants).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        time_emb_dim: int = 256,
        action_dim: int = 0,
        tau_init: float = 1.0,
        state_chunk_dim: Optional[int] = None,
        learned_state_tokens: Optional[int] = None,
        use_rope: bool = True,
        time_embedding_mode: str = "legacy_fourier",
        time_delta_transform: str = "normalized",
        time_delta_scale: Optional[float] = None,
        mean_prediction_mode: str = "monte_carlo",
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_register_tokens = num_register_tokens
        self.action_dim = action_dim

        self.use_rope = use_rope
        if time_embedding_mode not in {"legacy_fourier", "time2vec", "bounded_lowfreq_fourier"}:
            raise ValueError(f"Unknown time_embedding_mode={time_embedding_mode!r}")
        if mean_prediction_mode not in {"monte_carlo", "zero_noise"}:
            raise ValueError(f"Unknown mean_prediction_mode={mean_prediction_mode!r}")
        self.time_embedding_mode = time_embedding_mode
        self.time_delta_transform = time_delta_transform
        self.time_delta_scale = float(time_delta_scale) if time_delta_scale is not None else float(tau_init) * math.log(2.0)
        self.mean_prediction_mode = mean_prediction_mode

        # DiT v2: dim-split src/noise into multiple state tokens to let attention
        # learn "gene-module / latent-subspace" interactions (GPT 5.4 Pro rec'd).
        # Default None → legacy single-token mode (back-compat with old ckpts).
        self.state_chunk_dim = state_chunk_dim
        self.learned_state_tokens = learned_state_tokens
        if learned_state_tokens is not None and learned_state_tokens > 1:
            self.tokenizer_mode = "learned"
            self.num_state_tokens = int(learned_state_tokens)
            self.src_proj = nn.Linear(dim, self.num_state_tokens * hidden_dim)
            self.noise_proj = nn.Linear(dim, self.num_state_tokens * hidden_dim)
            self.out_proj_dim = dim
            self.padded_dim = dim
        elif state_chunk_dim is None or state_chunk_dim >= dim:
            self.tokenizer_mode = "single"
            self.num_state_tokens = 1
            self.src_proj = nn.Linear(dim, hidden_dim)
            self.noise_proj = nn.Linear(dim, hidden_dim)
            self.out_proj_dim = dim
            self.padded_dim = dim
        else:
            self.tokenizer_mode = "hard_chunk"
            import math as _math
            self.num_state_tokens = _math.ceil(dim / state_chunk_dim)
            self.padded_dim = self.num_state_tokens * state_chunk_dim
            self.src_proj = nn.Linear(state_chunk_dim, hidden_dim)
            self.noise_proj = nn.Linear(state_chunk_dim, hidden_dim)
            self.out_proj_dim = state_chunk_dim

        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_dim) * 0.02)

        # Optional action token (Stage 3). Kept present so state_dict stays stable.
        if action_dim > 0:
            self.action_proj = nn.Linear(action_dim, hidden_dim)
        else:
            self.action_proj = None

        # Conditioning: time + optional action summary (action also attends in sequence)
        if time_embedding_mode == "legacy_fourier":
            self.time_embed = FourierTimeEmbedding(hidden_dim, n_freqs=time_emb_dim // 2)
        else:
            # Delay this import because Waddington-DiT reuses the blocks above.
            from cellworldmodel.model.waddington_dit_1d import (
                BoundedLowFreqFourierTimeEmbedding,
                Time2VecDeltaEmbedding,
            )
            if time_embedding_mode == "time2vec":
                self.time_embed = Time2VecDeltaEmbedding(
                    hidden_dim, n_periodic=max(1, time_emb_dim // 2),
                    delta_scale=self.time_delta_scale, transform=time_delta_transform,
                )
            else:
                self.time_embed = BoundedLowFreqFourierTimeEmbedding(
                    hidden_dim, delta_scale=self.time_delta_scale, transform=time_delta_transform,
                )
        if action_dim > 0:
            self.action_cond = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # α(Δ) gate (shared with BR / PC variants)
        self.alpha_gate = AlphaGate(tau_init=tau_init)

        # RoPE
        # state_tokens × 2 (src + noise) + register + optional action
        max_tokens = self.num_state_tokens * 2 + num_register_tokens + (1 if action_dim > 0 else 0)
        head_dim = hidden_dim // num_heads
        self.rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=max(max_tokens, 8))
        self._max_tokens = max_tokens

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Output head: take src token(s), project back to dim with adaLN modulation.
        # With state_chunk_dim, we project each src token → chunk_dim then concat.
        self.final_norm = RMSNorm(hidden_dim)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        self.final_linear = nn.Linear(hidden_dim, self.out_proj_dim)

        # Optional potential head U_theta(z, Δ, a) for Waddington landscape loss.
        # Small MLP on source-state + time embedding → scalar.
        # Disabled at init (returns 0), activated by caller via `use_potential=True`
        # in downhill_loss. Keeps state_dict stable if unused.
        self.U_net = nn.Sequential(
            nn.Linear(dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        def _basic(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic)

        # Zero-init adaLN so initial forward = identity shift (no modulation)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        # Small final linear so Δz starts small (not zero, to preserve gradient)
        nn.init.normal_(self.final_linear.weight, std=0.02)
        nn.init.zeros_(self.final_linear.bias)

    def _build_tokens(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
        action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fold (B, K) → (B*K, tokens, H).

        Source token repeats over K noise samples; register tokens are shared.
        """
        B, K, D = epsilon.shape
        BK = B * K

        if self.tokenizer_mode == "single":
            src_token = self.src_proj(z).unsqueeze(1)  # (B, 1, H)
            src_tokens = src_token.expand(B, K, -1).reshape(BK, 1, self.hidden_dim)
            noise_tokens = self.noise_proj(epsilon.reshape(BK, D)).unsqueeze(1)  # (BK, 1, H)
        elif self.tokenizer_mode == "learned":
            src_b = self.src_proj(z).view(B, self.num_state_tokens, self.hidden_dim)
            src_tokens = src_b.unsqueeze(1).expand(B, K, -1, -1).reshape(
                BK, self.num_state_tokens, self.hidden_dim,
            )
            noise_tokens = self.noise_proj(epsilon.reshape(BK, D)).view(
                BK, self.num_state_tokens, self.hidden_dim,
            )
        else:
            # Pad and chunk dim → (B, S, C) and (BK, S, C)
            pad = self.padded_dim - D
            if pad > 0:
                z_pad = F.pad(z, (0, pad))
                e_pad = F.pad(epsilon.reshape(BK, D), (0, pad))
            else:
                z_pad = z
                e_pad = epsilon.reshape(BK, D)
            z_chunks = z_pad.view(B, self.num_state_tokens, self.state_chunk_dim)
            e_chunks = e_pad.view(BK, self.num_state_tokens, self.state_chunk_dim)
            src_b = self.src_proj(z_chunks)  # (B, S, H)
            src_tokens = src_b.unsqueeze(1).expand(B, K, -1, -1).reshape(
                BK, self.num_state_tokens, self.hidden_dim,
            )
            noise_tokens = self.noise_proj(e_chunks)  # (BK, S, H)

        reg = self.register_tokens.expand(BK, -1, -1)  # (BK, R, H)

        pieces = [src_tokens, reg, noise_tokens]

        if self.action_proj is not None and action is not None:
            act_token = self.action_proj(action).unsqueeze(1).expand(B, K, -1).reshape(
                BK, 1, self.hidden_dim,
            )
            pieces.append(act_token)

        return torch.cat(pieces, dim=1)  # (BK, tokens, H)

    def _make_cond(self, delta: torch.Tensor, action: Optional[torch.Tensor], BK: int) -> torch.Tensor:
        """Conditioning vector c for adaLN-Zero. Shape (BK, H)."""
        c = self.time_embed(delta)  # (B, H)
        if self.action_proj is not None and action is not None:
            c = c + self.action_cond(action)
        # Expand to BK along K dim
        B = c.shape[0]
        K = BK // B
        return c.unsqueeze(1).expand(B, K, -1).reshape(BK, -1)

    def forward(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        epsilon: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            z: (B, dim) source cells
            delta: (B,) time intervals
            epsilon: (B, K, dim) noise samples
            action: (B, action_dim) optional

        Returns:
            ẑ: (B, K, dim)
        """
        B, K, D = epsilon.shape
        assert z.shape == (B, D), f"z shape {z.shape} != ({B}, {D})"
        assert delta.shape == (B,), f"delta shape {delta.shape} != ({B},)"

        BK = B * K
        tokens = self._build_tokens(z, epsilon, action)  # (BK, T, H)
        c = self._make_cond(delta, action, BK)            # (BK, H)
        if self.use_rope:
            rope_cos, rope_sin = self.rope(tokens.shape[1], tokens.device)
        else:
            rope_cos = rope_sin = None

        x = tokens
        for block in self.blocks:
            x = block(x, c, rope_cos, rope_sin)

        # Take source token(s), modulate, project to Δz
        shift, scale = self.final_modulation(c).chunk(2, dim=1)
        if self.tokenizer_mode == "single":
            src_out = x[:, 0, :]  # (BK, H)
            src_out = self.final_norm(src_out) * (1 + scale) + shift
            delta_z = self.final_linear(src_out)  # (BK, dim)
        elif self.tokenizer_mode == "learned":
            src_out = x[:, :self.num_state_tokens, :]  # (BK, S, H)
            src_out = self.final_norm(src_out) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            delta_tokens = self.final_linear(src_out)  # (BK, S, dim)
            delta_z = delta_tokens.mean(dim=1)  # dense learned slots vote on full residual
        else:
            src_out = x[:, :self.num_state_tokens, :]  # (BK, S, H)
            src_out = self.final_norm(src_out) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            delta_chunks = self.final_linear(src_out)  # (BK, S, chunk_dim)
            delta_z = delta_chunks.reshape(BK, -1)[:, :D]  # drop padding

        delta_z = delta_z.view(B, K, D)
        alpha = self.alpha_gate(delta)  # (B,)
        z_hat = z[:, None, :] + alpha[:, None, None] * delta_z
        return z_hat

    def compute_potential(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Waddington landscape U_theta(z, Δ, a) -> (B,) scalar potential.

        Used by downhill_loss (M10) to enforce U(ẑ) ≤ U(z).
        """
        t_emb = self.time_embed(delta)  # (B, H)
        if self.action_proj is not None and action is not None:
            t_emb = t_emb + self.action_cond(action)
        x = torch.cat([z, t_emb], dim=-1)  # (B, dim + H)
        return self.U_net(x).squeeze(-1)

    def predict_mean(
        self,
        z: torch.Tensor,
        delta: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        n_mc: int = 32,
        antithetic: bool = False,
    ) -> torch.Tensor:
        """Monte Carlo mean, or an explicitly configured zero-noise representative.

        DiT-1D's residual depends on ε through self-attention (no closed-form
        zero-mean centering like BR). So we approximate mean by averaging K
        forward passes by default. ``zero_noise`` evaluates one zero-noise
        residual without consuming RNG; it is not the mathematical expectation
        of this nonlinear generator. The method name retains the trainer API.
        """
        B = z.shape[0]
        if self.mean_prediction_mode == "zero_noise":
            eps = torch.zeros(B, 1, self.dim, device=z.device, dtype=z.dtype)
            return self.forward(z, delta, eps, action).squeeze(1)
        if antithetic:
            half = max(1, n_mc // 2)
            eps_half = torch.randn(B, half, self.dim, device=z.device, dtype=z.dtype)
            eps = torch.cat([eps_half, -eps_half], dim=1)
            if eps.shape[1] > n_mc:
                eps = eps[:, :n_mc, :]
        else:
            eps = torch.randn(B, n_mc, self.dim, device=z.device, dtype=z.dtype)
        return self.forward(z, delta, eps, action).mean(dim=1)


def DriftDiT1D_Tiny(dim: int, **kwargs) -> DriftDiT1D:
    """Tiny: 256 hidden, 6 depth, 4 heads, ~3M params (for Benchmark-scale smoke)."""
    return DriftDiT1D(
        dim=dim, hidden_dim=256, depth=6, num_heads=4,
        num_register_tokens=4, mlp_ratio=4.0, **kwargs,
    )


def DriftDiT1D_Small(dim: int, **kwargs) -> DriftDiT1D:
    """Small (default): 384 hidden, 12 depth, 6 heads, ~15M params (mouse embryo 2.4M target)."""
    return DriftDiT1D(
        dim=dim, hidden_dim=384, depth=12, num_heads=6,
        num_register_tokens=4, mlp_ratio=4.0, **kwargs,
    )


def DriftDiT1D_Base(dim: int, **kwargs) -> DriftDiT1D:
    """Base: 512 hidden, 16 depth, 8 heads, ~50M params (post-NeurIPS scale-up)."""
    return DriftDiT1D(
        dim=dim, hidden_dim=512, depth=16, num_heads=8,
        num_register_tokens=8, mlp_ratio=4.0, **kwargs,
    )


DRIFT_DIT_1D_MODELS = {
    "tiny": DriftDiT1D_Tiny,
    "small": DriftDiT1D_Small,
    "base": DriftDiT1D_Base,
}
