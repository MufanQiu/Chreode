"""Population-level perturbation predictors with biological control structure."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PopulationPredictorOutput:
    z: torch.Tensor
    aux: dict[str, float]
    tensors: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class PopulationPredictorConfig:
    route: str
    latent_dim: int
    action_dim: int
    n_programs: int = 8
    disable_kick: bool = False
    disable_field: bool = False
    flat_action: bool = False
    adapter_components: str = "full"
    k_samples: int = 2
    calibrate_potential: bool = False
    rollout_steps: int = 4
    disable_rollout: bool = False
    disable_action_time: bool = False
    virtual_time_min: float = 0.25
    virtual_time_max: float = 1.75
    locked_time_transform: str = "log_bounded"
    locked_time_scale: float = 30.0


@dataclass(frozen=True)
class GeneGraphPriorConfig:
    mode: str = "none"
    edge_index: torch.Tensor | None = None
    edge_weight: torch.Tensor | None = None
    basis_weight: float = 0.0
    output_weight: float = 0.0


@dataclass(frozen=True)
class ResponseDecoderConfig:
    response_decoder: str = "none"
    n_genes: int = 0
    latent_dim: int = 0
    action_dim: int = 0
    response_programs: int = 32
    use_sparse_programs: bool = False
    nonnegative_basis: bool = False
    use_set_context: bool = False
    graph_prior: GeneGraphPriorConfig | None = None
    graph_layers: int = 2


def _zero_last_linear(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return


class ProgramCoefficientHead(nn.Module):
    def __init__(self, action_dim: int, n_programs: int, sparse: bool = False) -> None:
        super().__init__()
        self.sparse = bool(sparse)
        self.net = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, max(action_dim, n_programs)),
            nn.SiLU(),
            nn.Linear(max(action_dim, n_programs), n_programs),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        logits = self.net(action)
        if not self.sparse:
            return torch.softmax(logits, dim=-1)
        return sparsemax(logits, dim=-1)


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted = torch.sort(z, descending=True, dim=dim).values
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.dim()
    view[dim] = -1
    k = k.view(view)
    z_cumsum = z_sorted.cumsum(dim)
    support = 1 + k * z_sorted > z_cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (z_cumsum.gather(dim, k_z.long() - 1) - 1) / k_z.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


class ActionConditionedFieldSurgery(nn.Module):
    """Fast kick plus program-conditioned low-rank modification of a base field."""

    def __init__(
        self,
        *,
        base_transition: nn.Module | None,
        latent_dim: int,
        action_dim: int,
        n_programs: int = 8,
        disable_kick: bool = False,
        disable_field: bool = False,
        flat_action: bool = False,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_transition = base_transition
        self.disable_kick = bool(disable_kick)
        self.disable_field = bool(disable_field)
        self.flat_action = bool(flat_action)
        self.n_programs = int(n_programs)
        if self.base_transition is not None and freeze_base:
            for param in self.base_transition.parameters():
                param.requires_grad_(False)
            self.base_transition.eval()
        hidden = max(int(latent_dim) * 2, 128)
        self.kick_net = nn.Sequential(
            nn.LayerNorm(int(latent_dim) + int(action_dim)),
            nn.Linear(int(latent_dim) + int(action_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(latent_dim)),
        )
        self.coeff_head = ProgramCoefficientHead(int(action_dim), int(n_programs))
        self.field_basis = nn.Sequential(
            nn.LayerNorm(int(latent_dim)),
            nn.Linear(int(latent_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(n_programs) * int(latent_dim)),
        )
        self.flat_field = nn.Sequential(
            nn.LayerNorm(int(latent_dim) + int(action_dim)),
            nn.Linear(int(latent_dim) + int(action_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(latent_dim)),
        )
        self.tau_gate = nn.Sequential(
            nn.LayerNorm(int(latent_dim) + int(action_dim)),
            nn.Linear(int(latent_dim) + int(action_dim), hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        _zero_last_linear(self.kick_net)
        _zero_last_linear(self.field_basis)
        _zero_last_linear(self.flat_field)

    def _base_rollout(self, z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tau = 1.0 + torch.nn.functional.softplus(self.tau_gate(torch.cat([z, action], dim=-1))).squeeze(-1)
        if self.base_transition is None:
            return z, tau
        out = self.base_transition.predict_mean(z, tau, action=None, n_mc=2)
        return out, tau

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> PopulationPredictorOutput:
        h = torch.cat([z, action], dim=-1)
        if self.disable_kick:
            kick = torch.zeros_like(z)
            z0 = z
        else:
            kick = self.kick_net(h)
            z0 = z + kick
        z_base, tau = self._base_rollout(z0, action)
        if self.disable_field:
            field = torch.zeros_like(z)
            coeff_entropy = torch.zeros((), device=z.device)
        elif self.flat_action:
            field = self.flat_field(torch.cat([z_base, action], dim=-1))
            coeff_entropy = torch.zeros((), device=z.device)
        else:
            coeff = self.coeff_head(action)
            basis = self.field_basis(z_base).view(z.shape[0], self.n_programs, z.shape[1])
            field = torch.sum(coeff.unsqueeze(-1) * basis, dim=1)
            coeff_entropy = -(coeff * torch.log(coeff.clamp_min(1e-8))).sum(dim=1).mean()
        pred = z_base + field
        return PopulationPredictorOutput(pred, {
            "kick_norm": float(kick.norm(dim=1).mean().detach().cpu()),
            "field_norm": float(field.norm(dim=1).mean().detach().cpu()),
            "tau_mean": float(tau.mean().detach().cpu()),
            "program_entropy": float(coeff_entropy.detach().cpu()),
        })


class InternalWaddingtonFieldSurgery(nn.Module):
    """Action-conditioned adapters inside the Waddington residual components.

    This module differs from ActionConditionedFieldSurgery: action modifies the
    potential, curl factors, and noise scale before the Waddington update is
    formed, instead of adding a residual after a frozen rollout.
    """

    def __init__(
        self,
        *,
        base_transition: nn.Module,
        latent_dim: int,
        action_dim: int,
        n_programs: int = 8,
        components: str = "full",
        disable_kick: bool = False,
        calibrate_potential: bool = False,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        required = ("_features", "_curl_features", "_curl_from_features", "U_head", "alpha_gate", "sigma")
        missing = [name for name in required if not hasattr(base_transition, name)]
        if missing:
            raise ValueError(f"InternalWaddingtonFieldSurgery requires WaddingtonDiT-like base; missing {missing}")
        if components not in {"full", "u", "s", "sigma"}:
            raise ValueError("components must be one of {'full', 'u', 's', 'sigma'}")
        self.base_transition = base_transition
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.n_programs = int(n_programs)
        self.components = components
        self.disable_kick = bool(disable_kick)
        self.calibrate_potential = bool(calibrate_potential)
        if freeze_base:
            for param in self.base_transition.parameters():
                param.requires_grad_(False)
            self.base_transition.eval()
        hidden_dim = int(getattr(base_transition, "hidden_dim"))
        curl_rank = int(getattr(base_transition, "curl_rank"))
        hidden = max(self.latent_dim * 2, 128)
        self.coeff_head = ProgramCoefficientHead(self.action_dim, self.n_programs)
        self.u_basis = nn.Parameter(torch.zeros(self.n_programs, hidden_dim))
        self.p_basis = nn.Parameter(torch.zeros(self.n_programs, self.latent_dim, curl_rank))
        self.q_basis = nn.Parameter(torch.zeros(self.n_programs, self.latent_dim, curl_rank))
        self.sigma_basis = nn.Parameter(torch.zeros(self.n_programs, self.latent_dim))
        self.kick_net = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.action_dim),
            nn.Linear(self.latent_dim + self.action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.latent_dim),
        )
        self.tau_gate = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.action_dim),
            nn.Linear(self.latent_dim + self.action_dim, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.u_gain_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.action_dim),
            nn.Linear(self.latent_dim + self.action_dim, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        _zero_last_linear(self.kick_net)

    def _uses_u(self) -> bool:
        return self.components in {"full", "u"}

    def _uses_s(self) -> bool:
        return self.components in {"full", "s"}

    def _uses_sigma(self) -> bool:
        return self.components in {"full", "sigma"}

    @staticmethod
    def _curl_apply(p_mat: torch.Tensor, q_mat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        qz = torch.einsum("bdk,bd->bk", q_mat, z)
        pz = torch.einsum("bdk,bd->bk", p_mat, z)
        return torch.einsum("bdk,bk->bd", p_mat, qz) - torch.einsum("bdk,bk->bd", q_mat, pz)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> PopulationPredictorOutput:
        h_action = torch.cat([z, action], dim=-1)
        if self.disable_kick:
            kick = torch.zeros_like(z)
            z0 = z
        else:
            kick = self.kick_net(h_action)
            z0 = z + kick
        tau = 1.0 + torch.nn.functional.softplus(self.tau_gate(torch.cat([z0, action], dim=-1))).squeeze(-1)
        coeff = self.coeff_head(action)
        with torch.enable_grad():
            z_req = z0.requires_grad_(True)
            h = self.base_transition._features(z_req, tau, action=None)
            h_curl = self.base_transition._curl_features(z_req, tau, h, action=None)
            base_potential = self.base_transition.U_head(h).sum()
            if self._uses_u():
                u_vec = torch.einsum("bk,kh->bh", coeff, self.u_basis)
                delta_potential = (h * u_vec).sum()
            else:
                delta_potential = base_potential * 0.0
            grad_u = torch.autograd.grad(
                base_potential + delta_potential,
                z_req,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            _base_s, p_mat, q_mat = self.base_transition._curl_from_features(h_curl, z_req)
            if self._uses_s():
                p_delta = torch.einsum("bk,kdr->bdr", coeff, self.p_basis)
                q_delta = torch.einsum("bk,kdr->bdr", coeff, self.q_basis)
                p_mat = p_mat + p_delta
                q_mat = q_mat + q_delta
            curl = self._curl_apply(p_mat, q_mat, z_req)
            alpha = self.base_transition.alpha_gate(tau)
            if self.calibrate_potential:
                u_gain = torch.nn.functional.softplus(self.u_gain_head(torch.cat([z_req, action], dim=-1))).squeeze(-1)
            else:
                u_gain = torch.ones_like(alpha)
            det = -u_gain[:, None] * grad_u + curl
            if self._uses_sigma() and self.training:
                sigma_delta = torch.einsum("bk,kd->bd", coeff, self.sigma_basis)
                sigma = self.base_transition.sigma[None, :] * torch.exp(sigma_delta.clamp(-3.0, 3.0))
                det = det + sigma * torch.randn_like(det)
            pred = z_req + alpha[:, None] * det
        entropy = -(coeff * torch.log(coeff.clamp_min(1e-8))).sum(dim=1).mean()
        u_norm = torch.einsum("bk,kh->bh", coeff, self.u_basis).norm(dim=1).mean()
        s_norm = (
            torch.einsum("bk,kdr->bdr", coeff, self.p_basis).norm(dim=(1, 2)).mean()
            + torch.einsum("bk,kdr->bdr", coeff, self.q_basis).norm(dim=(1, 2)).mean()
        )
        sigma_norm = torch.einsum("bk,kd->bd", coeff, self.sigma_basis).norm(dim=1).mean()
        return PopulationPredictorOutput(pred, {
            "kick_norm": float(kick.norm(dim=1).mean().detach().cpu()),
            "tau_mean": float(tau.mean().detach().cpu()),
            "program_entropy": float(entropy.detach().cpu()),
            "u_adapter_norm": float(u_norm.detach().cpu()),
            "s_adapter_norm": float(s_norm.detach().cpu()),
            "sigma_adapter_norm": float(sigma_norm.detach().cpu()),
            "u_gain_mean": float(u_gain.mean().detach().cpu()),
        })


class KTVURolloutPredictor(nn.Module):
    """Kick + virtual action-time + delta-U rollout over a frozen Waddington field."""

    def __init__(
        self,
        *,
        base_transition: nn.Module,
        latent_dim: int,
        action_dim: int,
        n_programs: int = 8,
        rollout_steps: int = 4,
        disable_kick: bool = False,
        disable_field: bool = False,
        disable_rollout: bool = False,
        disable_action_time: bool = False,
        virtual_time_min: float = 0.25,
        virtual_time_max: float = 1.75,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        required = ("_features", "_curl_features", "_curl_from_features", "U_head", "alpha_gate")
        missing = [name for name in required if not hasattr(base_transition, name)]
        if missing:
            raise ValueError(f"KTVURolloutPredictor requires WaddingtonDiT-like base; missing {missing}")
        self.base_transition = base_transition
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.n_programs = int(n_programs)
        self.rollout_steps = max(1, int(rollout_steps))
        self.disable_kick = bool(disable_kick)
        self.disable_field = bool(disable_field)
        self.disable_rollout = bool(disable_rollout)
        self.disable_action_time = bool(disable_action_time)
        self.virtual_time_min = float(virtual_time_min)
        self.virtual_time_max = float(virtual_time_max)
        if self.virtual_time_max <= self.virtual_time_min:
            raise ValueError("virtual_time_max must be larger than virtual_time_min")
        if freeze_base:
            for param in self.base_transition.parameters():
                param.requires_grad_(False)
            self.base_transition.eval()
        hidden = max(self.latent_dim * 2, 128)
        self.coeff_head = ProgramCoefficientHead(self.action_dim, self.n_programs)
        self.kick_net = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.action_dim),
            nn.Linear(self.latent_dim + self.action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.latent_dim),
        )
        self.time_net = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.action_dim),
            nn.Linear(self.latent_dim + self.action_dim, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.delta_u_basis = nn.Sequential(
            nn.LayerNorm(self.latent_dim + 1),
            nn.Linear(self.latent_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_programs),
        )
        _zero_last_linear(self.kick_net)
        _zero_last_linear(self.delta_u_basis)

    def _virtual_time(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.disable_action_time:
            return torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        logits = self.time_net(torch.cat([z, action], dim=-1)).squeeze(-1)
        span = self.virtual_time_max - self.virtual_time_min
        return self.virtual_time_min + span * torch.sigmoid(logits)

    def _delta_potential(self, z: torch.Tensor, delta: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        if self.disable_field:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        basis = self.delta_u_basis(torch.cat([z, delta[:, None]], dim=-1))
        return (basis * coeff).sum()

    def _field_step(self, z: torch.Tensor, delta: torch.Tensor, coeff: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.enable_grad():
            z_req = z.requires_grad_(True)
            h = self.base_transition._features(z_req, delta, action=None)
            h_curl = self.base_transition._curl_features(z_req, delta, h, action=None)
            base_potential = self.base_transition.U_head(h).sum()
            delta_potential = self._delta_potential(z_req, delta, coeff)
            grad_u = torch.autograd.grad(
                base_potential + delta_potential,
                z_req,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            curl, _p_mat, _q_mat = self.base_transition._curl_from_features(h_curl, z_req)
            alpha = self.base_transition.alpha_gate(delta)
            pred = z_req + alpha[:, None] * (-grad_u + curl)
        return pred, {
            "alpha": alpha,
            "delta_potential": delta_potential.detach(),
            "curl_norm": curl.norm(dim=1).detach(),
            "grad_norm": grad_u.norm(dim=1).detach(),
        }

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> PopulationPredictorOutput:
        coeff = self.coeff_head(action)
        h = torch.cat([z, action], dim=-1)
        if self.disable_kick:
            kick = torch.zeros_like(z)
        else:
            kick = self.kick_net(h)
        z_cur = z + kick
        total_time = self._virtual_time(z_cur, action)
        if self.disable_rollout:
            step_stats: list[dict[str, torch.Tensor]] = []
        else:
            dt = total_time / float(self.rollout_steps)
            step_stats = []
            for _ in range(self.rollout_steps):
                z_cur, stats = self._field_step(z_cur, dt, coeff)
                step_stats.append(stats)
        entropy = -(coeff * torch.log(coeff.clamp_min(1e-8))).sum(dim=1).mean()
        delta_u_norm = torch.zeros((), device=z.device)
        curl_norm = torch.zeros((), device=z.device)
        grad_norm = torch.zeros((), device=z.device)
        if step_stats:
            delta_u_norm = torch.stack([s["delta_potential"].abs() for s in step_stats]).mean()
            curl_norm = torch.stack([s["curl_norm"].mean() for s in step_stats]).mean()
            grad_norm = torch.stack([s["grad_norm"].mean() for s in step_stats]).mean()
        return PopulationPredictorOutput(
            z_cur,
            {
                "kick_norm": float(kick.norm(dim=1).mean().detach().cpu()),
                "virtual_time_mean": float(total_time.mean().detach().cpu()),
                "virtual_time_std": float(total_time.std(unbiased=False).detach().cpu()),
                "rollout_steps": float(0 if self.disable_rollout else self.rollout_steps),
                "program_entropy": float(entropy.detach().cpu()),
                "delta_u_abs": float(delta_u_norm.detach().cpu()),
                "rollout_curl_norm": float(curl_norm.detach().cpu()),
                "rollout_grad_norm": float(grad_norm.detach().cpu()),
            },
            tensors={"program_coeff": coeff},
        )


class NativeUBridgePredictor(nn.Module):
    """Native U-branch bridge: kick + action-time + FiLM adapter on h_U."""

    def __init__(
        self,
        *,
        base_transition: nn.Module,
        latent_dim: int,
        action_dim: int,
        n_programs: int = 8,
        rollout_steps: int = 1,
        disable_kick: bool = False,
        disable_field: bool = False,
        disable_action_time: bool = False,
        virtual_time_min: float = 0.25,
        virtual_time_max: float = 1.75,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        required = ("_features", "_curl_features", "_curl_from_features", "U_head", "alpha_gate")
        missing = [name for name in required if not hasattr(base_transition, name)]
        if missing:
            raise ValueError(f"NativeUBridgePredictor requires WaddingtonDiT-like base; missing {missing}")
        self.base_transition = base_transition
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.n_programs = int(n_programs)
        self.rollout_steps = max(1, int(rollout_steps))
        self.disable_kick = bool(disable_kick)
        self.disable_field = bool(disable_field)
        self.disable_action_time = bool(disable_action_time)
        self.virtual_time_min = float(virtual_time_min)
        self.virtual_time_max = float(virtual_time_max)
        if self.virtual_time_max <= self.virtual_time_min:
            raise ValueError("virtual_time_max must be larger than virtual_time_min")
        if freeze_base:
            for param in self.base_transition.parameters():
                param.requires_grad_(False)
            self.base_transition.eval()
        hidden_dim = int(getattr(base_transition, "hidden_dim"))
        hidden = max(self.latent_dim * 2, 128)
        self.coeff_head = ProgramCoefficientHead(self.action_dim, self.n_programs)
        self.kick_net = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.action_dim),
            nn.Linear(self.latent_dim + self.action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.latent_dim),
        )
        self.time_net = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.action_dim),
            nn.Linear(self.latent_dim + self.action_dim, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.gamma_basis = nn.Parameter(torch.zeros(self.n_programs, hidden_dim))
        _zero_last_linear(self.kick_net)

    def _virtual_time(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.disable_action_time:
            return torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        logits = self.time_net(torch.cat([z, action], dim=-1)).squeeze(-1)
        span = self.virtual_time_max - self.virtual_time_min
        return self.virtual_time_min + span * torch.sigmoid(logits)

    def _adapt_u_features(self, h: torch.Tensor, coeff: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.disable_field:
            gamma = torch.zeros_like(h)
            return h, gamma
        gamma = torch.einsum("bp,ph->bh", coeff, self.gamma_basis)
        return h * (1.0 + gamma), gamma

    def _step(self, z: torch.Tensor, delta: torch.Tensor, coeff: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.enable_grad():
            z_req = z.requires_grad_(True)
            h_u = self.base_transition._features(z_req, delta, action=None)
            h_u_adapted, gamma = self._adapt_u_features(h_u, coeff)
            potential = self.base_transition.U_head(h_u_adapted).sum()
            grad_u = torch.autograd.grad(
                potential,
                z_req,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            h_curl = self.base_transition._curl_features(z_req, delta, h_u, action=None)
            curl, _p_mat, _q_mat = self.base_transition._curl_from_features(h_curl, z_req)
            alpha = self.base_transition.alpha_gate(delta)
            pred = z_req + alpha[:, None] * (-grad_u + curl)
        return pred, {
            "alpha": alpha.detach(),
            "gamma_norm": gamma.norm(dim=1).detach(),
            "curl_norm": curl.norm(dim=1).detach(),
            "grad_norm": grad_u.norm(dim=1).detach(),
        }

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> PopulationPredictorOutput:
        coeff = self.coeff_head(action)
        if self.disable_kick:
            kick = torch.zeros_like(z)
        else:
            kick = self.kick_net(torch.cat([z, action], dim=-1))
        z_cur = z + kick
        total_time = self._virtual_time(z_cur, action)
        dt = total_time / float(self.rollout_steps)
        step_stats = []
        for _ in range(self.rollout_steps):
            z_cur, stats = self._step(z_cur, dt, coeff)
            step_stats.append(stats)
        entropy = -(coeff * torch.log(coeff.clamp_min(1e-8))).sum(dim=1).mean()
        gamma_norm = torch.stack([s["gamma_norm"].mean() for s in step_stats]).mean()
        curl_norm = torch.stack([s["curl_norm"].mean() for s in step_stats]).mean()
        grad_norm = torch.stack([s["grad_norm"].mean() for s in step_stats]).mean()
        low = (total_time < self.virtual_time_min + 1e-3).to(total_time.dtype).mean()
        high = (total_time > self.virtual_time_max - 1e-3).to(total_time.dtype).mean()
        return PopulationPredictorOutput(
            z_cur,
            {
                "kick_norm": float(kick.norm(dim=1).mean().detach().cpu()),
                "virtual_time_mean": float(total_time.mean().detach().cpu()),
                "virtual_time_std": float(total_time.std(unbiased=False).detach().cpu()),
                "virtual_time_low_frac": float(low.detach().cpu()),
                "virtual_time_high_frac": float(high.detach().cpu()),
                "rollout_steps": float(self.rollout_steps),
                "program_entropy": float(entropy.detach().cpu()),
                "u_gamma_norm": float(gamma_norm.detach().cpu()),
                "rollout_curl_norm": float(curl_norm.detach().cpu()),
                "rollout_grad_norm": float(grad_norm.detach().cpu()),
            },
            tensors={"program_coeff": coeff},
        )


class TimeLockedNativeUBridgePredictor(NativeUBridgePredictor):
    """Native U-bridge whose W-DiT rollout time is locked to biological Delta.

    `NativeUBridgePredictor` learns a bounded virtual time from state/action.
    That is flexible, but it can bypass the perturbation-as-temporal-transition
    hypothesis. This variant keeps the action-conditioned U-branch adapter but
    makes rollout time an explicit monotone function of the observed Delta.
    """

    requires_delta = True

    def __init__(
        self,
        *,
        locked_time_transform: str = "log_bounded",
        locked_time_scale: float = 30.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if locked_time_transform not in {"raw", "divide", "bounded", "log_bounded"}:
            raise ValueError("locked_time_transform must be raw/divide/bounded/log_bounded")
        self.locked_time_transform = str(locked_time_transform)
        self.locked_time_scale = float(max(float(locked_time_scale), 1e-6))

    def _locked_time(self, delta: torch.Tensor) -> torch.Tensor:
        delta = delta.to(dtype=torch.float32).clamp_min(0.0)
        scale = torch.tensor(self.locked_time_scale, device=delta.device, dtype=delta.dtype)
        if self.locked_time_transform == "raw":
            return delta
        if self.locked_time_transform == "divide":
            return delta / scale
        if self.locked_time_transform == "bounded":
            x = (delta / scale).clamp(0.0, 1.0)
        else:
            x = (torch.log1p(delta) / torch.log1p(scale)).clamp(0.0, 1.0)
        span = self.virtual_time_max - self.virtual_time_min
        return self.virtual_time_min + span * x

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        delta: torch.Tensor | None = None,
    ) -> PopulationPredictorOutput:
        if delta is None:
            raise ValueError("TimeLockedNativeUBridgePredictor requires delta")
        coeff = self.coeff_head(action)
        if self.disable_kick:
            kick = torch.zeros_like(z)
        else:
            kick = self.kick_net(torch.cat([z, action], dim=-1))
        z_cur = z + kick
        total_time = self._locked_time(delta.to(device=z.device, dtype=z.dtype))
        dt = total_time / float(self.rollout_steps)
        step_stats = []
        for _ in range(self.rollout_steps):
            z_cur, stats = self._step(z_cur, dt, coeff)
            step_stats.append(stats)
        entropy = -(coeff * torch.log(coeff.clamp_min(1e-8))).sum(dim=1).mean()
        gamma_norm = torch.stack([s["gamma_norm"].mean() for s in step_stats]).mean()
        curl_norm = torch.stack([s["curl_norm"].mean() for s in step_stats]).mean()
        grad_norm = torch.stack([s["grad_norm"].mean() for s in step_stats]).mean()
        low = (total_time < self.virtual_time_min + 1e-3).to(total_time.dtype).mean()
        high = (total_time > self.virtual_time_max - 1e-3).to(total_time.dtype).mean()
        return PopulationPredictorOutput(
            z_cur,
            {
                "kick_norm": float(kick.norm(dim=1).mean().detach().cpu()),
                "locked_time_mean": float(total_time.mean().detach().cpu()),
                "locked_time_std": float(total_time.std(unbiased=False).detach().cpu()),
                "locked_time_low_frac": float(low.detach().cpu()),
                "locked_time_high_frac": float(high.detach().cpu()),
                "rollout_steps": float(self.rollout_steps),
                "program_entropy": float(entropy.detach().cpu()),
                "u_gamma_norm": float(gamma_norm.detach().cpu()),
                "rollout_curl_norm": float(curl_norm.detach().cpu()),
                "rollout_grad_norm": float(grad_norm.detach().cpu()),
            },
            tensors={"program_coeff": coeff},
        )


class SetFlowPredictor(nn.Module):
    """Program-conditioned set flow predictor with optional frozen base rollout."""

    def __init__(
        self,
        *,
        base_transition: nn.Module | None,
        latent_dim: int,
        action_dim: int,
        n_programs: int = 8,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_transition = base_transition
        if self.base_transition is not None and freeze_base:
            for param in self.base_transition.parameters():
                param.requires_grad_(False)
            self.base_transition.eval()
        hidden = max(int(latent_dim) * 2, 128)
        self.coeff_head = ProgramCoefficientHead(int(action_dim), int(n_programs))
        self.context_net = nn.Sequential(
            nn.LayerNorm(int(latent_dim) * 2),
            nn.Linear(int(latent_dim) * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(latent_dim)),
        )
        self.program_proj = nn.Linear(int(n_programs), int(latent_dim))
        self.flow_net = nn.Sequential(
            nn.LayerNorm(int(latent_dim) * 3),
            nn.Linear(int(latent_dim) * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(latent_dim)),
        )
        _zero_last_linear(self.flow_net)

    def _base_rollout(self, z: torch.Tensor) -> torch.Tensor:
        if self.base_transition is None:
            return z
        delta = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        with torch.no_grad():
            return self.base_transition.predict_mean(z, delta, action=None, n_mc=2)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> PopulationPredictorOutput:
        z_base = self._base_rollout(z)
        mean = z.mean(dim=0, keepdim=True).expand_as(z)
        std = z.std(dim=0, keepdim=True, unbiased=False).expand_as(z)
        context = self.context_net(torch.cat([mean, std], dim=-1))
        coeff = self.coeff_head(action)
        program = self.program_proj(coeff)
        residual = self.flow_net(torch.cat([z_base, context, program], dim=-1))
        pred = z_base + residual
        entropy = -(coeff * torch.log(coeff.clamp_min(1e-8))).sum(dim=1).mean()
        return PopulationPredictorOutput(pred, {
            "flow_norm": float(residual.norm(dim=1).mean().detach().cpu()),
            "program_entropy": float(entropy.detach().cpu()),
            "context_norm": float(context.norm(dim=1).mean().detach().cpu()),
        })


class HybridRolloutPopulationPredictor(nn.Module):
    """Population-route version of the existing hybrid kick-rollout head."""

    def __init__(
        self,
        *,
        base_transition: nn.Module,
        latent_dim: int,
        action_dim: int,
        k_samples: int = 2,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_transition = base_transition
        self.k_samples = int(k_samples)
        if freeze_base:
            for param in self.base_transition.parameters():
                param.requires_grad_(False)
            self.base_transition.eval()
        hidden = max(int(latent_dim) * 2, 128)
        self.kick_net = nn.Sequential(
            nn.LayerNorm(int(latent_dim) + int(action_dim)),
            nn.Linear(int(latent_dim) + int(action_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(latent_dim)),
        )
        self.gate_net = nn.Sequential(
            nn.LayerNorm(int(latent_dim) + int(action_dim)),
            nn.Linear(int(latent_dim) + int(action_dim), hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        _zero_last_linear(self.kick_net)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> PopulationPredictorOutput:
        h = torch.cat([z, action], dim=-1)
        kick = self.kick_net(h)
        z_kick = z + kick
        delta = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        z_dyn = self.base_transition.predict_mean(z_kick, delta, action=None, n_mc=self.k_samples)
        gamma = torch.sigmoid(self.gate_net(h))
        pred = (1.0 - gamma) * z_kick + gamma * z_dyn
        return PopulationPredictorOutput(pred, {
            "kick_norm": float(kick.norm(dim=1).mean().detach().cpu()),
            "gate_mean": float(gamma.mean().detach().cpu()),
        })


class GeneProgramResponseDecoder(nn.Module):
    """Gene-program residual decoder gated by perturbation and cell state."""

    def __init__(
        self,
        *,
        n_genes: int,
        latent_dim: int,
        action_dim: int,
        n_programs: int = 32,
        use_sparse_programs: bool = False,
        nonnegative_basis: bool = False,
        use_set_context: bool = False,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_programs = int(n_programs)
        self.nonnegative_basis = bool(nonnegative_basis)
        self.use_set_context = bool(use_set_context)
        hidden = max(int(latent_dim) * 2, 128)
        self.coeff_head = ProgramCoefficientHead(int(action_dim), int(n_programs), sparse=bool(use_sparse_programs))
        state_in = int(latent_dim) * (3 if self.use_set_context else 1)
        self.state_gate = nn.Sequential(
            nn.LayerNorm(state_in),
            nn.Linear(state_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(n_programs)),
            nn.Tanh(),
        )
        self.program_to_gene = nn.Parameter(torch.zeros(int(n_programs), int(n_genes)))

    def forward(self, coarse_x: torch.Tensor, z_ctx: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        coeff = self.coeff_head(action)
        if self.use_set_context:
            mean = z_ctx.mean(dim=0, keepdim=True).expand_as(z_ctx)
            std = z_ctx.std(dim=0, keepdim=True, unbiased=False).expand_as(z_ctx)
            state_in = torch.cat([z_ctx, mean, std], dim=-1)
        else:
            state_in = z_ctx
        state = self.state_gate(state_in)
        program_activity = coeff * state
        basis = torch.nn.functional.softplus(self.program_to_gene) if self.nonnegative_basis else self.program_to_gene
        delta = program_activity @ basis
        out = coarse_x + delta
        entropy = -(coeff * torch.log(coeff.clamp_min(1e-8))).sum(dim=1).mean()
        return out, {
            "program_decoder_delta_norm": float(delta.norm(dim=1).mean().detach().cpu()),
            "program_decoder_entropy": float(entropy.detach().cpu()),
        }

    def program_scores(self, x: torch.Tensor, control_mean: torch.Tensor) -> torch.Tensor:
        basis = torch.nn.functional.softplus(self.program_to_gene) if self.nonnegative_basis else self.program_to_gene
        return (x - control_mean[None, :]) @ basis.transpose(0, 1)


class SharedCoefficientResponseDecoder(nn.Module):
    """Observation head using action coefficients produced by the field predictor."""

    requires_predictor_tensors = True

    def __init__(
        self,
        *,
        n_genes: int,
        latent_dim: int,
        n_programs: int = 8,
        nonnegative_basis: bool = False,
        use_set_context: bool = False,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_programs = int(n_programs)
        self.nonnegative_basis = bool(nonnegative_basis)
        self.use_set_context = bool(use_set_context)
        hidden = max(int(latent_dim) * 2, 128)
        state_in = int(latent_dim) * (3 if self.use_set_context else 1)
        self.state_gate = nn.Sequential(
            nn.LayerNorm(state_in),
            nn.Linear(state_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_programs),
            nn.Tanh(),
        )
        self.program_to_gene = nn.Parameter(torch.zeros(self.n_programs, self.n_genes))

    def forward(
        self,
        coarse_x: torch.Tensor,
        z_ctx: torch.Tensor,
        action: torch.Tensor,
        predictor_tensors: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del action
        if predictor_tensors is None or "program_coeff" not in predictor_tensors:
            raise ValueError("SharedCoefficientResponseDecoder requires predictor_tensors['program_coeff']")
        coeff = predictor_tensors["program_coeff"]
        if self.use_set_context:
            mean = z_ctx.mean(dim=0, keepdim=True).expand_as(z_ctx)
            std = z_ctx.std(dim=0, keepdim=True, unbiased=False).expand_as(z_ctx)
            state_in = torch.cat([z_ctx, mean, std], dim=-1)
        else:
            state_in = z_ctx
        activity = coeff * self.state_gate(state_in)
        basis = torch.nn.functional.softplus(self.program_to_gene) if self.nonnegative_basis else self.program_to_gene
        delta = activity @ basis
        return coarse_x + delta, {
            "shared_decoder_delta_norm": float(delta.norm(dim=1).mean().detach().cpu()),
            "shared_decoder_coeff_entropy": float((-(coeff * torch.log(coeff.clamp_min(1e-8))).sum(dim=1).mean()).detach().cpu()),
        }

    def program_scores(self, x: torch.Tensor, control_mean: torch.Tensor) -> torch.Tensor:
        basis = torch.nn.functional.softplus(self.program_to_gene) if self.nonnegative_basis else self.program_to_gene
        return (x - control_mean[None, :]) @ basis.transpose(0, 1)


class SharedSignedResponseDecoder(nn.Module):
    """Signed up/down observation head using predictor-owned action coefficients."""

    requires_predictor_tensors = True

    def __init__(
        self,
        *,
        n_genes: int,
        latent_dim: int,
        n_programs: int = 8,
        graph_prior: GeneGraphPriorConfig | None = None,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_programs = int(n_programs)
        graph_prior = graph_prior or GeneGraphPriorConfig()
        self.graph_mode = str(graph_prior.mode)
        self.graph_basis_weight = float(graph_prior.basis_weight)
        if graph_prior.edge_index is not None and graph_prior.edge_weight is not None:
            self.register_buffer("graph_edge_index", graph_prior.edge_index.long(), persistent=False)
            self.register_buffer("graph_edge_weight", graph_prior.edge_weight.float(), persistent=False)
        else:
            self.register_buffer("graph_edge_index", torch.empty(2, 0, dtype=torch.long), persistent=False)
            self.register_buffer("graph_edge_weight", torch.empty(0, dtype=torch.float32), persistent=False)
        hidden = max(int(latent_dim) * 2, 128)
        self.state_up = nn.Sequential(
            nn.LayerNorm(int(latent_dim)),
            nn.Linear(int(latent_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_programs),
        )
        self.state_down = nn.Sequential(
            nn.LayerNorm(int(latent_dim)),
            nn.Linear(int(latent_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_programs),
        )
        self.up_basis = nn.Parameter(torch.zeros(self.n_programs, self.n_genes))
        self.down_basis = nn.Parameter(torch.zeros(self.n_programs, self.n_genes))

    def _has_graph(self) -> bool:
        return self.graph_mode != "none" and self.graph_edge_index.numel() > 0 and self.graph_basis_weight

    def _smooth_basis(self, basis: torch.Tensor) -> torch.Tensor:
        if not self._has_graph():
            return basis
        adj = torch.sparse_coo_tensor(
            self.graph_edge_index.to(device=basis.device),
            self.graph_edge_weight.to(device=basis.device, dtype=basis.dtype),
            size=(self.n_genes, self.n_genes),
            device=basis.device,
        ).coalesce()
        smooth = torch.sparse.mm(adj, basis.transpose(0, 1)).transpose(0, 1)
        return basis + self.graph_basis_weight * smooth

    def forward(
        self,
        coarse_x: torch.Tensor,
        z_ctx: torch.Tensor,
        action: torch.Tensor,
        predictor_tensors: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del action
        if predictor_tensors is None or "program_coeff" not in predictor_tensors:
            raise ValueError("SharedSignedResponseDecoder requires predictor_tensors['program_coeff']")
        coeff = predictor_tensors["program_coeff"]
        up = torch.nn.functional.softplus(self.state_up(z_ctx) - 5.0) * coeff
        down = torch.nn.functional.softplus(self.state_down(z_ctx) - 5.0) * coeff
        up_basis = self._smooth_basis(torch.nn.functional.softplus(self.up_basis))
        down_basis = self._smooth_basis(torch.nn.functional.softplus(self.down_basis))
        delta = up @ up_basis - down @ down_basis
        return coarse_x + delta, {
            "shared_signed_delta_norm": float(delta.norm(dim=1).mean().detach().cpu()),
            "shared_signed_coeff_entropy": float((-(coeff * torch.log(coeff.clamp_min(1e-8))).sum(dim=1).mean()).detach().cpu()),
            "shared_signed_up_norm": float(up.norm(dim=1).mean().detach().cpu()),
            "shared_signed_down_norm": float(down.norm(dim=1).mean().detach().cpu()),
        }

    def program_scores(self, x: torch.Tensor, control_mean: torch.Tensor) -> torch.Tensor:
        basis = self._smooth_basis(torch.nn.functional.softplus(self.up_basis + self.down_basis))
        return (x - control_mean[None, :]) @ basis.transpose(0, 1)


class GeneTokenResponseDecoder(nn.Module):
    """Lightweight gene-token residual decoder: delta_g = <q(z,a), e_g>."""

    def __init__(
        self,
        *,
        n_genes: int,
        latent_dim: int,
        action_dim: int,
        token_dim: int = 64,
        use_sparse_programs: bool = False,
        nonnegative_basis: bool = False,
        use_set_context: bool = False,
        graph_prior: GeneGraphPriorConfig | None = None,
    ) -> None:
        super().__init__()
        self.nonnegative_basis = bool(nonnegative_basis)
        self.use_set_context = bool(use_set_context)
        graph_prior = graph_prior or GeneGraphPriorConfig()
        if graph_prior.mode not in {"none", "basis", "output", "both"}:
            raise ValueError("graph_prior.mode must be one of {'none', 'basis', 'output', 'both'}")
        self.graph_mode = str(graph_prior.mode)
        self.graph_basis_weight = float(graph_prior.basis_weight)
        self.graph_output_weight = float(graph_prior.output_weight)
        self.gene_emb = nn.Parameter(torch.randn(int(n_genes), int(token_dim)) * 0.02)
        if graph_prior.edge_index is not None and graph_prior.edge_weight is not None:
            self.register_buffer("graph_edge_index", graph_prior.edge_index.long(), persistent=False)
            self.register_buffer("graph_edge_weight", graph_prior.edge_weight.float(), persistent=False)
        else:
            self.register_buffer("graph_edge_index", torch.empty(2, 0, dtype=torch.long), persistent=False)
            self.register_buffer("graph_edge_weight", torch.empty(0, dtype=torch.float32), persistent=False)
        hidden = max(int(latent_dim) * 2, 128)
        q_in = int(latent_dim) * (3 if self.use_set_context else 1) + int(action_dim)
        self.query = nn.Sequential(
            nn.LayerNorm(q_in),
            nn.Linear(q_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(token_dim)),
        )
        self.coeff_head = ProgramCoefficientHead(int(action_dim), int(token_dim), sparse=bool(use_sparse_programs))
        _zero_last_linear(self.query)

    def _has_graph(self) -> bool:
        return self.graph_mode != "none" and self.graph_edge_index.numel() > 0

    def _graph_adj(self, dtype: torch.dtype, device: torch.device, n_genes: int) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self.graph_edge_index.to(device=device),
            self.graph_edge_weight.to(device=device, dtype=dtype),
            size=(n_genes, n_genes),
            device=device,
        ).coalesce()

    def _gene_basis(self) -> torch.Tensor:
        basis = torch.nn.functional.softplus(self.gene_emb) if self.nonnegative_basis else self.gene_emb
        return self._smooth_gene_basis(basis)

    def _smooth_gene_basis(self, basis: torch.Tensor) -> torch.Tensor:
        if self._has_graph() and self.graph_mode in {"basis", "both"} and self.graph_basis_weight:
            n = basis.shape[0]
            adj = self._graph_adj(basis.dtype, basis.device, n)
            smooth = torch.sparse.mm(adj, basis)
            basis = basis + float(self.graph_basis_weight) * smooth
        return basis

    def _output_delta(self, delta: torch.Tensor) -> torch.Tensor:
        if not (self._has_graph() and self.graph_mode in {"output", "both"} and self.graph_output_weight):
            return delta
        n = delta.shape[1]
        adj = self._graph_adj(delta.dtype, delta.device, n)
        smooth = torch.sparse.mm(adj, delta.transpose(0, 1)).transpose(0, 1)
        return delta + float(self.graph_output_weight) * smooth

    def forward(self, coarse_x: torch.Tensor, z_ctx: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if self.use_set_context:
            mean = z_ctx.mean(dim=0, keepdim=True).expand_as(z_ctx)
            std = z_ctx.std(dim=0, keepdim=True, unbiased=False).expand_as(z_ctx)
            z_in = torch.cat([z_ctx, mean, std], dim=-1)
        else:
            z_in = z_ctx
        q = self.query(torch.cat([z_in, action], dim=-1))
        gate = self.coeff_head(action)
        q = q * gate
        basis = self._gene_basis()
        delta = q @ basis.transpose(0, 1)
        delta = self._output_delta(delta)
        return coarse_x + delta, {
            "gene_token_delta_norm": float(delta.norm(dim=1).mean().detach().cpu()),
            "gene_token_gate_entropy": float((-(gate * torch.log(gate.clamp_min(1e-8))).sum(dim=1).mean()).detach().cpu()),
            "gene_token_graph_basis_weight": self.graph_basis_weight if self.graph_mode in {"basis", "both"} else 0.0,
            "gene_token_graph_output_weight": self.graph_output_weight if self.graph_mode in {"output", "both"} else 0.0,
        }

    def program_scores(self, x: torch.Tensor, control_mean: torch.Tensor) -> torch.Tensor:
        basis = self._gene_basis()
        return (x - control_mean[None, :]) @ basis


class GeneGraphMPNNResponseDecoder(GeneTokenResponseDecoder):
    """Gene-token decoder with trainable message passing over the gene graph.

    The graph layers operate on the shared gene basis, not on a dense
    batch-by-gene tensor. This preserves the efficient low-rank readout while
    testing a stronger GEARS-style relational prior than fixed graph smoothing.
    """

    def __init__(
        self,
        *,
        n_genes: int,
        latent_dim: int,
        action_dim: int,
        token_dim: int = 64,
        use_sparse_programs: bool = False,
        nonnegative_basis: bool = False,
        use_set_context: bool = False,
        graph_prior: GeneGraphPriorConfig | None = None,
        graph_layers: int = 2,
    ) -> None:
        super().__init__(
            n_genes=n_genes,
            latent_dim=latent_dim,
            action_dim=action_dim,
            token_dim=token_dim,
            use_sparse_programs=use_sparse_programs,
            nonnegative_basis=nonnegative_basis,
            use_set_context=use_set_context,
            graph_prior=graph_prior,
        )
        self.graph_layers = int(graph_layers)
        self.graph_mpnn_weight = float((graph_prior or GeneGraphPriorConfig()).basis_weight or 0.05)
        self.graph_projs = nn.ModuleList([nn.Linear(int(token_dim), int(token_dim), bias=False) for _ in range(self.graph_layers)])
        self.graph_norms = nn.ModuleList([nn.LayerNorm(int(token_dim)) for _ in range(self.graph_layers)])

    def _gene_basis(self) -> torch.Tensor:
        basis = torch.nn.functional.softplus(self.gene_emb) if self.nonnegative_basis else self.gene_emb
        if not self._has_graph():
            return basis
        adj = self._graph_adj(basis.dtype, basis.device, basis.shape[0])
        for proj, norm in zip(self.graph_projs, self.graph_norms, strict=True):
            msg = torch.sparse.mm(adj, basis)
            basis = norm(basis + self.graph_mpnn_weight * torch.nn.functional.silu(proj(msg)))
        return basis

    def forward(self, coarse_x: torch.Tensor, z_ctx: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pred, info = super().forward(coarse_x, z_ctx, action)
        info["gene_graph_mpnn_layers"] = float(self.graph_layers)
        info["gene_graph_mpnn_weight"] = float(self.graph_mpnn_weight)
        return pred, info


class GeneMPNNResidualResponseDecoder(GeneGraphMPNNResponseDecoder):
    """Base graph-smoothed gene-token decoder plus a small MPNN residual."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.residual_logit = nn.Parameter(torch.tensor(-3.0))

    def _mpnn_basis(self) -> torch.Tensor:
        basis = torch.nn.functional.softplus(self.gene_emb) if self.nonnegative_basis else self.gene_emb
        if not self._has_graph():
            return basis
        adj = self._graph_adj(basis.dtype, basis.device, basis.shape[0])
        for proj, norm in zip(self.graph_projs, self.graph_norms, strict=True):
            msg = torch.sparse.mm(adj, basis)
            basis = norm(basis + self.graph_mpnn_weight * torch.nn.functional.silu(proj(msg)))
        return basis

    def forward(self, coarse_x: torch.Tensor, z_ctx: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if self.use_set_context:
            mean = z_ctx.mean(dim=0, keepdim=True).expand_as(z_ctx)
            std = z_ctx.std(dim=0, keepdim=True, unbiased=False).expand_as(z_ctx)
            z_in = torch.cat([z_ctx, mean, std], dim=-1)
        else:
            z_in = z_ctx
        q = self.query(torch.cat([z_in, action], dim=-1))
        gate = self.coeff_head(action)
        q = q * gate
        base_delta = q @ GeneTokenResponseDecoder._gene_basis(self).transpose(0, 1)
        base_delta = self._output_delta(base_delta)
        residual_delta = q @ self._mpnn_basis().transpose(0, 1)
        gamma = torch.sigmoid(self.residual_logit)
        delta = base_delta + gamma * residual_delta
        return coarse_x + delta, {
            "gene_token_delta_norm": float(delta.norm(dim=1).mean().detach().cpu()),
            "gene_token_gate_entropy": float((-(gate * torch.log(gate.clamp_min(1e-8))).sum(dim=1).mean()).detach().cpu()),
            "gene_graph_mpnn_layers": float(self.graph_layers),
            "gene_graph_mpnn_weight": float(self.graph_mpnn_weight),
            "gene_graph_mpnn_residual_gamma": float(gamma.detach().cpu()),
        }


class SignedGeneTokenResponseDecoder(GeneTokenResponseDecoder):
    """Signed up/down gene-token decoder with nonnegative response programs."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        token_dim = int(self.gene_emb.shape[1])
        self.down_gene_emb = nn.Parameter(torch.randn_like(self.gene_emb) * 0.02)
        q_in = int(self.query[0].normalized_shape[0])
        hidden = int(self.query[1].out_features)
        self.down_query = nn.Sequential(
            nn.LayerNorm(q_in),
            nn.Linear(q_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, token_dim),
        )
        _zero_last_linear(self.down_query)

    def _down_basis(self) -> torch.Tensor:
        basis = torch.nn.functional.softplus(self.down_gene_emb)
        return self._smooth_gene_basis(basis)

    def forward(self, coarse_x: torch.Tensor, z_ctx: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if self.use_set_context:
            mean = z_ctx.mean(dim=0, keepdim=True).expand_as(z_ctx)
            std = z_ctx.std(dim=0, keepdim=True, unbiased=False).expand_as(z_ctx)
            z_in = torch.cat([z_ctx, mean, std], dim=-1)
        else:
            z_in = z_ctx
        h = torch.cat([z_in, action], dim=-1)
        gate = self.coeff_head(action)
        up = torch.nn.functional.softplus(self.query(h) - 5.0) * gate
        down = torch.nn.functional.softplus(self.down_query(h) - 5.0) * gate
        delta = up @ self._gene_basis().transpose(0, 1) - down @ self._down_basis().transpose(0, 1)
        delta = self._output_delta(delta)
        return coarse_x + delta, {
            "gene_token_delta_norm": float(delta.norm(dim=1).mean().detach().cpu()),
            "gene_token_gate_entropy": float((-(gate * torch.log(gate.clamp_min(1e-8))).sum(dim=1).mean()).detach().cpu()),
            "signed_up_norm": float(up.norm(dim=1).mean().detach().cpu()),
            "signed_down_norm": float(down.norm(dim=1).mean().detach().cpu()),
        }


class SignedGeneMPNNResidualResponseDecoder(SignedGeneTokenResponseDecoder):
    """Signed up/down decoder plus a small graph-MPNN residual branch."""

    def __init__(self, *, graph_layers: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        token_dim = int(self.gene_emb.shape[1])
        self.graph_layers = int(graph_layers)
        self.graph_mpnn_weight = float(self.graph_basis_weight or 0.05)
        self.graph_projs = nn.ModuleList([nn.Linear(token_dim, token_dim, bias=False) for _ in range(self.graph_layers)])
        self.graph_norms = nn.ModuleList([nn.LayerNorm(token_dim) for _ in range(self.graph_layers)])
        self.residual_logit = nn.Parameter(torch.tensor(-3.0))

    def _mpnn_basis(self) -> torch.Tensor:
        basis = torch.nn.functional.softplus(self.gene_emb)
        if not self._has_graph():
            return basis
        adj = self._graph_adj(basis.dtype, basis.device, basis.shape[0])
        for proj, norm in zip(self.graph_projs, self.graph_norms, strict=True):
            msg = torch.sparse.mm(adj, basis)
            basis = norm(basis + self.graph_mpnn_weight * torch.nn.functional.silu(proj(msg)))
        return basis

    def forward(self, coarse_x: torch.Tensor, z_ctx: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pred, info = super().forward(coarse_x, z_ctx, action)
        if self.use_set_context:
            mean = z_ctx.mean(dim=0, keepdim=True).expand_as(z_ctx)
            std = z_ctx.std(dim=0, keepdim=True, unbiased=False).expand_as(z_ctx)
            z_in = torch.cat([z_ctx, mean, std], dim=-1)
        else:
            z_in = z_ctx
        gate = self.coeff_head(action)
        q = self.query(torch.cat([z_in, action], dim=-1)) * gate
        residual_delta = q @ self._mpnn_basis().transpose(0, 1)
        gamma = torch.sigmoid(self.residual_logit)
        pred = pred + gamma * residual_delta
        info["gene_graph_mpnn_layers"] = float(self.graph_layers)
        info["gene_graph_mpnn_weight"] = float(self.graph_mpnn_weight)
        info["gene_graph_mpnn_residual_gamma"] = float(gamma.detach().cpu())
        return pred, info


def build_population_predictor(
    *,
    config: PopulationPredictorConfig,
    base_transition: nn.Module | None,
) -> nn.Module:
    if config.route == "route1_field":
        return ActionConditionedFieldSurgery(
            base_transition=base_transition,
            latent_dim=int(config.latent_dim),
            action_dim=int(config.action_dim),
            n_programs=int(config.n_programs),
            disable_kick=bool(config.disable_kick),
            disable_field=bool(config.disable_field),
            flat_action=bool(config.flat_action),
        )
    if config.route == "route1_internal":
        if base_transition is None:
            raise ValueError("route1_internal requires a Waddington-DiT base_transition")
        return InternalWaddingtonFieldSurgery(
            base_transition=base_transition,
            latent_dim=int(config.latent_dim),
            action_dim=int(config.action_dim),
            n_programs=int(config.n_programs),
            components=str(config.adapter_components),
            disable_kick=bool(config.disable_kick),
            calibrate_potential=bool(config.calibrate_potential),
        )
    if config.route == "ktvu_rollout":
        if base_transition is None:
            raise ValueError("ktvu_rollout requires a Waddington-DiT base_transition")
        return KTVURolloutPredictor(
            base_transition=base_transition,
            latent_dim=int(config.latent_dim),
            action_dim=int(config.action_dim),
            n_programs=int(config.n_programs),
            rollout_steps=int(config.rollout_steps),
            disable_kick=bool(config.disable_kick),
            disable_field=bool(config.disable_field),
            disable_rollout=bool(config.disable_rollout),
            disable_action_time=bool(config.disable_action_time),
            virtual_time_min=float(config.virtual_time_min),
            virtual_time_max=float(config.virtual_time_max),
        )
    if config.route == "native_u_bridge":
        if base_transition is None:
            raise ValueError("native_u_bridge requires a Waddington-DiT base_transition")
        return NativeUBridgePredictor(
            base_transition=base_transition,
            latent_dim=int(config.latent_dim),
            action_dim=int(config.action_dim),
            n_programs=int(config.n_programs),
            rollout_steps=int(config.rollout_steps),
            disable_kick=bool(config.disable_kick),
            disable_field=bool(config.disable_field),
            disable_action_time=bool(config.disable_action_time),
            virtual_time_min=float(config.virtual_time_min),
            virtual_time_max=float(config.virtual_time_max),
        )
    if config.route == "time_locked_native_u":
        if base_transition is None:
            raise ValueError("time_locked_native_u requires a Waddington-DiT base_transition")
        return TimeLockedNativeUBridgePredictor(
            base_transition=base_transition,
            latent_dim=int(config.latent_dim),
            action_dim=int(config.action_dim),
            n_programs=int(config.n_programs),
            rollout_steps=int(config.rollout_steps),
            disable_kick=bool(config.disable_kick),
            disable_field=bool(config.disable_field),
            disable_action_time=True,
            virtual_time_min=float(config.virtual_time_min),
            virtual_time_max=float(config.virtual_time_max),
            locked_time_transform=str(config.locked_time_transform),
            locked_time_scale=float(config.locked_time_scale),
        )
    if config.route == "route2_setflow":
        return SetFlowPredictor(
            base_transition=base_transition,
            latent_dim=int(config.latent_dim),
            action_dim=int(config.action_dim),
            n_programs=int(config.n_programs),
        )
    if config.route == "hybrid_rollout":
        if base_transition is None:
            raise ValueError("hybrid_rollout requires a base_transition")
        return HybridRolloutPopulationPredictor(
            base_transition=base_transition,
            latent_dim=int(config.latent_dim),
            action_dim=int(config.action_dim),
            k_samples=int(config.k_samples),
        )
    raise ValueError(f"Unknown population perturbation route={config.route!r}")


def build_response_decoder(config: ResponseDecoderConfig) -> nn.Module | None:
    if config.response_decoder == "none":
        return None
    if config.response_decoder == "program":
        return GeneProgramResponseDecoder(
            n_genes=int(config.n_genes),
            latent_dim=int(config.latent_dim),
            action_dim=int(config.action_dim),
            n_programs=int(config.response_programs),
            use_sparse_programs=bool(config.use_sparse_programs),
            nonnegative_basis=bool(config.nonnegative_basis),
            use_set_context=bool(config.use_set_context),
        )
    if config.response_decoder == "shared_program":
        return SharedCoefficientResponseDecoder(
            n_genes=int(config.n_genes),
            latent_dim=int(config.latent_dim),
            n_programs=int(config.response_programs),
            nonnegative_basis=bool(config.nonnegative_basis),
            use_set_context=bool(config.use_set_context),
        )
    if config.response_decoder == "shared_signed":
        return SharedSignedResponseDecoder(
            n_genes=int(config.n_genes),
            latent_dim=int(config.latent_dim),
            n_programs=int(config.response_programs),
            graph_prior=config.graph_prior,
        )
    if config.response_decoder in {"gene_token", "gene_mpnn", "gene_mpnn_residual", "signed_gene_token", "signed_gene_mpnn_residual"}:
        if config.response_decoder == "gene_mpnn":
            cls = GeneGraphMPNNResponseDecoder
        elif config.response_decoder == "gene_mpnn_residual":
            cls = GeneMPNNResidualResponseDecoder
        elif config.response_decoder == "signed_gene_token":
            cls = SignedGeneTokenResponseDecoder
        elif config.response_decoder == "signed_gene_mpnn_residual":
            cls = SignedGeneMPNNResidualResponseDecoder
        else:
            cls = GeneTokenResponseDecoder
        return cls(
            n_genes=int(config.n_genes),
            latent_dim=int(config.latent_dim),
            action_dim=int(config.action_dim),
            token_dim=int(config.response_programs),
            use_sparse_programs=bool(config.use_sparse_programs),
            nonnegative_basis=bool(config.nonnegative_basis),
            use_set_context=bool(config.use_set_context),
            graph_prior=config.graph_prior,
            **({"graph_layers": int(config.graph_layers)} if config.response_decoder in {"gene_mpnn", "gene_mpnn_residual", "signed_gene_mpnn_residual"} else {}),
        )
    raise ValueError(f"Unknown response_decoder={config.response_decoder!r}")
