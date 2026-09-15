"""Action-conditioned state shifts and potential corrections."""
from __future__ import annotations

import torch
from torch import nn
from cellworldmodel.benchmark.common_metrics import sinkhorn_w2

class MLP(nn.Module):
    def __init__(self, dims: list[int], zero_last: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        if zero_last:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x)

class FieldSurgeryAdapter(nn.Module):
    """Fast kick + program-mixture low-rank potential reshaping."""

    def __init__(self, dim: int, n_conditions: int, k_programs: int = 8,
                 emb_dim: int = 32, basis_hidden: int = 64, kick_hidden: int = 128,
                 delta_conditioned: bool = False):
        super().__init__()
        self.n_conditions = n_conditions
        self.shared_slot = n_conditions  # used by the no-action arms
        self.cond_emb = nn.Embedding(n_conditions + 1, emb_dim)
        self.delta_conditioned = delta_conditioned
        # optionally let the program mixture depend on the horizon, so the
        # reshaped landscape can differ between d4 and d6 predictions
        self.prog_head = nn.Linear(emb_dim + (1 if delta_conditioned else 0), k_programs)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.kick = MLP([dim + emb_dim, kick_hidden, dim], zero_last=True)
        self.basis = nn.ModuleList(
            MLP([dim, basis_hidden, basis_hidden, 1]) for _ in range(k_programs)
        )

    def kicked(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        e = self.cond_emb(cond)
        return z + self.kick(torch.cat([z, e], dim=-1))

    def potential_correction(self, z: torch.Tensor, cond: torch.Tensor,
                             delta: torch.Tensor | None = None) -> torch.Tensor:
        """-grad_z of gamma * sum_k c_k(a[,Delta]) b_k(z);  [B, D]."""
        e = self.cond_emb(cond)
        if self.delta_conditioned:
            d = (delta if delta is not None else torch.zeros(z.shape[0], device=z.device))
            e = torch.cat([e, d[:, None] / 4.0], dim=-1)  # scale ~[0.5, 1]
        coeff = torch.softmax(self.prog_head(e), dim=-1)  # [B, K]
        with torch.enable_grad():
            zin = z if z.requires_grad else z.detach().requires_grad_(True)
            u = torch.stack([b(zin).squeeze(-1) for b in self.basis], dim=-1)  # [B, K]
            u_mix = (coeff * u).sum(-1).sum()
            grad = torch.autograd.grad(u_mix, zin, create_graph=self.training)[0]
        return -self.gamma * grad

class SurgeryModel(nn.Module):
    """Frozen-code base + external surgery. forward(z, delta, eps, cond) -> [B,K,D]."""

    def __init__(self, base: nn.Module, adapter: FieldSurgeryAdapter, tune_base: bool):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.tune_base = tune_base
        for p in self.base.parameters():
            p.requires_grad_(tune_base)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.tune_base:
            self.base.eval()  # a frozen base must not flip into train-mode behavior
        return self

    def forward(self, z, delta, eps, cond):
        z_plus = self.adapter.kicked(z, cond)
        out = self.base(z_plus, delta, eps)                       # [B, K, D]
        alpha = self.base.alpha_gate(delta)                       # [B]
        corr = self.adapter.potential_correction(z_plus, cond, delta)  # [B, D]
        return out + alpha[:, None, None] * corr[:, None, :]

class StaticShiftModel(nn.Module):
    """scGen-style: zhat = z + delta_a. No base, no time gate, no noise."""

    def __init__(self, dim: int, n_conditions: int):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(n_conditions + 1, dim))

    def forward(self, z, delta, eps, cond):
        out = z + self.delta[cond]
        return out[:, None, :].expand(-1, eps.shape[1], -1).contiguous()

def multiscale_mmd(x: torch.Tensor, y: torch.Tensor,
                   scales=(0.01, 0.1, 1.0, 10.0, 100.0)) -> torch.Tensor:
    def pdist2(a, b):
        return (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T

    dxx, dyy, dxy = pdist2(x, x), pdist2(y, y), pdist2(x, y)
    out = x.new_zeros(())
    for s in scales:
        out = out + (torch.exp(-dxx / (2 * s)).mean()
                     + torch.exp(-dyy / (2 * s)).mean()
                     - 2 * torch.exp(-dxy / (2 * s)).mean())
    return out

def population_loss(pred: torch.Tensor, target: torch.Tensor,
                    w_w2: float = 1.0, w_mmd: float = 1.0,
                    transport_objective: str = "legacy",
                    transport_blur: float = 0.05) -> tuple[torch.Tensor, dict]:
    if transport_objective == "legacy":
        w2 = sinkhorn_w2(pred, target, epsilon=0.05, num_iters=100)
    else:
        from cellworldmodel.training.transport_objective import training_transport

        w2 = training_transport(pred, target, {
            "transport_objective": transport_objective, "transport_blur": transport_blur})
    mmd = multiscale_mmd(pred, target)
    return w_w2 * w2 + w_mmd * mmd, {"w2": float(w2), "mmd": float(mmd)}
