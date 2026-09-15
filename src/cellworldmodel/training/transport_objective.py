"""Optional transport objectives; the historical metric kernel stays unchanged."""
from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint

from cellworldmodel.benchmark.common_metrics import _pairwise_sq_dists, sinkhorn_w2


def converged_proxy(x, y, *, epsilon=0.05, max_iters=20000, tolerance=1e-4,
                    weight_x=None, return_info=False):
    """Original entropic transport cost, with both marginal residuals checked.

    Checkpointed iteration blocks preserve derivatives through the plan. No
    detached-plan gradient approximation is used. Failure to converge is fatal.
    """
    if epsilon <= 0 or tolerance <= 0 or max_iters < 1:
        raise ValueError("epsilon, tolerance and max_iters must be positive")
    cost = _pairwise_sq_dists(x, y)
    # Double-precision potentials avoid cancellation at low regularization.
    kernel = -cost.double() / epsilon
    a = (torch.full((len(x),), 1 / len(x), device=x.device, dtype=torch.float64)
         if weight_x is None else weight_x.double().clamp_min(1e-12))
    a = a / a.sum()
    b = torch.full((len(y),), 1 / len(y), device=y.device, dtype=torch.float64)
    log_a, log_b = a.log(), b.log()
    u, v = torch.zeros_like(a), torch.zeros_like(b)

    def block(k, lu, lv, la, lb, count):
        for _ in range(count):
            lu = la - torch.logsumexp(k + lv[None, :], dim=1)
            lv = lb - torch.logsumexp(k + lu[:, None], dim=0)
        return lu, lv

    residual = math.inf
    for start in range(0, max_iters, 50):
        count = min(50, max_iters - start)
        u, v = checkpoint(block, kernel, u, v, log_a, log_b, count,
                          use_reentrant=False, preserve_rng_state=False)
        with torch.no_grad():
            plan = torch.exp(u[:, None] + kernel + v[None, :])
            source_error = (plan.sum(1) - a).abs().sum().item()
            target_error = (plan.sum(0) - b).abs().sum().item()
            residual = max(source_error, target_error)
        if residual <= tolerance:
            break
    if not math.isfinite(residual) or residual > tolerance:
        raise RuntimeError(f"Transport did not converge in {max_iters} iterations: "
                           f"marginal L1={residual}, tolerance={tolerance}")
    plan = torch.exp(u[:, None] + kernel + v[None, :])
    value = (plan * cost).sum().to(x.dtype)
    info = {"iterations": start + count, "source_marginal_l1": source_error,
            "target_marginal_l1": target_error, "tolerance": tolerance}
    return (value, info) if return_info else value


def training_transport(x, y, cfg, *, weight_x=None):
    objective = cfg.get("transport_objective", "legacy")
    if objective == "legacy":
        return sinkhorn_w2(x, y, epsilon=cfg["sinkhorn_eps"], num_iters=50,
                           weight_x=weight_x)
    if objective == "converged_proxy":
        return converged_proxy(x, y, epsilon=cfg["sinkhorn_eps"],
                               max_iters=int(cfg.get("transport_max_iters", 20000)),
                               tolerance=float(cfg.get("transport_tolerance", 1e-4)),
                               weight_x=weight_x)
    if objective == "sinkhorn_divergence":
        from geomloss import SamplesLoss

        blur = float(cfg.get("transport_blur", 0.05))
        if not math.isfinite(blur) or blur <= 0:
            raise ValueError("transport_blur must be finite and positive")
        loss = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=0.9, debias=True,
                           backend="tensorized")
        if weight_x is None:
            return loss(x, y)
        a = weight_x.to(x).clamp_min(1e-12)
        a = a / a.sum()
        b = torch.full((len(y),), 1 / len(y), device=y.device, dtype=y.dtype)
        return loss(a, x, b, y)
    raise ValueError(f"Unknown transport_objective={objective!r}")
