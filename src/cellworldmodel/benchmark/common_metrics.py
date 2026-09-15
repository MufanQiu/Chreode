"""Distributional metrics for population-level cell state comparison.

Implements:
  - MMD² with multi-sigma RBF kernel (unbiased estimator)
  - W₂ via Sinkhorn (entropic regularization) and exact OT via POT
  - W₁ via exact OT

All functions accept PyTorch tensors and are differentiable (unless noted).
Used for Stage 2 training losses (MMD/W2) and benchmark evaluation (W1/W2/MMD).
"""
from __future__ import annotations

import warnings

from typing import Sequence

import numpy as np
import torch


def _pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances. Returns (m, n) tensor."""
    # torch.cdist is numerically stable for this
    return torch.cdist(X, Y, p=2) ** 2


@torch.no_grad()
def median_heuristic_sigmas(
    X: torch.Tensor,
    scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    max_samples: int = 512,
) -> list[float]:
    """Compute sigma values via median heuristic on pairwise distances.

    Returns sigma (not sigma²) such that typical kernel bandwidth ranges cover
    the data scale. Used for multi-sigma MMD.
    """
    n = X.shape[0]
    if n > max_samples:
        idx = torch.randperm(n, device=X.device)[:max_samples]
        Z = X[idx]
    else:
        Z = X

    D2 = _pairwise_sq_dists(Z, Z)
    # Exclude diagonal
    mask = ~torch.eye(Z.shape[0], dtype=torch.bool, device=Z.device)
    med = torch.median(D2[mask]).clamp_min(1e-12)

    return [float(torch.sqrt(torch.tensor(s, device=Z.device) * med).item()) for s in scales]


def mmd2_unbiased_multi_sigma(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigmas: Sequence[float] | None = None,
    scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """Unbiased MMD² estimator with multi-sigma RBF kernel.

    MMD²(X, Y) = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]
    where k(a, b) = exp(-||a-b||² / (2σ²)).

    If sigmas is None, uses median heuristic on X's pairwise distances.
    Final MMD² is the mean over all sigmas (multi-scale).

    Args:
        X: (m, d) samples from distribution P
        Y: (n, d) samples from distribution Q
        sigmas: list of kernel bandwidths. If None, computed via median heuristic.
        scales: scales applied to median-based sigma when sigmas is None.

    Returns:
        scalar tensor, unbiased MMD² (differentiable in X, Y)
    """
    if sigmas is None:
        sigmas = median_heuristic_sigmas(X, scales=scales)

    m, n = X.shape[0], Y.shape[0]
    Dxx = _pairwise_sq_dists(X, X)
    Dyy = _pairwise_sq_dists(Y, Y)
    Dxy = _pairwise_sq_dists(X, Y)

    vals = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * (sigma**2) + 1e-12)
        Kxx = torch.exp(-beta * Dxx)
        Kyy = torch.exp(-beta * Dyy)
        Kxy = torch.exp(-beta * Dxy)

        # Unbiased estimator: exclude diagonal from Kxx and Kyy
        term_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1) + 1e-12)
        term_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1) + 1e-12)
        term_xy = Kxy.mean()

        vals.append(term_xx + term_yy - 2.0 * term_xy)

    return torch.stack(vals).mean()


def sinkhorn_w2(
    X: torch.Tensor,
    Y: torch.Tensor,
    epsilon: float = 0.05,
    num_iters: int = 100,
    blur: float | None = None,
    weight_x: torch.Tensor | None = None,
    weight_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Entropic-regularized Wasserstein-2 distance via Sinkhorn iterations.

    Uses log-domain Sinkhorn for numerical stability. Returns the squared
    Wasserstein distance estimate (not its square root). Differentiable.

    Args:
        X: (m, d) samples
        Y: (n, d) samples
        epsilon: regularization strength (smaller = closer to exact W2)
        num_iters: number of Sinkhorn iterations
        blur: alias for epsilon, for geomloss compatibility

    Returns:
        scalar tensor, entropic W2² estimate
    """
    if blur is not None:
        epsilon = blur

    m, n = X.shape[0], Y.shape[0]
    device = X.device
    dtype = X.dtype

    # Marginals (log-space). Defaults to uniform; optional weights are
    # normalized here so callers can pass arbitrary positive masses.
    if weight_x is None:
        log_a = torch.full((m,), -np.log(m), device=device, dtype=dtype)
    else:
        weight_x = weight_x.to(device=device, dtype=dtype).clamp_min(1e-12)
        log_a = torch.log(weight_x / weight_x.sum())
    if weight_y is None:
        log_b = torch.full((n,), -np.log(n), device=device, dtype=dtype)
    else:
        weight_y = weight_y.to(device=device, dtype=dtype).clamp_min(1e-12)
        log_b = torch.log(weight_y / weight_y.sum())

    # Cost = squared Euclidean distance
    C = _pairwise_sq_dists(X, Y)

    # Log-domain Sinkhorn
    log_K = -C / epsilon  # (m, n)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(num_iters):
        # Update u: u = a / (K v)  =>  log_u = log_a - logsumexp(log_K + log_v, dim=1)
        log_u = log_a - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u[:, None], dim=0)

    # Transport plan: Π = exp(log_u[:,None] + log_K + log_v[None,:])
    log_pi = log_u[:, None] + log_K + log_v[None, :]
    pi = torch.exp(log_pi)

    # W2² estimate = <Π, C>
    return (pi * C).sum()


def exact_w_pq(
    X: torch.Tensor,
    Y: torch.Tensor,
    p: int = 2,
    weight_x: torch.Tensor | np.ndarray | None = None,
    weight_y: torch.Tensor | np.ndarray | None = None,
    num_iter_max: int = 10_000_000,
) -> float:
    """Exact p-Wasserstein distance via POT (non-differentiable).

    Uses POT's `ot.emd2` on squared distances for W2² or raw distances for W1.
    Returns a Python float.

    The iteration cap matters. POT's default (100k) stops the network simplex
    before optimality on the sample sizes used here and returns the unconverged
    value with only a warning: measured on a Weinreb test split, the same pair
    of clouds gives 4.90 at n=4000 and 4.25 at n=5650 unconverged, against 4.83
    and 4.07 converged (1.5% and 4.6% high). Since the cap binds at larger n,
    an arm scored against more cells is penalised for that alone, which is the
    opposite of a fair comparison. We therefore raise the cap and refuse to
    return a value that is still unconverged rather than reporting it silently.

    Args:
        X: (m, d) samples
        Y: (n, d) samples
        p: 1 for W1, 2 for W2
        num_iter_max: network-simplex iteration cap handed to POT

    Returns:
        float: W_p distance (NOT W_p² — we take the p-th root)

    Raises:
        RuntimeError: if the solver still reports non-optimality at that cap.
    """
    try:
        import ot
    except ImportError as e:
        raise ImportError("POT (pip install pot) required for exact W_p") from e

    m, n = X.shape[0], Y.shape[0]
    def _weights(value, size: int, name: str) -> np.ndarray:
        if value is None:
            return np.ones(size, dtype=np.float64) / size
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        arr = arr.astype(np.float64, copy=False).reshape(-1)
        if arr.shape[0] != size:
            raise ValueError(f"{name} has length {arr.shape[0]}, expected {size}")
        if not np.isfinite(arr).all() or np.any(arr <= 0):
            raise ValueError(f"{name} must contain finite positive masses")
        return arr / arr.sum()

    a = _weights(weight_x, m, "weight_x")
    b = _weights(weight_y, n, "weight_y")

    X_np = X.detach().cpu().numpy().astype(np.float64)
    Y_np = Y.detach().cpu().numpy().astype(np.float64)

    if p == 1:
        M = ot.dist(X_np, Y_np, metric="euclidean")
    elif p == 2:
        M = ot.dist(X_np, Y_np, metric="sqeuclidean")
    else:
        raise ValueError(f"Unsupported p={p}, only 1 or 2")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = float(ot.emd2(a, b, M, numItermax=int(num_iter_max)))
        unconverged = [w for w in caught if "numItermax" in str(w.message)]
    for w in caught:
        if w not in unconverged:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if unconverged:
        raise RuntimeError(
            f"exact W_{p} did not converge at numItermax={num_iter_max} for "
            f"m={m}, n={n}; the returned value would be biased high. Raise the "
            "cap or score on fewer cells."
        )
    return value if p == 1 else float(np.sqrt(value))


def compute_all_distributional_metrics(
    X_pred: torch.Tensor,
    Y_true: torch.Tensor,
    mmd_scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 100,
    max_samples_for_exact: int = 2000,
    weight_pred: torch.Tensor | None = None,
    weight_true: torch.Tensor | None = None,
    pred_mass_per_source: torch.Tensor | np.ndarray | None = None,
    source_total_mass: float | None = None,
    target_total_mass: float | None = None,
) -> dict[str, float]:
    """Compute all distribution-level metrics between pred and true populations.

    Returns a dict with keys:
      - mmd2: unbiased MMD² (multi-sigma RBF)
      - sinkhorn_w2: entropic W2² (differentiable, approximate)
      - w1: exact 1-Wasserstein distance (sqrt-less, via POT)
      - w2: exact 2-Wasserstein distance (sqrt-taken, via POT)

    For populations larger than max_samples_for_exact, W1/W2 are computed on a
    random subsample to keep runtime manageable.
    """
    with torch.no_grad():
        mmd2 = float(mmd2_unbiased_multi_sigma(X_pred, Y_true, scales=mmd_scales).item())
        sinkhorn = float(
            sinkhorn_w2(
                X_pred,
                Y_true,
                epsilon=sinkhorn_epsilon,
                num_iters=sinkhorn_iters,
                weight_x=weight_pred,
                weight_y=weight_true,
            ).item()
        )

        # Exact W1/W2 on subsample if needed
        m, n = X_pred.shape[0], Y_true.shape[0]
        if m > max_samples_for_exact or n > max_samples_for_exact:
            idx_p = torch.randperm(m, device=X_pred.device)[:max_samples_for_exact]
            idx_t = torch.randperm(n, device=Y_true.device)[:max_samples_for_exact]
            X_sub, Y_sub = X_pred[idx_p], Y_true[idx_t]
            weight_pred_sub = None if weight_pred is None else weight_pred[idx_p]
            weight_true_sub = None if weight_true is None else weight_true[idx_t]
        else:
            X_sub, Y_sub = X_pred, Y_true
            weight_pred_sub = weight_pred
            weight_true_sub = weight_true

        w1 = exact_w_pq(
            X_sub,
            Y_sub,
            p=1,
            weight_x=weight_pred_sub,
            weight_y=weight_true_sub,
        )
        w2 = exact_w_pq(
            X_sub,
            Y_sub,
            p=2,
            weight_x=weight_pred_sub,
            weight_y=weight_true_sub,
        )

    result = {
        "mmd2": mmd2,
        "sinkhorn_w2": sinkhorn,
        "w1": w1,
        "w2": w2,
    }
    if pred_mass_per_source is not None:
        if source_total_mass is None or target_total_mass is None:
            raise ValueError(
                "source_total_mass and target_total_mass are required with "
                "pred_mass_per_source"
            )
        result.update(
            mass_metrics(
                pred_mass_per_source,
                source_total_mass=source_total_mass,
                target_total_mass=target_total_mass,
            )
        )
    return result


def mass_metrics(
    pred_mass_per_source: torch.Tensor | np.ndarray,
    *,
    source_total_mass: float,
    target_total_mass: float,
) -> dict[str, float]:
    """Compute total-mass variation and relative mass error.

    Per-source masses are defined relative to unit source-cell mass, so their
    mean is the predicted target/source total-mass ratio.
    """
    if source_total_mass <= 0 or target_total_mass <= 0:
        raise ValueError("source_total_mass and target_total_mass must be positive")
    if isinstance(pred_mass_per_source, torch.Tensor):
        pred = pred_mass_per_source.detach().cpu().numpy()
    else:
        pred = np.asarray(pred_mass_per_source)
    pred = pred.astype(np.float64, copy=False).reshape(-1)
    if pred.size == 0:
        raise ValueError("pred_mass_per_source must not be empty")
    if not np.isfinite(pred).all() or np.any(pred <= 0):
        raise ValueError("pred_mass_per_source must contain finite positive masses")

    pred_relative_mass = float(pred.mean())
    true_relative_mass = float(target_total_mass) / float(source_total_mass)
    tmv = abs(pred_relative_mass - true_relative_mass)
    rme = tmv / true_relative_mass
    return {
        "pred_relative_mass": pred_relative_mass,
        "true_relative_mass": true_relative_mass,
        "tmv": float(tmv),
        "rme": float(rme),
    }


# =============================================================================
# BranchSBM-style evaluation (for apples-to-apples comparison with paper Table 3)
# =============================================================================
# Differences from our protocol:
#   1. W1/W2 computed on top-2 PCs ONLY (not full dimensions)
#   2. MMD uses fixed σ_list = [0.01, 0.1, 1, 10, 100] (not median heuristic)
#   3. MMD is the "biased" estimator (includes diagonal) (not unbiased)
#   4. 5 random subsample "trials" to n_min, report mean ± std
#
# Source: 3rdparty/BranchSBM/src/branch_flow_net_test.py::compute_distribution_distances
#         3rdparty/BranchSBM/src/utils.py::mix_rbf_mmd2


def _mix_rbf_kernel_biased(X: torch.Tensor, Y: torch.Tensor, sigma_list: Sequence[float]):
    """Compute mixed RBF kernel matrices K_XX, K_XY, K_YY as in BranchSBM utils.py.

    Note: This requires X.size(0) == Y.size(0) per BranchSBM's assertion, because
    both are stacked into Z and squared-distance is computed jointly.
    """
    assert X.size(0) == Y.size(0), (
        f"BranchSBM MMD requires equal-size inputs, got {X.size(0)} vs {Y.size(0)}"
    )
    m = X.size(0)
    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()  # squared distances

    K = torch.zeros_like(exponent)
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K = K + torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:]


def branchsbm_mmd2_biased_fixed_sigma(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_list: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> torch.Tensor:
    """BranchSBM-style biased MMD² with fixed σ list.

    Replicates `mix_rbf_mmd2(..., biased=True)` from 3rdparty/BranchSBM/src/utils.py.
    "Biased" includes diagonal terms (K_XX.mean() instead of off-diagonal average).

    Sum over σ_list (NOT mean like our multi-sigma version).
    """
    K_XX, K_XY, K_YY = _mix_rbf_kernel_biased(X, Y, sigma_list)
    # Biased MMD²: mean includes diagonal
    mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return mmd2


def compute_branchsbm_style_metrics(
    X_pred: torch.Tensor,
    Y_true: torch.Tensor,
    n_trials: int = 5,
    n_top_pcs: int = 2,
    mmd_sigma_list: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    seed: int = 42,
    skip_w1: bool = False,
) -> dict[str, float]:
    """BranchSBM-style evaluation protocol (matches paper Table 2/3).

    Protocol (verbatim from `branch_flow_net_test.py`):
      - Subsample both to n_min = min(|X_pred|, |Y_true|)
      - W1/W2 computed on top `n_top_pcs` dimensions (paper uses 2)
      - MMD computed on FULL dimensions (biased, fixed σ_list)
      - Repeat n_trials times with different random subsamples
      - Report mean ± std

    ``skip_w1`` drops the two exact W1 solves per trial. They are the same cost
    as the W2 solves and are unused by callers that report W2 and MMD, which
    roughly halves the wall clock on large clouds. The W1 keys are absent from
    the result rather than present and wrong.

    Returns dict with keys prefixed `branchsbm_`:
      - branchsbm_w1_top2_mean / _std: W1 on top-N PCs (paper metric)
      - branchsbm_w2_top2_mean / _std: W2 on top-N PCs (paper metric)
      - branchsbm_mmd_full_mean / _std: biased MMD on all dims (paper metric)
      - branchsbm_w1_full_mean / _std: W1 on all dims (paper's "_full" variant)
      - branchsbm_w2_full_mean / _std: W2 on all dims (paper's "_full" variant)
    """
    rng = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        m, n = X_pred.shape[0], Y_true.shape[0]
        n_min = min(m, n)
        D = X_pred.shape[1]
        k = min(n_top_pcs, D)

        w1_top_trials, w2_top_trials = [], []
        w1_full_trials, w2_full_trials = [], []
        mmd_trials = []

        X_cpu = X_pred.detach().cpu()
        Y_cpu = Y_true.detach().cpu()

        for _ in range(n_trials):
            perm_p = torch.randperm(m, generator=rng)[:n_min]
            perm_t = torch.randperm(n, generator=rng)[:n_min]
            X_sub = X_cpu[perm_p]
            Y_sub = Y_cpu[perm_t]

            # W1/W2 on top-k PCs
            X_top = X_sub[:, :k]
            Y_top = Y_sub[:, :k]
            if not skip_w1:
                w1_top_trials.append(exact_w_pq(X_top, Y_top, p=1))
            w2_top_trials.append(exact_w_pq(X_top, Y_top, p=2))

            # W1/W2 on full dims (BranchSBM "_full" variant, also stored)
            if not skip_w1:
                w1_full_trials.append(exact_w_pq(X_sub, Y_sub, p=1))
            w2_full_trials.append(exact_w_pq(X_sub, Y_sub, p=2))

            # MMD on full dims (biased, fixed σ)
            mmd_val = float(branchsbm_mmd2_biased_fixed_sigma(X_sub, Y_sub, mmd_sigma_list).item())
            mmd_trials.append(mmd_val)

        import numpy as _np
        def _ms(arr):
            a = _np.asarray(arr, dtype=_np.float64)
            # ddof=1 matches BranchSBM's np.std(..., ddof=1) convention
            if len(a) > 1:
                return float(a.mean()), float(a.std(ddof=1))
            return float(a.mean()), 0.0

        w2_top_m, w2_top_s = _ms(w2_top_trials)
        w2_full_m, w2_full_s = _ms(w2_full_trials)
        mmd_m, mmd_s = _ms(mmd_trials)

    out = {
        f"branchsbm_w2_top{k}_mean": w2_top_m,
        f"branchsbm_w2_top{k}_std": w2_top_s,
        "branchsbm_mmd_full_mean": mmd_m,
        "branchsbm_mmd_full_std": mmd_s,
        "branchsbm_w2_full_mean": w2_full_m,
        "branchsbm_w2_full_std": w2_full_s,
        "n_trials": n_trials,
        "n_subsample": n_min,
    }
    if not skip_w1:
        w1_top_m, w1_top_s = _ms(w1_top_trials)
        w1_full_m, w1_full_s = _ms(w1_full_trials)
        out[f"branchsbm_w1_top{k}_mean"] = w1_top_m
        out[f"branchsbm_w1_top{k}_std"] = w1_top_s
        out["branchsbm_w1_full_mean"] = w1_full_m
        out["branchsbm_w1_full_std"] = w1_full_s
    return out


def compute_dual_protocol_metrics(
    X_pred: torch.Tensor,
    Y_true: torch.Tensor,
    n_trials: int = 5,
    n_top_pcs: int = 2,
    mmd_sigma_list: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    mmd_scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 100,
    max_samples_for_exact: int = 2000,
    seed: int = 42,
    weight_pred: torch.Tensor | None = None,
    weight_true: torch.Tensor | None = None,
    pred_mass_per_source: torch.Tensor | np.ndarray | None = None,
    source_total_mass: float | None = None,
    target_total_mass: float | None = None,
) -> dict[str, float]:
    """Compute BOTH evaluation protocols side-by-side.

    Returns a dict merging:
      - branchsbm_* (paper protocol, for direct Table 2/3 comparison)
      - ours_* (stricter protocol, full-dim + unbiased + median heuristic)

    Key naming rule: ALL branchsbm-style metrics start with `branchsbm_`,
    ALL our stricter metrics start with `ours_`. Never mix naming.
    """
    # BranchSBM-style
    bsbm = compute_branchsbm_style_metrics(
        X_pred, Y_true, n_trials=n_trials, n_top_pcs=n_top_pcs,
        mmd_sigma_list=mmd_sigma_list, seed=seed,
    )
    # Our stricter protocol (full dim, unbiased MMD, median heuristic)
    ours_raw = compute_all_distributional_metrics(
        X_pred, Y_true,
        mmd_scales=mmd_scales,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iters=sinkhorn_iters,
        max_samples_for_exact=max_samples_for_exact,
        weight_pred=weight_pred,
        weight_true=weight_true,
        pred_mass_per_source=pred_mass_per_source,
        source_total_mass=source_total_mass,
        target_total_mass=target_total_mass,
    )
    ours = {f"ours_{k}": v for k, v in ours_raw.items()}
    # Rename for clarity:
    ours["ours_mmd2_unbiased_median"] = ours.pop("ours_mmd2")
    ours["ours_w1_full"] = ours.pop("ours_w1")
    ours["ours_w2_full"] = ours.pop("ours_w2")
    ours["ours_sinkhorn_w2_full"] = ours.pop("ours_sinkhorn_w2")

    return {**bsbm, **ours}


# =============================================================================
# Per-branch evaluation (for OT-free models without explicit branch structure)
# =============================================================================
# For BranchSBM's per-branch metrics (Table 4/5 branch_1, branch_2, ...), we
# need to assign each prediction to one of the known target branches. Since
# OT-free M1/M2 produces an unlabeled prediction population, we use "post-hoc
# nearest-target-cluster assignment":
#
#   for each pred_i, find its nearest target cell y_j (L2), inherit y_j's
#   branch label. Then split pred by label and compute per-branch metrics.
#
# This is an honest evaluation because:
#   - target labels are ground truth (Leiden/KMeans on actual target cells)
#   - if a pred is far from ALL targets, it gets assigned wherever it's
#     least-far, and the per-branch metric will show that mismatch
#   - pooled (combined) metric is unaffected


@torch.no_grad()
def assign_by_nearest_target(
    X_pred: torch.Tensor,
    Y_true: torch.Tensor,
    target_labels: torch.Tensor | "np.ndarray",
) -> torch.Tensor:
    """Assign each pred to nearest target's label (L2 distance).

    Args:
        X_pred: (N_pred, D) prediction samples
        Y_true: (N_true, D) target samples (ground truth)
        target_labels: (N_true,) integer label for each target cell

    Returns:
        (N_pred,) integer label (one per pred, inherited from nearest target)
    """
    if not torch.is_tensor(target_labels):
        target_labels = torch.tensor(target_labels, dtype=torch.long)
    # chunk over pred to avoid OOM when both N_pred and N_true are large
    chunk = 512
    out = []
    for i in range(0, X_pred.shape[0], chunk):
        dists = torch.cdist(X_pred[i:i + chunk], Y_true, p=2)  # (chunk, N_true)
        nn_idx = dists.argmin(dim=1)  # (chunk,)
        out.append(target_labels[nn_idx.cpu()])
    return torch.cat(out)


def compute_per_branch_dual_protocol_metrics(
    X_pred: torch.Tensor,
    Y_true: torch.Tensor,
    target_labels: "np.ndarray | torch.Tensor",
    n_trials: int = 5,
    n_top_pcs: int = 2,
    mmd_sigma_list: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    mmd_scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 100,
    max_samples_for_exact: int = 2000,
    seed: int = 42,
    min_cells_per_branch: int = 10,
) -> dict[str, dict[str, float]]:
    """Compute dual-protocol metrics per branch + combined (pooled).

    Assigns each pred to nearest target's label, then splits by label.
    Returns nested dict:
        {
          "combined": {branchsbm_*..., ours_*...},
          "branch_0": {...},
          "branch_1": {...},
          ...
        }

    Skips branches with fewer than `min_cells_per_branch` pred or true cells.
    """
    pred_labels = assign_by_nearest_target(X_pred, Y_true, target_labels)

    if not torch.is_tensor(target_labels):
        target_labels_t = torch.tensor(target_labels, dtype=torch.long)
    else:
        target_labels_t = target_labels.long()

    unique_labels = torch.unique(target_labels_t).tolist()

    results: dict[str, dict[str, float]] = {}

    # Pooled (combined) — same as before
    results["combined"] = compute_dual_protocol_metrics(
        X_pred, Y_true,
        n_trials=n_trials, n_top_pcs=n_top_pcs,
        mmd_sigma_list=mmd_sigma_list, mmd_scales=mmd_scales,
        sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iters=sinkhorn_iters,
        max_samples_for_exact=max_samples_for_exact, seed=seed,
    )

    # Per-branch
    for lab in unique_labels:
        X_sub = X_pred[pred_labels == lab]
        Y_sub = Y_true[target_labels_t == lab]
        if X_sub.shape[0] < min_cells_per_branch or Y_sub.shape[0] < min_cells_per_branch:
            results[f"branch_{int(lab)}"] = {
                "n_pred": int(X_sub.shape[0]),
                "n_true": int(Y_sub.shape[0]),
                "skipped": True,
            }
            continue
        results[f"branch_{int(lab)}"] = compute_dual_protocol_metrics(
            X_sub, Y_sub,
            n_trials=n_trials, n_top_pcs=n_top_pcs,
            mmd_sigma_list=mmd_sigma_list, mmd_scales=mmd_scales,
            sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iters=sinkhorn_iters,
            max_samples_for_exact=max_samples_for_exact, seed=seed,
        )
        results[f"branch_{int(lab)}"]["n_pred"] = int(X_sub.shape[0])
        results[f"branch_{int(lab)}"]["n_true"] = int(Y_sub.shape[0])

    return results
