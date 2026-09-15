"""Drifting field V loss for Cell World Model.

Adapted from Kaiming He's Drifting Model (arXiv:2602.04770), `drifting.py`.
Key changes for cell dynamics:
  1. Positives = target population (not "any real image")
  2. Negatives = other generated predictions in batch
  3. Feature space = L2Norm(Whiten(z)) instead of ImageNet encoder
  4. Preserves the original stopgrad loss structure (critical!)

The training loss is:

    L_drift = E[||φ(ẑ) - sg(φ(ẑ) + V)||²]

where V is computed via compute_V_multi_temperature. Numerically L_drift = ||V||²
but gradient-wise, stopgrad ensures V is a *fixed target direction* — we train
the model to move ẑ toward ẑ + V, not to minimize V's own magnitude
(which would be susceptible to reward hacking / V collapsing to 0 trivially).

See agent/human-review/drifting-biological-motivation.typ for full derivation.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch


def compute_V(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
    mask_self: bool = True,
    balance_sample_counts: bool = False,
) -> torch.Tensor:
    """Compute single-temperature drifting field V (Algorithm 2 from paper).

    For each generated sample x_i, V points in the direction that x_i should
    move to better match the true distribution. When q_θ = p_data, V = 0.

    Args:
        x: (N, D) generated samples in feature space
        y_pos: (N_pos, D) positive (real data) samples in feature space
        y_neg: (N_neg, D) negative (generated) samples in feature space
        temperature: softmax temperature τ (smaller = sharper / more local)
        mask_self: if True and N==N_neg, mask x-to-itself distance (assumes
            y_neg includes the same x samples)
        balance_sample_counts: if True, subtract log(N_pos) / log(N_neg) from
            the corresponding logits. This treats positive and negative
            empirical supports as equal-mass distributions even when represented
            by different sample counts. The correction is symmetric under
            swapping pos/neg groups, so it preserves the anti-symmetry invariant.

    Returns:
        V: (N, D) drifting field vectors
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]
    device = x.device

    # 1. Pairwise L2 distances
    dist_pos = torch.cdist(x, y_pos, p=2)  # (N, N_pos)
    dist_neg = torch.cdist(x, y_neg, p=2)  # (N, N_neg)

    # 2. Mask self-distances if y_neg contains x
    if mask_self and N == N_neg:
        mask = torch.eye(N, device=device) * 1e6
        dist_neg = dist_neg + mask

    # 3. Logits = -distance / temperature
    logit_pos = -dist_pos / temperature
    logit_neg = -dist_neg / temperature
    if balance_sample_counts:
        logit_pos = logit_pos - torch.log(torch.as_tensor(float(N_pos), device=device, dtype=x.dtype))
        logit_neg = logit_neg - torch.log(torch.as_tensor(float(N_neg), device=device, dtype=x.dtype))

    # 4. Concat for joint softmax normalization
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # (N, N_pos + N_neg)

    # 5. Double softmax + geometric mean (doubly-stochastic-like, prevents mode collapse)
    A_row = torch.softmax(logit, dim=1)  # per-generated-sample attention over all refs
    A_col = torch.softmax(logit, dim=0)  # per-ref attention shared across generated samples
    A = torch.sqrt(A_row * A_col + 1e-12)

    A_pos = A[:, :N_pos]  # (N, N_pos)
    A_neg = A[:, N_pos:]  # (N, N_neg)

    # 6. Cross-weighting: only contested regions (pos + neg coexist) get strong signal
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

    # 7. V = attraction - repulsion
    drift_pos = W_pos @ y_pos  # (N, D)
    drift_neg = W_neg @ y_neg  # (N, D)

    V = drift_pos - drift_neg
    return V


def compute_V_multi_temperature(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
    mask_self: bool = True,
    normalize_each: bool = True,
    balance_sample_counts: bool = False,
) -> torch.Tensor:
    """Multi-temperature drifting field (Sec A.6 of paper).

    Sums V from multiple τ values, each normalized to E[||V||²] ~ 1, to capture
    structure at multiple scales simultaneously:
      - τ=0.02: microstate (fine single-cell level)
      - τ=0.05: mesostate (module/pathway neighborhood)
      - τ=0.2: macrostate (lineage allocation)

    Args:
        x, y_pos, y_neg: feature vectors
        temperatures: list of τ values
        mask_self: see compute_V
        normalize_each: if True, normalize each τ's V to unit RMS before summing
        balance_sample_counts: pass-through to compute_V

    Returns:
        V_total: (N, D) combined drifting field
    """
    V_total = torch.zeros_like(x)
    for tau in temperatures:
        V_tau = compute_V(
            x, y_pos, y_neg, tau, mask_self,
            balance_sample_counts=balance_sample_counts,
        )
        if normalize_each:
            # Normalize so E[||V||²] ~ 1 (for this τ)
            V_norm = torch.sqrt(torch.mean(V_tau**2) + 1e-8)
            V_tau = V_tau / V_norm
        V_total = V_total + V_tau
    return V_total


def median_heuristic_temperatures(
    X: torch.Tensor,
    multipliers: Sequence[float] = (0.2, 0.5, 1.5),
    max_samples: int = 512,
) -> list[float]:
    """Compute τ via median heuristic on pairwise distances (D6 finding).

    Paper's fixed τ=(0.02, 0.05, 0.2) works on ImageNet features but *saturates*
    on 30–50D scRNA-seq due to concentration of measure (all pairs concentrate
    near √(2D)). We verified median-heuristic τ=(0.2, 0.5, 1.5) × d_med gives
    proper multi-scale coverage on Mouse/Clonidine/Trametinib/Veres.

    Args:
        X: (N, D) reference population (typically whiten+scaled target cells)
        multipliers: scales applied to d_median
        max_samples: subsample cap for pairwise distance computation

    Returns:
        list of τ values
    """
    n = X.shape[0]
    if n > max_samples:
        idx = torch.randperm(n, device=X.device)[:max_samples]
        Z = X[idx]
    else:
        Z = X
    D = torch.cdist(Z, Z, p=2)
    mask = ~torch.eye(Z.shape[0], dtype=torch.bool, device=Z.device)
    d_med = D[mask].median().clamp_min(1e-12).item()
    return [float(m * d_med) for m in multipliers]


def normalize_features(
    features: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    target_dist_sqrt_D: bool = True,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """Whiten + rescale features so typical pairwise distance is ~sqrt(D).

    This is a prerequisite for multi-temperature V to behave correctly — τ
    values [0.02, 0.05, 0.2] only make sense when feature distances are
    standardized.

    Args:
        features: (N, D)
        mean, std: optional precomputed per-feature stats (for consistency
            across batches at eval time)
        scale: optional precomputed scale factor
        target_dist_sqrt_D: if True, rescale so average pairwise dist ≈ √D

    Returns:
        normalized features, scale, mean, std
    """
    D = features.shape[1]
    target_dist = D**0.5 if target_dist_sqrt_D else 1.0

    with torch.no_grad():
        if mean is None:
            mean = features.mean(dim=0, keepdim=True)
        if std is None:
            std = features.std(dim=0, keepdim=True) + 1e-8

    features_std = (features - mean) / std

    if scale is None:
        with torch.no_grad():
            n_sample = min(features.shape[0], 256)
            idx = torch.randperm(features.shape[0], device=features.device)[:n_sample]
            subset = features_std[idx]
            dists = torch.cdist(subset, subset, p=2)
            mask = ~torch.eye(n_sample, dtype=torch.bool, device=features.device)
            avg_dist = dists[mask].mean()
            scale = (target_dist / (avg_dist + 1e-8)).item()

    return features_std * scale, scale, mean, std


def drift_stopgrad_loss(
    phi_gen: torch.Tensor,
    phi_pos: torch.Tensor,
    phi_neg: Optional[torch.Tensor] = None,
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
    normalize_each_tau: bool = True,
    balance_sample_counts: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Drifting Model's stopgrad training loss.

        L = E[||φ(ẑ) - sg(φ(ẑ) + V)||²]

    Numerically equals ||V||² but its gradient flows ONLY through φ(ẑ), not
    through V. This implements the correct "move φ(ẑ) toward ẑ+V direction"
    interpretation of Drifting Model (see Sec 3.3 of the paper).

    Args:
        phi_gen: (N, D) feature of generated samples (requires grad)
        phi_pos: (N_pos, D) feature of target population (positives)
        phi_neg: (N_neg, D) feature of negatives. If None, use phi_gen itself.
        temperatures: list of τ values for multi-temperature V
        normalize_each_tau: normalize each τ's V contribution
        balance_sample_counts: if True, count-correct pos/neg logits

    Returns:
        loss: scalar stopgrad loss
        info: dict with diagnostic metrics
          - drift_norm: RMS of V (the standard Drifting Model diagnostic)
          - loss_value: same as loss but detached
    """
    if phi_neg is None:
        phi_neg = phi_gen  # use generated samples as their own negatives

    # Compute V (NOT part of gradient graph — model learns via stopgrad target)
    V = compute_V_multi_temperature(
        phi_gen,
        phi_pos,
        phi_neg,
        temperatures=temperatures,
        mask_self=(phi_neg is phi_gen),
        normalize_each=normalize_each_tau,
        balance_sample_counts=balance_sample_counts,
    )

    # Critical: stopgrad version
    # target = sg(phi_gen + V)
    # loss = || phi_gen - target ||²
    # Gradient flows only through phi_gen (model parameters), not through V
    target = (phi_gen + V).detach()
    loss = torch.mean((phi_gen - target) ** 2)

    # Diagnostic: drift norm (model is trained when this → 0)
    drift_norm = torch.sqrt(torch.mean(V**2) + 1e-12).detach().item()

    info = {
        "drift_norm": drift_norm,
        "loss_value": loss.detach().item(),
    }
    return loss, info


def drift_stopgrad_loss_from_raw(
    z_gen: torch.Tensor,
    z_pos: torch.Tensor,
    z_neg: Optional[torch.Tensor] = None,
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
    normalize_features_first: bool = True,
    feature_stats: Optional[dict] = None,
    balance_sample_counts: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Convenience wrapper: normalizes raw z features first, then computes V loss.

    Uses the paper's recommended L2Norm(Whiten(z)) feature space when
    `normalize_features_first=True`.

    For benchmark experiments we typically use this with raw latent z directly,
    letting normalize_features handle whitening/rescaling.

    Args:
        z_gen, z_pos, z_neg: raw (unnormalized) cell state vectors
        temperatures: multi-τ
        normalize_features_first: if True, run normalize_features on the joint
            distribution and apply consistent transformation
        feature_stats: optional {mean, std, scale} from prior batches to use
            consistent feature normalization across training
        balance_sample_counts: if True, count-correct pos/neg logits

    Returns:
        loss, info (info includes feature stats used, so callers can cache them)
    """
    # Track whether caller wants neg = gen (self-negatives, needs self-mask).
    # Bug fix 2026-04-21 (GPT review): normalize creates new tensors so
    # `phi_neg is phi_gen` became False even when z_neg was originally z_gen,
    # silently disabling mask_self inside drift_stopgrad_loss. Track explicitly.
    neg_is_self = z_neg is None
    if neg_is_self:
        z_neg = z_gen

    if not normalize_features_first:
        # Pass phi_neg=None so drift_stopgrad_loss treats as self-negatives
        phi_neg_arg = None if neg_is_self else z_neg
        return drift_stopgrad_loss(
            z_gen, z_pos, phi_neg_arg, temperatures,
            balance_sample_counts=balance_sample_counts,
        )

    # Fit normalization stats on target (real) population — this is the
    # reference space we want predictions to match
    if feature_stats is None:
        _, scale, mean, std = normalize_features(z_pos)
        feature_stats = {"mean": mean, "std": std, "scale": scale}

    # Apply same normalization to gen, pos, neg
    mean = feature_stats["mean"]
    std = feature_stats["std"]
    scale = feature_stats["scale"]

    phi_gen = ((z_gen - mean) / std) * scale
    phi_pos = ((z_pos - mean) / std) * scale
    # Critical: pass phi_neg=None (not phi_neg tensor) so drift_stopgrad_loss
    # uses phi_gen as its own negatives AND triggers mask_self=True.
    if neg_is_self:
        phi_neg_arg = None
    else:
        phi_neg_arg = ((z_neg - mean) / std) * scale

    loss, info = drift_stopgrad_loss(
        phi_gen, phi_pos, phi_neg_arg, temperatures,
        balance_sample_counts=balance_sample_counts,
    )
    info["feature_stats"] = feature_stats
    return loss, info
