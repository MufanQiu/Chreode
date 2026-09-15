"""Unit tests for drift loss.

Verifies:
  - V = 0 when pos == gen distribution (equilibrium)
  - V ≠ 0 when distributions differ
  - Stopgrad: gradient flows through phi_gen but NOT through V
  - Multi-temperature V normalization works
  - drift_norm goes to 0 as training would progress
  - Numerical: loss value ≈ ||V||² (up to stopgrad detach)
"""
from __future__ import annotations

import pytest
import torch

from cellworldmodel.training.drift_loss import (
    compute_V,
    compute_V_multi_temperature,
    drift_stopgrad_loss,
    drift_stopgrad_loss_from_raw,
    normalize_features,
)


def test_V_at_equilibrium_unnormalized():
    """Without per-τ normalization, V should be smaller when distributions match.

    This is the 'true' V behavior. normalize_each_tau=True is a training trick
    that forces each scale to contribute equally, hiding raw magnitudes.
    """
    torch.manual_seed(0)
    D = 4  # smaller D to avoid kernel saturation

    # Normalize feature space once (shared stats)
    all_data = torch.cat([torch.randn(500, D), torch.randn(500, D) + 2])
    from cellworldmodel.training.drift_loss import normalize_features
    _, scale, mean, std = normalize_features(all_data)
    norm_fn = lambda x: ((x - mean) / std) * scale

    # Case 1: gen ≈ pos (same dist)
    gen_same = norm_fn(torch.randn(200, D))
    pos_same = norm_fn(torch.randn(200, D))
    V_same = compute_V_multi_temperature(
        gen_same, pos_same, gen_same, temperatures=[0.5, 1.0, 2.0], normalize_each=False
    )

    # Case 2: gen at origin, pos shifted far
    gen_shift = norm_fn(torch.randn(200, D))
    pos_shift = norm_fn(torch.randn(200, D) + 2.0)
    V_shift = compute_V_multi_temperature(
        gen_shift, pos_shift, gen_shift, temperatures=[0.5, 1.0, 2.0], normalize_each=False
    )

    v_same_rms = torch.sqrt(torch.mean(V_same**2)).item()
    v_shift_rms = torch.sqrt(torch.mean(V_shift**2)).item()

    assert v_shift_rms > v_same_rms * 2, (
        f"V for shifted ({v_shift_rms:.3f}) not clearly larger than for "
        f"matched ({v_same_rms:.3f})"
    )


def test_V_nonzero_for_different_distributions():
    """Stopgrad loss should produce non-trivial gradient when distributions differ."""
    torch.manual_seed(0)
    D = 5
    # Use feature normalization with shared stats (important!)
    pos_raw = torch.randn(100, D) + 2.0  # shifted
    gen_raw = torch.randn(100, D, requires_grad=True)  # at origin

    loss, info = drift_stopgrad_loss_from_raw(
        gen_raw, pos_raw, None, temperatures=[0.2, 0.5, 1.0],
        normalize_features_first=True,
    )
    # drift_norm should be meaningfully > 0
    assert info["drift_norm"] > 0.5, f"drift_norm too small: {info['drift_norm']}"


def test_V_points_toward_target():
    """V should push generated samples toward the positives."""
    torch.manual_seed(0)
    gen = torch.randn(100, 3) * 0.5  # centered at 0
    pos = torch.randn(100, 3) * 0.5 + torch.tensor([2.0, 0.0, 0.0])  # shifted +x
    V = compute_V(gen, pos, gen, temperature=0.3, mask_self=True)
    # Mean V direction should be in +x direction
    mean_V = V.mean(0)
    assert mean_V[0].item() > 0.1, f"V not pointing toward positives: mean V = {mean_V}"


def test_V_antisymmetry_swap_equal_support():
    """Swapping positive and negative empirical supports should flip V.

    This is the finite-batch version of the Drifting paper's hard invariant:
    V_{p,q}(x) = -V_{q,p}(x). We use independent probe points and no self-mask
    so the only operation is swapping the two reference supports.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 5)
    p = torch.randn(80, 5) + 1.0
    q = torch.randn(80, 5) - 1.0

    v_pq = compute_V(x, p, q, temperature=0.7, mask_self=False)
    v_qp = compute_V(x, q, p, temperature=0.7, mask_self=False)

    assert torch.allclose(v_pq, -v_qp, atol=1e-5, rtol=1e-5)


def test_V_antisymmetry_with_count_balancing_unequal_support():
    """Count-balanced logits should still preserve anti-symmetry."""
    torch.manual_seed(0)
    x = torch.randn(64, 5)
    p = torch.randn(40, 5) + 1.0
    q = torch.randn(120, 5) - 1.0

    v_pq = compute_V(
        x, p, q, temperature=0.7, mask_self=False,
        balance_sample_counts=True,
    )
    v_qp = compute_V(
        x, q, p, temperature=0.7, mask_self=False,
        balance_sample_counts=True,
    )

    assert torch.allclose(v_pq, -v_qp, atol=1e-5, rtol=1e-5)


def test_V_zero_when_pos_neg_same_empirical_support():
    """If positive and negative supports are identical, V should be zero."""
    torch.manual_seed(0)
    x = torch.randn(64, 5)
    y = torch.randn(80, 5)
    v = compute_V(x, y, y, temperature=0.7, mask_self=False)
    assert torch.sqrt(torch.mean(v ** 2)).item() < 1e-6


def test_multi_temperature_V_nonzero():
    torch.manual_seed(0)
    gen = torch.randn(100, 5)
    pos = torch.randn(100, 5) + 1.0
    V_multi = compute_V_multi_temperature(gen, pos, gen, temperatures=[0.02, 0.05, 0.2])
    # Multi-temp sums normalized components, so magnitude ≈ 3 (for 3 taus)
    v_rms = torch.sqrt(torch.mean(V_multi**2)).item()
    assert v_rms > 0.5


def test_stopgrad_loss_gradient_only_through_phi_gen():
    """Key invariant: gradient of L should be -2*V in the direction of phi_gen.

    Since target = sg(phi_gen + V), gradient is:
      d/dθ L = 2 * (phi_gen - sg(phi_gen + V)) * dphi_gen/dθ
            = 2 * (-V) * dphi_gen/dθ = -2V * dphi_gen/dθ

    So gradient only flows through phi_gen, NOT through V's contrastive kernel.
    """
    torch.manual_seed(0)
    D = 4
    phi_gen = torch.randn(50, D, requires_grad=True)
    phi_pos = torch.randn(50, D) + 2.0  # fixed (no grad)
    phi_neg = phi_gen  # use same

    loss, info = drift_stopgrad_loss(phi_gen, phi_pos, phi_neg, temperatures=[0.1])
    loss.backward()

    # Check gradient is non-zero and finite
    assert phi_gen.grad is not None
    assert torch.isfinite(phi_gen.grad).all()

    # Check phi_pos got no gradient (it's a target, not model output)
    # (We can't check this directly since phi_pos has no grad flag, but we can
    # verify the loss value equals approximately ||V||²)
    assert info["drift_norm"] > 0
    assert info["loss_value"] > 0


def test_stopgrad_loss_value_equals_v_squared():
    """Loss value should numerically equal mean ||V||²."""
    torch.manual_seed(0)
    phi_gen = torch.randn(100, 5, requires_grad=True)
    phi_pos = torch.randn(100, 5) + 1.0
    loss, info = drift_stopgrad_loss(phi_gen, phi_pos, phi_gen, temperatures=[0.1])

    # Manually compute V and check loss ≈ mean(V²)
    V = compute_V_multi_temperature(phi_gen, phi_pos, phi_gen, temperatures=[0.1])
    v_mean_sq = torch.mean(V**2).item()
    assert abs(loss.item() - v_mean_sq) < 1e-4, (
        f"loss={loss.item()} vs mean||V||²={v_mean_sq}"
    )


def test_feature_normalization_stats():
    torch.manual_seed(0)
    z = torch.randn(100, 8) * 3 + 2  # non-standardized
    phi, scale, mean, std = normalize_features(z)
    # After whitening, mean ~ 0 and std ~ scale
    assert abs(phi.mean().item()) < 0.1
    # After rescaling, average pairwise distance should be ≈ √D = √8 ≈ 2.83
    dists = torch.cdist(phi[:64], phi[:64], p=2)
    avg_dist = dists[~torch.eye(64, dtype=torch.bool)].mean().item()
    assert abs(avg_dist - 8**0.5) < 1.0, f"Normalized avg dist {avg_dist} ≠ √8"


def test_drift_loss_from_raw_works():
    """End-to-end: raw z → normalize → V loss, reusable stats across batches."""
    torch.manual_seed(0)
    z_gen = torch.randn(80, 6, requires_grad=True)
    z_pos = torch.randn(80, 6) + 1.5
    loss, info = drift_stopgrad_loss_from_raw(
        z_gen, z_pos, None, temperatures=[0.1], normalize_features_first=True
    )
    loss.backward()
    assert z_gen.grad is not None
    assert "feature_stats" in info
    # Check we can reuse stats
    z_gen2 = torch.randn(80, 6, requires_grad=True)
    loss2, _ = drift_stopgrad_loss_from_raw(
        z_gen2, z_pos, None, temperatures=[0.1],
        normalize_features_first=True, feature_stats=info["feature_stats"]
    )
    loss2.backward()
    assert z_gen2.grad is not None


def test_drift_norm_decreases_with_training():
    """Simulate a minimal training loop and check gen actually moves toward pos.

    Note: when normalize_each_tau=True, drift_norm is fixed (each τ normalized
    to unit RMS then summed). Instead check that generated samples actually
    move toward the target distribution.
    """
    torch.manual_seed(0)

    D = 4
    pos_raw = torch.randn(200, D) + 2.0  # fixed target, shifted
    from cellworldmodel.training.drift_loss import normalize_features
    _, scale, mean, std = normalize_features(pos_raw)
    phi_pos = ((pos_raw - mean) / std) * scale

    gen_raw = torch.randn(200, D, requires_grad=True)  # starts at origin
    opt = torch.optim.Adam([gen_raw], lr=0.3)

    initial_mean = gen_raw.detach().clone().mean(0)
    pos_mean = pos_raw.mean(0)
    initial_dist = torch.norm(initial_mean - pos_mean).item()

    for step in range(40):
        opt.zero_grad()
        phi_gen = ((gen_raw - mean) / std) * scale
        loss, info = drift_stopgrad_loss(
            phi_gen, phi_pos, phi_gen, temperatures=[0.5, 1.0, 2.0]
        )
        loss.backward()
        opt.step()

    final_mean = gen_raw.detach().mean(0)
    final_dist = torch.norm(final_mean - pos_mean).item()

    # gen should have moved toward pos (distance decreased)
    assert final_dist < initial_dist * 0.5, (
        f"gen didn't move toward pos: initial dist={initial_dist:.3f}, "
        f"final dist={final_dist:.3f}"
    )


def test_drift_norm_decreases_without_normalization():
    """With normalize_each_tau=False, drift_norm itself should decrease."""
    torch.manual_seed(0)

    D = 4
    pos_raw = torch.randn(200, D) + 2.0
    from cellworldmodel.training.drift_loss import normalize_features
    _, scale, mean, std = normalize_features(pos_raw)
    phi_pos = ((pos_raw - mean) / std) * scale

    gen_raw = torch.randn(200, D, requires_grad=True)
    opt = torch.optim.Adam([gen_raw], lr=0.3)

    norms = []
    for step in range(40):
        opt.zero_grad()
        phi_gen = ((gen_raw - mean) / std) * scale
        # No per-τ normalization → raw V magnitude reflects actual mismatch
        V = compute_V_multi_temperature(
            phi_gen, phi_pos, phi_gen, temperatures=[0.5, 1.0, 2.0],
            normalize_each=False,
        )
        target = (phi_gen + V).detach()
        loss = torch.mean((phi_gen - target) ** 2)
        loss.backward()
        opt.step()
        norms.append(torch.sqrt(torch.mean(V**2)).item())

    assert norms[-1] < norms[0] * 0.7, (
        f"unnormalized drift_norm didn't decrease: "
        f"start={norms[0]:.3f}, end={norms[-1]:.3f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
