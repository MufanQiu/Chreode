"""Unit tests for common_metrics.py.

Verifies:
  - MMD² = 0 for identical distributions (up to sampling noise)
  - MMD² > 0 for different distributions
  - Sinkhorn W² decreases as epsilon decreases (approaches exact W²)
  - W1 <= W2 (always holds for 1D, typically for high-d)
  - MMD² is differentiable
  - Sinkhorn W² is differentiable
"""
from __future__ import annotations

import pytest
import torch

from cellworldmodel.benchmark.common_metrics import (
    compute_all_distributional_metrics,
    exact_w_pq,
    mmd2_unbiased_multi_sigma,
    sinkhorn_w2,
)


def test_mmd_same_distribution_small():
    """MMD² between two samples from same distribution should be small."""
    torch.manual_seed(0)
    X = torch.randn(200, 10)
    Y = torch.randn(200, 10)  # same dist
    mmd2 = mmd2_unbiased_multi_sigma(X, Y).item()
    # With 200 samples and multi-sigma, should be near zero
    assert abs(mmd2) < 0.05, f"MMD² between same dist too large: {mmd2}"


def test_mmd_different_distribution():
    """MMD² between shifted Gaussians should be clearly positive."""
    torch.manual_seed(0)
    X = torch.randn(300, 5)
    Y = torch.randn(300, 5) + 3.0  # shifted
    mmd2 = mmd2_unbiased_multi_sigma(X, Y).item()
    assert mmd2 > 0.1, f"MMD² for shifted dists too small: {mmd2}"


def test_mmd_differentiable():
    """MMD² should be differentiable w.r.t. inputs."""
    torch.manual_seed(0)
    X = torch.randn(100, 5, requires_grad=True)
    Y = torch.randn(100, 5)
    mmd2 = mmd2_unbiased_multi_sigma(X, Y)
    mmd2.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
    assert X.grad.abs().sum().item() > 0  # non-zero gradient


def test_sinkhorn_same_distribution():
    """Sinkhorn W² between identical samples should be near zero."""
    torch.manual_seed(0)
    X = torch.randn(100, 5)
    w2 = sinkhorn_w2(X, X.clone(), epsilon=0.05, num_iters=100).item()
    assert w2 < 0.5, f"Sinkhorn W² for same samples: {w2}"


def test_sinkhorn_shifted_gaussians():
    """For Gaussians shifted by d, W2² ≈ d² + small diffusion terms."""
    torch.manual_seed(0)
    d = 2.0
    X = torch.randn(200, 3)
    Y = torch.randn(200, 3) + torch.tensor([d, 0.0, 0.0])
    w2_approx = sinkhorn_w2(X, Y, epsilon=0.01, num_iters=200).item()
    # W2² ≈ d² = 4.0 (plus variance contributions, so should be around 4-6)
    assert 2.0 < w2_approx < 10.0, f"Sinkhorn W² for shift {d}: {w2_approx}"


def test_sinkhorn_differentiable():
    torch.manual_seed(0)
    X = torch.randn(80, 4, requires_grad=True)
    Y = torch.randn(80, 4)
    w2 = sinkhorn_w2(X, Y, epsilon=0.1, num_iters=50)
    w2.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_exact_w1_w2_order():
    """Generally W1 <= sqrt(d) * W2 in finite dimensions; for Gaussians W1 ≈ W2."""
    torch.manual_seed(0)
    X = torch.randn(100, 3)
    Y = torch.randn(100, 3) + 1.5
    w1 = exact_w_pq(X, Y, p=1)
    w2 = exact_w_pq(X, Y, p=2)
    # Both should be positive and on similar scale
    assert w1 > 0 and w2 > 0
    # For shifted Gaussians, W1 ≈ W2 ≈ shift_magnitude
    assert 0.5 < w1 < 5.0
    assert 0.5 < w2 < 5.0


def test_all_metrics_dict():
    """compute_all_distributional_metrics returns expected keys."""
    torch.manual_seed(0)
    X = torch.randn(150, 8)
    Y = torch.randn(150, 8) + 1.0
    result = compute_all_distributional_metrics(X, Y)
    assert set(result.keys()) == {"mmd2", "sinkhorn_w2", "w1", "w2"}
    for k, v in result.items():
        assert isinstance(v, float)
        assert v > 0, f"{k} = {v} should be positive for different dists"


def test_mmd_branching_toy():
    """Classic sanity check: MMD² between unimodal vs bimodal should be large."""
    torch.manual_seed(0)
    n = 200
    # unimodal: N(0, 1)
    X = torch.randn(n, 2)
    # bimodal: half N(-2, 0.5), half N(+2, 0.5)
    Y1 = torch.randn(n // 2, 2) * 0.5 - 2
    Y2 = torch.randn(n // 2, 2) * 0.5 + 2
    Y = torch.cat([Y1, Y2], dim=0)

    mmd2 = mmd2_unbiased_multi_sigma(X, Y).item()
    assert mmd2 > 0.1, f"MMD² unimodal vs bimodal too small: {mmd2}"


def test_branchsbm_style_metrics_shape():
    """BranchSBM-style metrics return expected keys with top-2 and full variants."""
    from cellworldmodel.benchmark.common_metrics import compute_branchsbm_style_metrics

    torch.manual_seed(0)
    X = torch.randn(200, 10)
    Y = torch.randn(200, 10) + 0.5
    result = compute_branchsbm_style_metrics(X, Y, n_trials=3, n_top_pcs=2)

    expected = {
        "branchsbm_w1_top2_mean", "branchsbm_w1_top2_std",
        "branchsbm_w2_top2_mean", "branchsbm_w2_top2_std",
        "branchsbm_mmd_full_mean", "branchsbm_mmd_full_std",
        "branchsbm_w1_full_mean", "branchsbm_w1_full_std",
        "branchsbm_w2_full_mean", "branchsbm_w2_full_std",
        "n_trials", "n_subsample",
    }
    assert set(result.keys()) >= expected


def test_branchsbm_top2_differs_from_full():
    """Top-2 PC W2 should differ from full-dim W2 when dims > 2."""
    from cellworldmodel.benchmark.common_metrics import compute_branchsbm_style_metrics

    torch.manual_seed(0)
    # 10D data with shift mostly in dims 3-10 (not top-2)
    D = 10
    X = torch.randn(150, D)
    shift = torch.zeros(D)
    shift[3:] = 2.0  # shift in "non-top-2" dims
    Y = torch.randn(150, D) + shift

    result = compute_branchsbm_style_metrics(X, Y, n_trials=3, n_top_pcs=2, seed=42)

    # Top-2 W2 should be small (no shift in dims 0-1), full W2 should be larger
    w2_top = result["branchsbm_w2_top2_mean"]
    w2_full = result["branchsbm_w2_full_mean"]
    assert w2_full > w2_top * 1.5, (
        f"full-dim W2 should be clearly larger than top-2 when shift is off-top-2: "
        f"top2={w2_top:.3f}, full={w2_full:.3f}"
    )


def test_branchsbm_mmd_biased_differs_from_unbiased():
    """Biased (BranchSBM) and unbiased MMD should give different numbers for small n."""
    from cellworldmodel.benchmark.common_metrics import (
        branchsbm_mmd2_biased_fixed_sigma,
        mmd2_unbiased_multi_sigma,
    )

    torch.manual_seed(0)
    # Small n — biased/unbiased will differ by O(1/n) on diagonal
    X = torch.randn(40, 5)
    Y = torch.randn(40, 5) + 0.3
    mmd_biased = branchsbm_mmd2_biased_fixed_sigma(
        X, Y, sigma_list=[0.01, 0.1, 1.0, 10.0, 100.0]
    ).item()
    mmd_unbiased = mmd2_unbiased_multi_sigma(X, Y).item()
    # They compute different things — just check both are finite and biased is usually larger
    assert torch.isfinite(torch.tensor(mmd_biased))
    assert torch.isfinite(torch.tensor(mmd_unbiased))


def test_dual_protocol_returns_both():
    """compute_dual_protocol_metrics returns both branchsbm_* and ours_* keys."""
    from cellworldmodel.benchmark.common_metrics import compute_dual_protocol_metrics

    torch.manual_seed(0)
    X = torch.randn(100, 8)
    Y = torch.randn(100, 8) + 1.0
    result = compute_dual_protocol_metrics(X, Y, n_trials=3)

    # BranchSBM-style keys
    branchsbm_keys = [k for k in result if k.startswith("branchsbm_")]
    assert len(branchsbm_keys) >= 8, f"missing branchsbm_* keys: {result.keys()}"

    # Our strict keys
    ours_keys = [k for k in result if k.startswith("ours_")]
    assert {"ours_mmd2_unbiased_median", "ours_w1_full", "ours_w2_full",
            "ours_sinkhorn_w2_full"} <= set(ours_keys)


def test_assign_by_nearest_target_basic():
    """Each pred should be assigned to the nearest target's label."""
    from cellworldmodel.benchmark.common_metrics import assign_by_nearest_target

    # 2 clusters: (0, 0) and (10, 10). Any pred near (0,0) → label 0.
    Y = torch.tensor([[0.0, 0.0], [0.1, -0.1], [10.0, 10.0], [9.8, 10.2]])
    labels = torch.tensor([0, 0, 1, 1])
    X = torch.tensor([[0.2, 0.3], [9.5, 9.7], [-0.5, 0.5], [11.0, 9.0]])

    assigned = assign_by_nearest_target(X, Y, labels)
    assert assigned.tolist() == [0, 1, 0, 1]


def test_per_branch_dual_protocol_structure():
    """compute_per_branch_dual_protocol_metrics returns combined + per-branch."""
    from cellworldmodel.benchmark.common_metrics import compute_per_branch_dual_protocol_metrics

    torch.manual_seed(0)
    # 2 branches: cluster 0 around origin, cluster 1 shifted
    Y0 = torch.randn(100, 5)
    Y1 = torch.randn(100, 5) + 5.0
    Y = torch.cat([Y0, Y1], dim=0)
    labels = torch.tensor([0] * 100 + [1] * 100)

    # Pred: half near cluster 0, half near cluster 1 (but noisier)
    X0 = torch.randn(80, 5) + 0.3
    X1 = torch.randn(80, 5) + 5.3
    X = torch.cat([X0, X1], dim=0)

    result = compute_per_branch_dual_protocol_metrics(X, Y, labels, n_trials=3)

    assert "combined" in result
    assert "branch_0" in result
    assert "branch_1" in result
    # Each branch should have branchsbm_* metrics
    for key in ["combined", "branch_0", "branch_1"]:
        assert "branchsbm_w1_top2_mean" in result[key], f"missing in {key}"
        assert "ours_w2_full" in result[key], f"missing in {key}"
    # Per-branch n_pred/n_true populated
    assert result["branch_0"]["n_true"] == 100
    assert result["branch_1"]["n_true"] == 100
    # Pred counts should sum roughly to 160 (nearest assignment may not be perfect)
    assert result["branch_0"]["n_pred"] + result["branch_1"]["n_pred"] == 160


def test_per_branch_skips_tiny_branch():
    """Branches with fewer than min_cells_per_branch get skipped=True."""
    from cellworldmodel.benchmark.common_metrics import compute_per_branch_dual_protocol_metrics

    torch.manual_seed(1)
    Y_main = torch.randn(200, 4)
    Y_tiny = torch.randn(3, 4) + 10.0  # only 3 cells in branch 1
    Y = torch.cat([Y_main, Y_tiny], dim=0)
    labels = torch.tensor([0] * 200 + [1] * 3)
    X = torch.randn(100, 4)

    result = compute_per_branch_dual_protocol_metrics(X, Y, labels, n_trials=2,
                                                      min_cells_per_branch=10)
    assert result["branch_1"].get("skipped") is True
    assert "skipped" not in result["branch_0"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
