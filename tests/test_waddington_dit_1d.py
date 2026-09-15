from __future__ import annotations

import torch

from cellworldmodel.model.waddington_dit_1d import WaddingtonDiT1D


def make_model(dim: int = 8) -> WaddingtonDiT1D:
    return WaddingtonDiT1D(
        dim=dim,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        num_register_tokens=2,
        time_emb_dim=64,
        curl_rank=3,
        tau_init=1.0,
    )


def test_forward_shape_and_zero_delta_identity():
    torch.manual_seed(0)
    model = make_model(dim=8)
    z = torch.randn(4, 8)
    eps = torch.randn(4, 5, 8)
    out = model(z, torch.ones(4), eps)
    assert out.shape == (4, 5, 8)
    out0 = model(z, torch.zeros(4), eps)
    assert torch.allclose(out0, z[:, None, :].expand_as(out0), atol=1e-6)


def test_explicit_noise_term_changes_samples():
    torch.manual_seed(0)
    model = make_model(dim=6)
    z = torch.randn(3, 6)
    delta = torch.ones(3)
    eps1 = torch.randn(3, 4, 6)
    eps2 = torch.randn(3, 4, 6)
    out1 = model(z, delta, eps1)
    out2 = model(z, delta, eps2)
    assert not torch.allclose(out1, out2)


def test_curl_component_is_perpendicular_to_state():
    torch.manual_seed(0)
    model = make_model(dim=7)
    z = torch.randn(5, 7)
    delta = torch.ones(5)
    h = model._features(z, delta)
    s_z, _, _ = model._curl_from_features(h, z)
    dot = (z * s_z).sum(-1)
    assert dot.abs().max().item() < 1e-4


def test_gradient_flows_through_potential_and_curl_heads():
    torch.manual_seed(0)
    model = make_model(dim=6)
    model.train()
    z = torch.randn(4, 6)
    delta = torch.rand(4) + 0.1
    eps = torch.randn(4, 3, 6)
    out = model(z, delta, eps)
    loss = out.pow(2).mean()
    loss.backward()
    assert model.U_head.weight.grad is not None
    assert model.U_head.weight.grad.abs().sum().item() > 0
    assert model.A_head.weight.grad is not None
    assert model.A_head.weight.grad.abs().sum().item() > 0


def test_predict_mean_is_noise_free_and_potential_shape():
    torch.manual_seed(0)
    model = make_model(dim=5)
    z = torch.randn(3, 5)
    delta = torch.ones(3)
    mean1 = model.predict_mean(z, delta)
    mean2 = model.predict_mean(z, delta, n_mc=2, antithetic=True)
    assert mean1.shape == (3, 5)
    assert torch.allclose(mean1, mean2, atol=1e-6)
    pot = model.compute_potential(z, delta)
    assert pot.shape == (3,)


def test_cayley_rotation_preserves_norm():
    torch.manual_seed(0)
    model = make_model(dim=6)
    z = torch.randn(4, 6)
    delta = torch.ones(4)
    alpha = model.alpha_gate(delta)
    h = model._features(z, delta)
    _, p_mat, q_mat = model._curl_from_features(h, z)
    rotated = model._cayley_rotate(z, alpha, p_mat, q_mat)
    assert torch.allclose(
        torch.linalg.norm(rotated, dim=-1),
        torch.linalg.norm(z, dim=-1),
        atol=1e-4,
        rtol=1e-4,
    )


def test_cayley_direct_and_residual_forms_are_equivalent():
    torch.manual_seed(0)
    model = make_model(dim=6)
    z = torch.randn(4, 6)
    delta = torch.rand(4) + 0.1
    eps = torch.randn(4, 3, 6)
    model.curl_update = "cayley_direct"
    direct = model(z, delta, eps)
    model.curl_update = "cayley_residual"
    residual = model(z, delta, eps)
    assert torch.allclose(direct, residual, atol=1e-5, rtol=1e-5)


def test_cayley_update_receives_gradients():
    torch.manual_seed(0)
    model = make_model(dim=6)
    model.curl_update = "cayley_direct"
    z = torch.randn(4, 6)
    delta = torch.rand(4) + 0.1
    eps = torch.randn(4, 3, 6)
    out = model(z, delta, eps)
    out.pow(2).mean().backward()
    assert model.A_head.weight.grad is not None
    assert model.A_head.weight.grad.abs().sum().item() > 0


def test_hybrid_delta_interpolates_additive_and_cayley():
    torch.manual_seed(0)
    model = make_model(dim=6)
    z = torch.randn(4, 6)
    eps = torch.randn(4, 3, 6)
    model.hybrid_delta0 = 10.0
    model.hybrid_slope = 10.0
    model.curl_update = "hybrid_delta"
    low = model(z, torch.full((4,), 1.0), eps)
    model.curl_update = "additive"
    additive = model(z, torch.full((4,), 1.0), eps)
    assert torch.allclose(low, additive, atol=1e-3, rtol=1e-3)
    model.curl_update = "hybrid_delta"
    high = model(z, torch.full((4,), 30.0), eps)
    model.curl_update = "cayley_direct"
    cayley = model(z, torch.full((4,), 30.0), eps)
    assert torch.allclose(high, cayley, atol=1e-3, rtol=1e-3)


def test_hard_delta_switches_from_additive_to_cayley_residual():
    torch.manual_seed(0)
    model = make_model(dim=6)
    z = torch.randn(4, 6)
    eps = torch.randn(4, 3, 6)
    model.hard_delta0 = 10.0
    model.curl_update = "hard_delta_cayley_residual"
    low = model(z, torch.full((4,), 1.0), eps)
    model.curl_update = "additive"
    additive = model(z, torch.full((4,), 1.0), eps)
    assert torch.allclose(low, additive, atol=1e-6, rtol=1e-6)
    model.curl_update = "hard_delta_cayley_residual"
    high = model(z, torch.full((4,), 30.0), eps)
    model.curl_update = "cayley_residual"
    cayley = model(z, torch.full((4,), 30.0), eps)
    assert torch.allclose(high, cayley, atol=1e-6, rtol=1e-6)


def test_state_only_curl_ignores_delta_for_curl_factors():
    torch.manual_seed(0)
    model = make_model(dim=6)
    model.curl_time_mode = "state_only"
    z = torch.randn(4, 6)
    h1 = model._curl_features(z, torch.full((4,), 1.0), model._features(z, torch.full((4,), 1.0)))
    h2 = model._curl_features(z, torch.full((4,), 30.0), model._features(z, torch.full((4,), 30.0)))
    assert torch.allclose(h1, h2, atol=1e-6)


def test_separate_curl_time_embedding_uses_distinct_conditioning():
    torch.manual_seed(0)
    model = WaddingtonDiT1D(
        dim=6,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        num_register_tokens=2,
        time_emb_dim=64,
        curl_rank=3,
        tau_init=8.0,
        time_embedding_mode="legacy_fourier",
        curl_time_mode="separate",
        curl_time_embedding_mode="bounded_lowfreq_fourier",
    )
    delta = torch.full((4,), 8.0)
    h_full = model.time_embed(delta)
    h_curl = model.curl_time_embed(delta)
    assert h_curl.shape == h_full.shape
    assert not torch.allclose(h_curl, h_full)


def test_waddington_regularization_has_gradients():
    torch.manual_seed(0)
    model = make_model(dim=6)
    z = torch.randn(4, 6)
    delta = torch.ones(4)
    reg = model.waddington_regularization(z, delta)
    loss = reg["wdit_curl_sq"] + reg["wdit_a_fro"]
    loss.backward()
    assert model.A_head.weight.grad is not None
    assert model.A_head.weight.grad.abs().sum().item() > 0


def test_lowfreq_time_embedding_forward_and_scale():
    torch.manual_seed(0)
    model = WaddingtonDiT1D(
        dim=6,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        num_register_tokens=2,
        time_emb_dim=64,
        curl_rank=3,
        tau_init=8.0,
        time_embedding_mode="bounded_lowfreq_fourier",
    )
    assert abs(model.time_delta_scale - 8.0 * torch.log(torch.tensor(2.0)).item()) < 1e-6
    z = torch.randn(4, 6)
    eps = torch.randn(4, 3, 6)
    out = model(z, torch.tensor([0.0, 1.0, 8.0, 16.0]), eps)
    assert out.shape == (4, 3, 6)
    assert torch.isfinite(out).all()


def test_time2vec_time_embedding_has_trainable_frequency_gradients():
    torch.manual_seed(0)
    model = WaddingtonDiT1D(
        dim=6,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        num_register_tokens=2,
        time_emb_dim=64,
        curl_rank=3,
        tau_init=8.0,
        time_embedding_mode="time2vec",
    )
    emb = model.time_embed(torch.tensor([1.0, 2.0, 4.0, 8.0]))
    emb.pow(2).mean().backward()
    assert model.time_embed.raw_omega.grad is not None
    assert model.time_embed.raw_omega.grad.abs().sum().item() > 0
