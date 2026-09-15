from __future__ import annotations

import torch

from cellworldmodel.training.loss_balancer import (
    LossComponent,
    build_loss_balancer,
)


def make_components(x: torch.Tensor):
    return [
        LossComponent("mmd", (x - 1).pow(2).mean(), 1.0),
        LossComponent("w2", (2 * x + 1).pow(2).mean(), 1.0),
        LossComponent("drift", (x + 0.5).pow(2).mean(), 1.0),
        LossComponent("down", (x - 0.25).pow(2).mean(), 0.1),
    ]


def test_fixed_balancer_uses_base_weights():
    x = torch.tensor([0.5, -0.25], requires_grad=True)
    components = make_components(x)
    balancer = build_loss_balancer({"loss_balancer": "fixed"}, [c.name for c in components])
    loss, info = balancer.combine(components, step=0)
    expected = sum(c.base_weight * c.loss for c in components)
    assert torch.allclose(loss, expected)
    assert info["loss_weight/mmd"] == 1.0
    assert abs(info["loss_weight/down"] - 0.1) < 1e-6


def test_uncertainty_balancer_has_trainable_parameters():
    x = torch.tensor([0.5, -0.25], requires_grad=True)
    components = make_components(x)
    balancer = build_loss_balancer({"loss_balancer": "uncertainty"}, [c.name for c in components])
    loss, info = balancer.combine(components, step=0)
    loss.backward()
    params = list(balancer.parameters())
    assert params
    assert all(p.grad is not None for p in params)
    assert "loss_log_var/mmd" in info


def test_stateful_balancers_return_finite_weights():
    for name in ("dwa", "relobralo", "rlw"):
        x = torch.tensor([0.5, -0.25], requires_grad=True)
        components = make_components(x)
        cfg = {"loss_balancer": name, "loss_balancer_temperature": 1.0}
        balancer = build_loss_balancer(cfg, [c.name for c in components], seed=0)
        for step in range(3):
            loss, info = balancer.combine(components, step=step)
            assert torch.isfinite(loss)
            assert all(torch.isfinite(torch.tensor(v)) for k, v in info.items() if "loss_weight/" in k)


def test_gradnorm_lite_changes_weights_from_gradient_norms():
    x = torch.tensor([0.5, -0.25], requires_grad=True)
    components = make_components(x)
    balancer = build_loss_balancer({"loss_balancer": "gradnorm_lite"}, [c.name for c in components])
    loss, info = balancer.combine(components, step=0, model_params=[x])
    assert torch.isfinite(loss)
    assert "loss_grad_norm/mmd" in info
    assert "loss_weight/mmd" in info
