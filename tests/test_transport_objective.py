import pytest
import torch

from cellworldmodel.training.transport_objective import training_transport


def test_sinkhorn_divergence_debiases_self_and_has_finite_shift_gradient():
    x = torch.tensor([[0., 0.], [.1, 0.], [.2, .1]], requires_grad=True)
    cfg = {'transport_objective': 'sinkhorn_divergence', 'transport_blur': .05}
    self_loss = training_transport(x, x.detach(), cfg)
    assert abs(self_loss.item()) < 1e-7
    loss = training_transport(x, x.detach() + 1., cfg)
    gradient, = torch.autograd.grad(loss, x)
    assert loss.item() > .5
    assert torch.isfinite(gradient).all() and gradient.norm().item() > .1
    weighted = training_transport(x, x.detach() + 1., cfg, weight_x=torch.ones(len(x)))
    torch.testing.assert_close(weighted, loss)


@pytest.mark.parametrize('blur', [0., -1., float('nan')])
def test_sinkhorn_divergence_rejects_invalid_blur(blur):
    x = torch.zeros(2, 2)
    with pytest.raises(ValueError, match='finite and positive'):
        training_transport(x, x, {'transport_objective': 'sinkhorn_divergence',
                                 'transport_blur': blur})
