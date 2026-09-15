import numpy as np
import pytest
import torch

from cellworldmodel.training.benchmark_loop import train_method
from cellworldmodel.training.transition_sampler import TimepointTransitionSampler


class Adapter:
    dim = 2
    timepoints = [0., 1.]
    coords_by_t = {t: np.random.default_rng(int(t)).normal(size=(24, 2)).astype('float32') for t in timepoints}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(2))

    def forward(self, source, delta, noise):
        return source[:, None, :] + self.offset + .1 * noise


@pytest.mark.parametrize('objective', ['legacy', 'sinkhorn_divergence'])
def test_exact_budget_continues_after_stop_but_freezes_selection(objective):
    cfg = dict(lr=.01, batch_size=4, K=2, sinkhorn_eps=.1,
               lambda_mmd=1., lambda_w2=1., grad_clip=1.,
               transport_objective=objective, transport_blur=.05)
    scores = [1., 2., .1, .05]
    snapshots = {}

    def save(epoch, model, info):
        snapshots[epoch + 1] = model.offset.detach().clone()

    model = Model()
    history = train_method('m1', Adapter(), model, torch.device('cpu'), cfg, epochs=8, seed=0,
                           sampler=TimepointTransitionSampler(Adapter(), split_seed=42),
                           validation_callback=lambda ep: {'val_w2_mean': scores[ep]},
                           validation_every=1, early_stopping_patience=1,
                           checkpoint_callback=save, checkpoint_updates=[1, 2, 4])
    assert set(snapshots) == {1, 2, 4}
    assert [h['epoch'] for h in history if h.get('event') == 'early_stopping'] == [1]
    assert history[-1]['event'] == 'best_validation' and history[-1]['epoch'] == 0
    assert torch.equal(model.offset, snapshots[1])
    assert not torch.equal(snapshots[2], snapshots[4])
