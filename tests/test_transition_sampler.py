import numpy as np

from cellworldmodel.training.transition_sampler import TimepointTransitionSampler


class DummyAdapter:
    def __init__(self):
        self.coords_by_t = {
            0.0: np.arange(30, dtype=np.float32).reshape(10, 3),
            1.0: np.arange(60, dtype=np.float32).reshape(20, 3),
            2.0: np.arange(90, dtype=np.float32).reshape(30, 3),
        }
        self.timepoints = [0.0, 1.0, 2.0]
        self.dim = 3


def test_sampler_batch_shapes_and_transition_metadata():
    sampler = TimepointTransitionSampler(DummyAdapter(), split_seed=123)
    rng = np.random.default_rng(5)
    batch = sampler.sample_train_batch(batch_size=4, rng=rng)
    assert batch.source.shape == (4, 3)
    assert batch.target.shape == (4, 3)
    assert batch.target_t > batch.source_t
    assert batch.delta == batch.target_t - batch.source_t


def test_sampler_reference_target_uses_train_split():
    sampler = TimepointTransitionSampler(DummyAdapter(), split_seed=123, reference_target_times=[1.0, 2.0])
    ref = sampler.reference_target(split="train")
    expected = len(sampler.splits[1.0].train) + len(sampler.splits[2.0].train)
    assert ref.shape == (expected, 3)


def test_endpoint_probability_targets_endpoint_pair():
    sampler = TimepointTransitionSampler(DummyAdapter(), split_seed=123, endpoint_prob=0.5)
    assert sampler.endpoint_pair == (0.0, 2.0)
    endpoint_idx = sampler.pairs.index((0.0, 2.0))
    assert sampler.pair_probs[endpoint_idx] == 0.5
