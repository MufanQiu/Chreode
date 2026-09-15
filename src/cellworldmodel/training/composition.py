"""Population semigroup consistency on observed training transitions."""
from __future__ import annotations

import torch


def valid_two_hop_paths(sampler):
    pairs = {pair for i, pair in enumerate(sampler.pairs)
             if sampler.pair_probs is None or sampler.pair_probs[i] > 0}
    return sorted((start, middle, end) for start, middle in pairs
                  for end in sampler.timepoints
                  if start < middle < end and (middle, end) in pairs
                  and (start, end) in pairs)


def two_hop_clouds(model, source, delta1, delta2, k):
    """Pair direct/first-hop noise; continue each particle with fresh K=1 noise."""
    n, dim = source.shape
    eps = torch.randn(n, k, dim, device=source.device, dtype=source.dtype)
    delta = source.new_full((n,), float(delta1 + delta2))
    direct = model(source, delta, eps).reshape(-1, dim)
    first = model(source, source.new_full((n,), float(delta1)), eps).reshape(-1, dim)
    next_eps = torch.randn(len(first), 1, dim, device=source.device, dtype=source.dtype)
    composed = model(first, source.new_full((len(first),), float(delta2)), next_eps,
                     preserve_source_grad=True)
    return direct, composed.reshape(-1, dim)
