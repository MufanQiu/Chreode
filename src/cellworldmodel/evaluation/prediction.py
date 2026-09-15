"""Prediction helpers shared by downstream evaluation tasks."""
from __future__ import annotations

import torch


@torch.no_grad()
def predict_at_delta(model, src: torch.Tensor, delta: float, K: int, dim: int,
                     device: torch.device, eval_batch: int = 512) -> torch.Tensor:
    preds = []
    for i in range(0, src.shape[0], eval_batch):
        src_b = src[i:i + eval_batch]
        eps = torch.randn(src_b.shape[0], K, dim, device=device)
        delta_t = torch.full((src_b.shape[0],), float(delta), device=device, dtype=src_b.dtype)
        pred_b = model(src_b, delta_t, eps).reshape(-1, dim)
        preds.append(pred_b)
    return torch.cat(preds, dim=0)
