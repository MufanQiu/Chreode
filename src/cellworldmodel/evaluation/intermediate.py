"""Intermediate-time evaluation for timepoint benchmark adapters."""
from __future__ import annotations

import torch

from cellworldmodel.benchmark.branchsbm_adapter import MouseHematopoiesisAdapter, VeresAdapter
from cellworldmodel.benchmark.common_metrics import compute_dual_protocol_metrics
from cellworldmodel.evaluation.prediction import predict_at_delta
from cellworldmodel.training.transition_sampler import TimepointTransitionSampler


def get_intermediate_target(adapter, t: float) -> torch.Tensor:
    """Get ground-truth cells at an intermediate timepoint."""
    if isinstance(adapter, VeresAdapter):
        return adapter.get_intermediate(int(t))
    if isinstance(adapter, MouseHematopoiesisAdapter):
        if float(t) in adapter.coords_by_t:
            return torch.from_numpy(adapter.coords_by_t[float(t)])
        raise ValueError(f"Mouse: no timepoint {t} (available: {adapter.timepoints})")
    if hasattr(adapter, "coords_by_t"):
        key = float(t) if float(t) in adapter.coords_by_t else int(t) if int(t) in adapter.coords_by_t else None
        if key is not None:
            return torch.from_numpy(adapter.coords_by_t[key])
    raise ValueError(f"Dataset type {type(adapter).__name__} does not support intermediate eval at t={t}")


@torch.no_grad()
def evaluate_intermediate(model, adapter, device, cfg: dict, seed: int,
                          n_source_subsample: int = 1000, max_preds: int = 2000,
                          max_target: int = 4000,
                          sampler: TimepointTransitionSampler | None = None) -> dict:
    """Evaluate predictions at every non-source timepoint."""
    model.eval()
    torch.manual_seed(seed + 1)
    src_time = adapter.timepoints[0]

    if sampler is not None:
        src_full = sampler.get_population(src_time, split="test")
    else:
        test_batch = adapter.get_transition(split="test")
        src_full = test_batch.source
    n_src = src_full.shape[0]
    if n_src > n_source_subsample:
        idx = torch.randperm(n_src)[:n_source_subsample]
        src = src_full[idx].to(device)
    else:
        src = src_full.to(device)

    results: dict[str, dict] = {}
    for t in adapter.timepoints:
        if t == src_time:
            continue
        delta = float(t) - float(src_time)
        try:
            tgt_np = get_intermediate_target(adapter, float(t))
        except ValueError as exc:
            print(f"  t={t}: skipped ({exc})")
            continue
        if sampler is not None:
            tgt = sampler.get_population(float(t), split="test").to(device)
        else:
            tgt = tgt_np.to(device) if torch.is_tensor(tgt_np) else torch.from_numpy(tgt_np).to(device)
        if tgt.shape[0] > max_target:
            idx_t = torch.randperm(tgt.shape[0], device=device)[:max_target]
            tgt = tgt[idx_t]

        preds = predict_at_delta(model, src, delta, cfg["K"], adapter.dim, device)
        pred_sample_weight = None
        if str(cfg.get("growth_mode", "frozen")) == "learned" and hasattr(model, "predict_mass"):
            delta_t = torch.full((src.shape[0],), delta, device=device, dtype=src.dtype)
            pred_mass = model.predict_mass(src, delta_t)
            pred_sample_weight = pred_mass.repeat_interleave(cfg["K"])
        if preds.shape[0] > max_preds:
            idx_p = torch.randperm(preds.shape[0], device=device)[:max_preds]
            preds = preds[idx_p]
            if pred_sample_weight is not None:
                pred_sample_weight = pred_sample_weight[idx_p]

        if getattr(adapter, "standardize_metrics", False):
            preds = adapter.transform_for_metrics(preds)
            tgt = adapter.transform_for_metrics(tgt)

        metrics = compute_dual_protocol_metrics(
            preds,
            tgt,
            seed=seed,
            weight_pred=pred_sample_weight,
        )
        results[f"t={t}"] = {
            "delta": delta,
            "n_pred": int(preds.shape[0]),
            "n_true": int(tgt.shape[0]),
            "branchsbm_w1_top2_mean": metrics["branchsbm_w1_top2_mean"],
            "branchsbm_w2_top2_mean": metrics["branchsbm_w2_top2_mean"],
            "branchsbm_mmd_full_mean": metrics["branchsbm_mmd_full_mean"],
            "branchsbm_w1_full_mean": metrics["branchsbm_w1_full_mean"],
            "branchsbm_w2_full_mean": metrics["branchsbm_w2_full_mean"],
            "ours_w1_full": metrics["ours_w1_full"],
            "ours_w2_full": metrics["ours_w2_full"],
            "ours_mmd2_unbiased_median": metrics["ours_mmd2_unbiased_median"],
        }
        print(
            f"  t={t:>3} (Δ={delta:>4.1f}): "
            f"W1_top2={metrics['branchsbm_w1_top2_mean']:.3f}, "
            f"W2_top2={metrics['branchsbm_w2_top2_mean']:.3f}, "
            f"W1_full={metrics['branchsbm_w1_full_mean']:.3f}, "
            f"W2_full={metrics['branchsbm_w2_full_mean']:.3f}, "
            f"MMD_full={metrics['branchsbm_mmd_full_mean']:.4f}"
        )
    return results
