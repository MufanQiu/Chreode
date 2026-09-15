"""Shared helpers for loading foundation latent transition checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.foundation.dynamics_train import build_foundation_dynamics_cfg
from cellworldmodel.training.checkpointing import MODEL_CONFIG_KEYS


def load_foundation_transition(
    *,
    checkpoint: str | Path | None,
    latent_dim: int,
    device: torch.device | str,
    experiment: str,
    dit_size: str,
    batch_size: int,
    k_samples: int,
    lr: float,
    action_dim: int = 0,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a W-DiT transition and optionally load matching checkpoint weights.

    The foundation checkpoints were produced across a few config schema
    versions. This loader centralizes the shape-matched restore logic so
    downstream scripts do not grow their own slightly different checkpoint
    adapters.
    """
    device = torch.device(device)
    ckpt: dict[str, Any] | None = None
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ckpt_cfg = dict(ckpt.get("config", {}))
    else:
        ckpt_cfg = {}

    if "train_cfg" in ckpt_cfg:
        method = str(ckpt_cfg.get("method", "m10"))
        train_cfg = dict(ckpt_cfg["train_cfg"])
        tau_init = float(ckpt_cfg.get("tau_init", 1.0))
    else:
        method, train_cfg, tau_init = build_foundation_dynamics_cfg(
            experiment=experiment,
            dit_size=dit_size,
            batch_size=int(batch_size),
            k_samples=int(k_samples),
            lr=float(lr),
        )
        # Downstream-benchmark checkpoints (run_intermediate_eval --save-checkpoint)
        # store their model config at the top-level "cfg" key rather than under
        # config.train_cfg. Without merging the shape-relevant keys, arms whose
        # config differs from the foundation recipe (e.g. scratch) restore only
        # the shape-coincident subset and silently keep random weights elsewhere.
        if ckpt is not None and isinstance(ckpt.get("cfg"), dict):
            train_cfg.update({k: v for k, v in ckpt["cfg"].items() if k in MODEL_CONFIG_KEYS})
            method = str(ckpt.get("method", method))
            try:
                tau_init = float(ckpt.get("tau_init", tau_init))
            except (TypeError, ValueError):
                pass
    train_cfg["action_dim"] = int(action_dim)
    train_cfg["loss_balancer"] = "fixed"
    model = build_model(method, int(latent_dim), train_cfg, tau_init=tau_init).to(device)

    if ckpt is None:
        model.eval()
        return model, {
            "loaded": 0,
            "skipped": 0,
            "base": "random_wdit",
            "method": method,
        }

    source = ckpt.get("model_state_dict", ckpt)
    target = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    target.update(matched)
    model.load_state_dict(target)
    model.eval()
    return model, {
        "loaded": int(len(matched)),
        "skipped": int(len(source) - len(matched)),
        "base": str(checkpoint),
        "method": method,
    }
