"""Checkpoint helpers for benchmark models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


MODEL_CONFIG_KEYS = {
    "hidden_dim",
    "n_layers",
    "time_emb_dim",
    "noise_dim",
    "dit_size",
    "waddington_dit",
    "curl_rank",
    "disable_rope",
    "state_chunk_dim",
    "learned_state_tokens",
    "dit_time_embedding",
    "dit_time_delta_transform",
    "dit_time_delta_scale",
    "dit_mean_prediction",
    "wdit_curl_update",
    "wdit_curl_time_mode",
    "wdit_hybrid_delta0",
    "wdit_hybrid_slope",
    "wdit_hard_delta0",
    "wdit_time_embedding",
    "wdit_time_delta_transform",
    "wdit_time_delta_scale",
    "wdit_curl_time_embedding",
    "wdit_curl_time_delta_transform",
    "wdit_curl_time_delta_scale",
    "growth_integration_steps",
}


def build_checkpoint_payload(model, **metadata: Any) -> dict[str, Any]:
    payload = dict(metadata)
    payload["model_state_dict"] = model.state_dict()
    payload["schema_version"] = 1
    return payload


def save_model_checkpoint(path: str | Path, model, **metadata: Any) -> Path:
    return save_checkpoint_payload(path, build_checkpoint_payload(model, **metadata))


def save_checkpoint_payload(path: str | Path, payload: dict[str, Any], *, provenance=None) -> Path:
    """Shared verification for model, adapter, and multi-network checkpoints."""
    if provenance is not None:
        from cellworldmodel.provenance.writer import attach_provenance
        payload = attach_provenance(payload, provenance)
    provenance = payload.get("provenance")
    if isinstance(provenance, dict) and provenance.get("schema_version") == 2:
        from cellworldmodel.provenance.models import ArtifactProvenance
        from cellworldmodel.provenance.writer import verify_inputs
        verify_inputs(ArtifactProvenance.model_validate(provenance))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_model_checkpoint(path: str | Path, map_location=None) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def checkpoint_model_config(path: str | Path) -> dict[str, Any]:
    """Return architecture-relevant config from a benchmark/foundation checkpoint."""
    ckpt = load_model_checkpoint(path, map_location="cpu")
    config = ckpt.get("config", {})
    train_cfg = config.get("train_cfg") if isinstance(config, dict) else None
    if isinstance(train_cfg, dict):
        return {k: v for k, v in train_cfg.items() if k in MODEL_CONFIG_KEYS}
    cfg = ckpt.get("cfg", {})
    if isinstance(cfg, dict):
        return {k: v for k, v in cfg.items() if k in MODEL_CONFIG_KEYS}
    return {}


def apply_model_config_from_checkpoint(cfg: dict, path: str | Path) -> dict[str, Any]:
    """Update `cfg` in-place with model-shape config from `path` and return applied keys."""
    updates = checkpoint_model_config(path)
    cfg.update(updates)
    return updates


def load_shape_matched_checkpoint(model, path: str | Path, *, min_match_ratio: float = 0.8) -> dict[str, Any]:
    """Load all checkpoint tensors whose names and shapes match `model`.

    This is useful for pretraining/fine-tuning transfers where downstream
    modules may add small heads. A low match ratio is treated as an error so an
    accidental dim/architecture mismatch cannot silently become a scratch run.
    """
    ckpt = load_model_checkpoint(path, map_location="cpu")
    source = ckpt.get("model_state_dict", ckpt)
    target = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    ratio = len(matched) / max(1, len(target))
    if ratio < float(min_match_ratio):
        skipped = [key for key in source if key not in matched][:12]
        raise RuntimeError(
            f"Only matched {len(matched)}/{len(target)} tensors from {path} "
            f"(ratio={ratio:.3f} < {min_match_ratio}); first skipped={skipped}"
        )
    target.update(matched)
    model.load_state_dict(target)
    return {
        "path": str(path),
        "loaded": int(len(matched)),
        "target": int(len(target)),
        "source": int(len(source)),
        "match_ratio": float(ratio),
        "skipped": int(len(source) - len(matched)),
    }
