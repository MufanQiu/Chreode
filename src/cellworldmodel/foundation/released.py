"""Strict, offline loading of versioned inference artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.foundation.encoder_input import transform_encoder_input
from cellworldmodel.foundation.vae_registry import build_foundation_vae
from cellworldmodel.model.intervention import SurgeryModel


class ReleasedIntervention(SurgeryModel):
    """Apply the released action mapping, including shared-slot controls."""

    def forward(self, z, delta, eps, cond):
        cond = torch.as_tensor(cond, device=z.device)
        if cond.ndim == 0:
            cond = cond.expand(z.shape[0])
        if cond.shape != (z.shape[0],) or cond.dtype not in (torch.int32, torch.int64):
            raise ValueError("condition IDs must be an integer scalar or one ID per cell")
        if torch.any(cond < 0) or torch.any(cond > len(self.condition_to_id)):
            raise ValueError("condition ID is outside the released mapping")
        if self.shared_condition:
            cond = torch.full_like(cond, self.adapter.shared_slot)
        return super().forward(z, delta, eps, cond)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_released_model(config: dict[str, Any]) -> torch.nn.Module:
    """Restore constructor settings, including nonpersistent time-scale buffers."""
    kind = config.get("kind")
    if kind == "encoder":
        required = {"architecture", "n_genes", "latent_dim", "leaf_to_id"}
        if required - config.keys():
            raise ValueError(f"encoder config lacks {sorted(required - config.keys())}")
        leaves = config["leaf_to_id"]
        if not isinstance(leaves, dict) or sorted(leaves.values()) != list(range(len(leaves))):
            raise ValueError("leaf_to_id must enumerate contiguous decoder covariates")
        return build_foundation_vae(
            config["architecture"], n_genes=_positive_int(config["n_genes"], "n_genes"),
            latent_dim=_positive_int(config["latent_dim"], "latent_dim"), n_batches=len(leaves),
        )
    if kind == "dynamics":
        if not all(key in config for key in ("method", "latent_dim", "tau_init", "model_cfg")):
            raise ValueError("dynamics config requires method, latent_dim, tau_init and model_cfg")
        cfg = dict(config["model_cfg"])
        required = {"hidden_dim", "n_layers", "time_emb_dim", "dit_size", "waddington_dit", "curl_rank",
                    "disable_rope", "wdit_curl_update", "wdit_curl_time_mode", "wdit_time_embedding",
                    "wdit_time_delta_transform", "wdit_time_delta_scale", "wdit_curl_time_embedding",
                    "wdit_curl_time_delta_transform", "wdit_curl_time_delta_scale", "action_dim", "growth_mode"}
        if required - cfg.keys():
            raise ValueError(f"model_cfg lacks {sorted(required - cfg.keys())}; no inferred time-scale defaults are allowed")
        tau = config["tau_init"]
        if isinstance(tau, bool) or not isinstance(tau, (float, int)) or not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau_init must be finite and positive")
        for key in ("wdit_time_delta_scale", "wdit_curl_time_delta_scale"):
            if not isinstance(cfg[key], (int, float)) or isinstance(cfg[key], bool) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        return build_model(config["method"], _positive_int(config["latent_dim"], "latent_dim"), cfg, tau_init=tau)
    if kind == "intervention":
        from cellworldmodel.model.intervention import FieldSurgeryAdapter, SurgeryModel

        base = build_released_model(config["base"])
        conditions = config["condition_to_id"]
        if not conditions or sorted(conditions.values()) != list(range(len(conditions))):
            raise ValueError("condition_to_id must enumerate contiguous action IDs")
        adapter = FieldSurgeryAdapter(config["base"]["latent_dim"], len(conditions), **config["adapter"])
        model = ReleasedIntervention(base, adapter, tune_base=config["tune_base"])
        model.condition_to_id = dict(conditions)
        model.shared_condition = config["shared_condition"]
        return model
    if kind == "rectangle":
        from cellworldmodel.model.rectangle import RectangleTransition

        scale = config["xy_scale"]
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("xy_scale must be finite and positive")
        if config["model_size"] != "small" or config["delta_scale"] != 2.5:
            raise ValueError("the released Rectangle model requires Small and delta_scale=2.5")
        return RectangleTransition(model_size="small", xy_scale=scale, seed=config["seed"],
                                   enabled=config["enabled"], include_heads=config["include_heads"])
    raise ValueError(f"unsupported released model kind: {kind!r}")


def read_release_manifest(directory: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(directory) / "release.json"
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("release manifest checksum mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), dict):
        raise ValueError("unsupported or incomplete release manifest")
    return manifest


def load_released_model(directory: str | Path, model_id: str, *, device: str | torch.device = "cpu",
                        expected_manifest_sha256: str | None = None) -> torch.nn.Module:
    root = Path(directory).resolve()
    manifest = read_release_manifest(root, expected_sha256=expected_manifest_sha256)
    if model_id not in manifest["models"]:
        raise ValueError(f"model {model_id!r} is not listed in this release")
    entry = manifest["models"][model_id]
    weights = (root / entry["weights"]).resolve()
    if not weights.is_relative_to(root):
        raise ValueError("model weight path leaves the release directory")
    if sha256(weights) != entry["sha256"]:
        raise ValueError(f"weight checksum mismatch for {model_id}")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("released weights must be a nonempty tensor-only state dictionary")
    model = build_released_model(entry["config"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    model.release_config = entry["config"]
    return model


@dataclass
class ChreodeBackbone:
    """A fixed encoder and a loaded one-step transition; methods return tensors."""

    encoder: torch.nn.Module
    dynamics: torch.nn.Module
    n_genes: int
    latent_dim: int

    @property
    def device(self) -> torch.device:
        return next(self.dynamics.parameters()).device

    def encode(self, expression, *, input_transform: str = "match-pretraining", batch_size: int = 256) -> torch.Tensor:
        """Encode aligned mapped counts, or explicitly supplied preprocessed input."""
        _positive_int(batch_size, "batch_size")
        if len(expression.shape) != 2 or expression.shape[1] != self.n_genes:
            raise ValueError(f"expression must have shape [cells, {self.n_genes}] in released vocabulary order")
        if expression.shape[0] == 0:
            return torch.empty((0, self.latent_dim), device=self.device)
        parts = []
        for start in range(0, expression.shape[0], batch_size):
            batch = expression[start:start + batch_size]
            if hasattr(batch, "toarray"):
                batch = batch.toarray()
            if isinstance(batch, torch.Tensor):
                batch = batch.detach().cpu().numpy()
            batch = transform_encoder_input(np.asarray(batch, dtype=np.float32), mode=input_transform)
            if not np.isfinite(batch).all():
                raise ValueError("encoder input contains non-finite values")
            with torch.no_grad():
                mean, _ = self.encoder.encode(torch.as_tensor(batch, dtype=torch.float32, device=self.device), None)
            parts.append(mean)
        return torch.cat(parts, dim=0)

    def _inputs(self, z, delta):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("the potential gradient needs local autograd; use torch.no_grad(), not torch.inference_mode()")
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if z.ndim != 2 or z.shape[1] != self.latent_dim or not torch.isfinite(z).all():
            raise ValueError(f"z must be finite with shape [cells, {self.latent_dim}]")
        delta = torch.as_tensor(delta, dtype=z.dtype, device=z.device)
        if delta.ndim == 0:
            delta = delta.expand(z.shape[0])
        if delta.shape != (z.shape[0],) or not torch.isfinite(delta).all() or torch.any(delta < 0):
            raise ValueError("delta must be a finite nonnegative scalar or one value per cell")
        return z, delta

    def sample(self, z, delta, *, k_samples: int = 8, noise=None, seed: int | None = None) -> torch.Tensor:
        """Return [cells, samples, latent_dim], with explicit noise supported for replay."""
        _positive_int(k_samples, "k_samples")
        z, delta = self._inputs(z, delta)
        shape = (z.shape[0], k_samples, self.latent_dim)
        if z.shape[0] == 0:
            return torch.empty(shape, device=z.device, dtype=z.dtype)
        if noise is None:
            generator = None if seed is None else torch.Generator(device=z.device).manual_seed(seed)
            noise = torch.randn(shape, device=z.device, dtype=z.dtype, generator=generator)
        else:
            noise = torch.as_tensor(noise, device=z.device, dtype=z.dtype)
            if noise.shape != shape or not torch.isfinite(noise).all():
                raise ValueError(f"noise must be finite with shape {shape}")
        with torch.no_grad():
            return self.dynamics(z, delta, noise)

    def predict(self, z, delta) -> torch.Tensor:
        """Return the deterministic conditional center, with shape [cells, latent_dim]."""
        z, delta = self._inputs(z, delta)
        noise = torch.zeros((z.shape[0], 1, self.latent_dim), device=z.device, dtype=z.dtype)
        return self.sample(z, delta, k_samples=1, noise=noise)[:, 0]

    def decode(self, z) -> torch.Tensor:
        """Decode through the shared/null-leaf decoder to normalized expression space."""
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if z.ndim < 2 or z.shape[-1] != self.latent_dim or not torch.isfinite(z).all():
            raise ValueError("z has an invalid latent dimension or non-finite values")
        shape = z.shape[:-1]
        with torch.no_grad():
            decoded = self.encoder.decode(z.reshape(-1, self.latent_dim), None)
        return decoded.reshape(*shape, self.n_genes)


def load_chreode_backbone(directory: str | Path, *, device: str | torch.device = "cpu",
                         encoder_id: str = "encoder", dynamics_id: str = "dynamics",
                         expected_manifest_sha256: str | None = None) -> ChreodeBackbone:
    encoder = load_released_model(directory, encoder_id, device=device, expected_manifest_sha256=expected_manifest_sha256)
    dynamics = load_released_model(directory, dynamics_id, device=device, expected_manifest_sha256=expected_manifest_sha256)
    ecfg, dcfg = encoder.release_config, dynamics.release_config
    if ecfg["kind"] != "encoder" or dcfg["kind"] != "dynamics" or ecfg["latent_dim"] != dcfg["latent_dim"]:
        raise ValueError("encoder and dynamics do not share a released state space")
    return ChreodeBackbone(encoder, dynamics, ecfg["n_genes"], ecfg["latent_dim"])
