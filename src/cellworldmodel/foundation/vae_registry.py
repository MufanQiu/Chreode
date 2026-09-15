"""Registry for foundation VAE architectures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn

from cellworldmodel.foundation.vae_model import (
    EncoderNoBatchDecoderResidualVAE,
    Log1pGaussianVAE,
    ResidualGaussianVAE,
    ScviStyleGaussianVAE,
    StateTokenGaussianVAE,
)


Builder = Callable[..., nn.Module]


@dataclass(frozen=True)
class VaeArchitectureSpec:
    name: str
    description: str
    builder: Builder
    default_kwargs: dict = field(default_factory=dict)

    def build(self, *, n_genes: int, latent_dim: int, n_batches: int = 0, **overrides) -> nn.Module:
        kwargs = dict(self.default_kwargs)
        kwargs.update(overrides)
        return self.builder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            n_batches=n_batches,
            **kwargs,
        )


VAE_ARCHITECTURES: dict[str, VaeArchitectureSpec] = {}


def register_vae_architecture(spec: VaeArchitectureSpec) -> None:
    if spec.name in VAE_ARCHITECTURES:
        raise ValueError(f"Duplicate VAE architecture: {spec.name}")
    VAE_ARCHITECTURES[spec.name] = spec


def get_vae_architecture(name: str) -> VaeArchitectureSpec:
    try:
        return VAE_ARCHITECTURES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown VAE architecture={name!r}; expected one of {sorted(VAE_ARCHITECTURES)}"
        ) from exc


def vae_architecture_names() -> list[str]:
    return sorted(VAE_ARCHITECTURES)


def build_foundation_vae(
    architecture: str,
    *,
    n_genes: int,
    latent_dim: int,
    n_batches: int = 0,
    **overrides,
) -> nn.Module:
    return get_vae_architecture(architecture).build(
        n_genes=n_genes,
        latent_dim=latent_dim,
        n_batches=n_batches,
        **overrides,
    )


register_vae_architecture(VaeArchitectureSpec(
    name="mlp512",
    description="Dense MLP Gaussian VAE baseline, hidden=512, layers=3.",
    builder=Log1pGaussianVAE,
    default_kwargs={"hidden_dim": 512, "n_layers": 3},
))

register_vae_architecture(VaeArchitectureSpec(
    name="scvi_fclayers1024",
    description="scVI-inspired FCLayers VAE with deep covariate injection.",
    builder=ScviStyleGaussianVAE,
    default_kwargs={"hidden_dim": 1024, "n_layers": 3},
))

register_vae_architecture(VaeArchitectureSpec(
    name="strict_scvi1024",
    description="Strict zero-shot VAE: encoder ignores batch, decoder has optional leaf residual.",
    builder=EncoderNoBatchDecoderResidualVAE,
    default_kwargs={"hidden_dim": 1024, "n_layers": 3, "decoder_batch_dropout": 0.3},
))

register_vae_architecture(VaeArchitectureSpec(
    name="resmlp1024",
    description="Residual MLP Gaussian VAE, hidden=1024, layers=4.",
    builder=ResidualGaussianVAE,
    default_kwargs={"hidden_dim": 1024, "n_layers": 4},
))

register_vae_architecture(VaeArchitectureSpec(
    name="state_token_small",
    description="STATE-inspired top-expressed-gene token encoder prototype.",
    builder=StateTokenGaussianVAE,
    default_kwargs={"hidden_dim": 256, "n_layers": 2, "n_heads": 4},
))
