"""Named pretraining objective protocols for foundation ablations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PretrainProtocolSpec:
    name: str
    objective: str
    description: str
    freezes_vae: bool = True
    uses_temporal_pairs: bool = False
    uses_action: bool = False


PRETRAIN_PROTOCOLS: dict[str, PretrainProtocolSpec] = {
    "vae_warmup": PretrainProtocolSpec(
        name="vae_warmup",
        objective="vae_reconstruction",
        description="Train only the VAE encoder/decoder reconstruction objective.",
        freezes_vae=False,
    ),
    "static_dit_reconstruction": PretrainProtocolSpec(
        name="static_dit_reconstruction",
        objective="static_reconstruction",
        description=(
            "Freeze VAE and train DiT as an unconditional latent bridge whose "
            "decoded output reconstructs the input cell."
        ),
    ),
    "temporal_dynamics": PretrainProtocolSpec(
        name="temporal_dynamics",
        objective="temporal_transition",
        description="Freeze VAE and train selected W-DiT on reference temporal population transitions.",
        uses_temporal_pairs=True,
    ),
}


def get_pretrain_protocol(name: str) -> PretrainProtocolSpec:
    try:
        return PRETRAIN_PROTOCOLS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown pretrain protocol={name!r}; expected one of {sorted(PRETRAIN_PROTOCOLS)}") from exc


def pretrain_protocol_names() -> list[str]:
    return sorted(PRETRAIN_PROTOCOLS)
