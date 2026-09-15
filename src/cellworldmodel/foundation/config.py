"""Configuration schema for reference foundation-model workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


VALID_GENE_ORDERS = {"mouse_unified_order_filtered"}
VALID_VAE_POLICIES = {"vae_b_log1p_normal"}
VALID_BATCH_STRATEGIES = {
    "b1_leaf_dataset",
    "b2_leaf_covariate",
    "b2_encoder_nobatch_decoder_residual",
    "b3_none",
}
VALID_TRANSITION_PAIR_POLICIES = {"all_ordered", "adjacent"}
VALID_WANDB_MODES = {"online", "offline", "disabled"}


@dataclass(frozen=True)
class GeneVocabConfig:
    source: str
    canonical_order: str = "mouse_unified_order_filtered"


@dataclass(frozen=True)
class SplitConfig:
    split_seed: int = 42
    heldout_families: tuple[str, ...] = ("GSE275562",)
    external: tuple[str, ...] = ("Norman",)
    split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)


@dataclass(frozen=True)
class VaeConfig:
    policy: str = "vae_b_log1p_normal"
    batch_strategy: str = "b1_leaf_dataset"
    smoke_batch_strategies: tuple[str, ...] = ("b1_leaf_dataset", "b3_none")
    latent_dims: tuple[int, ...] = (128, 256)
    hidden_dim: int = 512
    n_layers: int = 3
    batch_size: int = 4096
    smoke_batch_size: int = 256
    max_epochs: int = 100
    smoke_epochs: int = 10
    smoke_steps: int = 200
    qc_sample_size: int = 2048
    throughput_batch_sizes: tuple[int, ...] = (1024, 2048, 4096)
    throughput_steps: int = 50
    throughput_latent_dims: tuple[int, ...] = (128, 256)
    throughput_batch_strategies: tuple[str, ...] = ("b1_leaf_dataset",)
    arch_search: tuple[dict, ...] = field(default_factory=tuple)
    leaf_sampling_alpha: float = 0.5
    full_name: str = "scvi1024_l128_vae2"
    full_architecture: str = "scvi_fclayers1024"
    full_latent_dim: int = 128
    full_batch_strategy: str = "b1_leaf_dataset"
    full_batch_size: int = 2048
    full_epochs: int = 2
    latent_cache_splits: tuple[str, ...] = ("train", "val", "test")
    latent_cache_batch_size: int = 512
    latent_cache_shard_size: int = 50_000


@dataclass(frozen=True)
class DynamicsConfig:
    experiment: str = "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw"
    controls: tuple[str, ...] = (
        "g2a_m10_wdit_time2vecu_lowfreqcurl_adamw",
        "g2a_m10_md_adamw",
    )
    dit_size: str = "small"
    transition_pairs: str = "all_ordered"
    max_steps: int = 100_000
    smoke_steps: int = 20_000
    batch_size: int = 1024
    k_samples: int = 8
    pretrain_epoch_equivalent: int = 2
    static_objective: str = "static_dit_reconstruction"
    temporal_objective: str = "temporal_dynamics"
    static_name: str = "vae2_staticdit2"
    temporal_name: str = "vae2_dynamicsdit2"
    lr: float = 3e-4


@dataclass(frozen=True)
class ResourceConfig:
    partition: str = "blackwell,a100"
    cpus: int = 12
    mem_mb: int = 120_000
    runtime_min: int = 720
    gres: str = "gpu:1"


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = True
    project: str = "CellWorldModel"
    entity: str | None = None
    group: str = "foundation_reference_v1"
    mode: str = "online"
    tags: str = "foundation,reference,vae,wdit,strict_zero_shot"


@dataclass(frozen=True)
class FoundationConfig:
    output_root: str
    data_root: str
    gene_vocab: GeneVocabConfig
    splits: SplitConfig = field(default_factory=SplitConfig)
    vae: VaeConfig = field(default_factory=VaeConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.output_root)

    @property
    def data_path(self) -> Path:
        return Path(self.data_root)


def _as_tuple(value: Any, *, default: tuple = ()) -> tuple:
    if value is None:
        return default
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(str(x) for x in value)
    return (str(value),)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def load_foundation_config(path: str | Path) -> FoundationConfig:
    path = Path(path)
    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level foundation config must be a mapping")
    for key in ("output_root", "data_root", "gene_vocab"):
        if key not in raw:
            raise ValueError(f"Missing required key: {key}")

    gene_raw = _section(raw, "gene_vocab")
    if "source" not in gene_raw:
        raise ValueError("Missing required key: gene_vocab.source")
    gene_vocab = GeneVocabConfig(
        source=str(gene_raw["source"]),
        canonical_order=str(gene_raw.get("canonical_order", "mouse_unified_order_filtered")),
    )

    split_raw = _section(raw, "splits")
    strict_raw = _section(split_raw, "strict_zero_shot") if "strict_zero_shot" in split_raw else {}
    splits = SplitConfig(
        split_seed=int(split_raw.get("split_seed", 42)),
        heldout_families=_as_tuple(
            strict_raw.get("heldout_families", split_raw.get("heldout_families")),
            default=("GSE275562",),
        ),
        external=_as_tuple(
            strict_raw.get("external", split_raw.get("external")),
            default=("Norman",),
        ),
        split_ratios=tuple(float(x) for x in split_raw.get("split_ratios", (0.7, 0.1, 0.2))),
    )

    vae_raw = _section(raw, "vae")
    vae = VaeConfig(
        policy=str(vae_raw.get("policy", "vae_b_log1p_normal")),
        batch_strategy=str(vae_raw.get("batch_strategy", "b1_leaf_dataset")),
        smoke_batch_strategies=tuple(
            str(x) for x in vae_raw.get("smoke_batch_strategies", ["b1_leaf_dataset", "b3_none"])
        ),
        latent_dims=tuple(int(x) for x in vae_raw.get("latent_dims", [128, 256])),
        hidden_dim=int(vae_raw.get("hidden_dim", 512)),
        n_layers=int(vae_raw.get("n_layers", 3)),
        batch_size=int(vae_raw.get("batch_size", 4096)),
        smoke_batch_size=int(vae_raw.get("smoke_batch_size", 256)),
        max_epochs=int(vae_raw.get("max_epochs", 100)),
        smoke_epochs=int(vae_raw.get("smoke_epochs", 10)),
        smoke_steps=int(vae_raw.get("smoke_steps", 200)),
        qc_sample_size=int(vae_raw.get("qc_sample_size", 2048)),
        throughput_batch_sizes=tuple(int(x) for x in vae_raw.get("throughput_batch_sizes", [1024, 2048, 4096])),
        throughput_steps=int(vae_raw.get("throughput_steps", 50)),
        throughput_latent_dims=tuple(int(x) for x in vae_raw.get("throughput_latent_dims", [128, 256])),
        throughput_batch_strategies=tuple(str(x) for x in vae_raw.get("throughput_batch_strategies", ["b1_leaf_dataset"])),
        arch_search=tuple(dict(x) for x in vae_raw.get("arch_search", [])),
        leaf_sampling_alpha=float(vae_raw.get("leaf_sampling_alpha", 0.5)),
        full_name=str(vae_raw.get("full_name", "scvi1024_l128_vae2")),
        full_architecture=str(vae_raw.get("full_architecture", "scvi_fclayers1024")),
        full_latent_dim=int(vae_raw.get("full_latent_dim", 128)),
        full_batch_strategy=str(vae_raw.get("full_batch_strategy", "b1_leaf_dataset")),
        full_batch_size=int(vae_raw.get("full_batch_size", 2048)),
        full_epochs=int(vae_raw.get("full_epochs", 2)),
        latent_cache_splits=tuple(str(x) for x in vae_raw.get("latent_cache_splits", ["train", "val", "test"])),
        latent_cache_batch_size=int(vae_raw.get("latent_cache_batch_size", 512)),
        latent_cache_shard_size=int(vae_raw.get("latent_cache_shard_size", 50_000)),
    )

    dyn_raw = _section(raw, "dynamics")
    dynamics = DynamicsConfig(
        experiment=str(dyn_raw.get("experiment", DynamicsConfig.experiment)),
        controls=tuple(str(x) for x in dyn_raw.get("controls", list(DynamicsConfig.controls))),
        dit_size=str(dyn_raw.get("dit_size", "small")),
        transition_pairs=str(dyn_raw.get("transition_pairs", "all_ordered")),
        max_steps=int(dyn_raw.get("max_steps", 100_000)),
        smoke_steps=int(dyn_raw.get("smoke_steps", 20_000)),
        batch_size=int(dyn_raw.get("batch_size", 1024)),
        k_samples=int(dyn_raw.get("K", dyn_raw.get("k_samples", 8))),
        pretrain_epoch_equivalent=int(dyn_raw.get("pretrain_epoch_equivalent", 2)),
        static_objective=str(dyn_raw.get("static_objective", "static_dit_reconstruction")),
        temporal_objective=str(dyn_raw.get("temporal_objective", "temporal_dynamics")),
        static_name=str(dyn_raw.get("static_name", "vae2_staticdit2")),
        temporal_name=str(dyn_raw.get("temporal_name", "vae2_dynamicsdit2")),
        lr=float(dyn_raw.get("lr", 3e-4)),
    )

    res_raw = _section(raw, "resources")
    resources = ResourceConfig(
        partition=str(res_raw.get("partition", "blackwell,a100")),
        cpus=int(res_raw.get("cpus", 12)),
        mem_mb=int(res_raw.get("mem_mb", 120_000)),
        runtime_min=int(res_raw.get("runtime_min", 720)),
        gres=str(res_raw.get("gres", "gpu:1")),
    )

    wandb_raw = _section(raw, "wandb")
    wandb = WandbConfig(
        enabled=bool(wandb_raw.get("enabled", True)),
        project=str(wandb_raw.get("project", "CellWorldModel")),
        entity=wandb_raw.get("entity"),
        group=str(wandb_raw.get("group", "foundation_reference_v1")),
        mode=str(wandb_raw.get("mode", "online")),
        tags=str(wandb_raw.get("tags", "foundation,reference,vae,wdit,strict_zero_shot")),
    )

    cfg = FoundationConfig(
        output_root=str(raw["output_root"]),
        data_root=str(raw["data_root"]),
        gene_vocab=gene_vocab,
        splits=splits,
        vae=vae,
        dynamics=dynamics,
        resources=resources,
        wandb=wandb,
    )
    validate_foundation_config(cfg)
    return cfg


def validate_foundation_config(cfg: FoundationConfig) -> None:
    errors: list[str] = []
    if cfg.gene_vocab.canonical_order not in VALID_GENE_ORDERS:
        errors.append(f"gene_vocab.canonical_order must be one of {sorted(VALID_GENE_ORDERS)}")
    if cfg.vae.policy not in VALID_VAE_POLICIES:
        errors.append(f"vae.policy must be one of {sorted(VALID_VAE_POLICIES)}")
    if cfg.vae.batch_strategy not in VALID_BATCH_STRATEGIES:
        errors.append(f"vae.batch_strategy must be one of {sorted(VALID_BATCH_STRATEGIES)}")
    if cfg.vae.full_batch_strategy not in VALID_BATCH_STRATEGIES:
        errors.append(f"vae.full_batch_strategy must be one of {sorted(VALID_BATCH_STRATEGIES)}")
    bad_smoke = [x for x in cfg.vae.smoke_batch_strategies if x not in VALID_BATCH_STRATEGIES]
    if bad_smoke:
        errors.append(f"vae.smoke_batch_strategies contains invalid values: {bad_smoke}")
    bad_throughput = [x for x in cfg.vae.throughput_batch_strategies if x not in VALID_BATCH_STRATEGIES]
    if bad_throughput:
        errors.append(f"vae.throughput_batch_strategies contains invalid values: {bad_throughput}")
    if not cfg.vae.latent_dims:
        errors.append("vae.latent_dims must not be empty")
    if any(int(x) <= 0 for x in cfg.vae.latent_dims):
        errors.append("vae.latent_dims must be positive")
    if any(int(x) <= 0 for x in cfg.vae.throughput_batch_sizes):
        errors.append("vae.throughput_batch_sizes must be positive")
    if any(int(x) <= 0 for x in cfg.vae.throughput_latent_dims):
        errors.append("vae.throughput_latent_dims must be positive")
    if cfg.vae.leaf_sampling_alpha < 0:
        errors.append("vae.leaf_sampling_alpha must be non-negative")
    bad_cache_splits = [x for x in cfg.vae.latent_cache_splits if x not in {"train", "val", "test", "heldout"}]
    if bad_cache_splits:
        errors.append(f"vae.latent_cache_splits contains invalid values: {bad_cache_splits}")
    if cfg.dynamics.transition_pairs not in VALID_TRANSITION_PAIR_POLICIES:
        errors.append(f"dynamics.transition_pairs must be one of {sorted(VALID_TRANSITION_PAIR_POLICIES)}")
    for key, value in (
        ("vae.batch_size", cfg.vae.batch_size),
        ("vae.smoke_batch_size", cfg.vae.smoke_batch_size),
        ("vae.max_epochs", cfg.vae.max_epochs),
        ("vae.smoke_epochs", cfg.vae.smoke_epochs),
        ("vae.smoke_steps", cfg.vae.smoke_steps),
        ("vae.qc_sample_size", cfg.vae.qc_sample_size),
        ("vae.throughput_steps", cfg.vae.throughput_steps),
        ("vae.full_latent_dim", cfg.vae.full_latent_dim),
        ("vae.full_batch_size", cfg.vae.full_batch_size),
        ("vae.full_epochs", cfg.vae.full_epochs),
        ("vae.latent_cache_batch_size", cfg.vae.latent_cache_batch_size),
        ("vae.latent_cache_shard_size", cfg.vae.latent_cache_shard_size),
        ("dynamics.max_steps", cfg.dynamics.max_steps),
        ("dynamics.smoke_steps", cfg.dynamics.smoke_steps),
        ("dynamics.batch_size", cfg.dynamics.batch_size),
        ("dynamics.k_samples", cfg.dynamics.k_samples),
        ("dynamics.pretrain_epoch_equivalent", cfg.dynamics.pretrain_epoch_equivalent),
        ("resources.cpus", cfg.resources.cpus),
        ("resources.mem_mb", cfg.resources.mem_mb),
        ("resources.runtime_min", cfg.resources.runtime_min),
    ):
        if int(value) <= 0:
            errors.append(f"{key} must be positive")
    if cfg.wandb.mode not in VALID_WANDB_MODES:
        errors.append(f"wandb.mode must be one of {sorted(VALID_WANDB_MODES)}")
    if len(cfg.splits.split_ratios) != 3:
        errors.append("splits.split_ratios must contain train/val/test ratios")
    elif any(x < 0 for x in cfg.splits.split_ratios) or sum(cfg.splits.split_ratios) <= 0:
        errors.append("splits.split_ratios must be non-negative and sum to a positive value")
    if errors:
        raise ValueError("; ".join(errors))
