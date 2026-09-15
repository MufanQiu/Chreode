"""Named benchmark experiment recipes.

The benchmark scripts still expose explicit CLI flags for quick ablations. This
module provides stable named recipes for configurations that should be reused
across larger runs and workflow managers.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class ModelRecipe:
    dit_size: str = "tiny"
    state_chunk_dim: int | None = None
    learned_state_tokens: int | None = None
    disable_rope: bool = False
    waddington_dit: bool = False
    curl_rank: int = 16
    wdit_curl_update: str = "additive"
    wdit_curl_time_mode: str = "full"
    wdit_hybrid_delta0: float = 36.0
    wdit_hybrid_slope: float = 0.25
    wdit_hard_delta0: float = 30.0
    wdit_time_embedding: str = "legacy_fourier"
    wdit_time_delta_transform: str = "normalized"
    wdit_time_delta_scale: float | None = None
    wdit_curl_time_embedding: str = "same"
    wdit_curl_time_delta_transform: str = "normalized"
    wdit_curl_time_delta_scale: float | None = None
    dit_time_embedding: str | None = None
    dit_time_delta_transform: str | None = None
    dit_time_delta_scale: float | None = None
    dit_mean_prediction: str | None = None

    def to_cfg(self) -> dict[str, Any]:
        cfg = {
            "dit_size": self.dit_size,
            "state_chunk_dim": self.state_chunk_dim,
            "learned_state_tokens": self.learned_state_tokens,
            "disable_rope": self.disable_rope,
            "waddington_dit": self.waddington_dit,
            "curl_rank": self.curl_rank,
            "wdit_curl_update": self.wdit_curl_update,
            "wdit_curl_time_mode": self.wdit_curl_time_mode,
            "wdit_hybrid_delta0": self.wdit_hybrid_delta0,
            "wdit_hybrid_slope": self.wdit_hybrid_slope,
            "wdit_hard_delta0": self.wdit_hard_delta0,
            "wdit_time_embedding": self.wdit_time_embedding,
            "wdit_time_delta_transform": self.wdit_time_delta_transform,
            "wdit_time_delta_scale": self.wdit_time_delta_scale,
            "wdit_curl_time_embedding": self.wdit_curl_time_embedding,
            "wdit_curl_time_delta_transform": self.wdit_curl_time_delta_transform,
            "wdit_curl_time_delta_scale": self.wdit_curl_time_delta_scale,
        }
        # Omit inactive options so every existing serialized recipe is unchanged.
        for key in ("dit_time_embedding", "dit_time_delta_transform", "dit_time_delta_scale", "dit_mean_prediction"):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        return cfg


@dataclass(frozen=True)
class TrainRecipe:
    batch_size: int = 256
    K: int = 8
    lambda_mmd: float | None = None
    lambda_w2: float | None = None
    lambda_drift: float | None = None
    lambda_down: float | None = None
    multi_delta: bool = False
    optimizer: str = "adam"
    weight_decay: float | None = None
    lr_schedule: str | None = None
    warmup_frac: float | None = None
    drift_pos_ratio: float | None = None
    drift_balance_sample_counts: bool = False
    ema_decay: float | None = None
    down_n_mc: int = 32
    down_antithetic: bool = False
    md_endpoint_prob: float | None = None
    lambda_wdit_a_fro: float = 0.0
    lambda_wdit_curl: float = 0.0
    loss_balancer: str = "fixed"
    loss_balancer_temperature: float = 1.0
    loss_balancer_lookback_prob: float = 0.9
    loss_balancer_alpha: float = 1.0
    loss_balancer_max_multiplier: float = 5.0

    def to_cfg(self) -> dict[str, Any]:
        cfg = {
            "batch_size": self.batch_size,
            "K": self.K,
            "multi_delta": self.multi_delta,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "lr_schedule": self.lr_schedule,
            "warmup_frac": self.warmup_frac,
            "drift_pos_ratio": self.drift_pos_ratio,
            "drift_balance_sample_counts": self.drift_balance_sample_counts,
            "ema_decay": self.ema_decay,
            "down_n_mc": self.down_n_mc,
            "down_antithetic": self.down_antithetic,
            "md_endpoint_prob": self.md_endpoint_prob,
            "lambda_wdit_a_fro": self.lambda_wdit_a_fro,
            "lambda_wdit_curl": self.lambda_wdit_curl,
            "loss_balancer": self.loss_balancer,
            "loss_balancer_temperature": self.loss_balancer_temperature,
            "loss_balancer_lookback_prob": self.loss_balancer_lookback_prob,
            "loss_balancer_alpha": self.loss_balancer_alpha,
            "loss_balancer_max_multiplier": self.loss_balancer_max_multiplier,
        }
        if self.lambda_mmd is not None:
            cfg["lambda_mmd"] = self.lambda_mmd
        if self.lambda_w2 is not None:
            cfg["lambda_w2"] = self.lambda_w2
        if self.lambda_drift is not None:
            cfg["lambda_drift"] = self.lambda_drift
        if self.lambda_down is not None:
            cfg["lambda_down"] = self.lambda_down
        return cfg


@dataclass(frozen=True)
class SplitRecipe:
    split_policy: str = "legacy"
    split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)

    def to_cfg(self) -> dict[str, Any]:
        return {
            "split_policy": self.split_policy,
            "split_ratios": self.split_ratios,
        }


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    method: str
    description: str
    model: ModelRecipe
    train: TrainRecipe
    split: SplitRecipe
    epochs: int | None = None
    save_checkpoint: bool = False

    @property
    def cfg(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        cfg.update(self.model.to_cfg())
        cfg.update(self.train.to_cfg())
        cfg.update(self.split.to_cfg())
        return cfg

    def apply_to_cfg(self, cfg: dict[str, Any]) -> None:
        cfg.update(self.cfg)


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "g2a_m10_md_adamw": ExperimentSpec(
        name="g2a_m10_md_adamw",
        method="m10",
        description=(
            "Selected high-budget MD Phase 2 recipe: M10 Tiny + canonical "
            "1-token RoPE + multi-delta + B512/5000 steps + AdamW/warmup cosine."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(dit_size="tiny"),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_adamw": ExperimentSpec(
        name="g2a_m10_wdit_adamw",
        method="m10",
        description=(
            "G2a recipe with explicit Waddington-DiT residual: "
            "R=-grad U+(A-A^T)z+sigma eps, same AdamW/multi-delta training."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(dit_size="tiny", waddington_dit=True, curl_rank=16),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_cayley_direct_adamw": ExperimentSpec(
        name="g2a_m10_wdit_cayley_direct_adamw",
        method="m10",
        description=(
            "Explicit Waddington-DiT with Cayley orthogonal curl update: "
            "z_hat = Cayley(alpha S) z + alpha(-grad U + sigma eps)."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_update="cayley_direct",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_cayley_residual_adamw": ExperimentSpec(
        name="g2a_m10_wdit_cayley_residual_adamw",
        method="m10",
        description=(
            "Explicit Waddington-DiT with Cayley curl written in residual form: "
            "z_hat = z + alpha(([Cayley(alpha S)z-z]/alpha) - grad U + sigma eps)."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_update="cayley_residual",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_hybrid_delta_adamw": ExperimentSpec(
        name="g2a_m10_wdit_hybrid_delta_adamw",
        method="m10",
        description=(
            "Explicit Waddington-DiT with delta-gated hybrid curl: additive "
            "within training-range, Cayley-dominant for long Delta."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_update="hybrid_delta",
            wdit_hybrid_delta0=36.0,
            wdit_hybrid_slope=0.25,
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_curlpen1e5_adamw": ExperimentSpec(
        name="g2a_m10_wdit_curlpen1e5_adamw",
        method="m10",
        description="Explicit additive W-DiT with curl norm penalty lambda=1e-5.",
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(dit_size="tiny", waddington_dit=True, curl_rank=16),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
            lambda_wdit_curl=1e-5,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_statecurl_adamw": ExperimentSpec(
        name="g2a_m10_wdit_statecurl_adamw",
        method="m10",
        description=(
            "Explicit additive W-DiT where the curl field is state-only; U/drift "
            "still receives Delta."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_time_mode="state_only",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_statecurl_cayley_residual_adamw": ExperimentSpec(
        name="g2a_m10_wdit_statecurl_cayley_residual_adamw",
        method="m10",
        description=(
            "Explicit W-DiT with state-only curl factors and Cayley residual update: "
            "S_theta=S_theta(z), while U/drift still receive Delta."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_update="cayley_residual",
            wdit_curl_time_mode="state_only",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_lowfreqtime_adamw": ExperimentSpec(
        name="g2a_m10_wdit_lowfreqtime_adamw",
        method="m10",
        description=(
            "Additive W-DiT with bounded low-frequency Fourier Delta embedding "
            "on dataset-normalized Delta."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_time_embedding="bounded_lowfreq_fourier",
            wdit_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_time2vec_adamw": ExperimentSpec(
        name="g2a_m10_wdit_time2vec_adamw",
        method="m10",
        description=(
            "Additive W-DiT with Time2Vec-style Delta embedding on "
            "dataset-normalized Delta."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_statecurl_cayley_lowfreqtime_adamw": ExperimentSpec(
        name="g2a_m10_wdit_statecurl_cayley_lowfreqtime_adamw",
        method="m10",
        description=(
            "State-only curl + Cayley residual W-DiT with bounded low-frequency "
            "Fourier Delta embedding."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_update="cayley_residual",
            wdit_curl_time_mode="state_only",
            wdit_time_embedding="bounded_lowfreq_fourier",
            wdit_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_statecurl_cayley_time2vec_adamw": ExperimentSpec(
        name="g2a_m10_wdit_statecurl_cayley_time2vec_adamw",
        method="m10",
        description=(
            "State-only curl + Cayley residual W-DiT with Time2Vec-style "
            "Delta embedding."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_update="cayley_residual",
            wdit_curl_time_mode="state_only",
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_lowfreq_hardcayley_adamw": ExperimentSpec(
        name="g2a_m10_wdit_lowfreq_hardcayley_adamw",
        method="m10",
        description=(
            "Additive W-DiT with bounded low-frequency Delta embedding during "
            "training, hard-switching to Cayley residual when Delta > 30."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_update="hard_delta_cayley_residual",
            wdit_hard_delta0=30.0,
            wdit_time_embedding="bounded_lowfreq_fourier",
            wdit_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_legacyu_lowfreqcurl_adamw": ExperimentSpec(
        name="g2a_m10_wdit_legacyu_lowfreqcurl_adamw",
        method="m10",
        description=(
            "Additive W-DiT with legacy Fourier Delta embedding for U/drift and "
            "bounded low-frequency Fourier Delta embedding for curl."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_time_mode="separate",
            wdit_time_embedding="legacy_fourier",
            wdit_curl_time_embedding="bounded_lowfreq_fourier",
            wdit_curl_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_time2vecu_lowfreqcurl_adamw": ExperimentSpec(
        name="g2a_m10_wdit_time2vecu_lowfreqcurl_adamw",
        method="m10",
        description=(
            "Additive W-DiT with Time2Vec Delta embedding for U/drift and "
            "bounded low-frequency Fourier Delta embedding for curl."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_time_mode="separate",
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
            wdit_curl_time_embedding="bounded_lowfreq_fourier",
            wdit_curl_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw": ExperimentSpec(
        name="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw",
        method="m10",
        description="Selected W-DiT with homoscedastic uncertainty loss weighting.",
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_time_mode="separate",
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
            wdit_curl_time_embedding="bounded_lowfreq_fourier",
            wdit_curl_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
            loss_balancer="uncertainty",
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_nodrift_adamw": ExperimentSpec(
        name="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_nodrift_adamw",
        method="m10",
        description="Selected W-DiT with the drifting-field loss disabled.",
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_time_mode="separate",
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
            wdit_curl_time_embedding="bounded_lowfreq_fourier",
            wdit_curl_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
            lambda_drift=0.0,
            loss_balancer="uncertainty",
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_nodown_adamw": ExperimentSpec(
        name="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_nodown_adamw",
        method="m10",
        description="Selected W-DiT with the downhill potential regularizer disabled.",
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_time_mode="separate",
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
            wdit_curl_time_embedding="bounded_lowfreq_fourier",
            wdit_curl_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
            lambda_down=0.0,
            loss_balancer="uncertainty",
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_nommd_adamw": ExperimentSpec(
        name="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_nommd_adamw",
        method="m10",
        description="Selected W-DiT with the MMD population-matching loss disabled.",
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_time_mode="separate",
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
            wdit_curl_time_embedding="bounded_lowfreq_fourier",
            wdit_curl_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
            lambda_mmd=0.0,
            lambda_w2=1.0,
            loss_balancer="uncertainty",
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_now2_adamw": ExperimentSpec(
        name="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_now2_adamw",
        method="m10",
        description="Selected W-DiT with the Sinkhorn W2 population-matching loss disabled.",
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_time_mode="separate",
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
            wdit_curl_time_embedding="bounded_lowfreq_fourier",
            wdit_curl_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
            lambda_mmd=1.0,
            lambda_w2=0.0,
            loss_balancer="uncertainty",
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_time2vecu_lowfreqcurl_relobralo_adamw": ExperimentSpec(
        name="g2a_m10_wdit_time2vecu_lowfreqcurl_relobralo_adamw",
        method="m10",
        description="Selected W-DiT with ReLoBRaLo-style relative loss balancing.",
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_time_mode="separate",
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
            wdit_curl_time_embedding="bounded_lowfreq_fourier",
            wdit_curl_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
            loss_balancer="relobralo",
            loss_balancer_temperature=1.0,
            loss_balancer_lookback_prob=0.9,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_time2vecu_lowfreqcurl_dwa_adamw": ExperimentSpec(
        name="g2a_m10_wdit_time2vecu_lowfreqcurl_dwa_adamw",
        method="m10",
        description="Selected W-DiT with Dynamic Weight Average loss balancing.",
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_time_mode="separate",
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
            wdit_curl_time_embedding="bounded_lowfreq_fourier",
            wdit_curl_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
            loss_balancer="dwa",
            loss_balancer_temperature=2.0,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_time2vecu_lowfreqcurl_gradnormlite_adamw": ExperimentSpec(
        name="g2a_m10_wdit_time2vecu_lowfreqcurl_gradnormlite_adamw",
        method="m10",
        description="Selected W-DiT with GradNorm-lite inverse gradient-norm loss balancing.",
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_time_mode="separate",
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
            wdit_curl_time_embedding="bounded_lowfreq_fourier",
            wdit_curl_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
            loss_balancer="gradnorm_lite",
            loss_balancer_max_multiplier=5.0,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_time2vecu_lowfreqcurl_rlw_adamw": ExperimentSpec(
        name="g2a_m10_wdit_time2vecu_lowfreqcurl_rlw_adamw",
        method="m10",
        description="Selected W-DiT with Random Loss Weighting sanity baseline.",
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_time_mode="separate",
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
            wdit_curl_time_embedding="bounded_lowfreq_fourier",
            wdit_curl_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
            loss_balancer="rlw",
            loss_balancer_alpha=1.0,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
    "g2a_m10_wdit_cayley_time2vecu_lowfreqcurl_adamw": ExperimentSpec(
        name="g2a_m10_wdit_cayley_time2vecu_lowfreqcurl_adamw",
        method="m10",
        description=(
            "Cayley-residual W-DiT with Time2Vec Delta embedding for U/drift "
            "and bounded low-frequency Fourier Delta embedding for curl."
        ),
        epochs=5000,
        save_checkpoint=True,
        model=ModelRecipe(
            dit_size="tiny",
            waddington_dit=True,
            curl_rank=16,
            wdit_curl_update="cayley_residual",
            wdit_curl_time_mode="separate",
            wdit_time_embedding="time2vec",
            wdit_time_delta_transform="normalized",
            wdit_curl_time_embedding="bounded_lowfreq_fourier",
            wdit_curl_time_delta_transform="normalized",
        ),
        train=TrainRecipe(
            batch_size=512,
            K=8,
            multi_delta=True,
            optimizer="adamw",
            weight_decay=0.01,
            lr_schedule="warmup_cosine",
            warmup_frac=0.05,
        ),
        split=SplitRecipe(split_policy="per_timepoint", split_ratios=(0.7, 0.1, 0.2)),
    ),
}


# Derive architecture comparisons from the selected recipe so that objective,
# optimizer and data-split fields cannot silently revert to legacy defaults.
_selected_time_recipe = EXPERIMENTS[
    "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw"
]
_unconstrained_name = "g2a_m10_unconstrained_dit_time2vec_zeronoise_uncertainty_adamw"
EXPERIMENTS[_unconstrained_name] = replace(
    _selected_time_recipe,
    name=_unconstrained_name,
    description=(
        "Unconstrained noise-token DiT residual with normalized Time2Vec, "
        "auxiliary potential and a zero-noise representative for the downhill "
        "term; selected full objective, not an analytic mean or equal-FLOP ablation."
    ),
    model=replace(
        _selected_time_recipe.model,
        waddington_dit=False,
        dit_time_embedding="time2vec",
        dit_time_delta_transform="normalized",
        dit_mean_prediction="zero_noise",
    ),
)
for _embedding_name, _embedding in (
    ("time2vec", "time2vec"),
    ("lowfreq", "bounded_lowfreq_fourier"),
):
    _name = f"g2a_m10_wdit_shared_{_embedding_name}_uncertainty_adamw"
    EXPERIMENTS[_name] = replace(
        _selected_time_recipe,
        name=_name,
        description=(
            f"Selected objective with shared {_embedding} potential/curl features; "
            "a time-parameterization comparison, not equal-FLOP training."
        ),
        model=replace(
            _selected_time_recipe.model,
            wdit_curl_time_mode="full",
            wdit_time_embedding=_embedding,
            wdit_curl_time_embedding="same",
        ),
    )
del _selected_time_recipe, _embedding_name, _embedding, _name


def experiment_names() -> list[str]:
    return sorted(EXPERIMENTS)


def get_experiment_spec(name: str | None) -> ExperimentSpec | None:
    if name is None:
        return None
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment={name!r}; expected one of {experiment_names()}") from exc


def add_experiment_arg(parser) -> None:
    parser.add_argument(
        "--experiment",
        choices=experiment_names(),
        default=None,
        help="Apply a named experiment recipe before explicit CLI overrides.",
    )


def apply_experiment_args(args, cfg: dict[str, Any]) -> tuple[str, int | None, bool]:
    """Apply `--experiment` to cfg and return method/epochs/checkpoint hints.

    Explicit CLI flags should be applied after this helper so they can override
    the named recipe for quick smoke tests.
    """
    spec = get_experiment_spec(getattr(args, "experiment", None))
    if spec is None:
        if getattr(args, "method", None) is None:
            raise ValueError("--method is required when --experiment is not set")
        return args.method, None, False

    if getattr(args, "method", None) is not None and args.method != spec.method:
        raise ValueError(
            f"--method {args.method!r} conflicts with experiment {spec.name!r} "
            f"which requires method {spec.method!r}"
        )
    spec.apply_to_cfg(cfg)
    return spec.method, spec.epochs, spec.save_checkpoint
