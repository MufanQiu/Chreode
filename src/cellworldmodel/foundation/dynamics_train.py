"""Step-based foundation DiT pretraining for A1/A2 perturbation ablations."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cellworldmodel.benchmark.common_metrics import mmd2_unbiased_multi_sigma, sinkhorn_w2
from cellworldmodel.benchmark.configs import DATASET_CONFIGS
from cellworldmodel.benchmark.experiment_registry import EXPERIMENTS
from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.foundation.config import FoundationConfig
from cellworldmodel.foundation.dynamics_dataset import (
    FoundationLatentTransitionDataset,
)
from cellworldmodel.foundation.expression_dataset import FoundationExpressionDataset
from cellworldmodel.foundation.latent_cache import LatentCacheDataset
from cellworldmodel.foundation.pretrain_protocols import get_pretrain_protocol
from cellworldmodel.foundation.transition_index import FoundationTransitionIdSampler, build_transition_index
from cellworldmodel.foundation.vae_eval import LoadedVAE, load_vae_checkpoint
from cellworldmodel.foundation.vae_train import batch_codes, mean_row_pearson
from cellworldmodel.model.pc_celldrift_bench import downhill_loss
from cellworldmodel.provenance.models import sha256_file
from cellworldmodel.training.benchmark_loop import build_optimizer, build_scheduler
from cellworldmodel.training.drift_loss import (
    drift_stopgrad_loss_from_raw,
    median_heuristic_temperatures,
    normalize_features,
)
from cellworldmodel.training.loss_balancer import (
    LossComponent,
    build_loss_balancer,
    select_gradnorm_params,
)
from cellworldmodel.script.wandb_utils import flatten_numeric, wandb_run_info, wandb_summary_update


@dataclass(frozen=True)
class FoundationDynamicsTrainOptions:
    catalog_dir: str | Path
    output_dir: str | Path
    objective: str
    max_steps: int
    batch_size: int
    seed: int
    vae_checkpoint: str | Path | None = None
    latent_cache_dir: str | Path | None = None
    transition_index_dir: str | Path | None = None
    experiment: str = "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw"
    dit_size: str = "small"
    k_samples: int = 8
    lr: float = 3e-4
    static_delta: float = 1.0
    checkpoint_every_steps: int = 1000
    log_every: int = 50
    device: str | None = None
    static_initialization: str = "legacy"
    train_transition_policy: str = "all_ordered"
    transport_objective: str = "legacy"
    transport_blur: float = 0.05
    transport_max_iters: int = 20000
    transport_tolerance: float = 1e-4


def build_foundation_dynamics_cfg(
    *,
    experiment: str,
    dit_size: str,
    batch_size: int,
    k_samples: int,
    lr: float,
    transition_index: pd.DataFrame | None = None,
) -> tuple[str, dict, float]:
    """Build benchmark-compatible model/loss config for foundation training."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown dynamics experiment={experiment!r}; expected one of {sorted(EXPERIMENTS)}")
    spec = EXPERIMENTS[experiment]
    cfg = dict(DATASET_CONFIGS["weinreb_scvi"])
    spec.apply_to_cfg(cfg)
    cfg.update({
        "dit_size": dit_size,
        "batch_size": int(batch_size),
        "K": int(k_samples),
        "lr": float(lr),
        "grad_clip": cfg.get("grad_clip", 1.0),
    })
    tau_init = 1.0
    if transition_index is not None and not transition_index.empty:
        deltas = transition_index["delta"].to_numpy(dtype=float)
        tau_init = max(float(np.median(deltas)) / np.log(2.0), 1e-6)
        max_delta = max(float(np.max(deltas)), 1e-6)
        cfg["wdit_time_delta_scale"] = max_delta
        cfg["wdit_curl_time_delta_scale"] = max_delta
        if cfg.get("dit_time_embedding", "legacy_fourier") != "legacy_fourier":
            cfg["dit_time_delta_scale"] = max_delta
    return spec.method, cfg, tau_init


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


class FoundationDynamicsTrainer:
    def __init__(self, cfg: FoundationConfig, options: FoundationDynamicsTrainOptions, wandb_run=None) -> None:
        if options.transport_objective not in {"legacy", "sinkhorn_divergence", "converged_proxy"}:
            raise ValueError(f"Unknown transport objective: {options.transport_objective}")
        if options.transport_objective != "legacy" and options.objective != "temporal_dynamics":
            raise ValueError("Transport objectives require temporal_dynamics")
        if options.train_transition_policy not in {"all_ordered", "leaf_endpoints"}:
            raise ValueError("train_transition_policy must be all_ordered or leaf_endpoints")
        if options.train_transition_policy != "all_ordered" and options.objective != "temporal_dynamics":
            raise ValueError("leaf_endpoints requires temporal_dynamics")
        if options.static_initialization not in {"legacy", "match-temporal"}:
            raise ValueError("static_initialization must be legacy or match-temporal")
        align_static = options.static_initialization == "match-temporal"
        if align_static and (options.objective != "static_dit_reconstruction"
                             or options.transition_index_dir is None):
            raise ValueError("match-temporal requires static_dit_reconstruction and transition_index_dir")
        self.cfg = cfg
        self.options = options
        self.wandb_run = wandb_run
        self.output_dir = Path(options.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(options.seed)
        torch.manual_seed(options.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(options.seed)
        self.device = torch.device(options.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.protocol = get_pretrain_protocol(options.objective)
        self.history: list[dict] = []

        self.loaded_vae: LoadedVAE | None = None
        self.expression_dataset: FoundationExpressionDataset | None = None
        self.latent_cache: LatentCacheDataset | None = None
        self.transition_latents: FoundationLatentTransitionDataset | None = None
        self.training_transition_sampling = None
        transition_index = None
        if options.objective == "static_dit_reconstruction":
            if options.vae_checkpoint is None:
                raise ValueError("vae_checkpoint is required for static_dit_reconstruction")
            self.loaded_vae = load_vae_checkpoint(options.vae_checkpoint, self.device)
            for param in self.loaded_vae.model.parameters():
                param.requires_grad_(False)
            self.expression_dataset = FoundationExpressionDataset(options.catalog_dir)
            latent_dim = int(self.loaded_vae.config["latent_dim"])
            if align_static:
                transition_index = pd.read_parquet(Path(options.transition_index_dir) / "transition_index.parquet")
                deltas = transition_index["delta"].to_numpy(dtype=float)
                if not len(deltas) or not np.isfinite(deltas).all() or not (deltas > 0).all():
                    raise ValueError("Static initialization alignment requires finite positive training horizons")
                if "split" not in transition_index or not (transition_index["split"] == "train").all():
                    raise ValueError("Static initialization alignment requires a train-only transition index")
        elif options.objective == "temporal_dynamics":
            if options.latent_cache_dir is None or options.transition_index_dir is None:
                raise ValueError("latent_cache_dir and transition_index_dir are required for temporal_dynamics")
            latent_cache = LatentCacheDataset(options.latent_cache_dir)
            self.latent_cache = latent_cache
            transition_index = pd.read_parquet(Path(options.transition_index_dir) / "transition_index.parquet")
            self.transition_latents = FoundationLatentTransitionDataset(
                latent_cache,
                str(options.catalog_dir),
                transition_index,
                split="train",
                leaf_sampling_alpha=float(cfg.vae.leaf_sampling_alpha),
            )
            if options.train_transition_policy == "leaf_endpoints":
                self.training_transition_sampling = self.transition_latents.id_sampler.restrict_to_leaf_endpoints()
            latent_dim = int(latent_cache.latent_dim)
        else:
            raise ValueError(f"Unsupported foundation dynamics objective={options.objective!r}")

        self.method, self.train_cfg, tau_init = build_foundation_dynamics_cfg(
            experiment=options.experiment,
            dit_size=options.dit_size,
            batch_size=options.batch_size,
            k_samples=options.k_samples,
            lr=options.lr,
            transition_index=transition_index,
        )
        if options.objective == "static_dit_reconstruction":
            self.train_cfg["loss_balancer"] = "fixed"
        if options.transport_objective != "legacy":
            self.train_cfg.update({
                "transport_objective": options.transport_objective,
                "transport_blur": float(options.transport_blur),
                "transport_max_iters": int(options.transport_max_iters),
                "transport_tolerance": float(options.transport_tolerance),
            })
        if align_static:
            # Loading the auxiliary VAE consumes RNG before its weights are restored.
            # Match the temporal trainer's model-construction boundary explicitly.
            torch.manual_seed(options.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(options.seed)
        self.model = build_model(self.method, latent_dim, self.train_cfg, tau_init=tau_init).to(self.device)
        self.initialization_alignment = None
        if align_static:
            initial_path = self.output_dir / "initial_backbone.pt"
            with initial_path.open("xb") as handle:
                torch.save({
                    "model_state_dict": {key: value.detach().cpu().clone()
                                         for key, value in self.model.state_dict().items()},
                    "config": {"objective": options.objective, "method": self.method,
                               "train_cfg": self.train_cfg, "latent_dim": latent_dim,
                               "seed": options.seed, "tau_init": float(tau_init), "step": 0},
                }, handle)
            self.initialization_alignment = {
                "mode": options.static_initialization,
                "transition_index": str(Path(options.transition_index_dir) / "transition_index.parquet"),
                "tau_init": float(tau_init),
                "initial_checkpoint": str(initial_path),
                "initial_checkpoint_sha256": sha256_file(initial_path),
                "effective_static_k": 1,
                "effective_static_noise": "zero",
                "static_delta": float(options.static_delta),
                "scope": "Aligned backbone initialization and time scales; supervision, source sampling, noise and balancing remain different recipes",
            }
        if options.objective == "temporal_dynamics":
            component_names = [
                name for name in ["mmd", "w2", "drift", "down"]
                if float(self.train_cfg.get(f"lambda_{name}", 1.0)) != 0.0
            ]
        else:
            component_names = ["recon"]
        self.loss_balancer = build_loss_balancer(self.train_cfg, component_names, seed=options.seed).to(self.device)
        self.opt = build_optimizer(self.model, self.train_cfg, extra_params=self.loss_balancer.parameters())
        self.scheduler = build_scheduler(self.opt, self.train_cfg, int(options.max_steps))
        self.gradnorm_params = (
            select_gradnorm_params(self.model) if self.loss_balancer.requires_model_params else None
        )
        self.feature_stats = None
        self.taus = (0.02, 0.05, 0.2)

    def _sample_val_transition_latents(self, batch_size: int):
        val_index = build_transition_index(
            self.options.catalog_dir,
            split="val",
            pair_policy=self.cfg.dynamics.transition_pairs,
        ).transitions
        if val_index.empty:
            return None
        sampler = FoundationTransitionIdSampler(
            self.options.catalog_dir,
            val_index,
            split="val",
            leaf_sampling_alpha=float(self.cfg.vae.leaf_sampling_alpha),
        )
        ids = sampler.sample(batch_size, self.rng)
        if self.latent_cache is not None:
            src = _to_tensor(self.latent_cache.load_ids(ids.source_ids), self.device)
            tgt = _to_tensor(self.latent_cache.load_ids(ids.target_ids), self.device)
        else:
            assert self.expression_dataset is not None
            assert self.loaded_vae is not None
            src_batch = self.expression_dataset.load_cells(ids.source_ids)
            tgt_batch = self.expression_dataset.load_cells(ids.target_ids)
            src_x = _to_tensor(src_batch.x, self.device)
            tgt_x = _to_tensor(tgt_batch.x, self.device)
            src_codes = tgt_codes = None
            if self.loaded_vae.leaf_to_id and bool(self.loaded_vae.config.get("encoder_uses_batch", bool(self.loaded_vae.leaf_to_id))):
                src_codes = batch_codes(src_batch.leaf_dataset, self.loaded_vae.leaf_to_id).to(self.device)
                tgt_codes = batch_codes(tgt_batch.leaf_dataset, self.loaded_vae.leaf_to_id).to(self.device)
            with torch.no_grad():
                src, _ = self.loaded_vae.model.encode(src_x, src_codes)
                tgt, _ = self.loaded_vae.model.encode(tgt_x, tgt_codes)
        delta = torch.full((src.shape[0],), float(ids.delta), device=self.device, dtype=src.dtype)
        return src, tgt, delta

    def evaluate_heldout_temporal(self, n_batches: int = 4) -> dict:
        rows = []
        for _ in range(int(n_batches)):
            batch = self._sample_val_transition_latents(min(int(self.options.batch_size), 512))
            if batch is None:
                break
            src, tgt, delta = batch
            with torch.no_grad():
                if self.options.objective == "temporal_dynamics":
                    eps = torch.randn(src.shape[0], int(self.options.k_samples), src.shape[1], device=self.device, dtype=src.dtype)
                    pred = self.model(src, delta, eps).reshape(-1, src.shape[1])
                else:
                    pred = self.model.predict_mean(src, delta, n_mc=1)
                rows.append({
                    "heldout_temporal_w2": float(sinkhorn_w2(pred, tgt, epsilon=float(self.train_cfg["sinkhorn_eps"]), num_iters=50).item()),
                    "heldout_temporal_mmd": float(mmd2_unbiased_multi_sigma(pred, tgt).item()),
                    "heldout_temporal_delta_pearson": mean_row_pearson(
                        (pred[:src.shape[0]] - src).detach(),
                        (tgt - src).detach(),
                    ),
                    "source_norm_mean": float(torch.linalg.norm(src, dim=1).mean().item()),
                    "pred_norm_mean": float(torch.linalg.norm(pred, dim=1).mean().item()),
                    "target_norm_mean": float(torch.linalg.norm(tgt, dim=1).mean().item()),
                })
        if not rows:
            return {}
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "heldout_temporal_eval.tsv", sep="\t", index=False)
        return {key: float(df[key].mean()) for key in df.columns}

    def _sample_static_expression(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.expression_dataset is not None
        assert self.loaded_vae is not None
        ids = self.expression_dataset.sample_cell_ids_balanced_by_leaf(
            "train",
            int(self.options.batch_size),
            self.rng,
            alpha=float(self.cfg.vae.leaf_sampling_alpha),
        )
        batch = self.expression_dataset.load_cells(ids)
        x = _to_tensor(batch.x, self.device)
        codes = None
        if self.loaded_vae.leaf_to_id:
            codes = batch_codes(batch.leaf_dataset, self.loaded_vae.leaf_to_id).to(self.device)
        return x, codes

    def static_step(self, step: int) -> dict:
        assert self.loaded_vae is not None
        x, codes = self._sample_static_expression()
        with torch.no_grad():
            z, _ = self.loaded_vae.model.encode(x, codes)
        delta = torch.full((z.shape[0],), float(self.options.static_delta), device=self.device, dtype=z.dtype)
        eps = torch.zeros(z.shape[0], 1, z.shape[1], device=self.device, dtype=z.dtype)
        z_hat = self.model(z, delta, eps).squeeze(1)
        recon = self.loaded_vae.model.decode(z_hat, codes)
        loss_recon = F.mse_loss(recon, x)
        loss, balance_info = self.loss_balancer.combine(
            [LossComponent("recon", loss_recon, 1.0)],
            step=step,
            model_params=self.gradnorm_params,
        )
        info = {
            "step": step + 1,
            "loss": float(loss.item()),
            "recon_loss": float(loss_recon.item()),
            "recon_pearson": mean_row_pearson(x.detach(), recon.detach()),
            "latent_l2": float(torch.linalg.norm((z_hat - z).detach(), dim=1).mean().item()),
            **balance_info,
        }
        self._opt_step(loss)
        return info

    def _ensure_temporal_feature_stats(self, target: torch.Tensor) -> None:
        if self.feature_stats is not None:
            return
        _, scale, mean, std = normalize_features(target)
        self.feature_stats = {"mean": mean, "std": std, "scale": scale}
        with torch.no_grad():
            phi_ref = (target - mean) / std * scale
            self.taus = median_heuristic_temperatures(phi_ref[:1024], multipliers=(0.2, 0.5, 1.5))

    def temporal_step(self, step: int) -> dict:
        assert self.transition_latents is not None
        batch = self.transition_latents.sample(int(self.options.batch_size), self.rng)
        src = _to_tensor(batch.source, self.device)
        tgt = _to_tensor(batch.target, self.device)
        delta = torch.full((src.shape[0],), float(batch.delta), device=self.device, dtype=src.dtype)
        eps = torch.randn(src.shape[0], int(self.options.k_samples), src.shape[1], device=self.device, dtype=src.dtype)
        z_hat = self.model(src, delta, eps)
        z_hat_flat = z_hat.reshape(-1, src.shape[1])

        loss_mmd = mmd2_unbiased_multi_sigma(z_hat_flat, tgt)
        if self.train_cfg.get("transport_objective", "legacy") == "legacy":
            loss_w2 = sinkhorn_w2(z_hat_flat, tgt, epsilon=float(self.train_cfg["sinkhorn_eps"]), num_iters=50)
        else:
            from cellworldmodel.training.transport_objective import training_transport

            loss_w2 = training_transport(z_hat_flat, tgt, self.train_cfg)
        self._ensure_temporal_feature_stats(tgt)
        loss_drift, drift_info = drift_stopgrad_loss_from_raw(
            z_gen=z_hat_flat,
            z_pos=tgt,
            z_neg=None,
            temperatures=self.taus,
            normalize_features_first=True,
            feature_stats=self.feature_stats,
            balance_sample_counts=bool(self.train_cfg.get("drift_balance_sample_counts", False)),
        )
        z_det = self.model.predict_mean(src, delta, n_mc=int(self.train_cfg.get("down_n_mc", 32)))
        loss_down = downhill_loss(self.model, src, z_det, delta)
        components = [
            LossComponent("mmd", loss_mmd, float(self.train_cfg["lambda_mmd"])),
            LossComponent("w2", loss_w2, float(self.train_cfg["lambda_w2"])),
            LossComponent("drift", loss_drift, float(self.train_cfg["lambda_drift"])),
            LossComponent("down", loss_down, float(self.train_cfg["lambda_down"])),
        ]
        components = [component for component in components if float(component.base_weight) != 0.0]
        loss, balance_info = self.loss_balancer.combine(
            components,
            step=step,
            model_params=self.gradnorm_params,
        )
        if hasattr(self.model, "waddington_regularization"):
            lambda_a = float(self.train_cfg.get("lambda_wdit_a_fro", 0.0) or 0.0)
            lambda_curl = float(self.train_cfg.get("lambda_wdit_curl", 0.0) or 0.0)
            if lambda_a > 0.0 or lambda_curl > 0.0:
                reg_info = self.model.waddington_regularization(src, delta)
                if lambda_a > 0.0:
                    loss = loss + lambda_a * reg_info["wdit_a_fro"]
                if lambda_curl > 0.0:
                    loss = loss + lambda_curl * reg_info["wdit_curl_sq"]
        info = {
            "step": step + 1,
            "loss": float(loss.item()),
            "mmd2": float(loss_mmd.item()),
            "w2_approx": float(loss_w2.item()),
            "drift_loss": float(drift_info["loss_value"]),
            "drift_norm": float(drift_info["drift_norm"]),
            "down_loss": float(loss_down.item()),
            "delta": float(batch.delta),
            "source_t": float(batch.source_t),
            "target_t": float(batch.target_t),
            "transition_id": int(batch.transition_id),
            **balance_info,
        }
        self._opt_step(loss)
        return info

    def _opt_step(self, loss: torch.Tensor) -> None:
        self.opt.zero_grad()
        loss.backward()
        if self.train_cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.train_cfg["grad_clip"]))
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def fit(self) -> dict:
        t0 = time.time()
        for step in range(int(self.options.max_steps)):
            if self.options.objective == "static_dit_reconstruction":
                info = self.static_step(step)
            else:
                info = self.temporal_step(step)
            info["lr"] = float(self.opt.param_groups[0]["lr"])
            if step == 0 or (step + 1) % int(self.options.log_every) == 0 or step + 1 == self.options.max_steps:
                self.history.append(info)
                if self.wandb_run is not None:
                    self.wandb_run.log(flatten_numeric({f"train/{k}": v for k, v in info.items()}), step=step + 1)
            if self.options.checkpoint_every_steps and (step + 1) % int(self.options.checkpoint_every_steps) == 0:
                self.save_checkpoint(self.output_dir / f"checkpoint_step_{step + 1}.pt", step=step + 1)
        eval_info = self.evaluate_heldout_temporal()
        return self.finalize(time.time() - t0, eval_info)

    def save_checkpoint(self, path: Path, *, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "loss_balancer_state_dict": self.loss_balancer.state_dict(),
            "config": {
                "objective": self.options.objective,
                "experiment": self.options.experiment,
                "method": self.method,
                "train_cfg": self.train_cfg,
                "latent_dim": int(self.model.dim),
                "step": int(step),
                **({"initialization_alignment": self.initialization_alignment}
                   if self.initialization_alignment is not None else {}),
                **({"training_transition_sampling": self.training_transition_sampling}
                   if self.training_transition_sampling is not None else {}),
            },
        }, path)

    def finalize(self, elapsed_s: float, eval_info: dict | None = None) -> dict:
        self.save_checkpoint(self.output_dir / "model.pt", step=int(self.options.max_steps))
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.output_dir / "history.tsv", sep="\t", index=False)
        summary = {
            "objective": self.options.objective,
            "experiment": self.options.experiment,
            "protocol": asdict(self.protocol),
            "max_steps": int(self.options.max_steps),
            "batch_size": int(self.options.batch_size),
            "k_samples": int(self.options.k_samples),
            "seed": int(self.options.seed),
            "device": str(self.device),
            "elapsed_s": float(elapsed_s),
            "steps_per_s": float(self.options.max_steps / elapsed_s) if elapsed_s > 0 else None,
            "history": self.history,
            **({"initialization_alignment": self.initialization_alignment}
               if self.initialization_alignment is not None else {}),
            **({"training_transition_sampling": self.training_transition_sampling}
               if self.training_transition_sampling is not None else {}),
            **(eval_info or {}),
        }
        if self.wandb_run is not None:
            wandb_summary_update(self.wandb_run, summary)
            summary["wandb"] = wandb_run_info(self.wandb_run)
        with (self.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return summary


def train_foundation_dynamics(
    cfg: FoundationConfig,
    options: FoundationDynamicsTrainOptions,
    wandb_run=None,
) -> dict:
    return FoundationDynamicsTrainer(cfg, options, wandb_run=wandb_run).fit()
