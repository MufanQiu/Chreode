"""Norman perturbation fine-tuning for A0/A1/A2 foundation ablations."""
from __future__ import annotations

import json
import time
import dataclasses
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cellworldmodel.benchmark.common_metrics import mmd2_unbiased_multi_sigma, sinkhorn_w2
from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.foundation.action import CategoricalPerturbationEncoder, GeneSetPerturbationEncoder
from cellworldmodel.foundation.config import FoundationConfig
from cellworldmodel.foundation.dynamics_train import build_foundation_dynamics_cfg
from cellworldmodel.foundation.perturbation_dataset import NormanPerturbationDataset
from cellworldmodel.foundation.vae_eval import load_vae_checkpoint
from cellworldmodel.foundation.vae_train import mean_row_pearson
from cellworldmodel.training.benchmark_loop import build_optimizer, build_scheduler
from cellworldmodel.script.wandb_utils import flatten_numeric, wandb_run_info, wandb_summary_update


@dataclass(frozen=True)
class PerturbationFineTuneOptions:
    catalog_dir: str | Path
    output_dir: str | Path
    norman_path: str | Path
    vae_checkpoint: str | Path
    init_checkpoint: str | Path | None
    init_name: str
    max_steps: int
    batch_size: int
    seed: int
    experiment: str = "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw"
    dit_size: str = "small"
    action_dim: int = 64
    k_samples: int = 8
    lr: float = 3e-4
    split_method: str = "additive"
    eval_every: int = 200
    checkpoint_every_steps: int = 1000
    device: str | None = None
    allow_unknown_batch: bool = False
    action_encoder: str = "geneset_deepset_v1"


def load_matching_state_dict(model: torch.nn.Module, checkpoint: str | Path | None) -> dict:
    if checkpoint is None:
        return {"loaded": 0, "skipped": 0}
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = ckpt.get("model_state_dict", ckpt)
    target = model.state_dict()
    matched = {k: v for k, v in source.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
    target.update(matched)
    model.load_state_dict(target)
    return {"loaded": int(len(matched)), "skipped": int(len(source) - len(matched))}


class PerturbationFineTuneTrainer:
    def __init__(self, cfg: FoundationConfig, options: PerturbationFineTuneOptions, wandb_run=None) -> None:
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
        self.vae = load_vae_checkpoint(options.vae_checkpoint, self.device)
        for param in self.vae.model.parameters():
            param.requires_grad_(False)
        self.dataset = NormanPerturbationDataset(
            options.norman_path,
            Path(options.catalog_dir) / "gene_vocab.parquet",
            split_method=options.split_method,
            split_seed=options.seed,
        )
        method, train_cfg, tau_init = build_foundation_dynamics_cfg(
            experiment=options.experiment,
            dit_size=options.dit_size,
            batch_size=options.batch_size,
            k_samples=options.k_samples,
            lr=options.lr,
        )
        train_cfg["action_dim"] = int(options.action_dim)
        train_cfg["loss_balancer"] = "fixed"
        self.method = method
        self.train_cfg = train_cfg
        if options.action_encoder == "categorical_perturbation":
            self.action_encoder = CategoricalPerturbationEncoder(len(self.dataset.perturbed_conditions), options.action_dim).to(self.device)
            self.condition_to_action_id = {cond: i for i, cond in enumerate(self.dataset.perturbed_conditions)}
        elif options.action_encoder == "geneset_deepset_v1":
            self.action_encoder = GeneSetPerturbationEncoder(self.dataset.n_genes, options.action_dim).to(self.device)
            self.condition_to_action_id = {}
        else:
            raise ValueError(f"Unknown action_encoder={options.action_encoder!r}")
        self.model = build_model(method, int(self.vae.config["latent_dim"]), train_cfg, tau_init=tau_init).to(self.device)
        self.load_info = load_matching_state_dict(self.model, options.init_checkpoint)
        params = list(self.model.parameters()) + list(self.action_encoder.parameters())
        self.opt = torch.optim.AdamW(params, lr=float(options.lr), betas=(0.9, 0.95), weight_decay=0.01)
        self.scheduler = build_scheduler(self.opt, train_cfg, int(options.max_steps))
        self.history: list[dict] = []

    def _encode(self, x: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(x).to(self.device)
        codes = None
        if self.vae.leaf_to_id and bool(self.vae.config.get("encoder_uses_batch", bool(self.vae.leaf_to_id))):
            if not self.options.allow_unknown_batch:
                raise ValueError(
                    "Current VAE checkpoint is batch-conditioned and Norman has no known leaf id. "
                    "Use --allow-unknown-batch only for engineering smoke, or use a no-batch/decoder-only VAE."
                )
            codes = torch.zeros(tensor.shape[0], dtype=torch.long, device=self.device)
        with torch.no_grad():
            z, _ = self.vae.model.encode(tensor, codes)
        return z

    def _action(self, batch, *, mode: str = "normal") -> torch.Tensor:
        if mode == "null":
            gene_ids = np.full_like(batch.gene_ids, self.dataset.n_genes)
            signs = np.zeros_like(batch.signs)
            modality_ids = np.zeros_like(batch.modality_ids)
            strengths = np.ones_like(batch.strengths)
            mask = np.zeros_like(batch.mask)
            batch = dataclasses.replace(
                batch,
                gene_ids=gene_ids,
                signs=signs,
                modality_ids=modality_ids,
                strengths=strengths,
                mask=mask,
            )
        elif mode == "shuffle":
            perm = self.rng.permutation(batch.gene_ids.shape[0])
            batch = dataclasses.replace(
                batch,
                gene_ids=batch.gene_ids[perm],
                signs=batch.signs[perm],
                modality_ids=batch.modality_ids[perm],
                strengths=batch.strengths[perm],
                mask=batch.mask[perm],
            )
        if self.options.action_encoder == "categorical_perturbation":
            ids = np.full(
                batch.gene_ids.shape[0],
                self.condition_to_action_id[batch.condition],
                dtype=np.int64,
            )
            if mode == "null":
                ids = np.zeros_like(ids)
            elif mode == "shuffle":
                ids = self.rng.permutation(ids)
            return self.action_encoder(torch.from_numpy(ids).to(self.device))
        return self.action_encoder(
            gene_ids=torch.from_numpy(batch.gene_ids).to(self.device),
            signs=torch.from_numpy(batch.signs).to(self.device),
            modality_ids=torch.from_numpy(batch.modality_ids).to(self.device),
            strengths=torch.from_numpy(batch.strengths).to(self.device),
            mask=torch.from_numpy(batch.mask).to(self.device),
        )

    def train_step(self, step: int) -> dict:
        batch = self.dataset.sample_batch(int(self.options.batch_size), self.rng, split="train")
        src = self._encode(batch.control_x)
        tgt = self._encode(batch.target_x)
        action = self._action(batch)
        delta = torch.ones(src.shape[0], device=self.device, dtype=src.dtype)
        eps = torch.randn(src.shape[0], int(self.options.k_samples), src.shape[1], device=self.device, dtype=src.dtype)
        pred = self.model(src, delta, eps, action=action).reshape(-1, src.shape[1])
        loss_mmd = mmd2_unbiased_multi_sigma(pred, tgt)
        loss_w2 = sinkhorn_w2(pred, tgt, epsilon=float(self.train_cfg["sinkhorn_eps"]), num_iters=50)
        loss = loss_mmd + loss_w2
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.action_encoder.parameters()), 1.0)
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return {
            "step": step + 1,
            "loss": float(loss.item()),
            "mmd2": float(loss_mmd.item()),
            "w2_approx": float(loss_w2.item()),
            "condition": batch.condition,
            "lr": float(self.opt.param_groups[0]["lr"]),
        }

    def _eval_condition(self, condition: str, action_mode: str) -> dict:
        batch = self.dataset.sample_batch(int(self.options.batch_size), self.rng, split="test", condition=condition)
        src = self._encode(batch.control_x)
        tgt = self._encode(batch.target_x)
        action = self._action(batch, mode=action_mode)
        delta = torch.ones(src.shape[0], device=self.device, dtype=src.dtype)
        with torch.no_grad():
            pred = self.model.predict_mean(src, delta, action=action, n_mc=int(self.options.k_samples))
        return {
            "condition": condition,
            "action_mode": action_mode,
            "latent_mse": float(torch.mean((pred - tgt) ** 2).item()),
            "latent_delta_pearson": mean_row_pearson((pred - src).detach(), (tgt - src).detach()),
            "n_genes_in_action": int(self.dataset.condition_gene_ids(condition)[-1].sum()),
            "is_unseen_combo": bool("+" in condition),
        }

    def evaluate(self, max_conditions: int = 8) -> dict:
        rows = []
        for condition in self.dataset.test_conditions[:max_conditions]:
            for action_mode in ("normal", "shuffle", "null"):
                rows.append(self._eval_condition(condition, action_mode))
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_csv(self.output_dir / "eval_conditions.tsv", sep="\t", index=False)
        normal = df[df["action_mode"] == "normal"] if not df.empty else df
        shuffle = df[df["action_mode"] == "shuffle"] if not df.empty else df
        null = df[df["action_mode"] == "null"] if not df.empty else df
        return {
            "eval_n_conditions": int(normal["condition"].nunique()) if not normal.empty else 0,
            "eval_latent_mse": float(normal["latent_mse"].mean()) if not normal.empty else None,
            "eval_latent_delta_pearson": float(normal["latent_delta_pearson"].mean()) if not normal.empty else None,
            "action_shuffle_latent_delta_pearson": (
                float(shuffle["latent_delta_pearson"].mean()) if not shuffle.empty else None
            ),
            "null_action_latent_delta_pearson": (
                float(null["latent_delta_pearson"].mean()) if not null.empty else None
            ),
            "action_shuffle_drop": (
                float(normal["latent_delta_pearson"].mean() - shuffle["latent_delta_pearson"].mean())
                if not normal.empty and not shuffle.empty else None
            ),
            "null_action_drop": (
                float(normal["latent_delta_pearson"].mean() - null["latent_delta_pearson"].mean())
                if not normal.empty and not null.empty else None
            ),
        }

    def fit(self) -> dict:
        t0 = time.time()
        eval_info = {}
        for step in range(int(self.options.max_steps)):
            info = self.train_step(step)
            if step == 0 or (step + 1) % int(self.options.eval_every) == 0 or step + 1 == self.options.max_steps:
                eval_info = self.evaluate()
                info.update(eval_info)
                self.history.append(info)
                if self.wandb_run is not None:
                    self.wandb_run.log(flatten_numeric({f"train/{k}": v for k, v in info.items()}), step=step + 1)
            if self.options.checkpoint_every_steps and (step + 1) % int(self.options.checkpoint_every_steps) == 0:
                self.save_checkpoint(self.output_dir / f"checkpoint_step_{step + 1}.pt", step=step + 1)
        return self.finalize(time.time() - t0, eval_info)

    def save_checkpoint(self, path: Path, *, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "action_encoder_state_dict": self.action_encoder.state_dict(),
            "config": {
                "init_name": self.options.init_name,
                "init_checkpoint": str(self.options.init_checkpoint) if self.options.init_checkpoint else None,
                "action_dim": int(self.options.action_dim),
                "n_actions": int(self.dataset.n_actions),
                "action_encoder": self.options.action_encoder,
                "step": int(step),
            },
        }, path)

    def finalize(self, elapsed_s: float, eval_info: dict) -> dict:
        self.save_checkpoint(self.output_dir / "model.pt", step=int(self.options.max_steps))
        pd.DataFrame(self.history).to_csv(self.output_dir / "history.tsv", sep="\t", index=False)
        summary = {
            "init_name": self.options.init_name,
            "init_checkpoint": str(self.options.init_checkpoint) if self.options.init_checkpoint else None,
            "load_info": self.load_info,
            "max_steps": int(self.options.max_steps),
            "batch_size": int(self.options.batch_size),
            "seed": int(self.options.seed),
            "elapsed_s": float(elapsed_s),
            "n_actions": int(self.dataset.n_actions),
            "action_encoder": self.options.action_encoder,
            "n_train_conditions": int(len(self.dataset.train_conditions)),
            "n_test_conditions": int(len(self.dataset.test_conditions)),
            **eval_info,
        }
        if self.wandb_run is not None:
            wandb_summary_update(self.wandb_run, summary)
            summary["wandb"] = wandb_run_info(self.wandb_run)
        with (self.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return summary


def train_perturbation_finetune(
    cfg: FoundationConfig,
    options: PerturbationFineTuneOptions,
    wandb_run=None,
) -> dict:
    return PerturbationFineTuneTrainer(cfg, options, wandb_run=wandb_run).fit()
