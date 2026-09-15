"""GEARS-split downstream fine-tuning for foundation perturbation models."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.foundation.action import GeneSetPerturbationEncoder
from cellworldmodel.foundation.dynamics_train import build_foundation_dynamics_cfg
from cellworldmodel.foundation.gears_downstream_dataset import (
    GearsDownstreamDataOptions,
    GearsDownstreamDataset,
)
from cellworldmodel.foundation.gears_downstream_eval import (
    GearsDownstreamEvalOptions,
    GearsDownstreamEvaluator,
)
from cellworldmodel.foundation.gears_losses import GearsLossComputer, GearsLossWeights
from cellworldmodel.foundation.io_utils import write_json
from cellworldmodel.foundation.perturbation_predictors import build_perturbation_predictor, transition_uses_action
from cellworldmodel.foundation.vae_eval import load_vae_checkpoint


@dataclass(frozen=True)
class GearsDownstreamTrainOptions:
    gears_adata: str | Path
    split: str | Path
    subgroup: str | Path | None
    gene_vocab: str | Path
    vae_checkpoint: str | Path
    output_dir: str | Path
    init_checkpoint: str | Path | None = None
    init_name: str = "random"
    experiment: str = "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw"
    dit_size: str = "small"
    max_steps: int = 1000
    batch_size: int = 256
    k_samples: int = 8
    action_dim: int = 64
    lr: float = 3e-4
    seed: int = 0
    device: str | None = None
    latent_weight: float = 0.1
    expr_weight: float = 1.0
    de_weight: float = 2.0
    delta_weight: float = 0.2
    direction_weight: float = 0.0
    top_k: int = 20
    model_type: str = "direct_action"
    eval_max_cells_per_condition: int | None = None
    condition_col: str = "condition"
    control_label: str = "ctrl"


class GearsDownstreamTrainer:
    def __init__(self, options: GearsDownstreamTrainOptions) -> None:
        self.options = options
        self.output_dir = Path(options.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(options.seed)
        torch.manual_seed(options.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(options.seed)
        self.device = torch.device(options.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dataset = GearsDownstreamDataset(GearsDownstreamDataOptions(
            gears_adata=options.gears_adata,
            split=options.split,
            subgroup=options.subgroup,
            gene_vocab=options.gene_vocab,
            top_k=options.top_k,
            condition_col=options.condition_col,
            control_label=options.control_label,
        ))
        self.vae = load_vae_checkpoint(options.vae_checkpoint, self.device)
        for param in self.vae.model.parameters():
            param.requires_grad_(False)
        method, train_cfg, tau_init = build_foundation_dynamics_cfg(
            experiment=options.experiment,
            dit_size=options.dit_size,
            batch_size=options.batch_size,
            k_samples=options.k_samples,
            lr=options.lr,
        )
        self.model_type = str(options.model_type)
        transition_action_dim = int(options.action_dim) if transition_uses_action(self.model_type) else 0
        train_cfg["action_dim"] = transition_action_dim
        train_cfg["loss_balancer"] = "fixed"
        self.model = build_model(method, int(self.vae.config["latent_dim"]), train_cfg, tau_init=tau_init).to(self.device)
        self.load_info = self._load_init(options.init_checkpoint)
        self.action_encoder = GeneSetPerturbationEncoder(self.dataset.n_genes, int(options.action_dim)).to(self.device)
        latent_dim = int(self.vae.config["latent_dim"])
        self.predictor = build_perturbation_predictor(
            model_type=self.model_type,
            transition_model=self.model,
            latent_dim=latent_dim,
            action_dim=int(options.action_dim),
            k_samples=int(options.k_samples),
        ).to(self.device)
        self.loss_computer = GearsLossComputer(GearsLossWeights(
            latent=float(options.latent_weight),
            expression=float(options.expr_weight),
            de=float(options.de_weight),
            delta=float(options.delta_weight),
            direction=float(options.direction_weight),
        ))
        params = list(self.predictor.parameters()) + list(self.action_encoder.parameters())
        self.opt = torch.optim.AdamW(params, lr=float(options.lr), betas=(0.9, 0.95), weight_decay=0.01)
        self.history: list[dict[str, Any]] = []

    def _load_init(self, checkpoint: str | Path | None) -> dict[str, int]:
        if checkpoint is None:
            return {"loaded": 0, "skipped": 0}
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = ckpt.get("model_state_dict", ckpt)
        target = self.model.state_dict()
        matched = {k: v for k, v in source.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
        target.update(matched)
        self.model.load_state_dict(target)
        return {"loaded": int(len(matched)), "skipped": int(len(source) - len(matched))}

    def _encode(self, x: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            z, _ = self.vae.model.encode(torch.from_numpy(x).to(self.device), None)
        return z

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.model.decode(z, None)

    def _action(self, condition: str, n: int) -> torch.Tensor:
        gene_ids, signs, modality_ids, strengths, mask = self.dataset.condition_gene_arrays(condition)
        return self.action_encoder(
            gene_ids=torch.from_numpy(np.repeat(gene_ids[None, :], n, axis=0)).to(self.device),
            signs=torch.from_numpy(np.repeat(signs[None, :], n, axis=0)).to(self.device),
            modality_ids=torch.from_numpy(np.repeat(modality_ids[None, :], n, axis=0)).to(self.device),
            strengths=torch.from_numpy(np.repeat(strengths[None, :], n, axis=0)).to(self.device),
            mask=torch.from_numpy(np.repeat(mask[None, :], n, axis=0)).to(self.device),
        )

    def _predict_z(self, src_z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        out = self.predictor(src_z, action)
        return out.z, out.aux

    def train_step(self, step: int) -> dict[str, Any]:
        condition = str(self.rng.choice(self.dataset.train_conditions))
        control_x, target_x = self.dataset.sample_batch(condition, int(self.options.batch_size), self.rng)
        src_z = self._encode(control_x)
        tgt_z = self._encode(target_x)
        action = self._action(condition, src_z.shape[0])
        pred_z, pred_info = self._predict_z(src_z, action)
        pred_x = self._decode(pred_z)
        target = torch.from_numpy(target_x).to(self.device)
        de_idx = torch.from_numpy(self.dataset.de_idx[condition]).to(self.device)
        ctrl_mean = torch.from_numpy(self.dataset.control_mean).to(self.device)
        loss, loss_info = self.loss_computer(
            pred_z=pred_z,
            target_z=tgt_z,
            pred_x=pred_x,
            target_x=target,
            control_mean=ctrl_mean,
            de_idx=de_idx,
        )
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.predictor.parameters()) + list(self.action_encoder.parameters()), 1.0)
        self.opt.step()
        return {
            "step": int(step + 1),
            "condition": condition,
            "loss": float(loss.item()),
            **loss_info,
            **pred_info,
        }

    def fit(self) -> dict[str, Any]:
        t0 = time.time()
        for step in range(int(self.options.max_steps)):
            row = self.train_step(step)
            self.history.append(row)
            if step == 0 or (step + 1) % 50 == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
        pd.DataFrame(self.history).to_csv(self.output_dir / "history.tsv", sep="\t", index=False)
        self.save_checkpoint(self.output_dir / "model.pt")
        pred_summary, shared_summary = GearsDownstreamEvaluator(
            dataset=self.dataset,
            rng=self.rng,
            encode=self._encode,
            decode=self._decode,
            action=self._action,
            predict_z=self._predict_z,
            options=GearsDownstreamEvalOptions(
                gears_adata=self.options.gears_adata,
                gene_vocab=self.options.gene_vocab,
                subgroup=self.options.subgroup,
                output_dir=self.output_dir,
                batch_size=int(self.options.batch_size),
                top_k=int(self.options.top_k),
                eval_max_cells_per_condition=self.options.eval_max_cells_per_condition,
            ),
        ).run()
        summary = {
            "init_name": self.options.init_name,
            "init_checkpoint": str(self.options.init_checkpoint) if self.options.init_checkpoint else None,
            "model_type": self.model_type,
            "load_info": self.load_info,
            "max_steps": int(self.options.max_steps),
            "batch_size": int(self.options.batch_size),
            "elapsed_s": float(time.time() - t0),
            "prediction": pred_summary,
            "shared_eval": shared_summary,
        }
        write_json(self.output_dir / "summary.json", summary)
        return summary

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "predictor_state_dict": self.predictor.state_dict(),
            "action_encoder_state_dict": self.action_encoder.state_dict(),
            "config": {
                "init_name": self.options.init_name,
                "model_type": self.model_type,
                "action_dim": int(self.options.action_dim),
                "step": int(self.options.max_steps),
            },
        }, path)


def train_gears_downstream(options: GearsDownstreamTrainOptions) -> dict[str, Any]:
    return GearsDownstreamTrainer(options).fit()
