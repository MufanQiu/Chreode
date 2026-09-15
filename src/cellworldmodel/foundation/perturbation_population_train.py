"""Population-level perturbation training for Route1/Route2 method probes."""
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
from cellworldmodel.foundation.gears_downstream_dataset import GearsDownstreamDataOptions, GearsDownstreamDataset
from cellworldmodel.foundation.gears_downstream_eval import GearsDownstreamEvalOptions, GearsDownstreamEvaluator
from cellworldmodel.foundation.io_utils import write_json
from cellworldmodel.foundation.perturbation_population_losses import (
    PopulationLossWeights,
    PopulationPerturbationLoss,
)
from cellworldmodel.foundation.perturbation_population_models import (
    GeneGraphPriorConfig,
    PopulationPredictorConfig,
    ResponseDecoderConfig,
    build_population_predictor,
    build_response_decoder,
)
from cellworldmodel.foundation.vae_eval import load_vae_checkpoint


@dataclass(frozen=True)
class PopulationPerturbationTrainOptions:
    gears_adata: str | Path
    split: str | Path
    subgroup: str | Path | None
    gene_vocab: str | Path
    vae_checkpoint: str | Path
    output_dir: str | Path
    route: str
    init_checkpoint: str | Path | None = None
    init_name: str = "random"
    experiment: str = "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw"
    dit_size: str = "small"
    max_steps: int = 1200
    set_size: int = 128
    eval_batch_size: int = 256
    k_samples: int = 2
    action_dim: int = 64
    n_programs: int = 8
    lr: float = 3e-4
    seed: int = 0
    device: str | None = None
    latent_mmd_weight: float = 1.0
    latent_w2_weight: float = 0.1
    expr_bulk_weight: float = 1.0
    de_bulk_weight: float = 2.0
    delta_cosine_weight: float = 0.2
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 50
    top_k: int = 20
    disable_kick: bool = False
    disable_field: bool = False
    flat_action: bool = False
    adapter_components: str = "full"
    calibrate_potential: bool = False
    response_decoder: str = "none"
    response_programs: int = 32
    sparse_programs: bool = False
    nonnegative_basis: bool = False
    set_context_decoder: bool = False
    program_loss_weight: float = 0.0
    train_fraction: float = 1.0
    gene_graph: str | Path | None = None
    graph_mode: str = "none"
    graph_weight: float = 0.0
    graph_basis_weight: float = 0.0
    graph_output_weight: float = 0.0
    graph_top_k: int = 0
    graph_self_loop: bool = True
    graph_layers: int = 2
    rollout_steps: int = 4
    disable_rollout: bool = False
    disable_action_time: bool = False
    virtual_time_min: float = 0.25
    virtual_time_max: float = 1.75
    eval_max_cells_per_condition: int | None = None
    condition_col: str = "condition"
    control_label: str = "ctrl"


class PopulationPerturbationTrainer:
    def __init__(self, options: PopulationPerturbationTrainOptions) -> None:
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
        self.train_condition_to_idx = self._build_train_condition_to_idx(float(options.train_fraction))
        self.vae = load_vae_checkpoint(options.vae_checkpoint, self.device)
        for param in self.vae.model.parameters():
            param.requires_grad_(False)
        self.base_transition, self.load_info = self._build_base_transition(options.init_checkpoint)
        self.action_encoder = GeneSetPerturbationEncoder(self.dataset.n_genes, int(options.action_dim)).to(self.device)
        latent_dim = int(self.vae.config["latent_dim"])
        graph_prior = self._build_gene_graph_prior()
        self.predictor = build_population_predictor(
            config=PopulationPredictorConfig(
                route=str(options.route),
                latent_dim=latent_dim,
                action_dim=int(options.action_dim),
                n_programs=int(options.n_programs),
                disable_kick=bool(options.disable_kick),
                disable_field=bool(options.disable_field),
                flat_action=bool(options.flat_action),
                adapter_components=str(options.adapter_components),
                k_samples=int(options.k_samples),
                calibrate_potential=bool(options.calibrate_potential),
                rollout_steps=int(options.rollout_steps),
                disable_rollout=bool(options.disable_rollout),
                disable_action_time=bool(options.disable_action_time),
                virtual_time_min=float(options.virtual_time_min),
                virtual_time_max=float(options.virtual_time_max),
            ),
            base_transition=self.base_transition,
        ).to(self.device)
        self.loss_computer = PopulationPerturbationLoss(PopulationLossWeights(
            latent_mmd=float(options.latent_mmd_weight),
            latent_w2=float(options.latent_w2_weight),
            expr_bulk=float(options.expr_bulk_weight),
            de_bulk=float(options.de_bulk_weight),
            delta_cosine=float(options.delta_cosine_weight),
            sinkhorn_eps=float(options.sinkhorn_eps),
            sinkhorn_iters=int(options.sinkhorn_iters),
        ))
        self.response_decoder = build_response_decoder(
            ResponseDecoderConfig(
                response_decoder=str(options.response_decoder),
                n_genes=self.dataset.n_genes,
                latent_dim=latent_dim,
                action_dim=int(options.action_dim),
                response_programs=int(options.response_programs),
                use_sparse_programs=bool(options.sparse_programs),
                nonnegative_basis=bool(options.nonnegative_basis),
                use_set_context=bool(options.set_context_decoder),
                graph_prior=graph_prior,
                graph_layers=int(options.graph_layers),
            )
        )
        if self.response_decoder is not None:
            self.response_decoder = self.response_decoder.to(self.device)
        params = list(self.predictor.parameters()) + list(self.action_encoder.parameters())
        if self.response_decoder is not None:
            params += list(self.response_decoder.parameters())
        self.trainable_params = [p for p in params if p.requires_grad]
        self.opt = torch.optim.AdamW(self.trainable_params, lr=float(options.lr), betas=(0.9, 0.95), weight_decay=0.01)
        self.history: list[dict[str, Any]] = []
        self._last_predictor_tensors: dict[str, torch.Tensor] = {}

    def _build_gene_graph_prior(self) -> GeneGraphPriorConfig:
        mode = str(self.options.graph_mode)
        legacy_weight = float(self.options.graph_weight)
        basis_weight = float(self.options.graph_basis_weight)
        output_weight = float(self.options.graph_output_weight)
        if mode == "none" and self.options.gene_graph is not None and legacy_weight > 0:
            mode = "basis"
        if mode == "basis" and basis_weight == 0 and legacy_weight > 0:
            basis_weight = legacy_weight
        if mode == "output" and output_weight == 0 and legacy_weight > 0:
            output_weight = legacy_weight
        if mode == "both" and legacy_weight > 0:
            basis_weight = basis_weight or legacy_weight
            output_weight = output_weight or legacy_weight
        if mode not in {"none", "basis", "output", "both"}:
            raise ValueError("graph_mode must be one of {'none', 'basis', 'output', 'both'}")
        if mode == "none" or self.options.gene_graph is None:
            return GeneGraphPriorConfig(mode="none")
        edge_index, edge_weight = self._load_gene_graph(
            self.options.gene_graph,
            top_k=int(self.options.graph_top_k),
            self_loop=bool(self.options.graph_self_loop),
        )
        return GeneGraphPriorConfig(
            mode=mode,
            edge_index=edge_index,
            edge_weight=edge_weight,
            basis_weight=basis_weight,
            output_weight=output_weight,
        )

    def _load_gene_graph(
        self,
        path: str | Path,
        *,
        top_k: int,
        self_loop: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        df = pd.read_csv(path)
        if not {"source", "target", "importance"}.issubset(df.columns):
            raise ValueError(f"gene graph must contain source,target,importance columns: {path}")
        gene_to_id = {str(g).upper(): i for i, g in enumerate(self.dataset.gene_names)}
        src = df["source"].astype(str).str.upper().map(gene_to_id)
        tgt = df["target"].astype(str).str.upper().map(gene_to_id)
        keep = src.notna() & tgt.notna()
        if not keep.any():
            return None, None
        src_arr = src[keep].to_numpy(dtype=np.int64)
        tgt_arr = tgt[keep].to_numpy(dtype=np.int64)
        w = df.loc[keep, "importance"].to_numpy(dtype=np.float32)
        edge_df = pd.DataFrame({
            "row": np.concatenate([src_arr, tgt_arr]),
            "col": np.concatenate([tgt_arr, src_arr]),
            "weight": np.concatenate([w, w]).astype(np.float32),
        })
        edge_df = edge_df[edge_df["row"] != edge_df["col"]]
        edge_df = edge_df.groupby(["row", "col"], as_index=False)["weight"].max()
        if top_k > 0:
            edge_df = edge_df.sort_values(["row", "weight"], ascending=[True, False])
            edge_df = edge_df.groupby("row", group_keys=False).head(int(top_k))
        if self_loop:
            loop_df = pd.DataFrame({
                "row": np.arange(self.dataset.n_genes, dtype=np.int64),
                "col": np.arange(self.dataset.n_genes, dtype=np.int64),
                "weight": np.ones(self.dataset.n_genes, dtype=np.float32),
            })
            edge_df = pd.concat([edge_df, loop_df], ignore_index=True)
        row = edge_df["row"].to_numpy(dtype=np.int64)
        col = edge_df["col"].to_numpy(dtype=np.int64)
        weight = edge_df["weight"].to_numpy(dtype=np.float32)
        row_sum = np.bincount(row, weights=weight, minlength=self.dataset.n_genes).astype(np.float32)
        norm = np.divide(weight, row_sum[row], out=np.zeros_like(weight), where=row_sum[row] > 0)
        edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)
        edge_weight = torch.tensor(norm, dtype=torch.float32)
        return edge_index, edge_weight

    def _build_train_condition_to_idx(self, train_fraction: float) -> dict[str, np.ndarray]:
        if train_fraction <= 0 or train_fraction > 1:
            raise ValueError("train_fraction must be in (0, 1]")
        out: dict[str, np.ndarray] = {}
        for condition in self.dataset.train_conditions:
            idx = np.asarray(self.dataset.condition_to_idx[condition], dtype=np.int64)
            if train_fraction < 1.0:
                n = max(1, int(np.ceil(len(idx) * train_fraction)))
                idx = self.rng.choice(idx, size=n, replace=False)
            out[condition] = idx
        return out

    def _build_base_transition(self, checkpoint: str | Path | None):
        if checkpoint is None:
            if self.options.route in {"ktvu_rollout", "native_u_bridge", "route1_internal"}:
                method, train_cfg, tau_init = build_foundation_dynamics_cfg(
                    experiment=self.options.experiment,
                    dit_size=self.options.dit_size,
                    batch_size=self.options.set_size,
                    k_samples=self.options.k_samples,
                    lr=self.options.lr,
                )
                train_cfg["action_dim"] = 0
                train_cfg["loss_balancer"] = "fixed"
                model = build_model(method, int(self.vae.config["latent_dim"]), train_cfg, tau_init=tau_init).to(self.device)
                model.eval()
                return model, {"loaded": 0, "skipped": 0, "base": "random_wdit"}
            return None, {"loaded": 0, "skipped": 0, "base": "identity"}
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ckpt_cfg = dict(ckpt.get("config", {}))
        if "train_cfg" in ckpt_cfg:
            method = str(ckpt_cfg.get("method", "m10"))
            train_cfg = dict(ckpt_cfg["train_cfg"])
            tau_init = 1.0
        else:
            method, train_cfg, tau_init = build_foundation_dynamics_cfg(
                experiment=self.options.experiment,
                dit_size=self.options.dit_size,
                batch_size=self.options.set_size,
                k_samples=self.options.k_samples,
                lr=self.options.lr,
            )
        train_cfg["action_dim"] = 0
        train_cfg["loss_balancer"] = "fixed"
        model = build_model(method, int(self.vae.config["latent_dim"]), train_cfg, tau_init=tau_init).to(self.device)
        source = ckpt.get("model_state_dict", ckpt)
        target = model.state_dict()
        matched = {k: v for k, v in source.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
        target.update(matched)
        model.load_state_dict(target)
        model.eval()
        return model, {"loaded": int(len(matched)), "skipped": int(len(source) - len(matched)), "base": str(checkpoint)}

    def _encode(self, x: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            z, _ = self.vae.model.encode(torch.from_numpy(x).to(self.device), None)
        return z

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.model.decode(z, None)

    def _predict_x(self, condition: str, control_x: np.ndarray, pred_z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        del condition, control_x
        coarse = self._decode(pred_z)
        if self.response_decoder is None:
            return coarse, {}
        if getattr(self.response_decoder, "requires_predictor_tensors", False):
            pred_x, info = self.response_decoder(coarse, pred_z, action, self._last_predictor_tensors)
        else:
            pred_x, info = self.response_decoder(coarse, pred_z, action)
        return pred_x, info

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
        self._last_predictor_tensors = out.tensors or {}
        return out.z, out.aux

    def train_step(self, step: int) -> dict[str, Any]:
        condition = str(self.rng.choice(self.dataset.train_conditions))
        control_x, target_x = self.dataset.sample_set_pair(
            condition,
            int(self.options.set_size),
            self.rng,
            target_idx=self.train_condition_to_idx.get(condition),
        )
        src_z = self._encode(control_x)
        tgt_z = self._encode(target_x)
        action = self._action(condition, src_z.shape[0])
        pred_z, pred_info = self._predict_z(src_z, action)
        pred_x, decoder_info = self._predict_x(condition, control_x, pred_z, action)
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
        if self.response_decoder is not None and float(self.options.program_loss_weight) > 0.0 and hasattr(self.response_decoder, "program_scores"):
            pred_scores = self.response_decoder.program_scores(pred_x, ctrl_mean)
            target_scores = self.response_decoder.program_scores(target, ctrl_mean)
            pred_score_delta = pred_scores.mean(dim=0)
            target_score_delta = target_scores.mean(dim=0)
            program_loss = 1.0 - torch.nn.functional.cosine_similarity(
                pred_score_delta[None, :],
                target_score_delta[None, :],
            ).mean()
            loss = loss + float(self.options.program_loss_weight) * program_loss
            loss_info["program_loss"] = float(program_loss.item())
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
        self.opt.step()
        return {
            "step": int(step + 1),
            "condition": condition,
            "loss": float(loss.item()),
            **loss_info,
            **pred_info,
            **decoder_info,
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
            predict_x=lambda condition, control_x, pred_z, action: self._predict_x(condition, control_x, pred_z, action)[0],
            options=GearsDownstreamEvalOptions(
                gears_adata=self.options.gears_adata,
                gene_vocab=self.options.gene_vocab,
                subgroup=self.options.subgroup,
                output_dir=self.output_dir,
                batch_size=int(self.options.eval_batch_size),
                top_k=int(self.options.top_k),
                eval_max_cells_per_condition=self.options.eval_max_cells_per_condition,
            ),
        ).run()
        summary = {
            "route": self.options.route,
            "init_name": self.options.init_name,
            "init_checkpoint": str(self.options.init_checkpoint) if self.options.init_checkpoint else None,
            "load_info": self.load_info,
            "max_steps": int(self.options.max_steps),
            "set_size": int(self.options.set_size),
            "elapsed_s": float(time.time() - t0),
            "prediction": pred_summary,
            "shared_eval": shared_summary,
            "flags": {
                "disable_kick": bool(self.options.disable_kick),
                "disable_field": bool(self.options.disable_field),
                "flat_action": bool(self.options.flat_action),
                "adapter_components": str(self.options.adapter_components),
                "calibrate_potential": bool(self.options.calibrate_potential),
                "response_decoder": str(self.options.response_decoder),
                "response_programs": int(self.options.response_programs),
                "sparse_programs": bool(self.options.sparse_programs),
                "nonnegative_basis": bool(self.options.nonnegative_basis),
                "set_context_decoder": bool(self.options.set_context_decoder),
                "train_fraction": float(self.options.train_fraction),
                "gene_graph": str(self.options.gene_graph) if self.options.gene_graph is not None else None,
                "graph_mode": str(self.options.graph_mode),
                "graph_weight": float(self.options.graph_weight),
                "graph_basis_weight": float(self.options.graph_basis_weight),
                "graph_output_weight": float(self.options.graph_output_weight),
                "graph_top_k": int(self.options.graph_top_k),
                "graph_self_loop": bool(self.options.graph_self_loop),
                "graph_layers": int(self.options.graph_layers),
                "rollout_steps": int(self.options.rollout_steps),
                "disable_rollout": bool(self.options.disable_rollout),
                "disable_action_time": bool(self.options.disable_action_time),
                "virtual_time_min": float(self.options.virtual_time_min),
                "virtual_time_max": float(self.options.virtual_time_max),
            },
            "loss_weights": {
                "latent_mmd": float(self.options.latent_mmd_weight),
                "latent_w2": float(self.options.latent_w2_weight),
                "expr_bulk": float(self.options.expr_bulk_weight),
                "de_bulk": float(self.options.de_bulk_weight),
                "delta_cosine": float(self.options.delta_cosine_weight),
                "program_loss": float(self.options.program_loss_weight),
                "sinkhorn_eps": float(self.options.sinkhorn_eps),
                "sinkhorn_iters": int(self.options.sinkhorn_iters),
            },
        }
        write_json(self.output_dir / "summary.json", summary)
        return summary

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save({
            "predictor_state_dict": self.predictor.state_dict(),
            "action_encoder_state_dict": self.action_encoder.state_dict(),
            "response_decoder_state_dict": self.response_decoder.state_dict() if self.response_decoder is not None else None,
            "config": {
                "route": self.options.route,
                "init_name": self.options.init_name,
                "action_dim": int(self.options.action_dim),
                "n_programs": int(self.options.n_programs),
                "adapter_components": str(self.options.adapter_components),
                "calibrate_potential": bool(self.options.calibrate_potential),
                "response_decoder": str(self.options.response_decoder),
                "response_programs": int(self.options.response_programs),
                "sparse_programs": bool(self.options.sparse_programs),
                "nonnegative_basis": bool(self.options.nonnegative_basis),
                "set_context_decoder": bool(self.options.set_context_decoder),
                "train_fraction": float(self.options.train_fraction),
                "gene_graph": str(self.options.gene_graph) if self.options.gene_graph is not None else None,
                "graph_mode": str(self.options.graph_mode),
                "graph_weight": float(self.options.graph_weight),
                "graph_basis_weight": float(self.options.graph_basis_weight),
                "graph_output_weight": float(self.options.graph_output_weight),
                "graph_top_k": int(self.options.graph_top_k),
                "graph_self_loop": bool(self.options.graph_self_loop),
                "graph_layers": int(self.options.graph_layers),
                "rollout_steps": int(self.options.rollout_steps),
                "disable_rollout": bool(self.options.disable_rollout),
                "disable_action_time": bool(self.options.disable_action_time),
                "virtual_time_min": float(self.options.virtual_time_min),
                "virtual_time_max": float(self.options.virtual_time_max),
                "loss_weights": {
                    "latent_mmd": float(self.options.latent_mmd_weight),
                    "latent_w2": float(self.options.latent_w2_weight),
                    "expr_bulk": float(self.options.expr_bulk_weight),
                    "de_bulk": float(self.options.de_bulk_weight),
                    "delta_cosine": float(self.options.delta_cosine_weight),
                    "program_loss": float(self.options.program_loss_weight),
                },
                "step": int(self.options.max_steps),
            },
        }, path)


def train_population_perturbation(options: PopulationPerturbationTrainOptions) -> dict[str, Any]:
    return PopulationPerturbationTrainer(options).fit()
