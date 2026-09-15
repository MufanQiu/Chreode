"""Export foundation perturbation checkpoints into GEARS-style prediction arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.foundation.action import CategoricalPerturbationEncoder, GeneSetPerturbationEncoder
from cellworldmodel.foundation.dynamics_train import build_foundation_dynamics_cfg
from cellworldmodel.foundation.gears_downstream_dataset import load_gears_subgroup_conditions
from cellworldmodel.foundation.gene_space import foundation_gene_names_from_vocab
from cellworldmodel.foundation.io_utils import write_json
from cellworldmodel.foundation.perturbation_dataset import NormanPerturbationDataset, normalize_norman_condition
from cellworldmodel.foundation.perturbation_predictors import build_perturbation_predictor, transition_uses_action
from cellworldmodel.foundation.vae_eval import load_vae_checkpoint


@dataclass(frozen=True)
class FoundationGearsPredictionExportOptions:
    gears_adata: str | Path
    gene_vocab: str | Path
    vae_checkpoint: str | Path
    perturbation_checkpoint: str | Path
    output_dir: str | Path
    subgroup: str | Path | None = None
    experiment: str = "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw"
    dit_size: str = "small"
    batch_size: int = 256
    k_samples: int = 8
    action_dim: int = 64
    lr: float = 3e-4
    seed: int = 0
    device: str | None = None
    max_cells_per_condition: int | None = None
    condition_col: str = "condition"
    control_label: str = "ctrl"


class FoundationGearsPredictionExporter:
    def __init__(self, options: FoundationGearsPredictionExportOptions) -> None:
        self.options = options
        self.output_dir = Path(options.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(options.seed)
        torch.manual_seed(options.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(options.seed)
        self.device = torch.device(options.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dataset = NormanPerturbationDataset(
            options.gears_adata,
            options.gene_vocab,
            split_method="additive",
            split_seed=options.seed,
            condition_col=options.condition_col,
            control_label=options.control_label,
        )
        self.raw_conditions = self.dataset.adata.obs[options.condition_col].astype(str).to_numpy()
        self.control_idx = np.flatnonzero(self.raw_conditions == options.control_label)
        if len(self.control_idx) == 0:
            raise ValueError(f"No control cells found with label={options.control_label!r}")
        self.gene_names = foundation_gene_names_from_vocab(options.gene_vocab)
        self.vae = load_vae_checkpoint(options.vae_checkpoint, self.device)
        for param in self.vae.model.parameters():
            param.requires_grad_(False)
        self.predictor, self.action_encoder, self.load_info = self._load_perturbation_checkpoint()

    def _load_perturbation_checkpoint(self):
        ckpt = torch.load(self.options.perturbation_checkpoint, map_location="cpu", weights_only=False)
        ckpt_cfg = dict(ckpt.get("config", {}))
        model_type = str(ckpt_cfg.get("model_type", "direct_action"))
        action_dim = int(ckpt_cfg.get("action_dim", self.options.action_dim))
        action_encoder_name = str(ckpt_cfg.get("action_encoder", "geneset_deepset_v1"))
        method, train_cfg, tau_init = build_foundation_dynamics_cfg(
            experiment=self.options.experiment,
            dit_size=self.options.dit_size,
            batch_size=self.options.batch_size,
            k_samples=self.options.k_samples,
            lr=self.options.lr,
        )
        train_cfg["action_dim"] = action_dim if transition_uses_action(model_type) else 0
        train_cfg["loss_balancer"] = "fixed"
        model = build_model(method, int(self.vae.config["latent_dim"]), train_cfg, tau_init=tau_init)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        predictor = build_perturbation_predictor(
            model_type=model_type,
            transition_model=model,
            latent_dim=int(self.vae.config["latent_dim"]),
            action_dim=action_dim,
            k_samples=int(self.options.k_samples),
        ).to(self.device)
        predictor_info = {"model_type": model_type, "predictor_state_loaded": False}
        if "predictor_state_dict" in ckpt:
            missing, unexpected = predictor.load_state_dict(ckpt["predictor_state_dict"], strict=False)
            predictor_info.update({
                "predictor_state_loaded": True,
                "predictor_missing": list(missing),
                "predictor_unexpected": list(unexpected),
            })
        elif "kick_net_state_dict" in ckpt and hasattr(predictor, "kick_net"):
            predictor.kick_net.load_state_dict(ckpt["kick_net_state_dict"])  # type: ignore[attr-defined]
            predictor_info.update({
                "predictor_state_loaded": True,
                "predictor_legacy_keys": ["kick_net_state_dict"],
            })
            if "gate_net_state_dict" in ckpt and hasattr(predictor, "gate_net"):
                predictor.gate_net.load_state_dict(ckpt["gate_net_state_dict"])  # type: ignore[attr-defined]
                predictor_info["predictor_legacy_keys"].append("gate_net_state_dict")  # type: ignore[index]
        elif model_type != "direct_action":
            raise ValueError(
                f"Checkpoint uses model_type={model_type!r} but has no predictor_state_dict. "
                "Only legacy direct_action checkpoints, or legacy kick/gate checkpoints, "
                "can be exported without predictor_state_dict."
            )
        predictor.eval()
        if action_encoder_name == "geneset_deepset_v1":
            action_encoder = GeneSetPerturbationEncoder(self.dataset.n_genes, action_dim)
        elif action_encoder_name == "categorical_perturbation":
            action_encoder = CategoricalPerturbationEncoder(int(ckpt_cfg["n_actions"]), action_dim)
        else:
            raise ValueError(f"Unsupported action encoder in checkpoint: {action_encoder_name}")
        action_encoder.load_state_dict(ckpt["action_encoder_state_dict"])
        action_encoder.to(self.device).eval()
        return predictor, action_encoder, predictor_info

    def _encode(self, x: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            z, _ = self.vae.model.encode(torch.from_numpy(x).to(self.device), None)
        return z

    def _decode(self, z: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            x = self.vae.model.decode(z, None)
        return x.detach().cpu().numpy().astype(np.float32)

    def _action(self, condition: str, n: int) -> torch.Tensor:
        normalized = normalize_norman_condition(condition)
        gene_ids, signs, modality_ids, strengths, mask = self.dataset.condition_gene_ids(normalized)
        if isinstance(self.action_encoder, CategoricalPerturbationEncoder):
            raise ValueError("Categorical perturbation checkpoints cannot be exported to unseen GEARS conditions")
        return self.action_encoder(
            gene_ids=torch.from_numpy(np.repeat(gene_ids[None, :], n, axis=0)).to(self.device),
            signs=torch.from_numpy(np.repeat(signs[None, :], n, axis=0)).to(self.device),
            modality_ids=torch.from_numpy(np.repeat(modality_ids[None, :], n, axis=0)).to(self.device),
            strengths=torch.from_numpy(np.repeat(strengths[None, :], n, axis=0)).to(self.device),
            mask=torch.from_numpy(np.repeat(mask[None, :], n, axis=0)).to(self.device),
        )

    def _condition_rows(self, condition: str) -> np.ndarray:
        rows = np.flatnonzero(self.raw_conditions == condition)
        if len(rows) == 0:
            raise ValueError(f"No GEARS rows found for condition={condition!r}")
        if self.options.max_cells_per_condition is not None and len(rows) > int(self.options.max_cells_per_condition):
            rows = self.rng.choice(rows, size=int(self.options.max_cells_per_condition), replace=False)
        return np.asarray(rows, dtype=np.int64)

    def _predict_condition_mean(self, condition: str) -> tuple[np.ndarray, np.ndarray, int]:
        target_rows = self._condition_rows(condition)
        n = len(target_rows)
        control_rows = self.rng.choice(self.control_idx, size=n, replace=n > len(self.control_idx))
        truth_sum = np.zeros(len(self.gene_names), dtype=np.float64)
        pred_sum = np.zeros(len(self.gene_names), dtype=np.float64)
        for start in range(0, n, int(self.options.batch_size)):
            end = min(start + int(self.options.batch_size), n)
            target_x = self.dataset._load_rows(target_rows[start:end])
            control_x = self.dataset._load_rows(control_rows[start:end])
            truth_sum += target_x.sum(axis=0, dtype=np.float64)
            src_z = self._encode(control_x)
            action = self._action(condition, src_z.shape[0])
            with torch.no_grad():
                pred_z = self.predictor(src_z, action).z
            pred_x = self._decode(pred_z)
            pred_sum += pred_x.sum(axis=0, dtype=np.float64)
        return (pred_sum / n).astype(np.float32), (truth_sum / n).astype(np.float32), int(n)

    def run(self) -> dict[str, Any]:
        conditions = load_gears_subgroup_conditions(self.options.subgroup)
        if not conditions:
            conditions = sorted(c for c in np.unique(self.raw_conditions) if c != self.options.control_label)
        pred_rows = []
        truth_rows = []
        n_cells = []
        for condition in conditions:
            pred_mean, truth_mean, n = self._predict_condition_mean(condition)
            pred_rows.append(pred_mean)
            truth_rows.append(truth_mean)
            n_cells.append(n)
        out_npz = self.output_dir / "predictions.npz"
        np.savez_compressed(
            out_npz,
            pred=np.stack(pred_rows).astype(np.float32),
            truth=np.stack(truth_rows).astype(np.float32),
            conditions=np.asarray(conditions, dtype=str),
            gene_names=np.asarray(self.gene_names, dtype=str),
            n_cells=np.asarray(n_cells, dtype=np.int32),
        )
        summary = {
            "prediction": str(out_npz),
            "n_conditions": int(len(conditions)),
            "n_genes": int(len(self.gene_names)),
            "condition_mean_export": True,
            "max_cells_per_condition": self.options.max_cells_per_condition,
            "checkpoint": str(self.options.perturbation_checkpoint),
            "vae_checkpoint": str(self.options.vae_checkpoint),
            "load_info": self.load_info,
        }
        write_json(self.output_dir / "export_summary.json", summary)
        return summary


def export_foundation_gears_predictions(options: FoundationGearsPredictionExportOptions) -> dict[str, Any]:
    return FoundationGearsPredictionExporter(options).run()
