"""Prediction export and shared-vocabulary evaluation for GEARS downstream runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from cellworldmodel.foundation.gears_downstream_dataset import GearsDownstreamDataset
from cellworldmodel.foundation.gears_shared_eval import SharedVocabularyEvalOptions, run_shared_vocab_eval


EncodeFn = Callable[[np.ndarray], torch.Tensor]
DecodeFn = Callable[[torch.Tensor], torch.Tensor]
ActionFn = Callable[[str, int], torch.Tensor]
PredictFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict[str, float]]]
PredictXFn = Callable[[str, np.ndarray, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GearsDownstreamEvalOptions:
    gears_adata: str | Path
    gene_vocab: str | Path
    subgroup: str | Path | None
    output_dir: str | Path
    batch_size: int
    top_k: int = 20
    eval_max_cells_per_condition: int | None = None


class GearsDownstreamEvaluator:
    def __init__(
        self,
        *,
        dataset: GearsDownstreamDataset,
        rng: np.random.Generator,
        encode: EncodeFn,
        decode: DecodeFn,
        action: ActionFn,
        predict_z: PredictFn,
        options: GearsDownstreamEvalOptions,
        predict_x: PredictXFn | None = None,
    ) -> None:
        self.dataset = dataset
        self.rng = rng
        self.encode = encode
        self.decode = decode
        self.action = action
        self.predict_z = predict_z
        self.predict_x = predict_x
        self.options = options
        self.output_dir = Path(options.output_dir)

    def export_predictions(self, output_path: str | Path) -> dict:
        conditions = self.dataset.test_conditions
        pred_rows = []
        truth_rows = []
        n_cells = []
        for condition in conditions:
            control_x, target_x, n = self.dataset.condition_mean_arrays(
                condition,
                self.rng,
                max_cells=self.options.eval_max_cells_per_condition,
            )
            pred_sum = np.zeros(self.dataset.n_genes, dtype=np.float64)
            truth_sum = target_x.sum(axis=0, dtype=np.float64)
            for start in range(0, n, int(self.options.batch_size)):
                end = min(start + int(self.options.batch_size), n)
                with torch.no_grad():
                    src_z = self.encode(control_x[start:end])
                    action = self.action(condition, src_z.shape[0])
                    pred_z, _ = self.predict_z(src_z, action)
                    if self.predict_x is None:
                        pred_x_t = self.decode(pred_z)
                    else:
                        pred_x_t = self.predict_x(condition, control_x[start:end], pred_z, action)
                    pred_x = pred_x_t.detach().cpu().numpy()
                pred_sum += pred_x.sum(axis=0, dtype=np.float64)
            pred_rows.append((pred_sum / n).astype(np.float32))
            truth_rows.append((truth_sum / n).astype(np.float32))
            n_cells.append(n)
        np.savez_compressed(
            output_path,
            pred=np.stack(pred_rows).astype(np.float32),
            truth=np.stack(truth_rows).astype(np.float32),
            conditions=np.asarray(conditions, dtype=str),
            gene_names=np.asarray(self.dataset.gene_names, dtype=str),
            n_cells=np.asarray(n_cells, dtype=np.int32),
        )
        return {
            "path": str(output_path),
            "n_conditions": int(len(conditions)),
            "n_genes": int(self.dataset.n_genes),
        }

    def run(self) -> tuple[dict, dict]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pred_summary = self.export_predictions(self.output_dir / "predictions.npz")
        shared_summary = run_shared_vocab_eval(SharedVocabularyEvalOptions(
            gears_adata=self.options.gears_adata,
            prediction=self.output_dir / "predictions.npz",
            output_dir=self.output_dir / "shared_eval",
            ours_genes=self.options.gene_vocab,
            subgroup=self.options.subgroup,
            top_k=self.options.top_k,
        ))
        return pred_summary, shared_summary
