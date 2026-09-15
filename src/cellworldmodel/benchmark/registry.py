"""Registries for benchmark dataset adapters and model constructors."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from cellworldmodel.benchmark.branchsbm_adapter import (
    ClonidineAdapter,
    MouseHematopoiesisAdapter,
    NormanAdapter,
    TrametinibAdapter,
    VeresAdapter,
)
from cellworldmodel.benchmark.weinreb_hvg_adapter import WeinrebHVGAdapter
from cellworldmodel.model.br_celldrift_bench import BRCellDriftMLP
from cellworldmodel.model.drift_dit_1d import DRIFT_DIT_1D_MODELS
from cellworldmodel.model.pc_celldrift_bench import PCCellDriftMLP
from cellworldmodel.model.waddington_dit_1d import WADDINGTON_DIT_1D_MODELS


METHODS = {"m1", "m2", "m7", "m8", "m9", "m10"}


def build_adapter(dataset: str, pcs: int, seed: int):
    """Build dataset adapter. `seed` is the split seed."""
    if dataset == "mouse":
        return MouseHematopoiesisAdapter(seed=seed)
    if dataset == "clonidine":
        return ClonidineAdapter(pcs=pcs, seed=seed)
    if dataset == "trametinib":
        return TrametinibAdapter(pcs=pcs, seed=seed)
    if dataset == "veres":
        return VeresAdapter(seed=seed, dim=pcs)
    if dataset == "weinreb_hvg":
        return WeinrebHVGAdapter(seed=seed)
    if dataset == "weinreb_scvi":
        from cellworldmodel.benchmark.weinreb_scvi_adapter import WeinrebScVIAdapter
        return WeinrebScVIAdapter(seed=seed)
    if dataset == "veres_scvi":
        from cellworldmodel.benchmark.veres_scvi_adapter import VeresScVIAdapter
        return VeresScVIAdapter(seed=seed)
    if dataset == "paper_weinreb_scvi128":
        from cellworldmodel.benchmark.paper_bench_adapter import PaperBenchScVI128Adapter
        return PaperBenchScVI128Adapter("weinreb", seed=seed)
    if dataset == "paper_veres_scvi128":
        from cellworldmodel.benchmark.paper_bench_adapter import PaperBenchScVI128Adapter
        return PaperBenchScVI128Adapter("veres", seed=seed)
    if dataset.startswith("paper_native_"):
        # generic: paper_native_<name> reads <representation root>/<name>,
        # letting a benchmark that ships its own aligned coordinates reuse the
        # whole shared-representation protocol without a bespoke adapter.
        from cellworldmodel.benchmark.paper_bench_adapter import PaperBenchScVI128Adapter
        return PaperBenchScVI128Adapter(dataset[len("paper_native_"):], seed=seed)
    if dataset == "synthetic_growth":
        from cellworldmodel.benchmark.synthetic_growth_adapter import SyntheticGrowthAdapter
        return SyntheticGrowthAdapter(seed=seed)
    if dataset == "cellstream_sim_growth":
        from cellworldmodel.benchmark.cellstream_adapter import CellStreamSimGrowthAdapter
        return CellStreamSimGrowthAdapter(seed=seed)
    if dataset == "stvcr_rectangle_gene":
        from cellworldmodel.benchmark.synthetic_growth_adapter import StVCRRectangleGeneAdapter
        return StVCRRectangleGeneAdapter(seed=seed)
    if dataset == "norman":
        scdfm_path = (
            Path(__file__).parent.parent.parent.parent
            / "3rdparty" / "scDFM" / "data" / "norman" / "norman.h5ad"
        )
        data_path = str(scdfm_path) if scdfm_path.exists() else None
        return NormanAdapter(
            data_path=data_path,
            split_seed=seed,
            precomputed_pca_dim=pcs,
            split_method="additive",
            n_top_genes=5000,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def build_model(method: str, dim: int, cfg: dict, tau_init: float):
    """Instantiate model based on method."""
    common = dict(
        dim=dim,
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        time_emb_dim=cfg["time_emb_dim"],
        tau_init=tau_init,
    )
    if method in ("m1", "m7"):
        return BRCellDriftMLP(noise_dim=cfg["noise_dim"], **common)
    if method in ("m2", "m8"):
        return PCCellDriftMLP(curl_rank=16, **common)
    if method in ("m9", "m10"):
        dit_size = cfg.get("dit_size", "tiny")
        if dit_size not in DRIFT_DIT_1D_MODELS:
            raise ValueError(f"Unknown dit_size={dit_size!r}; expected one of {sorted(DRIFT_DIT_1D_MODELS)}")
        if bool(cfg.get("waddington_dit", False)):
            ctor = WADDINGTON_DIT_1D_MODELS[dit_size]
            return ctor(
                dim=dim,
                time_emb_dim=cfg["time_emb_dim"],
                tau_init=tau_init,
                curl_rank=int(cfg.get("curl_rank", 16)),
                use_rope=not bool(cfg.get("disable_rope", False)),
                curl_update=str(cfg.get("wdit_curl_update", "additive")),
                curl_time_mode=str(cfg.get("wdit_curl_time_mode", "full")),
                hybrid_delta0=float(cfg.get("wdit_hybrid_delta0", 36.0)),
                hybrid_slope=float(cfg.get("wdit_hybrid_slope", 0.25)),
                hard_delta0=float(cfg.get("wdit_hard_delta0", 30.0)),
                time_embedding_mode=str(cfg.get("wdit_time_embedding", "legacy_fourier")),
                time_delta_transform=str(cfg.get("wdit_time_delta_transform", "normalized")),
                time_delta_scale=cfg.get("wdit_time_delta_scale"),
                curl_time_embedding_mode=str(cfg.get("wdit_curl_time_embedding", "same")),
                curl_time_delta_transform=str(cfg.get("wdit_curl_time_delta_transform", "normalized")),
                curl_time_delta_scale=cfg.get("wdit_curl_time_delta_scale"),
                action_dim=int(cfg.get("action_dim", 0)),
                growth_head=str(cfg.get("growth_mode", "frozen")) == "learned",
                growth_log_mass_clip=float(cfg.get("growth_log_mass_clip", 8.0)),
                growth_integration_steps=int(cfg.get("growth_integration_steps", 1)),
            )
        ctor = DRIFT_DIT_1D_MODELS[dit_size]
        return ctor(
            dim=dim,
            time_emb_dim=cfg["time_emb_dim"],
            tau_init=tau_init,
            state_chunk_dim=cfg.get("state_chunk_dim"),
            learned_state_tokens=cfg.get("learned_state_tokens"),
            use_rope=not bool(cfg.get("disable_rope", False)),
            action_dim=int(cfg.get("action_dim", 0)),
            time_embedding_mode=str(cfg.get("dit_time_embedding", "legacy_fourier")),
            time_delta_transform=str(cfg.get("dit_time_delta_transform", "normalized")),
            time_delta_scale=cfg.get("dit_time_delta_scale"),
            mean_prediction_mode=str(cfg.get("dit_mean_prediction", "monte_carlo")),
        )
    raise ValueError(f"Unknown method: {method}")


def get_target_labels(dataset: str, adapter) -> Optional[np.ndarray]:
    if dataset in ("clonidine", "trametinib"):
        return adapter.get_target_cluster_labels()
    if dataset == "veres":
        return adapter.get_target_cluster_labels(n_clusters=11)
    if dataset == "mouse":
        from sklearn.cluster import KMeans
        tgt = adapter.coords_by_t[adapter.timepoints[-1]]
        return KMeans(n_clusters=2, random_state=42, n_init=10).fit(tgt).labels_.astype(np.int64)
    return None
