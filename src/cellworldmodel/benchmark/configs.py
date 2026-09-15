"""Per-dataset hyperparameter configs for M1/M2/M7/M8 benchmark runs.

Extracted from run_benchmark.py so both run_benchmark.py and
run_intermediate_eval.py can share without cross-script imports.
"""
from __future__ import annotations


DATASET_CONFIGS: dict[str, dict] = {
    "mouse": {
        "hidden_dim": 128, "n_layers": 3, "noise_dim": 8, "time_emb_dim": 32,
        "batch_size": 256, "K": 8, "lr": 1e-3,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": None, "default_epochs": 300,
    },
    "clonidine": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 4, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 0.5, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.1, "grad_clip": 1.0, "default_epochs": 500,
    },
    "trametinib": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 4, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 0.5, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.1, "grad_clip": 1.0, "default_epochs": 500,
    },
    "veres": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 16, "time_emb_dim": 32,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 500,
    },
    "norman": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 4, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 0.5, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.1, "grad_clip": 1.0, "default_epochs": 500,
    },
    "weinreb_hvg": {
        # PCA-50 space (same as PRESCIENT input), 3 timepoints (d2/d4/d6), delta=4.
        # Similar scale to Veres 30D. Used for apples-to-apples vs PRESCIENT.
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 500,
    },
    "weinreb_scvi": {
        # scVI 64-dim latent (ortholog-filtered, from output/scvi/v1_weinreb/).
        # Gaussian prior → d2→d6 shift expected 1-3 units (vs PCA-50 12.68).
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 500,
    },
    "veres_scvi": {
        # scVI 64-dim latent on Veres Stage 5 (ortholog-renamed, 51K × 13,660 → 64D).
        # 8 timepoints (CellWeek 0..7), shift ~2.8 units (vs BranchSBM PCA-30 ~8).
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 500,
    },
    "paper_weinreb_scvi128": {
        # Foundation VAE 128D latent exported in output/paper_bench/representations.
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 500,
    },
    "paper_veres_scvi128": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 500,
    },
    "synthetic_growth": {
        "hidden_dim": 128, "n_layers": 3, "noise_dim": 8, "time_emb_dim": 64,
        "batch_size": 128, "K": 4, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 0.2,
        "lambda_down": 0.05, "lambda_mass": 0.5,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 300,
        "optimizer": "adamw", "weight_decay": 0.01,
        "lr_schedule": "warmup_cosine", "warmup_frac": 0.05,
        "dit_size": "tiny", "waddington_dit": True,
        "growth_mode": "learned", "growth_head_warmup_epochs": 25,
        "multi_delta": True,
    },
    "cellstream_sim_growth": {
        "hidden_dim": 128, "n_layers": 3, "noise_dim": 8, "time_emb_dim": 64,
        "batch_size": 128, "K": 4, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 0.2,
        "lambda_down": 0.05, "lambda_mass": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 300,
        "optimizer": "adamw", "weight_decay": 0.01,
        "lr_schedule": "warmup_cosine", "warmup_frac": 0.05,
        "dit_size": "tiny", "waddington_dit": True,
        "growth_mode": "learned", "growth_head_warmup_epochs": 25,
        "multi_delta": True,
    },
    # Native-representation benchmarks (dataset ships its own aligned space).
    # Scratch-only: the pretrained backbone lives in the shared scVI-128 space
    # and cannot be initialised into these coordinates.
    "paper_native_zesta": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 16, "time_emb_dim": 32,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 5000,
        "optimizer": "adamw", "weight_decay": 0.01,
        "lr_schedule": "warmup_cosine", "warmup_frac": 0.05,
        "dit_size": "small", "waddington_dit": True,
        "multi_delta": True,
    },
    "stvcr_rectangle_gene": {
        "hidden_dim": 128, "n_layers": 3, "noise_dim": 8, "time_emb_dim": 64,
        "batch_size": 128, "K": 4, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 0.2,
        "lambda_down": 0.05, "lambda_mass": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 300,
        "optimizer": "adamw", "weight_decay": 0.01,
        "lr_schedule": "warmup_cosine", "warmup_frac": 0.05,
        "dit_size": "tiny", "waddington_dit": True,
        "growth_mode": "learned", "growth_head_warmup_epochs": 25,
        "multi_delta": True,
    },
}

DEFAULT_PCS = {
    "mouse": 2, "clonidine": 50, "trametinib": 50, "veres": 30,
    "norman": 128, "weinreb_hvg": 50, "weinreb_scvi": 64, "veres_scvi": 64,
    "paper_weinreb_scvi128": 128, "paper_veres_scvi128": 128,
    "synthetic_growth": 2,
    "cellstream_sim_growth": 6,
    "paper_native_zesta": 100,
    "stvcr_rectangle_gene": 3,
}


# Any paper_native_<name> export that has no bespoke entry falls back to this.
# Mirrors the paper_veres_scvi128 recipe, which is the downstream configuration
# every shared-representation benchmark has been run with.
PAPER_NATIVE_DEFAULT = {
    "hidden_dim": 512, "n_layers": 3, "noise_dim": 16, "time_emb_dim": 32,
    "batch_size": 256, "K": 8, "lr": 3e-4,
    "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
    "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 5000,
    "optimizer": "adamw", "weight_decay": 0.01,
    "lr_schedule": "warmup_cosine", "warmup_frac": 0.05,
    "dit_size": "small", "waddington_dit": True,
    "multi_delta": True,
}

# This named export uses the shared foundation encoder, not native ZESTA PCs.
# Match its time embedding to the pretrained 128D backbone without affecting
# other generic exports or the separate paper_native_zesta recipe.
DATASET_CONFIGS["paper_native_zestaspatial"] = {
    **PAPER_NATIVE_DEFAULT, "noise_dim": 32, "time_emb_dim": 64,
}


def dataset_config(dataset: str) -> dict:
    """Config for `dataset`, falling back for generic paper_native_ exports."""
    if dataset in DATASET_CONFIGS:
        return dict(DATASET_CONFIGS[dataset])
    if dataset.startswith("paper_native_"):
        return dict(PAPER_NATIVE_DEFAULT)
    raise KeyError(dataset)


def dataset_pcs(dataset: str, default: int = 128) -> int:
    if dataset in DEFAULT_PCS:
        return int(DEFAULT_PCS[dataset])
    if dataset.startswith("paper_native_"):
        return int(default)
    raise KeyError(dataset)
