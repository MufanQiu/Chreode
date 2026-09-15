"""Entry point for foundation-model workflow utilities."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from cellworldmodel.foundation.catalog import build_catalog, write_catalog
from cellworldmodel.foundation.config import load_foundation_config
from cellworldmodel.foundation.gene_vocab import build_gene_vocab, write_gene_vocab
from cellworldmodel.foundation.latent_cache import LatentCacheConfig, write_latent_cache
from cellworldmodel.foundation.perturbation_predictors import available_perturbation_predictors
from cellworldmodel.foundation.pretrain_protocols import pretrain_protocol_names
from cellworldmodel.foundation.transition_index import build_transition_index, write_transition_index
from cellworldmodel.foundation.vae_eval import EmbeddingEvalConfig, evaluate_vae_embedding
from cellworldmodel.foundation.vae_registry import vae_architecture_names
from cellworldmodel.foundation.vae_train import VaeTrainOptions, steps_per_epoch, train_vae_smoke
from cellworldmodel.script.wandb_utils import add_wandb_args, maybe_init_wandb


def cmd_validate_config(args: argparse.Namespace) -> None:
    cfg = load_foundation_config(args.config)
    if args.print_resolved:
        print(yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=False))
    else:
        print(
            "OK "
            f"{args.config}: output_root={cfg.output_root} "
            f"data_root={cfg.data_root} "
            f"vae_latent_dims={list(cfg.vae.latent_dims)} "
            f"dynamics={cfg.dynamics.experiment}"
        )


def cmd_build_gene_vocab(args: argparse.Namespace) -> None:
    cfg = load_foundation_config(args.config)
    output_dir = args.output_dir or (cfg.output_path / "catalog")
    manifest = write_gene_vocab(build_gene_vocab(cfg), output_dir)
    print(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False))


def cmd_build_catalog(args: argparse.Namespace) -> None:
    cfg = load_foundation_config(args.config)
    output_dir = args.output_dir or (cfg.output_path / "catalog")
    catalog = build_catalog(cfg, max_files=args.max_files)
    manifest = write_catalog(catalog, output_dir)
    print(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False))


def cmd_train_vae_smoke(args: argparse.Namespace) -> None:
    cfg = load_foundation_config(args.config)
    catalog_dir = args.catalog_dir or (cfg.output_path / "catalog")
    output_dir = args.output_dir or (
        cfg.output_path / "vae" / f"smoke_{args.batch_strategy}_latent{args.latent_dim}"
    )
    wandb_run = maybe_init_wandb(
        args,
        config={
            "foundation_config": args.config.as_posix(),
            "catalog_dir": str(catalog_dir),
            "architecture": args.architecture,
            "latent_dim": int(args.latent_dim),
            "batch_strategy": args.batch_strategy,
            "max_steps": int(args.max_steps),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
        },
        output_dir=output_dir,
        default_name=f"foundation-vae-{args.batch_strategy}-latent{args.latent_dim}",
        default_group=cfg.wandb.group,
    )
    qc = train_vae_smoke(
        cfg,
        VaeTrainOptions(
            catalog_dir=catalog_dir,
            output_dir=output_dir,
            latent_dim=args.latent_dim,
            batch_strategy=args.batch_strategy,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            architecture=args.architecture,
            beta_kl=args.beta_kl,
            lr=args.lr,
            device=args.device,
            qc_sample_size=args.qc_sample_size,
            silhouette_max_samples=args.silhouette_max_samples,
        ),
        wandb_run=wandb_run,
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(yaml.safe_dump(qc, sort_keys=False, allow_unicode=False))


def cmd_train_vae_full(args: argparse.Namespace) -> None:
    cfg = load_foundation_config(args.config)
    catalog_dir = args.catalog_dir or (cfg.output_path / "catalog")
    output_dir = args.output_dir or (cfg.output_path / "vae" / f"full_{args.name}")
    step_per_epoch = steps_per_epoch(catalog_dir, args.batch_size, split="train")
    max_steps = int(step_per_epoch * args.epochs)
    wandb_run = maybe_init_wandb(
        args,
        config={
            "foundation_config": args.config.as_posix(),
            "catalog_dir": str(catalog_dir),
            "name": args.name,
            "architecture": args.architecture,
            "latent_dim": int(args.latent_dim),
            "batch_strategy": args.batch_strategy,
            "epochs": int(args.epochs),
            "steps_per_epoch": int(step_per_epoch),
            "max_steps": int(max_steps),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
        },
        output_dir=output_dir,
        default_name=f"foundation-vae-full-{args.name}",
        default_group=cfg.wandb.group,
    )
    qc = train_vae_smoke(
        cfg,
        VaeTrainOptions(
            catalog_dir=catalog_dir,
            output_dir=output_dir,
            latent_dim=args.latent_dim,
            batch_strategy=args.batch_strategy,
            max_steps=max_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            architecture=args.architecture,
            beta_kl=args.beta_kl,
            lr=args.lr,
            device=args.device,
            qc_sample_size=args.qc_sample_size,
            silhouette_max_samples=args.silhouette_max_samples,
            checkpoint_every_steps=step_per_epoch,
            checkpoint_prefix="epoch",
            run_label=f"full_{args.name}",
        ),
        wandb_run=wandb_run,
    )
    qc["epochs"] = int(args.epochs)
    qc["steps_per_epoch"] = int(step_per_epoch)
    with (Path(output_dir) / "qc.json").open("w", encoding="utf-8") as handle:
        json.dump(qc, handle, indent=2, sort_keys=True)
    if wandb_run is not None:
        wandb_run.finish()
    print(yaml.safe_dump(qc, sort_keys=False, allow_unicode=False))


def cmd_eval_vae_embedding(args: argparse.Namespace) -> None:
    result = evaluate_vae_embedding(EmbeddingEvalConfig(
        checkpoint=str(args.checkpoint),
        catalog_dir=str(args.catalog_dir),
        output_dir=str(args.output_dir),
        split=args.split,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        save_embeddings=not args.no_save_embeddings,
        silhouette_max_samples=args.silhouette_max_samples,
        knn_k=args.knn_k,
        make_plots=not args.no_plots,
        leiden_resolution=args.leiden_resolution,
    ))
    print(yaml.safe_dump(result, sort_keys=False, allow_unicode=False))


def cmd_encode_latent_cache(args: argparse.Namespace) -> None:
    manifest = write_latent_cache(LatentCacheConfig(
        checkpoint=str(args.checkpoint),
        catalog_dir=str(args.catalog_dir),
        output_dir=str(args.output_dir),
        splits=tuple(args.splits),
        batch_size=int(args.batch_size),
        shard_size=int(args.shard_size),
        device=args.device,
        allow_unknown_batch=bool(args.allow_unknown_batch),
    ))
    print(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False))


def _resolve_max_steps(args: argparse.Namespace, cfg) -> int:
    if args.max_steps is not None:
        return int(args.max_steps)
    catalog_dir = args.catalog_dir or (cfg.output_path / "catalog")
    step_per_epoch = steps_per_epoch(catalog_dir, int(args.batch_size), split="train")
    return int(step_per_epoch * int(args.epoch_equivalent))


def cmd_train_foundation_dynamics(args: argparse.Namespace) -> None:
    from cellworldmodel.foundation.dynamics_train import FoundationDynamicsTrainOptions, train_foundation_dynamics

    cfg = load_foundation_config(args.config)
    catalog_dir = args.catalog_dir or (cfg.output_path / "catalog")
    max_steps = _resolve_max_steps(args, cfg)
    output_dir = args.output_dir or (cfg.output_path / "dynamics" / args.name)
    wandb_run = maybe_init_wandb(
        args,
        config={
            "foundation_config": args.config.as_posix(),
            "catalog_dir": str(catalog_dir),
            "name": args.name,
            "objective": args.objective,
            "experiment": args.experiment,
            "max_steps": int(max_steps),
            "batch_size": int(args.batch_size),
            "k_samples": int(args.k_samples),
            "seed": int(args.seed),
        },
        output_dir=output_dir,
        default_name=f"foundation-dynamics-{args.name}",
        default_group=cfg.wandb.group,
    )
    summary = train_foundation_dynamics(
        cfg,
        FoundationDynamicsTrainOptions(
            catalog_dir=catalog_dir,
            output_dir=output_dir,
            objective=args.objective,
            max_steps=max_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            vae_checkpoint=args.vae_checkpoint,
            latent_cache_dir=args.latent_cache_dir,
            transition_index_dir=args.transition_index_dir,
            experiment=args.experiment,
            dit_size=args.dit_size,
            k_samples=args.k_samples,
            lr=args.lr,
            static_delta=args.static_delta,
            static_initialization=args.static_initialization,
            train_transition_policy=getattr(args, "train_transition_policy", "all_ordered"),
            checkpoint_every_steps=args.checkpoint_every_steps,
            log_every=args.log_every,
            device=args.device,
            **{name: getattr(args, name) for name in (
                "transport_objective", "transport_blur", "transport_max_iters", "transport_tolerance"
            ) if getattr(args, name, None) is not None},
        ),
        wandb_run=wandb_run,
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=False))


def cmd_train_perturbation_finetune(args: argparse.Namespace) -> None:
    from cellworldmodel.foundation.perturbation_train import PerturbationFineTuneOptions, train_perturbation_finetune

    cfg = load_foundation_config(args.config)
    catalog_dir = args.catalog_dir or (cfg.output_path / "catalog")
    output_dir = args.output_dir or (cfg.output_path / "perturbation" / args.name)
    wandb_run = maybe_init_wandb(
        args,
        config={
            "foundation_config": args.config.as_posix(),
            "catalog_dir": str(catalog_dir),
            "name": args.name,
            "init_name": args.init_name,
            "max_steps": int(args.max_steps),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
        },
        output_dir=output_dir,
        default_name=f"foundation-perturb-{args.name}",
        default_group=cfg.wandb.group,
    )
    summary = train_perturbation_finetune(
        cfg,
        PerturbationFineTuneOptions(
            catalog_dir=catalog_dir,
            output_dir=output_dir,
            norman_path=args.norman_path,
            vae_checkpoint=args.vae_checkpoint,
            init_checkpoint=args.init_checkpoint,
            init_name=args.init_name,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            experiment=args.experiment,
            dit_size=args.dit_size,
            action_dim=args.action_dim,
            k_samples=args.k_samples,
            lr=args.lr,
            split_method=args.split_method,
            eval_every=args.eval_every,
            checkpoint_every_steps=args.checkpoint_every_steps,
            device=args.device,
            allow_unknown_batch=args.allow_unknown_batch,
            action_encoder=args.action_encoder,
        ),
        wandb_run=wandb_run,
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=False))


def cmd_run_perturbation_baselines(args: argparse.Namespace) -> None:
    from cellworldmodel.foundation.perturbation_baselines import (
        PerturbationBaselineOptions,
        run_perturbation_baselines,
    )

    cfg = load_foundation_config(args.config)
    catalog_dir = args.catalog_dir or (cfg.output_path / "catalog")
    output_dir = args.output_dir or (cfg.output_path / "perturbation" / "baselines")
    summary = run_perturbation_baselines(PerturbationBaselineOptions(
        catalog_dir=catalog_dir,
        output_dir=output_dir,
        norman_path=args.norman_path,
        vae_checkpoint=args.vae_checkpoint,
        dynamics_checkpoint=args.dynamics_checkpoint,
        split_method=args.split_method,
        seed=args.seed,
        max_cells_per_condition=args.max_cells_per_condition,
        eval_cells_per_condition=args.eval_cells_per_condition,
        z_score=args.z_score,
        ridge_alpha=args.ridge_alpha,
        device=args.device,
    ))
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=False))


def cmd_eval_gears_shared_vocab(args: argparse.Namespace) -> None:
    from cellworldmodel.foundation.gears_shared_eval import (
        SharedVocabularyEvalOptions,
        run_shared_vocab_eval,
    )

    summary = run_shared_vocab_eval(SharedVocabularyEvalOptions(
        gears_adata=args.gears_adata,
        prediction=args.prediction,
        gears_test_res=args.gears_test_res,
        output_dir=args.output_dir,
        ours_genes=args.ours_genes,
        subgroup=args.subgroup,
        top_k=args.top_k,
        condition_col=args.condition_col,
        control_label=args.control_label,
    ))
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=False))


def cmd_export_gears_foundation_predictions(args: argparse.Namespace) -> None:
    from cellworldmodel.foundation.gears_prediction_export import (
        FoundationGearsPredictionExportOptions,
        export_foundation_gears_predictions,
    )

    cfg = load_foundation_config(args.config)
    summary = export_foundation_gears_predictions(FoundationGearsPredictionExportOptions(
        gears_adata=args.gears_adata,
        gene_vocab=args.gene_vocab or (cfg.output_path / "catalog" / "gene_vocab.parquet"),
        vae_checkpoint=args.vae_checkpoint,
        perturbation_checkpoint=args.perturbation_checkpoint,
        output_dir=args.output_dir,
        subgroup=args.subgroup,
        experiment=args.experiment,
        dit_size=args.dit_size,
        batch_size=args.batch_size,
        k_samples=args.k_samples,
        action_dim=args.action_dim,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        max_cells_per_condition=args.max_cells_per_condition,
        condition_col=args.condition_col,
        control_label=args.control_label,
    ))
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=False))


def cmd_train_gears_downstream(args: argparse.Namespace) -> None:
    from cellworldmodel.foundation.gears_downstream_train import (
        GearsDownstreamTrainOptions,
        train_gears_downstream,
    )

    cfg = load_foundation_config(args.config)
    summary = train_gears_downstream(GearsDownstreamTrainOptions(
        gears_adata=args.gears_adata,
        split=args.split,
        subgroup=args.subgroup,
        gene_vocab=args.gene_vocab or (cfg.output_path / "catalog" / "gene_vocab.parquet"),
        vae_checkpoint=args.vae_checkpoint,
        init_checkpoint=args.init_checkpoint,
        init_name=args.init_name,
        output_dir=args.output_dir,
        experiment=args.experiment,
        dit_size=args.dit_size,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        k_samples=args.k_samples,
        action_dim=args.action_dim,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        latent_weight=args.latent_weight,
        expr_weight=args.expr_weight,
        de_weight=args.de_weight,
        delta_weight=args.delta_weight,
        direction_weight=args.direction_weight,
        top_k=args.top_k,
        model_type=args.model_type,
        eval_max_cells_per_condition=args.eval_max_cells_per_condition,
        condition_col=args.condition_col,
        control_label=args.control_label,
    ))
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=False))


def cmd_train_population_perturbation(args: argparse.Namespace) -> None:
    from cellworldmodel.foundation.perturbation_population_train import (
        PopulationPerturbationTrainOptions,
        train_population_perturbation,
    )

    cfg = load_foundation_config(args.config)
    summary = train_population_perturbation(PopulationPerturbationTrainOptions(
        gears_adata=args.gears_adata,
        split=args.split,
        subgroup=args.subgroup,
        gene_vocab=args.gene_vocab or (cfg.output_path / "catalog" / "gene_vocab.parquet"),
        vae_checkpoint=args.vae_checkpoint,
        init_checkpoint=args.init_checkpoint,
        init_name=args.init_name,
        output_dir=args.output_dir,
        route=args.route,
        experiment=args.experiment,
        dit_size=args.dit_size,
        max_steps=args.max_steps,
        set_size=args.set_size,
        eval_batch_size=args.eval_batch_size,
        k_samples=args.k_samples,
        action_dim=args.action_dim,
        n_programs=args.n_programs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        latent_mmd_weight=args.latent_mmd_weight,
        latent_w2_weight=args.latent_w2_weight,
        expr_bulk_weight=args.expr_bulk_weight,
        de_bulk_weight=args.de_bulk_weight,
        delta_cosine_weight=args.delta_cosine_weight,
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        top_k=args.top_k,
        disable_kick=args.disable_kick,
        disable_field=args.disable_field,
        flat_action=args.flat_action,
        adapter_components=args.adapter_components,
        calibrate_potential=args.calibrate_potential,
        response_decoder=args.response_decoder,
        response_programs=args.response_programs,
        sparse_programs=args.sparse_programs,
        nonnegative_basis=args.nonnegative_basis,
        set_context_decoder=args.set_context_decoder,
        program_loss_weight=args.program_loss_weight,
        train_fraction=args.train_fraction,
        gene_graph=args.gene_graph,
        graph_mode=args.graph_mode,
        graph_weight=args.graph_weight,
        graph_basis_weight=args.graph_basis_weight,
        graph_output_weight=args.graph_output_weight,
        graph_top_k=args.graph_top_k,
        graph_self_loop=not args.no_graph_self_loop,
        graph_layers=args.graph_layers,
        rollout_steps=args.rollout_steps,
        disable_rollout=args.disable_rollout,
        disable_action_time=args.disable_action_time,
        virtual_time_min=args.virtual_time_min,
        virtual_time_max=args.virtual_time_max,
        eval_max_cells_per_condition=args.eval_max_cells_per_condition,
        condition_col=args.condition_col,
        control_label=args.control_label,
    ))
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=False))


def cmd_build_transition_index(args: argparse.Namespace) -> None:
    index = build_transition_index(
        args.catalog_dir,
        split=args.split,
        pair_policy=args.pair_policy,
    )
    manifest = write_transition_index(index, args.output_dir)
    print(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Foundation-model workflow utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config", help="Validate a foundation workflow YAML config.")
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--print-resolved", action="store_true")
    validate.set_defaults(func=cmd_validate_config)

    gene_vocab = sub.add_parser("build-gene-vocab", help="Build canonical ortholog gene vocabulary.")
    gene_vocab.add_argument("--config", required=True, type=Path)
    gene_vocab.add_argument("--output-dir", type=Path, default=None)
    gene_vocab.set_defaults(func=cmd_build_gene_vocab)

    catalog = sub.add_parser("build-catalog", help="Build h5ad/cell metadata catalog.")
    catalog.add_argument("--config", required=True, type=Path)
    catalog.add_argument("--output-dir", type=Path, default=None)
    catalog.add_argument("--max-files", type=int, default=None, help="Debug limit for tests/smoke runs.")
    catalog.set_defaults(func=cmd_build_catalog)

    vae_smoke = sub.add_parser("train-vae-smoke", help="Train a short local VAE smoke run.")
    vae_smoke.add_argument("--config", required=True, type=Path)
    vae_smoke.add_argument("--catalog-dir", type=Path, default=None)
    vae_smoke.add_argument("--output-dir", type=Path, default=None)
    vae_smoke.add_argument("--latent-dim", type=int, required=True)
    vae_smoke.add_argument(
        "--architecture",
        default="mlp512",
        choices=vae_architecture_names(),
    )
    vae_smoke.add_argument(
        "--batch-strategy",
        choices=["b1_leaf_dataset", "b2_encoder_nobatch_decoder_residual", "b3_none"],
        required=True,
    )
    vae_smoke.add_argument("--max-steps", type=int, default=20)
    vae_smoke.add_argument("--batch-size", type=int, default=32)
    vae_smoke.add_argument("--seed", type=int, default=0)
    vae_smoke.add_argument("--beta-kl", type=float, default=1e-3)
    vae_smoke.add_argument("--lr", type=float, default=1e-3)
    vae_smoke.add_argument("--device", default=None)
    vae_smoke.add_argument("--qc-sample-size", type=int, default=2048)
    vae_smoke.add_argument("--silhouette-max-samples", type=int, default=2000)
    add_wandb_args(vae_smoke)
    vae_smoke.set_defaults(func=cmd_train_vae_smoke)

    eval_vae = sub.add_parser("eval-vae-embedding", help="Evaluate semantic structure of VAE embeddings.")
    eval_vae.add_argument("--checkpoint", required=True, type=Path)
    eval_vae.add_argument("--catalog-dir", required=True, type=Path)
    eval_vae.add_argument("--output-dir", required=True, type=Path)
    eval_vae.add_argument("--split", default="val", choices=["train", "val", "test", "heldout"])
    eval_vae.add_argument("--sample-size", type=int, default=10000)
    eval_vae.add_argument("--batch-size", type=int, default=512)
    eval_vae.add_argument("--seed", type=int, default=0)
    eval_vae.add_argument("--device", default="cpu")
    eval_vae.add_argument("--no-save-embeddings", action="store_true")
    eval_vae.add_argument("--no-plots", action="store_true")
    eval_vae.add_argument("--silhouette-max-samples", type=int, default=2000)
    eval_vae.add_argument("--knn-k", type=int, default=15)
    eval_vae.add_argument("--leiden-resolution", type=float, default=1.0)
    eval_vae.set_defaults(func=cmd_eval_vae_embedding)

    transition = sub.add_parser("build-transition-index", help="Build foundation transition index.")
    transition.add_argument("--catalog-dir", required=True, type=Path)
    transition.add_argument("--output-dir", required=True, type=Path)
    transition.add_argument("--split", default="train", choices=["train", "val", "test", "heldout"])
    transition.add_argument("--pair-policy", default="all_ordered", choices=["all_ordered", "adjacent", "endpoint"])
    transition.set_defaults(func=cmd_build_transition_index)

    vae_full = sub.add_parser("train-vae-full", help="Train a full/epoch-based local VAE run.")
    vae_full.add_argument("--config", required=True, type=Path)
    vae_full.add_argument("--catalog-dir", type=Path, default=None)
    vae_full.add_argument("--output-dir", type=Path, default=None)
    vae_full.add_argument("--name", default="scvi1024_l128_vae2")
    vae_full.add_argument("--latent-dim", type=int, required=True)
    vae_full.add_argument("--architecture", default="scvi_fclayers1024", choices=vae_architecture_names())
    vae_full.add_argument(
        "--batch-strategy",
        choices=["b1_leaf_dataset", "b2_encoder_nobatch_decoder_residual", "b3_none"],
        required=True,
    )
    vae_full.add_argument("--epochs", type=int, default=2)
    vae_full.add_argument("--batch-size", type=int, default=2048)
    vae_full.add_argument("--seed", type=int, default=0)
    vae_full.add_argument("--beta-kl", type=float, default=1e-3)
    vae_full.add_argument("--lr", type=float, default=1e-3)
    vae_full.add_argument("--device", default=None)
    vae_full.add_argument("--qc-sample-size", type=int, default=4096)
    vae_full.add_argument("--silhouette-max-samples", type=int, default=2000)
    add_wandb_args(vae_full)
    vae_full.set_defaults(func=cmd_train_vae_full)

    latent = sub.add_parser("encode-latent-cache", help="Encode split cells into a chunked frozen-VAE latent cache.")
    latent.add_argument("--checkpoint", required=True, type=Path)
    latent.add_argument("--catalog-dir", required=True, type=Path)
    latent.add_argument("--output-dir", required=True, type=Path)
    latent.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    latent.add_argument("--batch-size", type=int, default=512)
    latent.add_argument("--shard-size", type=int, default=50_000)
    latent.add_argument("--device", default="cuda")
    latent.add_argument("--allow-unknown-batch", action="store_true")
    latent.set_defaults(func=cmd_encode_latent_cache)

    protocols = sub.add_parser("list-pretrain-protocols", help="List foundation pretraining objective protocols.")
    protocols.set_defaults(func=lambda args: print(yaml.safe_dump(pretrain_protocol_names(), sort_keys=False)))

    dyn = sub.add_parser("train-foundation-dynamics", help="Train A1/A2 foundation DiT pretraining objectives.")
    dyn.add_argument("--config", required=True, type=Path)
    dyn.add_argument("--catalog-dir", type=Path, default=None)
    dyn.add_argument("--output-dir", type=Path, default=None)
    dyn.add_argument("--name", required=True)
    dyn.add_argument("--objective", required=True, choices=["static_dit_reconstruction", "temporal_dynamics"])
    dyn.add_argument("--vae-checkpoint", type=Path, default=None)
    dyn.add_argument("--latent-cache-dir", type=Path, default=None)
    dyn.add_argument("--transition-index-dir", type=Path, default=None)
    dyn.add_argument("--experiment", default="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw")
    dyn.add_argument("--dit-size", default="small", choices=["tiny", "small", "base"])
    dyn.add_argument("--max-steps", type=int, default=None)
    dyn.add_argument("--epoch-equivalent", type=int, default=2)
    dyn.add_argument("--batch-size", type=int, default=1024)
    dyn.add_argument("--k-samples", type=int, default=8)
    dyn.add_argument("--seed", type=int, default=0)
    dyn.add_argument("--lr", type=float, default=3e-4)
    dyn.add_argument("--static-delta", type=float, default=1.0)
    dyn.add_argument("--static-initialization", choices=["legacy", "match-temporal"], default="legacy",
                     help="Align only static backbone RNG and time scales with the supplied temporal training index.")
    dyn.add_argument("--train-transition-policy", choices=["all_ordered", "leaf_endpoints"], default="all_ordered",
                     help="Restrict temporal training to each leaf's endpoints while retaining ordered-pair leaf probabilities and model time scales.")
    dyn.add_argument("--checkpoint-every-steps", type=int, default=1000)
    dyn.add_argument("--log-every", type=int, default=50)
    dyn.add_argument("--device", default=None)
    dyn.add_argument("--transport-objective", default=None,
                     choices=["legacy", "sinkhorn_divergence", "converged_proxy"])
    dyn.add_argument("--transport-blur", type=float, default=None)
    dyn.add_argument("--transport-max-iters", type=int, default=None)
    dyn.add_argument("--transport-tolerance", type=float, default=None)
    add_wandb_args(dyn)
    dyn.set_defaults(func=cmd_train_foundation_dynamics)

    perturb = sub.add_parser("train-perturbation-finetune", help="Fine-tune action-conditioned transition model on Norman.")
    perturb.add_argument("--config", required=True, type=Path)
    perturb.add_argument("--catalog-dir", type=Path, default=None)
    perturb.add_argument("--output-dir", type=Path, default=None)
    perturb.add_argument("--name", required=True)
    perturb.add_argument("--init-name", required=True)
    perturb.add_argument("--norman-path", required=True, type=Path)
    perturb.add_argument("--vae-checkpoint", required=True, type=Path)
    perturb.add_argument("--init-checkpoint", type=Path, default=None)
    perturb.add_argument("--experiment", default="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw")
    perturb.add_argument("--dit-size", default="small", choices=["tiny", "small", "base"])
    perturb.add_argument("--max-steps", type=int, default=1000)
    perturb.add_argument("--batch-size", type=int, default=256)
    perturb.add_argument("--k-samples", type=int, default=8)
    perturb.add_argument("--action-dim", type=int, default=64)
    perturb.add_argument(
        "--action-encoder",
        default="geneset_deepset_v1",
        choices=["geneset_deepset_v1", "categorical_perturbation"],
    )
    perturb.add_argument("--seed", type=int, default=0)
    perturb.add_argument("--lr", type=float, default=3e-4)
    perturb.add_argument("--split-method", default="additive", choices=["additive", "holdout"])
    perturb.add_argument("--eval-every", type=int, default=200)
    perturb.add_argument("--checkpoint-every-steps", type=int, default=1000)
    perturb.add_argument("--device", default=None)
    perturb.add_argument("--allow-unknown-batch", action="store_true")
    add_wandb_args(perturb)
    perturb.set_defaults(func=cmd_train_perturbation_finetune)

    pbase = sub.add_parser("run-perturbation-baselines", help="Evaluate simple and PRESCIENT-like Norman baselines.")
    pbase.add_argument("--config", required=True, type=Path)
    pbase.add_argument("--catalog-dir", type=Path, default=None)
    pbase.add_argument("--output-dir", type=Path, default=None)
    pbase.add_argument("--norman-path", required=True, type=Path)
    pbase.add_argument("--vae-checkpoint", required=True, type=Path)
    pbase.add_argument("--dynamics-checkpoint", type=Path, default=None)
    pbase.add_argument("--split-method", default="additive", choices=["additive", "holdout"])
    pbase.add_argument("--seed", type=int, default=0)
    pbase.add_argument("--max-cells-per-condition", type=int, default=512)
    pbase.add_argument("--eval-cells-per-condition", type=int, default=256)
    pbase.add_argument("--z-score", type=float, default=5.0)
    pbase.add_argument("--ridge-alpha", type=float, default=1.0)
    pbase.add_argument("--device", default=None)
    pbase.set_defaults(func=cmd_run_perturbation_baselines)

    shared = sub.add_parser("eval-gears-shared-vocab", help="Evaluate predictions with shared-vocabulary GEARS DE metrics.")
    shared.add_argument("--gears-adata", required=True, type=Path)
    group = shared.add_mutually_exclusive_group(required=True)
    group.add_argument("--prediction", type=Path, default=None, help="NPZ with pred/truth/conditions/gene_names arrays.")
    group.add_argument("--gears-test-res", type=Path, default=None, help="GEARS test_res.pkl containing pred/truth/pert_cat.")
    shared.add_argument("--ours-genes", required=True, type=Path, help="Text/JSON/NPY/NPZ/Parquet gene list for our output vocabulary.")
    shared.add_argument("--subgroup", type=Path, default=None, help="GEARS simulation subgroup pickle.")
    shared.add_argument("--output-dir", required=True, type=Path)
    shared.add_argument("--top-k", type=int, default=20)
    shared.add_argument("--condition-col", default="condition")
    shared.add_argument("--control-label", default="ctrl")
    shared.set_defaults(func=cmd_eval_gears_shared_vocab)

    export = sub.add_parser("export-gears-foundation-predictions", help="Export foundation perturbation checkpoint predictions for GEARS shared-vocab eval.")
    export.add_argument("--config", required=True, type=Path)
    export.add_argument("--gears-adata", required=True, type=Path)
    export.add_argument("--gene-vocab", type=Path, default=None)
    export.add_argument("--vae-checkpoint", required=True, type=Path)
    export.add_argument("--perturbation-checkpoint", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument("--subgroup", type=Path, default=None)
    export.add_argument("--experiment", default="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw")
    export.add_argument("--dit-size", default="small", choices=["tiny", "small", "base"])
    export.add_argument("--batch-size", type=int, default=256)
    export.add_argument("--k-samples", type=int, default=8)
    export.add_argument("--action-dim", type=int, default=64)
    export.add_argument("--lr", type=float, default=3e-4)
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--device", default=None)
    export.add_argument("--max-cells-per-condition", type=int, default=None)
    export.add_argument("--condition-col", default="condition")
    export.add_argument("--control-label", default="ctrl")
    export.set_defaults(func=cmd_export_gears_foundation_predictions)

    gtrain = sub.add_parser("train-gears-downstream", help="Fine-tune foundation perturbation model on GEARS simulation split with shared-DE expression losses.")
    gtrain.add_argument("--config", required=True, type=Path)
    gtrain.add_argument("--gears-adata", required=True, type=Path)
    gtrain.add_argument("--split", required=True, type=Path)
    gtrain.add_argument("--subgroup", type=Path, default=None)
    gtrain.add_argument("--gene-vocab", type=Path, default=None)
    gtrain.add_argument("--vae-checkpoint", required=True, type=Path)
    gtrain.add_argument("--init-checkpoint", type=Path, default=None)
    gtrain.add_argument("--init-name", required=True)
    gtrain.add_argument("--output-dir", required=True, type=Path)
    gtrain.add_argument("--experiment", default="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw")
    gtrain.add_argument("--dit-size", default="small", choices=["tiny", "small", "base"])
    gtrain.add_argument("--max-steps", type=int, default=1000)
    gtrain.add_argument("--batch-size", type=int, default=256)
    gtrain.add_argument("--k-samples", type=int, default=8)
    gtrain.add_argument("--action-dim", type=int, default=64)
    gtrain.add_argument("--lr", type=float, default=3e-4)
    gtrain.add_argument("--seed", type=int, default=0)
    gtrain.add_argument("--device", default=None)
    gtrain.add_argument("--latent-weight", type=float, default=0.1)
    gtrain.add_argument("--expr-weight", type=float, default=1.0)
    gtrain.add_argument("--de-weight", type=float, default=2.0)
    gtrain.add_argument("--delta-weight", type=float, default=0.2)
    gtrain.add_argument("--direction-weight", type=float, default=0.0)
    gtrain.add_argument("--top-k", type=int, default=20)
    gtrain.add_argument(
        "--model-type",
        default="direct_action",
        choices=available_perturbation_predictors(),
    )
    gtrain.add_argument("--eval-max-cells-per-condition", type=int, default=None)
    gtrain.add_argument("--condition-col", default="condition")
    gtrain.add_argument("--control-label", default="ctrl")
    gtrain.set_defaults(func=cmd_train_gears_downstream)

    pop = sub.add_parser("train-population-perturbation", help="Train Route1/Route2 population-level perturbation predictors.")
    pop.add_argument("--config", required=True, type=Path)
    pop.add_argument("--gears-adata", required=True, type=Path)
    pop.add_argument("--split", required=True, type=Path)
    pop.add_argument("--subgroup", type=Path, default=None)
    pop.add_argument("--gene-vocab", type=Path, default=None)
    pop.add_argument("--vae-checkpoint", required=True, type=Path)
    pop.add_argument("--init-checkpoint", type=Path, default=None)
    pop.add_argument("--init-name", required=True)
    pop.add_argument("--output-dir", required=True, type=Path)
    pop.add_argument("--route", required=True, choices=["route1_field", "route1_internal", "route2_setflow", "hybrid_rollout", "ktvu_rollout", "native_u_bridge"])
    pop.add_argument("--experiment", default="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw")
    pop.add_argument("--dit-size", default="small", choices=["tiny", "small", "base"])
    pop.add_argument("--max-steps", type=int, default=1200)
    pop.add_argument("--set-size", type=int, default=128)
    pop.add_argument("--eval-batch-size", type=int, default=256)
    pop.add_argument("--k-samples", type=int, default=2)
    pop.add_argument("--action-dim", type=int, default=64)
    pop.add_argument("--n-programs", type=int, default=8)
    pop.add_argument("--lr", type=float, default=3e-4)
    pop.add_argument("--seed", type=int, default=0)
    pop.add_argument("--device", default=None)
    pop.add_argument("--latent-mmd-weight", type=float, default=1.0)
    pop.add_argument("--latent-w2-weight", type=float, default=0.1)
    pop.add_argument("--expr-bulk-weight", type=float, default=1.0)
    pop.add_argument("--de-bulk-weight", type=float, default=2.0)
    pop.add_argument("--delta-cosine-weight", type=float, default=0.2)
    pop.add_argument("--sinkhorn-eps", type=float, default=0.05)
    pop.add_argument("--sinkhorn-iters", type=int, default=50)
    pop.add_argument("--top-k", type=int, default=20)
    pop.add_argument("--disable-kick", action="store_true")
    pop.add_argument("--disable-field", action="store_true")
    pop.add_argument("--flat-action", action="store_true")
    pop.add_argument("--adapter-components", default="full", choices=["full", "u", "s", "sigma"])
    pop.add_argument("--calibrate-potential", action="store_true")
    pop.add_argument("--response-decoder", default="none", choices=["none", "program", "shared_program", "shared_signed", "gene_token", "gene_mpnn", "gene_mpnn_residual", "signed_gene_token", "signed_gene_mpnn_residual"])
    pop.add_argument("--response-programs", type=int, default=32)
    pop.add_argument("--sparse-programs", action="store_true")
    pop.add_argument("--nonnegative-basis", action="store_true")
    pop.add_argument("--set-context-decoder", action="store_true")
    pop.add_argument("--program-loss-weight", type=float, default=0.0)
    pop.add_argument("--train-fraction", type=float, default=1.0)
    pop.add_argument("--gene-graph", type=Path, default=None)
    pop.add_argument("--graph-mode", default="none", choices=["none", "basis", "output", "both"])
    pop.add_argument("--graph-weight", type=float, default=0.0)
    pop.add_argument("--graph-basis-weight", type=float, default=0.0)
    pop.add_argument("--graph-output-weight", type=float, default=0.0)
    pop.add_argument("--graph-top-k", type=int, default=0)
    pop.add_argument("--no-graph-self-loop", action="store_true")
    pop.add_argument("--graph-layers", type=int, default=2)
    pop.add_argument("--rollout-steps", type=int, default=4)
    pop.add_argument("--disable-rollout", action="store_true")
    pop.add_argument("--disable-action-time", action="store_true")
    pop.add_argument("--virtual-time-min", type=float, default=0.25)
    pop.add_argument("--virtual-time-max", type=float, default=1.75)
    pop.add_argument("--eval-max-cells-per-condition", type=int, default=None)
    pop.add_argument("--condition-col", default="condition")
    pop.add_argument("--control-label", default="ctrl")
    pop.set_defaults(func=cmd_train_population_perturbation)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
