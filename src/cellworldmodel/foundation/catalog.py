"""Build foundation catalog artifacts from reference h5ad metadata."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cellworldmodel.foundation.config import FoundationConfig
from cellworldmodel.foundation.gene_vocab import sha1_strings
from cellworldmodel.foundation.h5ad_meta import h5ad_shape, read_obs_column, read_var_names


TIME_PATTERNS = (
    re.compile(r"[dD]ay_(\d+)p(\d+)"),
    re.compile(r"[dD]ay_(\d+)\.(\d+)"),
    re.compile(r"[dD](\d+)p(\d+)"),
)


@dataclass(frozen=True)
class FoundationCatalog:
    h5ad_files: pd.DataFrame
    leaf_datasets: pd.DataFrame
    cell_index: pd.DataFrame
    manifest: dict


def parse_timepoint(filename: str) -> float | None:
    for pattern in TIME_PATTERNS:
        match = pattern.search(filename)
        if match:
            return float(f"{match.group(1)}.{match.group(2)}")
    return None


def top_dataset(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else path.name


def leaf_dataset_id(root: Path, leaf: Path) -> str:
    rel_parts = leaf.relative_to(root).parts
    dataset_name = None
    subdirs: list[str] = []
    for i, part in enumerate(rel_parts):
        if part.startswith(("GSE", "E-MTAB-", "GSM", "tome-mouse")):
            dataset_name = part
            for sub in rel_parts[i + 1:]:
                if sub not in {"h5ad_out", "all_nonzero_pairs", "nonzero_pairs_growthrate", "growth_rates"}:
                    subdirs.append(sub)
            break
    if dataset_name is None:
        dataset_name = leaf.name
    if subdirs:
        suffix = "_".join(subdirs)
        return f"{dataset_name}_{suffix[:30]}" if suffix else dataset_name
    return dataset_name


def discover_h5ad_files(data_root: str | Path) -> list[Path]:
    return sorted(Path(data_root).rglob("*.h5ad"))


def _safe_obs(handle: h5py.File, key: str, n: int) -> list[str]:
    if "obs" not in handle or key not in handle["obs"]:
        return [""] * n
    return read_obs_column(handle, key)


def inspect_h5ad_file(root: Path, path: Path, file_id: int) -> tuple[dict, pd.DataFrame]:
    leaf_dir = path.parent
    leaf = leaf_dataset_id(root, leaf_dir)
    top = top_dataset(root, leaf_dir)
    timepoint = parse_timepoint(path.name)
    with h5py.File(path, "r") as handle:
        n_obs, n_vars = h5ad_shape(handle)
        genes = read_var_names(handle)
        gene_hash = sha1_strings(genes)
        barcodes = _safe_obs(handle, "barcode", n_obs)
        cell_types = _safe_obs(handle, "cell_type", n_obs)
        treatments = _safe_obs(handle, "cell_treatment", n_obs)
        obs_time = _safe_obs(handle, "time_of_sampling", n_obs)
    file_row = {
        "file_id": file_id,
        "path": str(path),
        "top_dataset": top,
        "leaf_dataset": leaf,
        "leaf_dir": str(leaf_dir),
        "timepoint": timepoint,
        "n_obs": int(n_obs),
        "n_vars": int(n_vars),
        "file_size_bytes": int(path.stat().st_size),
        "gene_hash": gene_hash,
    }
    cell_rows = pd.DataFrame({
        "file_id": file_id,
        "local_cell_index": range(n_obs),
        "barcode": barcodes,
        "cell_type": cell_types,
        "cell_treatment": treatments,
        "time_of_sampling": obs_time,
        "timepoint": timepoint,
        "top_dataset": top,
        "leaf_dataset": leaf,
        "h5ad_path": str(path),
    })
    return file_row, cell_rows


def assign_catalog_splits(cell_index: pd.DataFrame, cfg: FoundationConfig) -> pd.DataFrame:
    out = cell_index.copy()
    out["foundation_split"] = ""
    heldout = set(cfg.splits.heldout_families)
    heldout_mask = out["top_dataset"].isin(heldout) | out["leaf_dataset"].isin(heldout)
    out.loc[heldout_mask, "foundation_split"] = "heldout"
    ratios = np.asarray(cfg.splits.split_ratios, dtype=float)
    ratios = ratios / ratios.sum()
    labels = np.asarray(["train", "val", "test"], dtype=object)
    for (file_id, timepoint), idx in out.loc[~heldout_mask].groupby(["file_id", "timepoint"]).groups.items():
        idx_arr = np.asarray(list(idx), dtype=np.int64)
        rng = np.random.default_rng(int(cfg.splits.split_seed) + int(file_id) * 1009)
        perm = idx_arr[rng.permutation(len(idx_arr))]
        n_train = int(round(ratios[0] * len(perm)))
        n_val = int(round(ratios[1] * len(perm)))
        n_train = min(n_train, len(perm))
        n_val = min(n_val, max(0, len(perm) - n_train))
        split_chunks = (
            (perm[:n_train], labels[0]),
            (perm[n_train:n_train + n_val], labels[1]),
            (perm[n_train + n_val:], labels[2]),
        )
        for chunk, label in split_chunks:
            if len(chunk):
                out.loc[chunk, "foundation_split"] = str(label)
    if (out["foundation_split"] == "").any():
        raise RuntimeError("Internal error: some cells were not assigned to a foundation split")
    return out


def build_catalog(cfg: FoundationConfig, max_files: int | None = None) -> FoundationCatalog:
    root = Path(cfg.data_root)
    h5ad_paths = discover_h5ad_files(root)
    if max_files is not None:
        h5ad_paths = h5ad_paths[:max_files]
    if not h5ad_paths:
        raise FileNotFoundError(f"No .h5ad files found under {root}")

    h5ad_rows = []
    cell_chunks = []
    for file_id, path in enumerate(h5ad_paths):
        file_row, cells = inspect_h5ad_file(root, path, file_id)
        h5ad_rows.append(file_row)
        cell_chunks.append(cells)
    h5ad_files = pd.DataFrame(h5ad_rows)
    cell_index = pd.concat(cell_chunks, ignore_index=True)
    cell_index.insert(0, "global_cell_id", range(len(cell_index)))
    cell_index = assign_catalog_splits(cell_index, cfg)
    leaf_datasets = (
        h5ad_files
        .groupby(["top_dataset", "leaf_dataset"], as_index=False)
        .agg(n_files=("path", "count"), n_cells=("n_obs", "sum"),
             min_time=("timepoint", "min"), max_time=("timepoint", "max"))
        .sort_values(["top_dataset", "leaf_dataset"])
        .reset_index(drop=True)
    )
    manifest = {
        "data_root": str(root),
        "n_h5ad_files": int(len(h5ad_files)),
        "n_cells": int(len(cell_index)),
        "n_leaf_datasets": int(len(leaf_datasets)),
        "n_genes_unique": int(h5ad_files["n_vars"].nunique()),
        "gene_hashes": sorted(h5ad_files["gene_hash"].unique().tolist()),
        "split_seed": int(cfg.splits.split_seed),
        "heldout_families": list(cfg.splits.heldout_families),
        "external": list(cfg.splits.external),
        "split_counts": {str(k): int(v) for k, v in cell_index["foundation_split"].value_counts().items()},
    }
    return FoundationCatalog(
        h5ad_files=h5ad_files,
        leaf_datasets=leaf_datasets,
        cell_index=cell_index,
        manifest=manifest,
    )


def write_catalog(catalog: FoundationCatalog, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog.h5ad_files.to_parquet(output_dir / "h5ad_files.parquet", index=False)
    catalog.leaf_datasets.to_parquet(output_dir / "leaf_datasets.parquet", index=False)
    catalog.cell_index.to_parquet(output_dir / "cell_index.parquet", index=False)
    split_manifest = {
        "split_seed": catalog.manifest["split_seed"],
        "split_counts": catalog.manifest["split_counts"],
        "heldout_families": catalog.manifest["heldout_families"],
        "external": catalog.manifest["external"],
    }
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, sort_keys=True)
    with (output_dir / "data_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(catalog.manifest, handle, indent=2, sort_keys=True)
    return catalog.manifest
