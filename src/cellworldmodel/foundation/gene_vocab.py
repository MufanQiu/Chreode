"""Canonical ortholog gene vocabulary construction."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import pandas as pd

from cellworldmodel.foundation.config import FoundationConfig
from cellworldmodel.foundation.h5ad_meta import read_var_names


@dataclass(frozen=True)
class GeneVocabResult:
    table: pd.DataFrame
    sha1: str
    n_duplicate_unified_genes_skipped: int = 0


def sha1_strings(values: list[str]) -> str:
    return hashlib.sha1("\0".join(values).encode("utf-8")).hexdigest()


def first_h5ad(data_root: str | Path) -> Path:
    paths = sorted(Path(data_root).rglob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No .h5ad files found under {data_root}")
    return paths[0]


def read_unified_gene_order(data_root: str | Path) -> list[str]:
    path = first_h5ad(data_root)
    with h5py.File(path, "r") as handle:
        return read_var_names(handle)


def build_gene_vocab(cfg: FoundationConfig) -> GeneVocabResult:
    ortholog_path = Path(cfg.gene_vocab.source)
    if not ortholog_path.exists():
        raise FileNotFoundError(f"Ortholog table not found: {ortholog_path}")
    ortholog = pd.read_parquet(ortholog_path)
    required = {"mouse_symbol", "human_symbol"}
    missing = required - set(ortholog.columns)
    if missing:
        raise ValueError(f"Ortholog table is missing required columns: {sorted(missing)}")
    ortholog = ortholog.dropna(subset=["mouse_symbol", "human_symbol"]).copy()
    ortholog["mouse_symbol"] = ortholog["mouse_symbol"].astype(str)
    ortholog["human_symbol"] = ortholog["human_symbol"].astype(str)
    ortholog = ortholog.drop_duplicates(subset=["mouse_symbol"], keep="first")

    unified_order = read_unified_gene_order(cfg.data_root)
    by_mouse = ortholog.set_index("mouse_symbol", drop=False)
    rows = []
    seen: set[str] = set()
    duplicate_skipped = 0
    for gene in unified_order:
        if gene in seen:
            duplicate_skipped += 1
            continue
        seen.add(gene)
        if gene not in by_mouse.index:
            continue
        row = by_mouse.loc[gene]
        rows.append({
            "canonical_index": len(rows),
            "canonical_gene": gene,
            "mouse_symbol": row["mouse_symbol"],
            "human_symbol": row["human_symbol"],
            "mouse_ensembl": row.get("mouse_ensembl"),
            "human_ensembl": row.get("human_ensembl"),
            "orthology_type": row.get("orthology_type"),
            "source": row.get("source"),
        })
    if not rows:
        raise ValueError("No unified genes overlap the ortholog table")
    table = pd.DataFrame(rows)
    digest = sha1_strings(table["canonical_gene"].astype(str).tolist())
    return GeneVocabResult(
        table=table,
        sha1=digest,
        n_duplicate_unified_genes_skipped=duplicate_skipped,
    )


def write_gene_vocab(result: GeneVocabResult, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "gene_vocab.parquet"
    manifest_path = output_dir / "gene_vocab_manifest.json"
    result.table.to_parquet(table_path, index=False)
    manifest = {
        "gene_vocab_path": str(table_path),
        "n_genes": int(len(result.table)),
        "sha1": result.sha1,
        "n_duplicate_unified_genes_skipped": int(result.n_duplicate_unified_genes_skipped),
        "first5": result.table["canonical_gene"].head(5).tolist(),
        "last5": result.table["canonical_gene"].tail(5).tolist(),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest
