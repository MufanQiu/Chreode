"""Gene-space utilities for foundation/GEARS perturbation evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def gene_key(value: str) -> str:
    return str(value).upper()


def foundation_gene_names_from_vocab(path: str | Path) -> list[str]:
    """Return one output gene symbol per foundation vocabulary row.

    Human symbol is preferred because GEARS Norman is human.  Mouse/canonical
    names are used only as fallback.
    """
    df = pd.read_parquet(path)
    if "human_symbol" in df:
        fallback = df["canonical_gene"] if "canonical_gene" in df else pd.Series([""] * len(df))
        return [
            str(human) if str(human) and str(human).upper() != "NAN" else str(canonical)
            for human, canonical in zip(df["human_symbol"].to_numpy(), fallback.to_numpy())
        ]
    if "gene_name" in df:
        return [str(x) for x in df["gene_name"].to_numpy()]
    if "symbol" in df:
        return [str(x) for x in df["symbol"].to_numpy()]
    if "canonical_gene" in df:
        return [str(x) for x in df["canonical_gene"].to_numpy()]
    raise ValueError(f"No recognized gene column found in {path}")


def load_gene_names(path: str | Path) -> list[str]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return foundation_gene_names_from_vocab(path)
    if suffix in {".npy", ".npz"}:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            for key in ("gene_names", "genes", "ours_gene_names"):
                if key in arr:
                    return [str(x) for x in arr[key]]
            raise ValueError(f"No gene-name array found in {path}")
        return [str(x) for x in arr]
    if suffix == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            for key in ("gene_names", "genes", "ours_gene_names"):
                if key in obj:
                    return [str(x) for x in obj[key]]
            raise ValueError(f"No gene-name list found in {path}")
        return [str(x) for x in obj]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def adata_gene_names(adata) -> list[str]:
    if "gene_name" in adata.var:
        return [str(x) for x in adata.var["gene_name"].to_numpy()]
    return [str(x) for x in adata.var_names.to_numpy()]


def build_gene_to_id(gene_vocab: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, row in gene_vocab.reset_index(drop=True).iterrows():
        for key in ("human_symbol", "canonical_gene", "gene_name", "symbol"):
            value = str(row.get(key, "")).upper()
            if value and value != "NAN":
                out.setdefault(value, int(idx))
    return out


def build_source_to_vocab(source_genes: Iterable[str], output_genes: Iterable[str]) -> np.ndarray:
    vocab_map = {
        gene_key(g): i
        for i, g in enumerate(output_genes)
        if str(g) and str(g).upper() != "NAN"
    }
    source = [str(g) for g in source_genes]
    mapping = np.full(len(source), -1, dtype=np.int32)
    for i, gene in enumerate(source):
        mapping[i] = vocab_map.get(gene_key(gene), -1)
    return mapping


def shared_gene_indexes(
    reference_genes: Iterable[str],
    prediction_genes: Iterable[str],
    allowed_genes: Iterable[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return shared gene names and indexes into prediction/reference arrays."""
    pred_map = {gene_key(g): i for i, g in enumerate(prediction_genes)}
    allowed = {gene_key(g) for g in allowed_genes if str(g) and str(g).upper() != "NAN"}
    shared_names: list[str] = []
    pred_idx: list[int] = []
    ref_idx: list[int] = []
    seen: set[str] = set()
    for i, gene in enumerate(reference_genes):
        key = gene_key(gene)
        if key in seen or key not in allowed or key not in pred_map:
            continue
        seen.add(key)
        shared_names.append(str(gene))
        pred_idx.append(pred_map[key])
        ref_idx.append(i)
    if not shared_names:
        raise ValueError("Shared gene vocabulary is empty")
    return shared_names, np.asarray(pred_idx, dtype=np.int64), np.asarray(ref_idx, dtype=np.int64)
