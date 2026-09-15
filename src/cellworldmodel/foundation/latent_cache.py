"""Chunked latent cache for frozen foundation VAE encoders."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cellworldmodel.foundation.expression_dataset import FoundationExpressionDataset
from cellworldmodel.foundation.vae_eval import LoadedVAE, load_vae_checkpoint


@dataclass(frozen=True)
class LatentCacheConfig:
    checkpoint: str
    catalog_dir: str
    output_dir: str
    splits: tuple[str, ...] = ("train", "val", "test")
    batch_size: int = 512
    shard_size: int = 50_000
    device: str = "cuda"
    allow_unknown_batch: bool = False


def _batch_codes(
    leaves: np.ndarray,
    leaf_to_id: dict[str, int],
    *,
    allow_unknown: bool,
) -> torch.Tensor | None:
    if not leaf_to_id:
        return None
    missing = sorted({str(x) for x in leaves if str(x) not in leaf_to_id})
    if missing and not allow_unknown:
        preview = ", ".join(missing[:5])
        raise ValueError(
            "Cannot encode cells with unseen leaf_dataset values for a batch-conditioned VAE: "
            f"{preview}. Use a no-batch/decoder-only encoder for strict heldout encoding, "
            "or pass allow_unknown_batch only for debugging."
        )
    return torch.tensor([leaf_to_id.get(str(x), 0) for x in leaves], dtype=torch.long)


class LatentCacheWriter:
    def __init__(self, config: LatentCacheConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dataset = FoundationExpressionDataset(config.catalog_dir)
        self.loaded = load_vae_checkpoint(config.checkpoint, self.device)

    def _encode_ids(self, ids: np.ndarray) -> np.ndarray:
        z_chunks = []
        for start in range(0, len(ids), int(self.config.batch_size)):
            chunk_ids = ids[start:start + int(self.config.batch_size)]
            batch = self.dataset.load_cells(chunk_ids)
            x = torch.from_numpy(batch.x).to(self.device)
            codes = None
            if bool(self.loaded.config.get("encoder_uses_batch", bool(self.loaded.leaf_to_id))):
                codes = _batch_codes(
                    batch.leaf_dataset,
                    self.loaded.leaf_to_id,
                    allow_unknown=bool(self.config.allow_unknown_batch),
                )
            if codes is not None:
                codes = codes.to(self.device)
            with torch.no_grad():
                mu, _ = self.loaded.model.encode(x, codes)
            z_chunks.append(mu.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(z_chunks, axis=0) if z_chunks else np.zeros((0, 0), dtype=np.float32)

    def write(self) -> dict:
        t0 = time.time()
        index_rows = []
        split_counts = {}
        shard_id = 0
        latent_dim = int(self.loaded.config["latent_dim"])
        for split in self.config.splits:
            ids = np.sort(self.dataset.cell_ids_for_split(split).astype(np.int64))
            split_counts[split] = int(len(ids))
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for shard_start in range(0, len(ids), int(self.config.shard_size)):
                shard_ids = ids[shard_start:shard_start + int(self.config.shard_size)]
                z = self._encode_ids(shard_ids)
                if z.shape[1] != latent_dim:
                    raise ValueError(f"Unexpected latent dim {z.shape[1]} != {latent_dim}")
                shard_name = f"shard_{shard_id:06d}.npy"
                shard_path = split_dir / shard_name
                np.save(shard_path, z)
                rows = self.dataset.cell_index_by_id.loc[shard_ids]
                for row_in_shard, global_cell_id in enumerate(shard_ids):
                    row = rows.loc[global_cell_id]
                    index_rows.append({
                        "global_cell_id": int(global_cell_id),
                        "foundation_split": str(split),
                        "leaf_dataset": str(row["leaf_dataset"]),
                        "top_dataset": str(row.get("top_dataset", "")),
                        "timepoint": float(row["timepoint"]),
                        "cell_type": str(row.get("cell_type", "")),
                        "shard_id": int(shard_id),
                        "shard_path": str(shard_path.relative_to(self.output_dir)),
                        "row_in_shard": int(row_in_shard),
                    })
                shard_id += 1
        index = pd.DataFrame(index_rows)
        index.to_parquet(self.output_dir / "index.parquet", index=False)
        manifest = {
            "config": asdict(self.config),
            "checkpoint_config": self.loaded.config,
            "latent_dim": latent_dim,
            "split_counts": split_counts,
            "n_cells": int(len(index)),
            "n_shards": int(shard_id),
            "index": str(self.output_dir / "index.parquet"),
            "elapsed_s": float(time.time() - t0),
        }
        with (self.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest


class LatentCacheDataset:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.index = pd.read_parquet(self.cache_dir / "index.parquet")
        self.index_by_id = self.index.set_index("global_cell_id", drop=False)
        with (self.cache_dir / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.latent_dim = int(self.manifest["latent_dim"])
        self._shard_cache: dict[str, np.ndarray] = {}

    def _load_shard(self, shard_path: str) -> np.ndarray:
        cached = self._shard_cache.get(shard_path)
        if cached is None:
            cached = np.load(self.cache_dir / shard_path, mmap_mode="r")
            self._shard_cache[shard_path] = cached
        return cached

    def load_ids(self, ids: np.ndarray | list[int]) -> np.ndarray:
        ids_arr = np.asarray(ids, dtype=np.int64)
        rows = self.index_by_id.loc[ids_arr].reset_index(drop=True)
        out = np.zeros((len(ids_arr), self.latent_dim), dtype=np.float32)
        for shard_path, group in rows.groupby("shard_path", sort=False):
            shard = self._load_shard(str(shard_path))
            positions = group.index.to_numpy(dtype=np.int64)
            row_idx = group["row_in_shard"].to_numpy(dtype=np.int64)
            out[positions] = np.asarray(shard[row_idx], dtype=np.float32)
        return out


def write_latent_cache(config: LatentCacheConfig) -> dict:
    return LatentCacheWriter(config).write()
