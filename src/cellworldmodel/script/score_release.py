"""Evaluate released temporal models using the manuscript's frozen scoring counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zlib

import numpy as np
import pandas as pd
import torch

from cellworldmodel.benchmark.common_metrics import compute_branchsbm_style_metrics
from cellworldmodel.foundation.released import load_released_model, read_release_manifest, sha256

def target_pool(meta: pd.DataFrame, t: float, max_cells: int) -> np.ndarray:
    """Fixed test cells for one target time, independent of arm and seed."""
    times = meta["time"].astype(float).to_numpy()
    splits = meta["split"].astype(str).to_numpy()
    ids = np.where(np.isclose(times, t) & (splits == "test"))[0]
    if len(ids) > max_cells:
        rng = np.random.default_rng(zlib.crc32(f"table1-target|{t}".encode()))
        ids = np.sort(rng.choice(ids, max_cells, replace=False))
    return ids

def subsample(x: np.ndarray, n: int, tag: str) -> np.ndarray:
    if x.shape[0] <= n:
        return x
    rng = np.random.default_rng(zlib.crc32(f"table1-cut|{tag}".encode()))
    return x[np.sort(rng.choice(x.shape[0], n, replace=False))]

@torch.no_grad()
def predict_tensor(model, source: torch.Tensor, delta: float, k: int,
                   generator: torch.Generator) -> torch.Tensor:
    """One complete resident-source query, including its required noise generation."""
    d = torch.full((source.shape[0],), float(delta), device=source.device, dtype=source.dtype)
    eps = torch.randn(source.shape[0], k, source.shape[1], device=source.device, generator=generator)
    return model(source, d, eps).reshape(-1, source.shape[1])

@torch.no_grad()
def predict(model, source: np.ndarray, delta: float, k: int, device, seed: int,
            batch: int = 512) -> np.ndarray:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    out = []
    for i in range(0, len(source), batch):
        zb = torch.from_numpy(source[i:i + batch]).to(device)
        out.append(predict_tensor(model, zb, delta, k, gen).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def score_release(directory, protocol_path, dataset, seed, *, device="cpu", check_reference=False,
                  mode="replay"):
    root = Path(directory)
    release = read_release_manifest(root)
    protocol = json.loads(Path(protocol_path).read_text())[dataset]
    if seed not in (0, 1, 2):
        raise ValueError("the published protocol has seeds 0, 1 and 2")
    data_root = root / "data" / protocol["data_directory"]
    for name in ("representations.npz", "metadata.tsv", "metric_standardization.npz", "split_manifest.json"):
        relative = (data_root / name).relative_to(root).as_posix()
        if sha256(data_root / name) != release["files"][relative]["sha256"]:
            raise ValueError(f"dataset checksum mismatch: {relative}")
    z = np.load(data_root / "representations.npz", allow_pickle=False)["scvi128"].astype(np.float32)
    meta = pd.read_csv(data_root / "metadata.tsv", sep="\t")
    if len(meta) != len(z):
        raise ValueError("representation and metadata row counts differ")
    times = sorted(float(t) for t in meta["time"].unique())
    if times[1:] != [row["target_time"] for row in protocol["seeds"][str(seed)]]:
        raise ValueError("requested target times differ from the frozen protocol")
    train = meta["split"].eq("train").to_numpy()
    mean = z[train].mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(z[train].std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    stats = np.load(data_root / "metric_standardization.npz", allow_pickle=False)
    np.testing.assert_array_equal(mean.reshape(-1), stats["mean"].reshape(-1))
    np.testing.assert_array_equal(std.reshape(-1), stats["std"].reshape(-1))
    source = z[np.isclose(meta["time"], times[0]) & meta["split"].eq("test").to_numpy()]
    if mode not in {"replay", "predict"}:
        raise ValueError("mode must be replay or predict")
    model, archived = None, None
    if mode == "predict":
        model = load_released_model(root, f"{dataset}_seed{seed}", device=device)
    else:
        reference = data_root / "reference" / f"seed{seed}.npz"
        entry = release["files"][reference.relative_to(root).as_posix()]
        if sha256(reference) != entry["sha256"]:
            raise ValueError("reference prediction archive checksum mismatch")
        with np.load(reference, allow_pickle=False) as data:
            archived = {key: data[key] for key in data.files}
    rows = []
    for expected in protocol["seeds"][str(seed)]:
        t, n = float(expected["target_time"]), int(expected["n_scored"])
        if model is not None:
            raw = predict(model, source, t-times[0], 8, torch.device(device), seed)
            cloud = ((raw-mean)/std).astype(np.float32)
        else:
            cloud = archived[str(t)]
        target = ((z[target_pool(meta, t, 20000)]-mean)/std).astype(np.float32)
        if min(len(cloud), len(target)) < n:
            raise ValueError("a cloud is smaller than the accepted comparison support")
        pred_cut = subsample(cloud, n, f"temporal|{seed}|{t}")
        target_cut = subsample(target, n, f"target|{t}")
        metrics = compute_branchsbm_style_metrics(torch.from_numpy(pred_cut), torch.from_numpy(target_cut),
                                                 seed=seed, skip_w1=True)
        target_variance = float(target_cut.var(axis=0).sum())
        row = {"seed": seed, "target_time": t, "n_scored": n,
               "w2": float(metrics["branchsbm_w2_full_mean"]),
               "mmd": float(metrics["branchsbm_mmd_full_mean"]),
               "variance_ratio": float(pred_cut.var(axis=0).sum()/target_variance)}
        row["reference_w2"] = expected["w2"]
        row["absolute_w2_difference"] = abs(row["w2"]-expected["w2"])
        if check_reference and row["absolute_w2_difference"] > protocol["reference_w2_atol"]:
            raise ValueError(f"reference W2 differs beyond tolerance at {t}: {row}")
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"dataset": dataset, "seed": seed, "rows": rows, "mode": mode,
            "inference_executed": mode == "predict",
            "mean_w2": float(np.mean([r["w2"] for r in rows])),
            "release_manifest_sha256": sha256(root/"release.json"),
            "protocol_sha256": sha256(protocol_path), "torch": torch.__version__, "device": str(device),
            "inference_device_name": (torch.cuda.get_device_name(torch.device(device))
                                      if mode == "predict" and torch.device(device).type == "cuda" else None),
            "scoring_device": "cpu"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset", required=True, choices=("weinreb", "veres", "zesta"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("replay", "predict"), default="replay",
                        help="replay scores archived model outputs; predict runs fresh checkpoint inference")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-reference", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output}")
    result = score_release(args.directory, args.protocol, args.dataset, args.seed,
                           device=args.device, check_reference=args.check_reference, mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
