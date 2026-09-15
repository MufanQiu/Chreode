"""Native gene, position and mass transitions and their joint objective."""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from geomloss import SamplesLoss
import torch
from torch import nn
from cellworldmodel.benchmark.experiment_registry import EXPERIMENTS
from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.model.niche_conditioning import NichePotentialCorrection
from cellworldmodel.model.spatial_state_heads import SpatialStateHeads
from cellworldmodel.model.waddington_dit_1d import WaddingtonDiT1D
from cellworldmodel.training.split_policy import build_timepoint_splits

SOURCE_TIME, TARGET_TIME = 0., 2.5
DELTA = TARGET_TIME - SOURCE_TIME
EXPERIMENT = "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw"
LOSS_CONFIG = {"loss": "sinkhorn", "p": 2, "blur": .1, "reach": 1.,
               "scaling": .9, "debias": True, "backend": "tensorized"}
GENE_WEIGHT, MAX_ABS_LOG_MASS = .5, 12.
SPLITS = ("train", "val", "test")

def finite(name, value):
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} is nonfinite")

def source_neighborhood(gene, xy, row_ids, neighbors=16):
    """Stable distance/row-ID order, excluding self, inside one source split."""
    if neighbors < 1 or len(row_ids) < 2:
        raise ValueError("source neighborhood needs at least two rows and positive k")
    row_ids = np.sort(np.asarray(row_ids, dtype=np.int64))
    k, chosen = min(neighbors, len(row_ids) - 1), []
    for start in range(0, len(row_ids), 256):
        query = row_ids[start:start + 256]
        distances = cdist(xy[query].astype(np.float64), xy[row_ids].astype(np.float64))
        distances[np.arange(len(query)), np.arange(start, start + len(query))] = np.inf
        chosen.append(row_ids[np.argsort(distances, axis=1, kind="stable")[:, :k]])
    neighbor_ids = np.concatenate(chosen)
    return neighbor_ids, gene[neighbor_ids].mean(axis=1, dtype=np.float32)

class RectangleData:
    def __init__(self, root: Path, coordinate_view: str):
        if coordinate_view != "simulation_truth":
            raise ValueError("this pilot requires simulation_truth known alignment")
        self.manifest = json.loads((root / "split_manifest.json").read_text())
        if (self.manifest.get("dataset") != "stvcr_rectangle_native_joint"
                or not self.manifest.get("representation_is_native_space")
                or self.manifest.get("gene_dim") != 3 or self.manifest.get("position_dim") != 2
                or self.manifest.get("encoder") is not None):
            raise ValueError("not the explicit native Rectangle export")
        with np.load(root / "representations.npz", allow_pickle=False) as archive:
            self.gene = archive["native_gene"]
            self.xy = archive["spatial_" + coordinate_view]
            rows = archive["row_ids"]
        self.meta = pd.read_csv(root / "metadata.tsv", sep="\t", dtype={"cell_id": str})
        n = self.manifest["n_cells"]
        if n != 9858 or self.gene.shape != (n, 3) or self.xy.shape != (n, 2):
            raise ValueError("full 9858-row native dimensions required")
        if self.gene.dtype != np.float32 or self.xy.dtype != np.float32:
            raise ValueError("export dtype must remain float32")
        if (len(self.meta) != n or not self.meta.cell_id.is_unique
                or not np.array_equal(rows, np.arange(n))
                or not np.array_equal(self.meta.row_id.to_numpy(), rows)):
            raise ValueError("row identity/order mismatch")
        if (not np.isfinite(self.gene).all() or not np.isfinite(self.xy).all()
                or not np.isfinite(self.meta.time).all()
                or set(self.meta.split.unique()) != set(SPLITS)):
            raise ValueError("invalid values or splits")
        times = self.meta.time.to_numpy()
        if sorted(np.unique(times).tolist()) != [0., .5, 1., 1.5, 2., 2.5]:
            raise ValueError("time support differs")
        rebuilt = build_timepoint_splits({t: self.gene[times == t] for t in np.unique(times)}, 42, (.7, .1, .2))
        for t in np.unique(times):
            pool = np.flatnonzero(times == t)
            for split in SPLITS:
                expected = np.sort(pool[rebuilt[t].get(split)])
                actual = np.flatnonzero((times == t) & (self.meta.split.to_numpy() == split))
                if not np.array_equal(expected, actual):
                    raise ValueError("export does not match the frozen split policy")
        self.source_ids = {s: np.flatnonzero((times == SOURCE_TIME) & (self.meta.split == s)) for s in SPLITS}
        self.target_ids = {s: np.flatnonzero((times == TARGET_TIME) & (self.meta.split == s)) for s in SPLITS}
        if ([len(self.source_ids[s]) for s in SPLITS] != [1050, 150, 300]
                or [len(self.target_ids[s]) for s in SPLITS] != [1420, 203, 405]):
            raise ValueError("transition split counts differ")
        with np.load(root / "metric_standardization.npz", allow_pickle=False) as archive:
            self.gene_mean = archive["gene_mean"].copy()
            self.gene_std = archive["gene_std"].copy()
            self.xy_center = archive[coordinate_view + "_center"].copy()
            self.xy_scale = float(archive[coordinate_view + "_isotropic_scale"])
        training = self.meta.split.to_numpy() == "train"
        expected_mean = self.gene[training].mean(axis=0).astype(np.float32)
        expected_std = np.maximum(self.gene[training].std(axis=0), 1e-6).astype(np.float32)
        train_xy = self.xy[training].astype(np.float64)
        expected_center = train_xy.mean(axis=0)
        expected_scale = max(float(np.sqrt(np.mean((train_xy - expected_center) ** 2))), 1e-6)
        if (not np.array_equal(expected_mean, self.gene_mean)
                or not np.array_equal(expected_std, self.gene_std)
                or not np.array_equal(expected_center, self.xy_center)
                or expected_scale != self.xy_scale):
            raise ValueError("statistics disagree with training-only export values")
        self.context = np.zeros((n, 3), dtype=np.float32)
        self.neighbor_ids = np.full((n, 16), -1, dtype=np.int64)
        for split in SPLITS:
            ids = self.source_ids[split]
            neighbors, context = source_neighborhood(self.gene, self.xy, ids, 16)
            self.neighbor_ids[ids], self.context[ids] = neighbors, context

    def ratio(self, split):
        return len(self.target_ids[split]) / len(self.source_ids[split])

    def batch(self, ids, device):
        ids = np.asarray(ids, dtype=np.int64)
        neighbor_ids = self.neighbor_ids[ids]
        if (neighbor_ids < 0).any():
            raise ValueError("a requested row is not a source cell")
        values = {"z": self.gene[ids], "x": self.xy[ids], "neighbor_z": self.gene[neighbor_ids],
                  "relative_x": self.xy[neighbor_ids] - self.xy[ids, None], "context": self.context[ids]}
        return {key: torch.from_numpy(value).to(device) for key, value in values.items()}

def architecture(size):
    cfg = EXPERIMENTS[EXPERIMENT].model.to_cfg()
    cfg.update({"dit_size": "small", "hidden_dim": 384, "n_layers": 12, "time_emb_dim": 64,
                "wdit_time_delta_scale": DELTA, "wdit_curl_time_delta_scale": DELTA})
    if size == "small":
        return cfg
    if size != "micro":
        raise ValueError("unknown model size")
    return {**cfg, "dit_size": "micro_engineering_only", "hidden_dim": 64,
            "n_layers": 2, "curl_rank": 2, "num_heads": 4, "num_register_tokens": 4}

class RectangleTransition(nn.Module):
    def __init__(self, *, model_size, xy_scale, seed, enabled=True, include_heads=True):
        super().__init__()
        torch.manual_seed(seed)
        self.model_config = architecture(model_size)
        cfg = self.model_config
        if model_size == "small":
            self.base = build_model("m10", 3, cfg, tau_init=DELTA / math.log(2))
        else:
            self.base = WaddingtonDiT1D(dim=3, hidden_dim=64, depth=2, num_heads=4,
                num_register_tokens=4, time_emb_dim=64, curl_rank=2, tau_init=DELTA / math.log(2),
                time_embedding_mode=cfg["wdit_time_embedding"], time_delta_scale=DELTA,
                curl_time_mode=cfg["wdit_curl_time_mode"],
                curl_time_embedding_mode=cfg["wdit_curl_time_embedding"], curl_time_delta_scale=DELTA)
        self.niche = NichePotentialCorrection(3, 3, delta_scale=DELTA, programs=8, hidden_dim=64)
        self.heads = (SpatialStateHeads(3, 3, delta_scale=DELTA, distance_scale=xy_scale,
                                       hidden_dim=64, init_seed=seed + 10, enabled=enabled)
                      if include_heads else None)
        if self.heads is not None and not enabled:
            self.heads.requires_grad_(False)

    def gene_forward(self, z, delta, epsilon, context):
        gene = self.base(z, delta, epsilon)
        return gene + self.base.alpha_gate(delta)[:, None, None] * self.niche(z, delta, context)[:, None]

    def forward(self, z, delta, epsilon, context, x=None, neighbor_z=None, relative_x=None, log_mass=None):
        gene = self.gene_forward(z, delta, epsilon, context)
        if self.heads is not None:
            x, log_mass = self.heads(z, delta, self.base.alpha_gate(delta), x=x, neighbor_z=neighbor_z,
                                    relative_x=relative_x, log_mass=log_mass, context=context)
        return gene, x, log_mass

class JointObjective:
    def __init__(self, data, transport_blur=None):
        self.data = data
        self.sinkhorn = SamplesLoss(**{**LOSS_CONFIG, **({"blur": transport_blur} if transport_blur is not None else {})})

    def standardized(self, gene, xy):
        g = (gene - gene.new_tensor(self.data.gene_mean)) / gene.new_tensor(self.data.gene_std)
        x = (xy - xy.new_tensor(self.data.xy_center)) / self.data.xy_scale
        return g, x

    def measure(self, gene, xy, log_mass):
        if gene.ndim != 3 or gene.shape[-1] != 3 or xy.shape != (len(gene), 2) or log_mass.shape != (len(gene),):
            raise ValueError("prediction shapes must be (B,K,3), (B,2), (B,)")
        if gene.shape[0] < 1 or gene.shape[1] < 1:
            raise ValueError("empty prediction measure")
        for name, value in (("gene", gene), ("position", xy), ("log_mass", log_mass)):
            finite(name, value)
        if torch.any(log_mass.abs() > MAX_ABS_LOG_MASS):
            raise ValueError("log_mass exceeds the fixed exp safety bound; no clipping is applied")
        particle_mass = log_mass.exp()
        mass_ratio = particle_mass.mean()
        # K descendants divide, rather than multiply, each source particle's mass.
        weights = (particle_mass[:, None].expand(-1, gene.shape[1]) / (gene.shape[0] * gene.shape[1])).reshape(-1)
        g, x = self.standardized(gene.reshape(-1, 3), xy[:, None].expand(-1, gene.shape[1], -1).reshape(-1, 2))
        # GeomLoss p=2 uses ||a-b||^2/2; sqrt(2*kappa) yields the declared fused cost.
        features = torch.cat((math.sqrt(2 * GENE_WEIGHT) * g, math.sqrt(2 * (1 - GENE_WEIGHT)) * x), dim=1)
        finite("prediction weights", weights)
        finite("joint features", features)
        return weights, features, g, x, mass_ratio

    def __call__(self, gene, xy, log_mass, target_gene, target_xy, target_ratio):
        if not math.isfinite(target_ratio) or target_ratio <= 0 or len(target_gene) < 1:
            raise ValueError("invalid target population ratio")
        weights, features, _, _, mass_ratio = self.measure(gene, xy, log_mass)
        target_g, target_x = self.standardized(target_gene, target_xy)
        finite("target gene", target_g)
        finite("target position", target_x)
        targets = torch.cat((math.sqrt(2 * GENE_WEIGHT) * target_g,
                             math.sqrt(2 * (1 - GENE_WEIGHT)) * target_x), dim=1)
        target_weights = torch.full_like(target_gene[:, 0], target_ratio / len(target_gene))
        sinkhorn = self.sinkhorn(weights, features, target_weights, targets)
        mass_loss = (mass_ratio / target_ratio - 1).square()
        loss = sinkhorn + mass_loss
        finite("joint objective", loss)
        return loss, {"joint_objective": float(loss.detach()), "unbalanced_sinkhorn_divergence": float(sinkhorn.detach()),
                      "relative_total_mass_error_squared": float(mass_loss.detach()),
                      "predicted_mass_ratio": float(mass_ratio.detach()), "target_mass_ratio": float(target_ratio)}
