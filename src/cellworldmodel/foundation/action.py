"""Action encoders for future perturbation fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    action_dim: int

    def forward(self, action_ids: torch.Tensor | None) -> torch.Tensor | None:
        raise NotImplementedError


class NullActionEncoder(ActionEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.action_dim = 0

    def forward(self, action_ids: torch.Tensor | None) -> torch.Tensor | None:
        del action_ids
        return None


class CategoricalPerturbationEncoder(ActionEncoder):
    def __init__(self, n_actions: int, action_dim: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.embedding = nn.Embedding(int(n_actions), int(action_dim))

    def forward(self, action_ids: torch.Tensor | None) -> torch.Tensor:
        if action_ids is None:
            raise ValueError("action_ids are required for CategoricalPerturbationEncoder")
        return self.embedding(action_ids.long())


class GeneSetPerturbationEncoder(ActionEncoder):
    """Permutation-invariant gene-set perturbation encoder.

    The encoder is condition-id-free: single and double perturbations are
    represented by perturbed gene ids plus sign/modality/strength metadata.
    """

    def __init__(
        self,
        n_genes: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_modalities: int = 4,
        gene_prior_dim: int = 0,
        gene_priors: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.unknown_gene_id = self.n_genes
        self.action_dim = int(action_dim)
        self.gene_residual = nn.Embedding(self.n_genes + 1, hidden_dim)
        nn.init.normal_(self.gene_residual.weight, std=0.02)
        self.modality_embedding = nn.Embedding(int(n_modalities), hidden_dim)
        self.strength_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        if gene_priors is not None:
            if gene_priors.shape[0] != self.n_genes + 1:
                raise ValueError("gene_priors must include n_genes + unknown rows")
            gene_prior_dim = int(gene_priors.shape[1])
            self.register_buffer("gene_priors", gene_priors.float(), persistent=True)
        else:
            self.register_buffer("gene_priors", torch.zeros(self.n_genes + 1, int(gene_prior_dim)), persistent=True)
        self.prior_proj = nn.Linear(int(gene_prior_dim), hidden_dim, bias=False) if int(gene_prior_dim) > 0 else None
        self.item_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cardinality_embedding = nn.Embedding(8, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

    def forward(
        self,
        gene_ids: torch.Tensor,
        signs: torch.Tensor,
        modality_ids: torch.Tensor,
        strengths: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        gene_ids = gene_ids.long().clamp(min=0, max=self.unknown_gene_id)
        mask_f = mask.to(dtype=torch.float32)
        base = self.gene_residual(gene_ids)
        if self.prior_proj is not None:
            base = base + self.prior_proj(self.gene_priors[gene_ids])
        item = (
            signs.to(dtype=base.dtype).unsqueeze(-1) * base
            + self.modality_embedding(modality_ids.long().clamp_min(0))
            + self.strength_proj(strengths.to(dtype=base.dtype).unsqueeze(-1))
        )
        item = self.item_mlp(item) * mask_f.unsqueeze(-1)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0).sqrt()
        h_set = item.sum(dim=1) / denom

        pair_terms = []
        max_items = item.shape[1]
        for i in range(max_items):
            for j in range(i + 1, max_items):
                pair_mask = (mask_f[:, i] * mask_f[:, j]).unsqueeze(-1)
                pair_in = torch.cat([
                    item[:, i] + item[:, j],
                    item[:, i] * item[:, j],
                    torch.abs(item[:, i] - item[:, j]),
                ], dim=-1)
                pair_terms.append(self.pair_mlp(pair_in) * pair_mask)
        if pair_terms:
            h_pair_raw = torch.stack(pair_terms, dim=1)
            pair_count = torch.stack([
                mask_f[:, i] * mask_f[:, j]
                for i in range(max_items)
                for j in range(i + 1, max_items)
            ], dim=1).sum(dim=1, keepdim=True).clamp_min(1.0).sqrt()
            h_pair = h_pair_raw.sum(dim=1) / pair_count
        else:
            h_pair = torch.zeros_like(h_set)
        card = mask_f.sum(dim=1).long().clamp(min=0, max=7)
        h_card = self.cardinality_embedding(card)
        return self.out(torch.cat([h_set, h_pair, h_card], dim=-1))


@dataclass(frozen=True)
class ActionEncoderSpec:
    name: str
    description: str


ACTION_ENCODERS = {
    "none": ActionEncoderSpec("none", "No action conditioning."),
    "categorical_perturbation": ActionEncoderSpec(
        "categorical_perturbation",
        "Debug-only learned condition-level perturbation embedding.",
    ),
    "geneset_deepset_v1": ActionEncoderSpec(
        "geneset_deepset_v1",
        "Gene-set perturbation encoder with sign/modality tokens and pair interactions.",
    ),
}
