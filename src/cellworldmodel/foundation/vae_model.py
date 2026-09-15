"""Lightweight local VAE for log1p-normalized foundation expression."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class InjectedFCLayers(nn.Module):
    """scVI-inspired fully-connected layers with covariate injection.

    scVI's FCLayers can inject categorical covariates into every hidden layer.
    This local variant keeps that design while staying compatible with our
    streaming log1p-normal VAE.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        cov_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cov_dim = int(cov_dim)
        layers = []
        dim = int(input_dim)
        for _ in range(max(1, int(n_layers))):
            layers.append(nn.Sequential(
                nn.Linear(dim + self.cov_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ))
            dim = hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, cov: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        for layer in self.layers:
            if self.cov_dim:
                if cov is None:
                    raise ValueError("covariates are required for injected covariate layers")
                h = torch.cat([h, cov], dim=-1)
            h = layer(h)
        return h


class Log1pGaussianVAE(nn.Module):
    """MLP VAE with optional batch conditioning for log1p expression."""
    encoder_uses_batch = True
    supports_null_decode = False

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 3,
        n_batches: int = 0,
        batch_emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.n_batches = int(n_batches)
        self.batch_emb_dim = int(batch_emb_dim) if n_batches > 0 else 0
        self.batch_embedding = (
            nn.Embedding(n_batches, self.batch_emb_dim) if n_batches > 0 else None
        )
        enc_in = self.n_genes + self.batch_emb_dim
        dec_in = self.latent_dim + self.batch_emb_dim
        self.encoder = self._mlp(enc_in, hidden_dim, n_layers)
        self.mu = nn.Linear(hidden_dim, self.latent_dim)
        self.logvar = nn.Linear(hidden_dim, self.latent_dim)
        self.decoder = self._mlp(dec_in, hidden_dim, n_layers)
        self.out = nn.Linear(hidden_dim, self.n_genes)

    @staticmethod
    def _mlp(input_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
        layers = []
        dim = input_dim
        for _ in range(max(1, int(n_layers))):
            layers.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
            dim = hidden_dim
        return nn.Sequential(*layers)

    def _batch_emb(self, batch: torch.Tensor | None, n: int, device) -> torch.Tensor | None:
        if self.batch_embedding is None:
            return None
        if batch is None:
            raise ValueError("batch labels are required when n_batches > 0")
        return self.batch_embedding(batch.to(device=device, dtype=torch.long).view(n))

    def encode(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self._batch_emb(batch, x.shape[0], x.device)
        enc_in = torch.cat([x, emb], dim=-1) if emb is not None else x
        h = self.encoder(enc_in)
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

    def decode(self, z: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        emb = self._batch_emb(batch, z.shape[0], z.device)
        dec_in = torch.cat([z, emb], dim=-1) if emb is not None else z
        h = self.decoder(dec_in)
        return self.out(h)

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x, batch)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        recon = self.decode(z, batch)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return {"recon": recon, "mu": mu, "logvar": logvar, "recon_loss": recon_loss, "kl": kl}


class ScviStyleGaussianVAE(nn.Module):
    """Local scVI-style Gaussian VAE with covariate injection."""
    encoder_uses_batch = True
    supports_null_decode = False

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int = 1024,
        n_layers: int = 3,
        n_batches: int = 0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.n_batches = int(n_batches)
        self.cov_dim = int(n_batches) if n_batches > 0 else 0
        self.encoder = InjectedFCLayers(self.n_genes, hidden_dim, n_layers, self.cov_dim, dropout)
        self.mu = nn.Linear(hidden_dim, self.latent_dim)
        self.logvar = nn.Linear(hidden_dim, self.latent_dim)
        self.decoder = InjectedFCLayers(self.latent_dim, hidden_dim, n_layers, self.cov_dim, dropout)
        self.out_mean = nn.Linear(hidden_dim, self.n_genes)
        self.out_log_scale = nn.Parameter(torch.zeros(self.n_genes))

    def _cov(self, batch: torch.Tensor | None, n: int, device) -> torch.Tensor | None:
        if self.n_batches <= 0:
            return None
        if batch is None:
            raise ValueError("batch labels are required when n_batches > 0")
        return F.one_hot(batch.to(device=device, dtype=torch.long).view(n), num_classes=self.n_batches).float()

    def encode(self, x: torch.Tensor, batch: torch.Tensor | None = None):
        cov = self._cov(batch, x.shape[0], x.device)
        h = self.encoder(x, cov)
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

    def decode(self, z: torch.Tensor, batch: torch.Tensor | None = None):
        cov = self._cov(batch, z.shape[0], z.device)
        h = self.decoder(z, cov)
        return self.out_mean(h)

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None):
        mu, logvar = self.encode(x, batch)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decode(z, batch)
        scale = F.softplus(self.out_log_scale).view(1, -1) + 1e-4
        recon_loss = (0.5 * ((x - recon) / scale).pow(2) + torch.log(scale)).mean()
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return {"recon": recon, "mu": mu, "logvar": logvar, "recon_loss": recon_loss, "kl": kl}


class ResidualGaussianVAE(Log1pGaussianVAE):
    """Wider residual MLP VAE."""

    @staticmethod
    def _mlp(input_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            *[ResidualBlock(hidden_dim, dropout=0.05) for _ in range(max(1, int(n_layers) - 1))],
        )


class EncoderNoBatchDecoderResidualVAE(nn.Module):
    """Strict zero-shot VAE: encoder ignores batch, decoder has optional batch residual.

    This supports external/heldout encoding without arbitrary batch ids:

        z ~ q_phi(z | x)
        x_hat = D0(z) + m * R_b(z)

    where m is dropped during training with probability decoder_batch_dropout,
    and external/null decode uses m=0 by passing batch=None.
    """

    encoder_uses_batch = False
    supports_null_decode = True

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int = 1024,
        n_layers: int = 3,
        n_batches: int = 0,
        dropout: float = 0.05,
        decoder_batch_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.n_batches = int(n_batches)
        self.decoder_batch_dropout = float(decoder_batch_dropout)
        self.encoder = InjectedFCLayers(self.n_genes, hidden_dim, n_layers, cov_dim=0, dropout=dropout)
        self.mu = nn.Linear(hidden_dim, self.latent_dim)
        self.logvar = nn.Linear(hidden_dim, self.latent_dim)
        self.decoder_shared = InjectedFCLayers(self.latent_dim, hidden_dim, n_layers, cov_dim=0, dropout=dropout)
        self.out_shared = nn.Linear(hidden_dim, self.n_genes)
        self.decoder_residual = (
            InjectedFCLayers(self.latent_dim, hidden_dim, max(1, n_layers - 1), cov_dim=self.n_batches, dropout=dropout)
            if self.n_batches > 0 else None
        )
        self.out_residual = nn.Linear(hidden_dim, self.n_genes) if self.n_batches > 0 else None
        self.out_log_scale = nn.Parameter(torch.zeros(self.n_genes))
        if self.out_residual is not None:
            nn.init.zeros_(self.out_residual.weight)
            nn.init.zeros_(self.out_residual.bias)

    def _cov(self, batch: torch.Tensor | None, n: int, device) -> torch.Tensor | None:
        if self.n_batches <= 0:
            return None
        if batch is None:
            return None
        return F.one_hot(batch.to(device=device, dtype=torch.long).view(n), num_classes=self.n_batches).float()

    def encode(self, x: torch.Tensor, batch: torch.Tensor | None = None):
        del batch
        h = self.encoder(x, None)
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

    def decode(self, z: torch.Tensor, batch: torch.Tensor | None = None):
        h = self.decoder_shared(z, None)
        recon = self.out_shared(h)
        cov = self._cov(batch, z.shape[0], z.device)
        if cov is None or self.decoder_residual is None or self.out_residual is None:
            return recon
        keep = z.new_ones(z.shape[0], 1)
        if self.training and self.decoder_batch_dropout > 0.0:
            keep = torch.bernoulli(keep * (1.0 - self.decoder_batch_dropout))
        residual = self.out_residual(self.decoder_residual(z, cov))
        return recon + keep * residual

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None):
        mu, logvar = self.encode(x, batch)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decode(z, batch)
        scale = F.softplus(self.out_log_scale).view(1, -1) + 1e-4
        recon_loss = (0.5 * ((x - recon) / scale).pow(2) + torch.log(scale)).mean()
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return {"recon": recon, "mu": mu, "logvar": logvar, "recon_loss": recon_loss, "kl": kl}


class StateTokenGaussianVAE(nn.Module):
    """STATE-inspired token encoder over top expressed genes.

    This is not a full STATE reproduction. It borrows the token projection +
    Transformer pooling idea for a compact VAE encoder, while keeping our
    existing full-gene MLP decoder for reconstruction.
    """
    encoder_uses_batch = False
    supports_null_decode = True

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        n_batches: int = 0,
        top_k: int = 512,
        batch_emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.n_batches = int(n_batches)
        self.top_k = int(top_k)
        self.batch_emb_dim = int(batch_emb_dim) if n_batches > 0 else 0
        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)
        self.value_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.batch_embedding = nn.Embedding(n_batches, self.batch_emb_dim) if n_batches > 0 else None
        dec_in = latent_dim + self.batch_emb_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            ResidualBlock(512, dropout=0.05),
            ResidualBlock(512, dropout=0.05),
            nn.Linear(512, n_genes),
        )

    def _batch_emb(self, batch: torch.Tensor | None, n: int, device):
        if self.batch_embedding is None:
            return None
        if batch is None:
            raise ValueError("batch labels are required when n_batches > 0")
        return self.batch_embedding(batch.to(device=device, dtype=torch.long).view(n))

    def encode(self, x: torch.Tensor, batch: torch.Tensor | None = None):
        values, gene_ids = torch.topk(x, k=min(self.top_k, x.shape[1]), dim=1)
        tokens = self.gene_embedding(gene_ids) + self.value_proj(values.unsqueeze(-1))
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        h = self.encoder(torch.cat([cls, tokens], dim=1))[:, 0]
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

    def decode(self, z: torch.Tensor, batch: torch.Tensor | None = None):
        emb = self._batch_emb(batch, z.shape[0], z.device)
        dec_in = torch.cat([z, emb], dim=-1) if emb is not None else z
        return self.decoder(dec_in)

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None):
        mu, logvar = self.encode(x, batch)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decode(z, batch)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return {"recon": recon, "mu": mu, "logvar": logvar, "recon_loss": recon_loss, "kl": kl}
