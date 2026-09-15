import torch
import numpy as np
import pandas as pd
import pytest

from cellworldmodel.foundation.vae_eval import EmbeddingEvalConfig, EmbeddingMetricSuite, compute_leiden_labels
from cellworldmodel.foundation.vae_model import Log1pGaussianVAE
from cellworldmodel.foundation.vae_registry import (
    build_foundation_vae,
    get_vae_architecture,
    vae_architecture_names,
)


def test_log1p_gaussian_vae_forward_unconditional():
    model = Log1pGaussianVAE(n_genes=7, latent_dim=3, hidden_dim=5, n_layers=1)
    x = torch.randn(4, 7)
    out = model(x)
    assert out["recon"].shape == (4, 7)
    assert out["mu"].shape == (4, 3)
    assert out["logvar"].shape == (4, 3)
    assert torch.isfinite(out["recon_loss"])
    assert torch.isfinite(out["kl"])


def test_log1p_gaussian_vae_forward_conditional():
    model = Log1pGaussianVAE(n_genes=7, latent_dim=3, hidden_dim=5, n_layers=1, n_batches=2)
    x = torch.randn(4, 7)
    batch = torch.tensor([0, 1, 0, 1])
    out = model(x, batch)
    assert out["recon"].shape == (4, 7)


def test_foundation_vae_architectures_forward():
    x = torch.rand(3, 11)
    batch = torch.tensor([0, 1, 0])
    for arch in vae_architecture_names():
        model = build_foundation_vae(arch, n_genes=11, latent_dim=5, n_batches=2)
        out = model(x, batch)
        assert out["recon"].shape == (3, 11)
        assert out["mu"].shape == (3, 5)
        assert torch.isfinite(out["recon_loss"])
        assert torch.isfinite(out["kl"])


def test_foundation_vae_registry_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown VAE architecture"):
        get_vae_architecture("missing_architecture")


def test_strict_vae_encoder_ignores_batch_and_supports_null_decode():
    model = build_foundation_vae("strict_scvi1024", n_genes=11, latent_dim=5, n_batches=3)
    model.eval()
    x = torch.rand(4, 11)
    batch0 = torch.zeros(4, dtype=torch.long)
    batch1 = torch.ones(4, dtype=torch.long)
    with torch.no_grad():
        mu0, _ = model.encode(x, batch0)
        mu1, _ = model.encode(x, batch1)
        recon_null = model.decode(mu0, None)
        recon_seen = model.decode(mu0, batch0)
    torch.testing.assert_close(mu0, mu1)
    assert recon_null.shape == (4, 11)
    assert recon_seen.shape == (4, 11)
    assert model.encoder_uses_batch is False
    assert model.supports_null_decode is True


def test_embedding_metric_suite_handles_leiden_labels():
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError:
        pytest.skip("Leiden optional dependencies are not installed")

    rng = np.random.default_rng(0)
    z = np.concatenate([
        rng.normal(loc=-1.0, scale=0.1, size=(10, 4)),
        rng.normal(loc=1.0, scale=0.1, size=(10, 4)),
    ]).astype(np.float32)
    meta = pd.DataFrame({
        "leaf_dataset": ["leaf_a"] * 10 + ["leaf_b"] * 10,
        "top_dataset": ["top"] * 20,
        "timepoint": ["1"] * 10 + ["2"] * 10,
        "cell_type": ["type_a"] * 10 + ["type_b"] * 10,
    })
    meta["leiden"] = compute_leiden_labels(z, n_neighbors=5, resolution=1.0, seed=0)

    suite = EmbeddingMetricSuite(EmbeddingEvalConfig(
        checkpoint="dummy.pt",
        catalog_dir="catalog",
        output_dir="eval",
        sample_size=20,
        knn_k=5,
    ))
    label_metrics = suite.label_metrics(z, meta)

    assert set(label_metrics["label"]) == {"leaf_dataset", "top_dataset", "timepoint", "cell_type", "leiden"}
    assert int(label_metrics.loc[label_metrics["label"] == "leiden", "n_classes"].iloc[0]) >= 1
    assert suite.basic_metrics(z)["latent_dim"] == 4
