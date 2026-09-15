import hashlib
import json

import pytest
import torch
from torch import nn

from cellworldmodel.foundation.released import ReleasedIntervention, load_released_model
from cellworldmodel.model.intervention import FieldSurgeryAdapter, SurgeryModel
from cellworldmodel.script.download_release import download_release

class Base(nn.Module):
    def alpha_gate(self, delta):
        return torch.ones_like(delta)

    def forward(self, z, delta, eps):
        return z[:, None, :].expand_as(eps)


def test_noaction_uses_shared_slot_even_when_requested_conditions_differ():
    torch.manual_seed(4)
    adapter = FieldSurgeryAdapter(2, 2, k_programs=2, emb_dim=3, basis_hidden=4, kick_hidden=4)
    nn.init.normal_(adapter.kick.net[-1].weight)
    raw = SurgeryModel(Base(), adapter, tune_base=False).eval()
    released = ReleasedIntervention(Base(), adapter, tune_base=False).eval()
    released.condition_to_id = {"a": 0, "b": 1}
    released.shared_condition = True
    z, delta, eps = torch.randn(2, 2), torch.ones(2), torch.zeros(2, 3, 2)
    shared = raw(z, delta, eps, torch.full((2,), 2))
    assert torch.equal(released(z, delta, eps, torch.tensor([0, 1])), shared)
    assert torch.equal(released(z, delta, eps, torch.tensor([1, 0])), shared)
    assert not torch.equal(raw(z, delta, eps, torch.zeros(2, dtype=torch.long)), shared)
    with pytest.raises(ValueError, match="outside"):
        released(z, delta, eps, 3)


def test_loading_rejects_checksum_and_directory_escape_before_construction(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"weight": torch.ones(2)}, path)
    entry = {"weights": path.name, "sha256": "0" * 64, "config": {}}
    manifest = {"schema_version": 1, "models": {"test": entry}}
    (tmp_path / "release.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        load_released_model(tmp_path, "test")
    entry["weights"] = "../weights.pt"
    (tmp_path / "release.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leaves"):
        load_released_model(tmp_path, "test")


def test_downloader_pins_revision_verifies_bytes_and_preserves_existing_files(tmp_path, monkeypatch):
    import cellworldmodel.script.download_release as module

    content = b"released tensor fixture"
    cached = tmp_path / "cached.pt"
    cached.write_bytes(content)
    manifest = {
        "schema_version": 1,
        "models": {"test": {"weights": "models/test.pt"}},
        "files": {"models/test.pt": {"sha256": hashlib.sha256(content).hexdigest(),
                                    "size_bytes": len(content), "repository": "weights"}},
        "repositories": {"weights": {"repo_id": "example/release", "repo_type": "model",
                                      "revision": "a" * 40}},
    }
    calls = []
    def fetch(**kwargs):
        calls.append(kwargs)
        return str(cached)
    monkeypatch.setattr(module, "hf_hub_download", fetch)
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest))
    output = tmp_path / "download"
    download_release(source, output, models=["test"])
    assert calls[0]["revision"] == "a" * 40
    assert (output / "models/test.pt").read_bytes() == content
    assert (output / "release.json").read_bytes() == source.read_bytes()
    (output / "models/test.pt").write_bytes(b"user changed this file")
    with pytest.raises(ValueError, match="existing file"):
        download_release(source, output, models=["test"])
    assert (output / "models/test.pt").read_bytes() == b"user changed this file"
    manifest["repositories"]["weights"]["revision"] = "main"
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="immutable"):
        download_release(source, tmp_path / "mutable")


def test_downloader_rejects_corrupt_cache(tmp_path, monkeypatch):
    import cellworldmodel.script.download_release as module
    cached = tmp_path / "bad"
    cached.write_bytes(b"bad")
    monkeypatch.setattr(module, "hf_hub_download", lambda **_: str(cached))
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps({
        "schema_version": 1, "models": {},
        "files": {"data/example/values.npz": {"sha256": "0" * 64, "size_bytes": 3, "repository": "data"}},
        "repositories": {"data": {"repo_id": "example/data", "repo_type": "dataset", "revision": "b"*40}},
    }))
    with pytest.raises(ValueError, match="download checksum"):
        download_release(source, tmp_path / "output")
    assert not (tmp_path / "output/data/example/values.npz").exists()
