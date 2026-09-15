# Chreode

Review source for the pretrained spatial cell world model.

This source archive has no Git history or author metadata. It contains the
inference and adaptation implementation, model and data validators, original
metric functions, and focused tests.

## Install

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Offline inference

Place the tensor checkpoints and release manifest in one local artifact directory.

```python
from cellworldmodel.foundation import load_chreode_backbone
model = load_chreode_backbone("artifacts/release", device="cpu")
# z is an array of raw 128-dimensional states.
prediction = model.predict(z, delta=2.0)
samples = model.sample(z, delta=2.0, k_samples=8, seed=0)
```

For expression input, align mapped counts to `artifacts/gene_vocab.parquet` in
canonical_index order. The encoder normalizes counts to 10,000 and applies log1p.
Potential gradients require local autograd; torch.no_grad is supported.

The original temporal prediction protocol uses a CUDA Blackwell GPU, with
CPU scoring. CPU and CUDA noise streams differ even with the same integer seed.
The score replay entry point evaluates archived prediction clouds and reports
whether inference was executed.

Intervention artifacts contain the adapted base and adapter together.
No-action controls map all requested conditions to the shared slot.
Rectangle uses its native gene, position and mass interface, with source-only
neighbors and explicit log_mass inputs.

Training provenance requires a Git checkout; inference is supported from this
archive. The anonymous review source does not embed named artifact-host URLs.
