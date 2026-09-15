# Release validation

Date: 2026-09-15. Implementation: Codex; actual backend model identity was not
independently verified. Independent review is recorded in the release pull request.

This release updates the manuscript's model implementation and adds explicit
offline inference, artifact checksums, download and scoring entry points. It does
not introduce a new scientific model or substitute newly trained results.

## Model and data checks

- All 23 released models were rebuilt from original checkpoint constructor
  metadata, exported to tensor-only state dictionaries and strictly restored.
- Original and exported tensors, nonpersistent buffers and fixed-input outputs
  agree exactly on the same Linux CPU environment (Python 3.10.19, PyTorch
  2.11.0+cu128). This is not a claim of cross-device bitwise equality.
- Required constructor sidecars are checksum-locked. Export receipts bind the
  source inventory and exporter, loader, intervention and Rectangle source hashes.
- Five processed benchmark datasets retain identical numerical arrays and row
  order. Nine original temporal prediction clouds are exported without modifying
  their values. Private path metadata is removed from JSON manifests.
- The publication inventory contains 52 files and 4,407,176,086 bytes. File sizes
  and SHA-256 checks pass; tensor pickle string fields and JSON/TSV text have no
  private path or internal identifier matches in the reviewed scan.

## Runtime checks

The encoder and backbone load on macOS CPU. Dense and sparse encoding agree;
fixed-noise sampling, deterministic prediction, decoding and the unconditioned
zero-horizon identity pass. A real-data adaptation check loads all 168 backbone
tensors and completes one update, validation selection and checkpoint saving.
That bounded check is not a rerun of the reported training experiment.

Focused tests cover the existing model/metric/encoder paths, transport objective,
budget selection, niche and spatial heads, no-action mapping, checksum errors,
path containment, immutable download revisions and corrupt cached downloads.
The two suites pass 101 tests in total, with one CUDA-only test skipped; 44 nested
subtests also pass. Complete declared dependencies install and `uv pip check`
reports no incompatibilities. macOS validation uses SciPy 1.13.1, matching the
original numerical environment; the tested SciPy 1.15.3 wheel had an import error.

## Score replay

The public replay command reproduces all 42 target scores across Weinreb, Veres
and ZESTA, each with three training seeds. Maximum absolute W2 difference from
the full-precision recorded values is 1.78e-15. The variance ratio preserves the
original scalar conversion; MMD may differ at float32 reduction precision.

Original temporal predictions used CUDA on an RTX PRO 6000 Blackwell, followed by
CPU scoring. A CPU generator with the same integer seed produces different
particles. Replay mode scores the original archived clouds; prediction mode runs
fresh checkpoint inference and labels that fact in its output.

The original GPU model was fully occupied during this validation. An additional
CUDA replay request was cancelled while pending and never ran. CUDA inference
replay remains unverified in this release check; no new GPU training was run.

## Scope

The model, data, loss and inference interfaces for intervention and Rectangle are
included. Historical private scheduling wrappers are excluded. Inference works
without a Git checkout; training provenance currently requires an editable Git
checkout. Main-table replay evaluates the Chreode artifacts; external baselines
retain their own implementations. Older configurations and reproduction notes are
explicitly marked as the legacy release.
