"""The one place a run writes down where its numbers came from.

The benchmark entry points use this module; other experiment families are
migrated explicitly and tracked in the development record.

The rule it enforces is small and blunt: if a run cannot say which
representation and which encoder it used, it does not get to write a results
file. That is the opposite of the pattern this repository currently uses --

    "encoder_provenance": getattr(adapter, "encoder_provenance", None)

-- which converts a missing attribute into a null on disk, produces a
complete-looking artifact, and defers the discovery by however long it takes
someone to audit. Six rows of the current main table are null for exactly this
reason.

Hashes are computed here rather than taken on trust. A path is a claim about
where bytes were; a digest is a claim about which bytes they were, and only the
second survives an export being rewritten in place.
"""
from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    CodeProvenance,
    ArtifactProvenance,
    EncoderProvenance,
    InputFileProvenance,
    RepresentationProvenance,
    RunProvenance,
    SplitProvenance,
    git_commit,
    sha256_file,
)


class ProvenanceUnavailable(RuntimeError):
    """Raised instead of writing an artifact that cannot say where it came from."""


def build_provenance(
    *,
    arm: str,
    run_dir: str | Path,
    representation_dir: str | Path,
    encoder: dict[str, Any] | None,
    split_policy: str | None,
    split_ratios: tuple[float, float, float] | list[float] | None,
    split_seed: int | None,
    n_trainable_params: int,
    backbone: str | None = None,
    work_repo: str | Path | None = None,
    core_repo: str | Path | None = None,
) -> RunProvenance:
    """Assemble and validate one run's provenance, hashing the inputs it read.

    Every argument that the reconciliation depends on is required and unchecked
    defaults are refused: `encoder=None` raises here rather than serialising to
    null. Callers that genuinely have no encoder -- native-coordinate benchmarks
    are the real case -- should say so with an explicit sentinel record rather
    than by omission, so that "no encoder" and "forgot to record the encoder"
    stay distinguishable downstream.
    """
    root = Path(representation_dir)
    if encoder is None:
        raise ProvenanceUnavailable(
            f"{arm}: no encoder provenance for {root}. If this benchmark has no "
            "Stage-1 encoder, pass an explicit sentinel instead of omitting it."
        )
    missing = [name for name, value in (
        ("split_policy", split_policy),
        ("split_ratios", split_ratios),
        ("split_seed", split_seed),
    ) if value is None]
    if missing:
        raise ProvenanceUnavailable(f"{arm}: split provenance incomplete, missing {missing}")

    npz, meta = root / "representations.npz", root / "metadata.tsv"
    for path in (npz, meta):
        if not path.exists():
            raise ProvenanceUnavailable(f"{arm}: {path} does not exist, cannot hash what was read")

    return RunProvenance(
        arm=arm,
        run_dir=str(run_dir),
        representation=RepresentationProvenance(
            representation_dir=str(root),
            representations_sha256=sha256_file(npz),
            metadata_sha256=sha256_file(meta),
            encoder=EncoderProvenance(**encoder),
        ),
        split=SplitProvenance(
            split_policy=str(split_policy),
            split_ratios=tuple(float(r) for r in split_ratios),  # type: ignore[arg-type]
            split_seed=int(split_seed),  # type: ignore[arg-type]
        ),
        n_trainable_params=int(n_trainable_params),
        backbone=backbone,
        code=CodeProvenance(
            work_commit=git_commit(work_repo) if work_repo else None,
            core_commit=git_commit(core_repo) if core_repo else None,
        ),
    )


def attach_provenance(payload: dict[str, Any], provenance: RunProvenance | ArtifactProvenance) -> dict[str, Any]:
    """Merge a validated provenance block into a results payload.

    Kept separate from writing so a caller can assemble metrics first and still
    fail before anything reaches disk.
    """
    return {**payload, "provenance": provenance.model_dump(mode="json")}


def write_results(path: str | Path, payload: dict[str, Any], provenance: RunProvenance | ArtifactProvenance) -> Path:
    """Write a results file that is, by construction, traceable.

    Writes through a temporary file so a crash mid-write leaves no artifact that
    looks complete; a half-written results.json that still parses is the same
    failure mode in a different costume.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(provenance, ArtifactProvenance):
        verify_inputs(provenance)
    serialized = json.dumps(attach_provenance(payload, provenance), indent=2,
                            sort_keys=True, default=str, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=target.name + ".", suffix=".partial",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target


def capture_inputs(*, operation: str, objective: str, files: dict[str, str | Path],
                   resolved_config: dict[str, Any], source_file: str | Path,
                   representation_manifest: str | Path | None = None,
                   representation_kind: str = "not_applicable",
                   captured_files: dict[str, InputFileProvenance] | None = None) -> ArtifactProvenance:
    """Capture the actual inputs before loading data or starting computation.

    No field is recovered from a previous results file and missing inputs fail
    here. A later write verifies that those bytes have not changed. Callers
    pass the complete resolved configuration, including seeds and selection
    policy, rather than a method or run name as a proxy for the objective.
    """
    inputs = dict(files)
    if representation_manifest is not None:
        inputs["representation_manifest"] = representation_manifest
    fingerprints = {}
    for name, value in inputs.items():
        requested = Path(value).absolute()
        path = requested.resolve(strict=True)
        if not path.is_file():
            raise ProvenanceUnavailable(f"{name}: required input is not a file: {path}")
        if captured_files is not None and name in captured_files:
            prior = captured_files[name]
            if prior.path != str(path) or prior.requested_path != str(requested):
                raise ProvenanceUnavailable(f"{name}: captured input path differs from loaded input")
            fingerprints[name] = prior
        else:
            fingerprints[name] = fingerprint_file(requested)
    encoder = None
    manifest = None
    encoder_lineage_status = ("not_applicable" if representation_kind in {"native", "not_applicable"}
                              else "unverified")
    if representation_kind == "encoded":
        if representation_manifest is None:
            raise ProvenanceUnavailable("encoded input requires its representation manifest")
        manifest = json.loads(Path(representation_manifest).read_text())
        if manifest.get("representation_is_native_space"):
            raise ProvenanceUnavailable("encoded input declared native by its manifest")
        if not isinstance(manifest.get("encoder"), dict):
            raise ProvenanceUnavailable("no encoder provenance in representation manifest")
        encoder = EncoderProvenance.model_validate(manifest["encoder"])
        if manifest.get("encoder_recorded_retrospectively"):
            encoder_lineage_status = "operator_retrospective"
        elif manifest.get("provenance", {}).get("schema_version") == 2 and "produced_files" in manifest:
            producer = ArtifactProvenance.model_validate(manifest["provenance"])
            checkpoint = producer.input_files.get("vae_checkpoint")
            vocabulary = producer.input_files.get("gene_vocab")
            if (producer.operation == "export" and checkpoint and vocabulary
                    and checkpoint.sha256 == encoder.vae_sha256
                    and vocabulary.sha256 == encoder.gene_vocab_sha256):
                encoder_lineage_status = "export_recorded"
    elif representation_kind == "native":
        manifest = json.loads(Path(representation_manifest).read_text()) if representation_manifest is not None else None
        if manifest is None or not manifest.get("representation_is_native_space"):
            raise ProvenanceUnavailable("native input requires an explicit native-space manifest")
    elif representation_manifest is not None:
        manifest = json.loads(Path(representation_manifest).read_text())
    if manifest is not None and "produced_files" in manifest:
        produced = manifest["produced_files"]
        if not isinstance(produced, dict):
            raise ProvenanceUnavailable("export manifest has malformed produced_files")
        for key in ("representations", "metadata", "metric_standardization"):
            if key not in fingerprints:
                continue
            observed = fingerprints[key]
            name = {"representations": "representations.npz", "metadata": "metadata.tsv",
                    "metric_standardization": "metric_standardization.npz"}[key]
            if name not in produced:
                raise ProvenanceUnavailable(f"export manifest lacks produced hash for {name}")
            expected = InputFileProvenance.model_validate(produced[name])
            if observed.sha256 != expected.sha256 or observed.size_bytes != expected.size_bytes:
                raise ProvenanceUnavailable(f"export manifest and consumed artifact disagree: {name}")
    source = Path(source_file).resolve(strict=True)
    try:
        root = subprocess.run(["git", "-C", str(source.parent), "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        diff = subprocess.run(["git", "-C", root, "diff", "HEAD", "--", "*.py"],
                              capture_output=True, check=True, timeout=10).stdout
        code = {"repository": root, "commit": git_commit(root),
                "tracked_python_diff_sha256": hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProvenanceUnavailable("source code must be a versioned task checkout") from exc
    code.update({"entrypoint": str(source), "entrypoint_sha256": sha256_file(source)})
    # The package can itself be a separate submodule. Hash the complete active
    # Python source, including untracked new modules; HEAD alone misses those.
    package = Path(__file__).resolve().parents[1]
    code["core_commit"] = git_commit(package)
    code["core_python_sha256"] = {
        str(p.relative_to(package)): sha256_file(p)
        for p in sorted(package.rglob("*.py")) if "__pycache__" not in p.parts
    }
    project_root = Path(root).resolve()
    imported = set()
    virtual_modules = {}
    for module in tuple(sys.modules.values()):
        # Do not invoke lazy module __getattr__ hooks while observing metadata
        # (notably torch.classes creates synthetic namespaces for unknown keys).
        filename = module.__dict__.get("__file__") if isinstance(module, types.ModuleType) else None
        if isinstance(filename, str):
            # torch.distributed.nn.jit uses compiler pseudo-filenames such as
            # <_remote_module_non_scriptable>.py; these are not relative source
            # files under cwd. Keep their declared identity, never resolve/hash
            # them as if a physical project file had been loaded.
            if filename.startswith("<") and ">" in filename:
                virtual_modules[module.__dict__.get("__name__", filename)] = filename
                continue
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(project_root) and not path.is_relative_to(package):
                imported.add(path)
    code["loaded_project_python_sha256"] = {
        str(p.relative_to(project_root)): sha256_file(p) for p in sorted(imported)}
    code["runtime_virtual_module_filenames"] = virtual_modules
    runtime = {"python": sys.version, "executable": sys.executable,
               "platform": platform.platform()}
    # Inspect already-loaded modules, without importing libraries or touching
    # random-number state while collecting metadata.
    runtime["packages"] = {name: str(getattr(sys.modules[name], "__version__", "unknown"))
                           for name in ("numpy", "torch", "pandas", "pydantic", "scipy", "ot")
                           if name in sys.modules}
    result = ArtifactProvenance(
        operation=operation, objective=objective,
        captured_at=datetime.now(timezone.utc).isoformat(), input_files=fingerprints,
        resolved_config=json.loads(json.dumps(resolved_config, default=str)), code=code,
        runtime=runtime, representation_kind=representation_kind, encoder=encoder,
        encoder_lineage_status=encoder_lineage_status)
    verify_inputs(result)
    return result


def verify_inputs(provenance: ArtifactProvenance) -> None:
    """Refuse to publish metrics after any required input was replaced."""
    verify_fingerprints(provenance.input_files)


def verify_fingerprints(fingerprints: dict[str, InputFileProvenance]) -> None:
    for name, item in fingerprints.items():
        path = Path(item.path)
        requested = Path(item.requested_path)
        if (not requested.is_file() or requested.resolve() != path
                or path.stat().st_size != item.size_bytes or sha256_file(path) != item.sha256):
            raise ProvenanceUnavailable(f"{name}: input changed after capture: {path}")


def fingerprint_file(path: str | Path) -> InputFileProvenance:
    requested = Path(path).absolute()
    path = requested.resolve(strict=True)
    return InputFileProvenance(path=str(path), requested_path=str(requested), sha256=sha256_file(path),
                               size_bytes=path.stat().st_size)


def fingerprint_files(files: dict[str, str | Path]) -> dict[str, InputFileProvenance]:
    captured = {name: fingerprint_file(path) for name, path in files.items()}
    verify_fingerprints(captured)
    return captured


def capture_adapter_run(*, adapter, operation: str, objective: str,
                        resolved_config: dict[str, Any], source_file: str | Path,
                        checkpoint_inputs: dict[str, str | Path], sampler=None,
                        captured_checkpoints: dict[str, InputFileProvenance] | None = None) -> ArtifactProvenance:
    """Record observed populations and the pools actually used by training.

    Exported paper data also supplies fingerprints captured before np.load.
    Adapters without an export manifest are recorded as observed-only, with
    upstream encoder lineage explicitly unverified, never as raw/native data.
    """
    import inspect
    import numpy as np
    from cellworldmodel.benchmark.branchsbm_adapter import (
        NormanAdapter, PerturbationAdapter, TimePointAdapter,
    )
    from cellworldmodel.training.benchmark_loop import build_transition_pairs

    def array_record(values):
        array = np.ascontiguousarray(values)
        return {"shape": list(array.shape), "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}

    def index_record(values):
        return array_record(np.asarray(values, dtype=np.int64))

    actual = {}
    training_pools = {}
    populations = getattr(adapter, "coords_by_t", None)
    if isinstance(populations, dict) and populations:
        for timepoint, values in sorted(populations.items()):
            entry = array_record(values)
            split_map = sampler.splits if sampler is not None else getattr(adapter, "splits_by_t", {})
            splits = split_map.get(timepoint)
            if splits is not None:
                key = "applied_sampler_splits" if sampler is not None else "exported_split_metadata_only"
                entry[key] = {name: index_record(getattr(splits, name))
                              for name in ("train", "val", "test")}
            actual[str(timepoint)] = entry
        if sampler is not None:
            pairs = sampler.pairs
        elif isinstance(adapter, TimePointAdapter):
            cfg = resolved_config.get("cfg", {})
            pairs = (build_transition_pairs(list(adapter.timepoints), cfg.get("md_endpoint_prob"))[0]
                     if cfg.get("multi_delta") else [(adapter.timepoints[0], adapter.timepoints[-1])])
        else:
            # Custom samplers must expose their actual index map; naming an
            # exported split cannot establish that a training loop applies it.
            raise ProvenanceUnavailable("adapter training pools require an explicit sampler")
        for source_t, target_t in pairs:
            if sampler is not None:
                src = sampler.splits[source_t].train
                tgt = sampler.splits[target_t].train
            else:
                src = (adapter.train_src_idx if source_t == adapter.timepoints[0]
                       else np.arange(len(populations[source_t])))
                tgt = np.arange(len(populations[target_t]))
            training_pools[f"{source_t}->{target_t}"] = {
                "source_indices": index_record(src), "target_indices": index_record(tgt),
                "policy": "sampler_train_split" if sampler is not None else "legacy_source_only_split_all_targets"}
    elif isinstance(adapter, PerturbationAdapter):
        actual = {"control": array_record(adapter._get_coords(adapter.control_df)),
                  "perturbed": array_record(adapter._get_coords(adapter.perturbed_df))}
        training_pools = {"control_to_perturbed": {
            "source_indices": index_record(adapter.train_src_idx),
            "target_indices": index_record(np.arange(len(adapter.perturbed_df))),
            "policy": "legacy_source_only_split_all_targets"}}
    elif isinstance(adapter, NormanAdapter):
        conditions = adapter.adata.obs[adapter.condition_col].astype(str).to_numpy()
        actual = {"cells": array_record(adapter._X_pca),
                  "condition_labels": {"sha256": hashlib.sha256(
                      json.dumps(conditions.tolist(), ensure_ascii=False).encode()).hexdigest()}}
        training_pools = {"condition_partition": {
            "train_conditions": list(adapter.train_conds), "test_conditions": list(adapter.test_conds),
            "source_indices": index_record(np.flatnonzero(adapter.control_mask)),
            "target_indices": index_record(np.flatnonzero(np.isin(conditions, adapter.train_conds))),
            "drift_reference_indices": index_record(np.flatnonzero(conditions == adapter.train_conds[0])),
            "policy": "legacy_condition_holdout"}}
    else:
        raise ProvenanceUnavailable(f"unsupported adapter evidence layout: {type(adapter).__name__}")
    config = {**resolved_config, "actual_adapter_data": actual, "actual_training_pools": training_pools}
    config["metric_standardization"] = {
        "enabled": bool(getattr(adapter, "standardize_metrics", False)),
        **{name: hashlib.sha256(np.ascontiguousarray(getattr(adapter, name)).tobytes()).hexdigest()
           for name in ("metric_mean", "metric_std") if hasattr(adapter, name)},
    }
    if sampler is not None:
        config["actual_transition_pairs"] = list(sampler.pairs)
    else:
        config["legacy_source_split"] = {
            name: {"count": len(getattr(adapter, name)), "sha256": hashlib.sha256(
                np.ascontiguousarray(getattr(adapter, name), dtype=np.int64).tobytes()).hexdigest()}
            for name in ("train_src_idx", "test_src_idx") if hasattr(adapter, name)}
    files = {"adapter_source": inspect.getfile(type(adapter)), **checkpoint_inputs}
    captured_files = {**getattr(adapter, "input_fingerprints", {}), **(captured_checkpoints or {})}
    root = getattr(adapter, "representation_dir", None)
    if root is None:
        config["upstream_representation_lineage"] = "unverified; observed adapter arrays only"
        return capture_inputs(operation=operation, objective=objective, files=files,
                              resolved_config=config, source_file=source_file,
                              representation_kind="adapter_observed", captured_files=captured_files)
    root = Path(root)
    config["representation_array_key"] = "scvi128"
    files.update({"representations": root / "representations.npz", "metadata": root / "metadata.tsv"})
    manifest_path = root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return capture_inputs(
        operation=operation, objective=objective, files=files, resolved_config=config,
        source_file=source_file, representation_manifest=manifest_path,
        representation_kind="native" if manifest.get("representation_is_native_space") else "encoded",
        captured_files=captured_files)


def capture_exported_inputs(*, representations: str | Path, metadata: str | Path,
                            operation: str, objective: str,
                            resolved_config: dict[str, Any], source_file: str | Path,
                            extra_inputs: dict[str, str | Path] | None = None) -> ArtifactProvenance:
    """The shared entry for scripts consuming an exported representation."""
    representations = Path(representations)
    manifest_path = representations.parent / "split_manifest.json"
    if not manifest_path.is_file():
        raise ProvenanceUnavailable(f"missing representation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    files = {"representations": representations, "metadata": metadata, **(extra_inputs or {})}
    config = dict(resolved_config)
    options = config.get("args", config)
    config["representation_array_key"] = str(options.get("space", "scvi128"))
    kind = "native" if manifest.get("representation_is_native_space") else "encoded"
    if kind == "encoded" and config["representation_array_key"] not in {"scvi128", "scvi128_z"}:
        kind = "export_component"
        config["upstream_representation_lineage"] = "component-specific encoder lineage unverified"
    return capture_inputs(
        operation=operation, objective=objective, files=files,
        resolved_config=config, source_file=source_file,
        representation_manifest=manifest_path,
        representation_kind=kind)
