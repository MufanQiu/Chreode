"""Typed provenance for benchmark artifacts, and the invariants a paper table must satisfy.

Three audits in a row traced a wrong number back to the same shape of bug: an
optional parameter carried a default, and the default pointed at a specific
historical artifact. The representation root did it, the encoder checkpoint did
it, and the backbone size did it. Each time the run completed, wrote a
plausible-looking results file, and the mistake surfaced weeks later.

Two things follow, and they need different machinery.

A schema catches "this field is missing, or silently defaulted". That is what
the models here are for: provenance fields are required, `extra="forbid"`
catches a typo'd key, and a null is a parse failure rather than a value. The
call sites that write `getattr(adapter, "encoder_provenance", None)` are exactly
what this replaces -- that expression turns a missing attribute into a null on
disk without anyone noticing.

A schema does not catch "these rows disagree with each other", because that is a
property of a set of records, not of any one record. Table 1 compares a 41M
model against 200k baselines, so a global "all fields equal" rule would be
wrong; what must hold is that every row shares one representation, one encoder
and one split, and that rows the author *claims* are capacity-matched really do
have the same parameter count. `reconcile_table.py` checks that.

Nor does a schema make a recorded value true. `representation_dir` being a
required string proves only that the field has a value; the file at that path
can be replaced in place. Recording the sha256 of the bytes actually loaded is
what turns a declaration into a verification, which is why the hash lives in the
model and `verify_against_disk` exists.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_HEX = 64


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Content hash of one file. Streamed, so a 60 MB representation is cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: str | Path) -> str | None:
    """HEAD of a working tree, or None when it is not a repository."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip() or None


class EncoderProvenance(BaseModel):
    """Which Stage-1 encoder produced a representation.

    Every field is required. The 2026-09-02 incident is precisely a run that
    would have passed a version of this model with optional fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    vae_checkpoint: str = Field(min_length=1)
    vae_sha256: str
    gene_vocab: str = Field(min_length=1)
    gene_vocab_sha256: str
    n_vocab_genes: int = Field(gt=0)

    @field_validator("vae_sha256", "gene_vocab_sha256")
    @classmethod
    def _hex_digest(cls, value: str) -> str:
        if len(value) != SHA256_HEX or not all(c in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"expected a 64-character sha256 hex digest, got {value!r}")
        return value.lower()


class RepresentationProvenance(BaseModel):
    """Which latent space a run trained and scored in.

    `representations_sha256` is the field that makes this a verification rather
    than an assertion: two runs can name the same directory and still have read
    different bytes if the export was rewritten between them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    representation_dir: str = Field(min_length=1)
    representations_sha256: str | None = None
    metadata_sha256: str | None = None
    encoder: EncoderProvenance

    @field_validator("representations_sha256", "metadata_sha256")
    @classmethod
    def _optional_hex_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return EncoderProvenance._hex_digest(value)

    def verify_against_disk(self) -> list[str]:
        """Recompute the hashes and report every mismatch.

        An empty list means the bytes on disk today are the bytes this run
        recorded. A missing hash is reported as unverifiable rather than as
        agreement, because "we did not record it" and "it matches" are the two
        states this whole exercise exists to distinguish.
        """
        problems: list[str] = []
        root = Path(self.representation_dir)
        for name, recorded in (("representations.npz", self.representations_sha256),
                               ("metadata.tsv", self.metadata_sha256)):
            path = root / name
            if recorded is None:
                problems.append(f"{name}: no hash recorded, cannot verify")
                continue
            if not path.exists():
                problems.append(f"{name}: recorded a hash but the file is gone ({path})")
                continue
            actual = sha256_file(path)
            if actual != recorded:
                problems.append(f"{name}: recorded {recorded[:12]}… but disk holds {actual[:12]}…")
        return problems


class SplitProvenance(BaseModel):
    """How cells were partitioned. Two runs on different splits are not comparable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split_policy: str = Field(min_length=1)
    split_ratios: tuple[float, float, float]
    split_seed: int

    @field_validator("split_ratios")
    @classmethod
    def _sums_to_one(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if not all(math.isfinite(r) and 0 <= r <= 1 for r in value):
            raise ValueError(f"split ratios must be finite probabilities, got {value}")
        if abs(sum(value) - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1, got {value} summing to {sum(value)}")
        return value

    @property
    def key(self) -> str:
        ratios = ",".join(f"{r:g}" for r in self.split_ratios)
        return f"{self.split_policy}({ratios})@{self.split_seed}"


class CodeProvenance(BaseModel):
    """Which code produced the artifact.

    The core package was fast-forwarded while a batch of jobs was in flight on
    2026-09-03; log wording is currently the only way to tell the two versions
    apart after the fact.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    work_commit: str | None = None
    core_commit: str | None = None


class RunProvenance(BaseModel):
    """Everything a paper table needs to know about one row's origin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: str = Field(min_length=1)
    run_dir: str = Field(min_length=1)
    representation: RepresentationProvenance
    split: SplitProvenance
    n_trainable_params: int = Field(gt=0)
    backbone: str | None = None
    code: CodeProvenance = CodeProvenance()

    @property
    def space_key(self) -> str:
        """What must be identical across every row of one table."""
        rep = self.representation
        digest = rep.representations_sha256 or f"path:{rep.representation_dir}"
        return f"{digest}|{rep.encoder.vae_sha256}|{self.split.key}"

    @property
    def capacity_key(self) -> str:
        """What must be identical across rows an author calls capacity-matched."""
        return f"{self.n_trainable_params}|{self.backbone or '?'}"


class InputFileProvenance(BaseModel):
    """A required input fingerprint captured before the input is consumed."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1)
    requested_path: str = Field(min_length=1)
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def _valid_digest(cls, value: str) -> str:
        return EncoderProvenance._hex_digest(value)


class ArtifactProvenance(BaseModel):
    """Common envelope for training, evaluation, and export artifacts.

    This does not infer historical training provenance from files read today.
    An evaluation records its checkpoint as an input; the checkpoint's own
    training evidence must still be reconciled separately.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[2] = 2
    operation: Literal["training", "evaluation", "export", "analysis"]
    objective: str = Field(min_length=1)
    captured_at: str = Field(min_length=1)
    input_files: dict[str, InputFileProvenance] = Field(min_length=1)
    resolved_config: dict[str, Any] = Field(min_length=1)
    code: dict[str, Any] = Field(min_length=1)
    runtime: dict[str, Any] = Field(min_length=1)
    # Explicit status: native data has no Stage-1 encoder, while an encoded
    # representation must carry a validated encoder record in its manifest.
    representation_kind: Literal["encoded", "native", "export_component", "adapter_observed", "not_applicable"]
    encoder: EncoderProvenance | None = None
    # Missing on older v2 records means unknown, never retrospectively verified.
    encoder_lineage_status: Literal["export_recorded", "operator_retrospective", "unverified", "not_applicable"] = "unverified"

    @model_validator(mode="after")
    def _encoder_contract(self):
        if self.representation_kind == "encoded" and self.encoder is None:
            raise ValueError("encoded representations require encoder provenance")
        if self.representation_kind != "encoded" and self.encoder is not None:
            raise ValueError("only encoded representations carry Stage-1 encoder provenance")
        if self.encoder_lineage_status in {"export_recorded", "operator_retrospective"} and self.representation_kind != "encoded":
            raise ValueError("encoder lineage evidence requires an encoded representation")
        if self.representation_kind == "encoded" and self.encoder_lineage_status == "not_applicable":
            raise ValueError("encoded representations cannot omit encoder lineage status")
        if self.representation_kind == "adapter_observed" and not self.resolved_config.get("actual_adapter_data"):
            raise ValueError("untraced adapters require observed data fingerprints")
        return self
