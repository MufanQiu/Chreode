"""Download a checksum-locked release from explicit Hugging Face revisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

from huggingface_hub import hf_hub_download

from cellworldmodel.foundation.released import sha256


def download_release(manifest_path, output, *, models=(), datasets=()):
    manifest_path, root = Path(manifest_path), Path(output).resolve()
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("a versioned release manifest with file checksums is required")
    unknown = set(models) - manifest["models"].keys()
    if unknown:
        raise ValueError(f"unknown model IDs: {sorted(unknown)}")
    available_data = {name.split("/")[1] for name in manifest["files"] if name.startswith("data/")}
    if set(datasets) - available_data:
        raise ValueError(f"unknown datasets: {sorted(set(datasets) - available_data)}")
    wanted = {manifest["models"][key]["weights"] for key in models}
    selected = {name: entry for name, entry in manifest["files"].items()
                if not (models or datasets) or name in wanted
                or any(name.startswith(f"data/{dataset}/") for dataset in datasets)}
    root.mkdir(parents=True, exist_ok=True)
    destination_manifest = root / "release.json"
    if destination_manifest.exists() and destination_manifest.read_bytes() != payload:
        raise ValueError("output already contains a different release manifest")
    for name, entry in selected.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or target == destination_manifest:
            raise ValueError(f"invalid release path: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"invalid file digest: {name}")
        location = manifest["repositories"][entry["repository"]]
        if not re.fullmatch(r"[0-9a-f]{40}", location["revision"]):
            raise ValueError("downloads require immutable full commit revisions")
        if target.exists():
            if target.stat().st_size != entry["size_bytes"] or sha256(target) != entry["sha256"]:
                raise ValueError(f"existing file does not match this release: {name}")
            continue
        cached = Path(hf_hub_download(repo_id=location["repo_id"], repo_type=location["repo_type"],
                                     revision=location["revision"], filename=name))
        if cached.stat().st_size != entry["size_bytes"] or sha256(cached) != entry["sha256"]:
            raise ValueError(f"download checksum mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            staging = Path(temporary.name)
        try:
            shutil.copyfile(cached, staging)
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)
        print(f"Verified {name}", flush=True)
    destination_manifest.write_bytes(payload)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    args = parser.parse_args()
    download_release(args.manifest, args.output, models=args.model, datasets=args.dataset)


if __name__ == "__main__":
    main()
