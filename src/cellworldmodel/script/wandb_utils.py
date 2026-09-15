"""Small optional W&B helpers for benchmark scripts."""
from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import Any


try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def add_wandb_args(parser) -> None:
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="CellWorldModel")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default=None, help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-notes", default=None)


def _tags(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def maybe_init_wandb(args, config: dict[str, Any], output_dir: Path, default_name: str,
                     default_group: str | None = None):
    if not getattr(args, "wandb", False):
        return None
    if wandb is None:
        raise RuntimeError("W&B logging requested but wandb is not installed")

    mode = getattr(args, "wandb_mode", "online")
    return wandb.init(
        project=getattr(args, "wandb_project", "CellWorldModel"),
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_name", None) or default_name,
        group=getattr(args, "wandb_group", None) or default_group,
        tags=_tags(getattr(args, "wandb_tags", None)),
        notes=getattr(args, "wandb_notes", None),
        config=config,
        dir=str(output_dir),
        mode=mode,
    )


def wandb_run_info(run) -> dict[str, Any]:
    if run is None:
        return {"enabled": False}
    local_files_dir = Path(run.dir).resolve()
    raw_path = getattr(run, "path", None)
    if isinstance(raw_path, (list, tuple)):
        run_path = "/".join(str(part) for part in raw_path)
    elif raw_path:
        run_path = str(raw_path)
    else:
        run_path = None
    return {
        "enabled": True,
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "run_path": run_path,
        "run_url": getattr(run, "url", None),
        "local_files_dir": str(local_files_dir),
        "local_run_dir": str(local_files_dir.parent),
    }


def _clean_key(key: Any) -> str:
    text = str(key)
    for old, new in (
        (" ", "_"), ("(", ""), (")", ""), ("[", ""), ("]", ""),
        ("{", ""), ("}", ""), (":", "_"), ("→", "to"),
    ):
        text = text.replace(old, new)
    return text


def flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}/{_clean_key(key)}" if prefix else _clean_key(key)
            out.update(flatten_numeric(value, next_prefix))
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}/{idx}" if prefix else str(idx)
            out.update(flatten_numeric(value, next_prefix))
    elif isinstance(obj, Number) and not isinstance(obj, bool):
        value = float(obj)
        if value == value and value not in (float("inf"), float("-inf")):
            out[prefix] = value
    return out


def wandb_log(run, metrics: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    numeric = flatten_numeric(metrics)
    if numeric:
        run.log(numeric, step=step)


def wandb_summary_update(run, metrics: dict[str, Any]) -> None:
    if run is None:
        return
    for key, value in flatten_numeric(metrics).items():
        run.summary[key] = value
