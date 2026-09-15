"""Small I/O helpers shared by foundation workflow modules."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


def jsonable(value: Any) -> Any:
    """Convert numpy/scipy-adjacent objects into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return None if np.isnan(x) or np.isinf(x) else x
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, indent=2, sort_keys=True)


def as_dense_array(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)
