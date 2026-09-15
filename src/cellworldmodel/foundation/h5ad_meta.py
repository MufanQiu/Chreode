"""Small h5ad metadata readers based on h5py.

These helpers intentionally avoid loading expression matrices. They are used by
foundation catalog builders where backed AnnData object construction would be
unnecessary overhead.
"""
from __future__ import annotations

from typing import Any

import h5py
import numpy as np


def decode_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", errors="replace")
    return str(value)


def read_h5_column(group: h5py.Group, key: str, max_values: int | None = None) -> list[str]:
    obj = group[key]
    slc = slice(None) if max_values is None else slice(0, max_values)
    if isinstance(obj, h5py.Group) and "categories" in obj and "codes" in obj:
        categories = [decode_value(v) for v in obj["categories"][:]]
        out = []
        for code in obj["codes"][slc]:
            code_int = int(code)
            out.append(categories[code_int] if 0 <= code_int < len(categories) else "")
        return out
    if isinstance(obj, h5py.Dataset):
        return [decode_value(v) for v in obj[slc]]
    return []


def read_var_names(handle: h5py.File, preferred_keys=("gene_name", "_index")) -> list[str]:
    var = handle["var"]
    for key in preferred_keys:
        if key in var:
            return read_h5_column(var, key)
    first_key = next(iter(var.keys()))
    return read_h5_column(var, first_key)


def read_obs_column(handle: h5py.File, key: str) -> list[str]:
    return read_h5_column(handle["obs"], key)


def h5ad_shape(handle: h5py.File) -> tuple[int, int]:
    x = handle["X"]
    if isinstance(x, h5py.Group):
        shape = x.attrs.get("shape")
        if shape is not None:
            return int(shape[0]), int(shape[1])
    return int(x.shape[0]), int(x.shape[1])
