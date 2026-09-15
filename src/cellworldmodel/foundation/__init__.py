"""Foundation-model training utilities for large multi-dataset time series."""


def load_chreode_backbone(*args, **kwargs):
    """Load a checksummed encoder and dynamics release without partial restores."""
    from .released import load_chreode_backbone as load
    return load(*args, **kwargs)
