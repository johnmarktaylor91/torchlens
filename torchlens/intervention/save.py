"""Persistence ownership for future TorchLens intervention specifications."""

from pathlib import Path
from typing import Any

from .errors import _not_implemented


def save_intervention(log: Any, path: str | Path, *args: Any, **kwargs: Any) -> Any:
    """Save a future intervention specification.

    Parameters
    ----------
    log:
        ModelLog-like object whose future intervention recipe will be saved.
    path:
        Destination path for the future specification.
    *args:
        Reserved positional arguments.
    **kwargs:
        Reserved keyword arguments.
    """

    return _not_implemented("save_intervention", "Phase 10")


def load_intervention_spec(path: str | Path, *args: Any, **kwargs: Any) -> Any:
    """Load a future intervention specification.

    Parameters
    ----------
    path:
        Path to a future intervention specification.
    *args:
        Reserved positional arguments.
    **kwargs:
        Reserved keyword arguments.
    """

    return _not_implemented("load_intervention_spec", "Phase 10")


__all__ = ["load_intervention_spec", "save_intervention"]
