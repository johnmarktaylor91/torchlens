"""Native backward capture, one private module per backend.

Each backend's module (``jax.py``, ``tf.py``, ``mlx.py``, ``paddle.py``,
``tinygrad.py``) is owned by that backend's backward lane and defines the hook
functions the capture lanes call:

- ``backward_ready_scope(session)``: context manager around the forward;
- ``retain_backward_state(trace, session)``: called at the end of the capture;
- ``record_backward(trace, loss, **backward_kwargs)``: run one backward pass and
  journal what it observed as backward events (``torchlens._ir.events``), which
  ``torchlens.backends._backward_neutral`` folds into the records;
- optionally ``capture_backward(trace)``: a context manager for backward calls
  the user drives; a module without it refuses that entry point.

The framework is imported inside those functions, never at module import.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["NATIVE_BACKWARD_BACKENDS", "native_backward_module"]

NATIVE_BACKWARD_BACKENDS: frozenset[str] = frozenset({"jax", "mlx", "paddle", "tf", "tinygrad"})
"""Backend names that may own a native backward module in this package."""


def native_backward_module(backend: str) -> ModuleType | None:
    """Return the native backward module for ``backend``, or ``None`` when none exists.

    Parameters
    ----------
    backend:
        Registered backend name (``"jax"``, ``"tf"``, ...).

    Returns
    -------
    ModuleType | None
        The imported module, or ``None`` when the backend is not listed or its
        module is not built yet. An import error raised inside an existing
        module propagates, so a broken module is never mistaken for a missing one.

    Examples
    --------
    >>> native_backward_module("torch") is None
    True
    """

    if backend not in NATIVE_BACKWARD_BACKENDS:
        return None
    module_name = f"{__name__}.{backend}"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise
