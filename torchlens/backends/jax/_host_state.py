"""Closed-over host-state check for JAX derived gradients."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def _reject_closed_over_host_state(fn: Callable[..., Any]) -> None:
    """Reject obvious closed-over host scalar or array state for derived grads.

    Parameters
    ----------
    fn
        Callable being differentiated.

    Returns
    -------
    None
        Returns when no referenced host scalar/array globals are found.
    """

    import jax
    import numpy as np

    closure = inspect.getclosurevars(fn)
    candidates = {**closure.nonlocals, **closure.globals}
    for name, value in candidates.items():
        module = inspect.getmodule(value)
        module_name = "" if module is None else module.__name__
        if module_name.startswith(("jax", "jaxlib", "numpy")):
            continue
        if inspect.ismodule(value) or inspect.isfunction(value) or inspect.isclass(value):
            continue
        if isinstance(value, (int, float, complex, bool, str, np.ndarray, jax.Array)):
            raise ValueError(
                "JAX derived gradients reject closed-over host-state values. "
                f"Pass {name!r} as an explicit params/input/static argument."
            )
