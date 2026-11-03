"""Closure-constant readability check for a JAX function-root capture.

A raw ``fn(params, *inputs)`` function root records each closure constant of its
closed jaxpr as a constant op with a payload, so every constant must carry a
readable buffer. The backend module imports ``_readable_closure_consts``.
"""

from __future__ import annotations

from typing import Any

from ...backends.registry import BackendUnsupportedError


def _readable_closure_consts(closed_jaxpr: Any) -> tuple[Any, ...]:
    """Closure constants of a function-root jaxpr, each with a readable buffer.

    Parameters
    ----------
    closed_jaxpr
        The derived closed jaxpr.

    Returns
    -------
    tuple[Any, ...]
        ``closed_jaxpr.consts`` in order (possibly empty).

    Raises
    ------
    BackendUnsupportedError
        ``constant_unrealizable`` when a constant is not a concrete array
        (a tracer or a deleted buffer): it could not be recorded with a
        payload, and the rule never records a parentless op without one.
    """

    import jax
    import numpy as np

    consts = tuple(getattr(closed_jaxpr, "consts", ()))
    tracer_type = getattr(getattr(jax, "core", None), "Tracer", ())
    for index, value in enumerate(consts):
        if isinstance(value, tracer_type) or not isinstance(value, (jax.Array, np.ndarray)):
            raise BackendUnsupportedError(
                f"JAX closure constant {index} of type {type(value).__name__} has no readable "
                "buffer, so it cannot be recorded as a constant op with a payload.",
                code="constant_unrealizable",
                remedy="pass the value as an explicit pytree leaf of fn(params, *inputs)",
            )
    return consts
