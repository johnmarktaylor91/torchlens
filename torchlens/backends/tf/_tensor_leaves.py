"""Tensor-leaf helpers shared by the TensorFlow intervention engine (file cap split).

Moved unchanged out of :mod:`.interventions`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _map_tensor_leaves(value: Any, tf: Any, fn: Callable[[Any], Any]) -> Any:
    """Apply ``fn`` to every tensor leaf of a wrapped-call return value.

    Parameters
    ----------
    value
        Wrapped-call return container or tensor.
    tf
        Imported TensorFlow module.
    fn
        Leaf transformation.

    Returns
    -------
    Any
        Value with transformed tensor leaves.
    """

    if _is_tf_tensor_like(value, tf):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_tensor_leaves(item, tf, fn) for item in value)
    if isinstance(value, list):
        return [_map_tensor_leaves(item, tf, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensor_leaves(item, tf, fn) for key, item in value.items()}
    return value


def _is_tf_tensor_like(value: Any, tf: Any) -> bool:
    """Return whether ``value`` is a TensorFlow tensor or variable.

    Parameters
    ----------
    value
        Candidate value.
    tf
        Imported TensorFlow module.

    Returns
    -------
    bool
        True for tensors and variables.
    """

    tensor_type = getattr(tf, "Tensor", None)
    variable_type = getattr(tf, "Variable", None)
    return bool(
        (tensor_type is not None and isinstance(value, tensor_type))
        or (variable_type is not None and isinstance(value, variable_type))
    )
