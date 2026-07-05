"""TensorFlow compatibility probes for private eager-capture APIs."""

from __future__ import annotations

from typing import Any
import warnings

from ..registry import BackendUnsupportedError

TFCapabilitySnapshot = dict[str, bool]
"""Stable mapping from TensorFlow capability flag name to availability."""

_warned_missing_capabilities: set[str] = set()


def _import_op_callbacks_module() -> Any | None:
    """Return TensorFlow's private op-callback module when available.

    Returns
    -------
    Any | None
        ``tensorflow.python.framework.op_callbacks`` or ``None``.
    """

    try:
        from tensorflow.python.framework import op_callbacks
    except ImportError:
        return None
    return op_callbacks


def _probe_op_callbacks() -> bool:
    """Return whether TensorFlow exposes eager op callbacks.

    Returns
    -------
    bool
        True when ``tensorflow.python.framework.op_callbacks`` imports.
    """

    return _import_op_callbacks_module() is not None


HAS_TF_OP_CALLBACKS: bool = _probe_op_callbacks()

_CAPABILITY_ATTRS: tuple[str, ...] = ("HAS_TF_OP_CALLBACKS",)


def mark_tf_capability_missing(capability_name: str, detail: str) -> None:
    """Mark a TensorFlow capability absent and emit at most one warning.

    Parameters
    ----------
    capability_name:
        Name of the module-level ``HAS_*`` flag to flip to ``False``.
    detail:
        Short user-facing detail describing the graceful degradation.

    Returns
    -------
    None
        The matching module-level flag is updated in place.
    """

    if capability_name not in _CAPABILITY_ATTRS:
        raise ValueError(f"unknown TensorFlow capability flag: {capability_name}")
    globals()[capability_name] = False
    if capability_name in _warned_missing_capabilities:
        return
    _warned_missing_capabilities.add(capability_name)
    warnings.warn(
        f"TorchLens TensorFlow capability {capability_name} is unavailable; {detail}",
        UserWarning,
        stacklevel=3,
    )


def get_tf_capability_snapshot() -> TFCapabilitySnapshot:
    """Return the current TensorFlow capability flags as a stable snapshot.

    Returns
    -------
    TFCapabilitySnapshot
        Mapping from ``HAS_*`` flag name to boolean availability.
    """

    return {name: bool(globals()[name]) for name in _CAPABILITY_ATTRS}


def get_op_callbacks_module() -> Any:
    """Return TensorFlow's private eager op-callback module.

    Returns
    -------
    Any
        TensorFlow op-callback module.

    Raises
    ------
    BackendUnsupportedError
        If the private op-callback module is unavailable. TensorFlow eager
        capture cannot record per-op events without it; graph-only fallback
        capture remains separate.
    """

    op_callbacks = _import_op_callbacks_module()
    if op_callbacks is None:
        mark_tf_capability_missing(
            "HAS_TF_OP_CALLBACKS",
            "TensorFlow eager op capture is unavailable",
        )
        raise BackendUnsupportedError(
            "TensorFlow eager capture requires tensorflow.python.framework.op_callbacks."
        )
    return op_callbacks


__all__ = [
    "HAS_TF_OP_CALLBACKS",
    "TFCapabilitySnapshot",
    "get_op_callbacks_module",
    "get_tf_capability_snapshot",
    "mark_tf_capability_missing",
]
