"""Shared runtime handle resolution for TorchLens records."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

_T = TypeVar("_T")


def source_model_from_trace(trace: Any) -> Any | None:
    """Return the live source model referenced by a Trace.

    Parameters
    ----------
    trace:
        Trace-like object carrying ``_source_model_ref``.

    Returns
    -------
    Any | None
        Live source model, or ``None`` when the model weakref is unavailable.
    """

    source_ref = getattr(trace, "_source_model_ref", None) if trace is not None else None
    if source_ref is None:
        return None
    return source_ref()


def runtime_handle_from_trace(trace: Any, resolver: Callable[[Any], _T | None]) -> _T | None:
    """Resolve a live runtime handle from a Trace source model.

    Parameters
    ----------
    trace:
        Trace-like owner carrying a source-model weakref.
    resolver:
        Callable that maps the live source model to the desired handle.

    Returns
    -------
    _T | None
        Resolved handle, or ``None`` when the source model or handle is unavailable.
    """

    model = source_model_from_trace(trace)
    if model is None:
        return None
    try:
        return resolver(model)
    except Exception:
        return None


def runtime_handle_from_owner(owner: Any, resolver: Callable[[Any], _T | None]) -> _T | None:
    """Resolve a live runtime handle from an already-live owner object.

    Parameters
    ----------
    owner:
        Live owner object, usually a Trace.
    resolver:
        Callable that maps the owner to the desired handle.

    Returns
    -------
    _T | None
        Resolved handle, or ``None`` when the owner or handle is unavailable.
    """

    if owner is None:
        return None
    try:
        return resolver(owner)
    except Exception:
        return None
