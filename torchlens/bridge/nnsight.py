"""nnsight bridge helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def from_trace(trace: Any) -> dict[str, Any]:
    """Normalize a cached nnsight-style trace into a TorchLens bridge payload.

    Parameters
    ----------
    trace:
        Mapping, object with ``to_dict()``, or object exposing ``nodes``.

    Returns
    -------
    dict[str, Any]
        Offline trace payload with a stable TorchLens bridge schema.
    """

    payload = _trace_payload(trace)
    nodes = payload.get("nodes", [])
    return {
        "schema": "torchlens.nnsight_trace.v1",
        "nodes": list(nodes) if isinstance(nodes, list) else nodes,
        "metadata": {key: value for key, value in payload.items() if key != "nodes"},
    }


def _trace_payload(trace: Any) -> dict[str, Any]:
    """Return a dictionary payload for supported trace-like objects.

    Parameters
    ----------
    trace:
        Trace-like object.

    Returns
    -------
    dict[str, Any]
        Trace dictionary.
    """

    if isinstance(trace, Mapping):
        return dict(trace)
    if hasattr(trace, "to_dict") and callable(trace.to_dict):
        result = trace.to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    nodes = getattr(trace, "nodes", None)
    if nodes is not None:
        return {"nodes": nodes}
    return {"nodes": [], "repr": repr(trace)}


__all__ = ["from_trace"]
