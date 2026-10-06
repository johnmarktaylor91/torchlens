"""nnsight bridge helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_SUPPORTED = (
    "a mapping, an object whose to_dict() returns a mapping, or an object exposing a "
    "non-None `nodes` attribute"
)


def from_trace(trace: Any) -> dict[str, Any]:
    """Normalize a cached nnsight-style trace into a TorchLens bridge payload.

    Parameters
    ----------
    trace:
        Mapping, object with ``to_dict()`` returning a mapping, or object
        exposing ``nodes``.

    Returns
    -------
    dict[str, Any]
        Offline trace payload with a stable TorchLens bridge schema.

    Raises
    ------
    TypeError
        If ``trace`` is none of the supported shapes. A live nnsight tracer
        (``with model.trace(...) as tracer``) exposes none of them, so it is
        refused rather than turned into an empty payload.
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

    Raises
    ------
    TypeError
        If ``trace`` matches no supported shape.
    """

    if isinstance(trace, Mapping):
        return dict(trace)
    to_dict = getattr(trace, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    nodes = getattr(trace, "nodes", None)
    if nodes is not None:
        return {"nodes": nodes}
    type_name = f"{type(trace).__module__}.{type(trace).__qualname__}"
    hint = ""
    if type(trace).__module__.split(".")[0] == "nnsight":
        hint = (
            " A live nnsight tracer carries no node list after the `with` block; "
            "save the values you need inside the trace (`.save()`) and pass them "
            "as a mapping, e.g. {'nodes': [...], ...}."
        )
    raise TypeError(
        f"torchlens.bridge.nnsight.from_trace supports {_SUPPORTED}; got {type_name}.{hint}"
    )


__all__ = ["from_trace"]
