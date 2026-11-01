"""Pure trace-fact readers for the structure-only discharge preflight and comparison."""

from __future__ import annotations

from typing import Any

from .._data_classes._trace_missing_attrs import is_live_trace


def _input_geometry(trace: Any) -> tuple[tuple[tuple[int, ...] | None, str], ...]:
    """The (shape, dtype) tuple of each input layer, for the preflight."""

    facts: list[tuple[tuple[int, ...] | None, str]] = []
    for label in getattr(trace, "input_layers", ()) or ():
        layer = trace[label]
        shape = getattr(layer, "shape", None)
        facts.append((tuple(shape) if shape else None, str(getattr(layer, "dtype", None))))
    return tuple(facts)


def _layer_claims(layer: Any) -> tuple[tuple[str, Any], ...]:
    """Extract the compared claim kinds from one layer record."""

    shape = getattr(layer, "shape", None)
    param_shapes = getattr(layer, "param_shapes", None) or ()
    return (
        ("shape", tuple(shape) if shape is not None else None),
        ("dtype", str(getattr(layer, "dtype", None))),
        # Layer.param_shapes is an ordered sequence of parameter shapes
        # (weight, bias, ...); order is part of the geometry claim.
        ("param_geometry", tuple(tuple(entry) for entry in param_shapes)),
    )


def _backend_version(trace: Any) -> Any:
    """The backend runtime version a live trace recorded, else ``None``."""

    return trace._provenance_backend_version if is_live_trace(trace) else None
