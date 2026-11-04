"""Structural fingerprint of a trace for the receptive-field solution caches.

:func:`_graph_revision` snapshots topology, shapes, roles, function identity
and the geometry arguments the rules read, by value, so a graph edit or an
in-place argument change invalidates a cached solution. The helpers import
nothing else from the package.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def _snapshot_value(value: object) -> str:
    """Return a mutation-proof, type-tagged token for one recorded argument value."""

    try:
        return f"{type(value).__name__}:{value!r}"
    except Exception:
        # Best-effort fallback: identity keeps the token stable for the same
        # object and still invalidates when the object is replaced outright.
        return f"{type(value).__name__}:@{id(value):x}"


def _snapshot_mapping(mapping: object) -> object:
    """Return an order-insensitive by-value token for one recorded mapping field."""

    if isinstance(mapping, Mapping):
        return tuple(sorted((str(key), _snapshot_value(value)) for key, value in mapping.items()))
    return _snapshot_value(mapping)


def _geometry_args_snapshot(op: Op) -> tuple[object, ...]:
    """Snapshot the recorded operation arguments that RF rules consume, by value.

    Geometry rules read kernel/stride/padding/dilation/output-size/etc. from
    ``op.func_config`` (``RuleContext.cfg``) with fallbacks into the captured
    non-tensor arguments (``RuleContext.arg``). The staleness fingerprint must
    capture BOTH surfaces by value: holding live references would compare a
    mutated dict against itself and never invalidate, so an in-place geometry
    change that preserves shape and topology would keep serving stale frozen
    descriptors from the trace-level solution cache while per-unit ``.at()``
    queries recompute fresh from the live values -- the two public RF paths
    would disagree.
    """

    return (
        _snapshot_mapping(getattr(op, "func_config", None)),
        tuple(_snapshot_value(value) for value in getattr(op, "non_tensor_pos_args", ()) or ()),
        _snapshot_mapping(getattr(op, "non_tensor_kwargs", None)),
    )


def _edge_uses_snapshot(op: Op) -> tuple[str, ...]:
    """Snapshot semantic edge records consumed by control-branch classification."""

    return tuple(_snapshot_value(record) for record in getattr(op, "edge_uses", ()) or ())


def _graph_revision(trace: Trace) -> tuple[object, ...]:
    """Build a stable structural fingerprint that invalidates cache after graph edits.

    Includes topology (parents/children), shape, role, function identity, and a
    by-value snapshot of the geometry arguments RF rules read, so an argument
    change with unchanged shape+topology still bumps the revision.

    ``op.shape`` is legitimately ``None`` for a non-tensor-valued op -- e.g. a
    JAX region boundary/projection pseudo-op, whose captured ``output`` is a
    tuple of values rather than one tensor (``_tensor_ref`` reports
    ``shape=None`` for exactly this "not a tensor" case). ``None`` is itself a
    stable, hashable fingerprint component, so it is included as-is instead of
    calling ``tuple()`` on it.
    """

    return tuple(
        (
            op.label,
            tuple(op.parents),
            tuple(op.children),
            tuple(op.shape) if op.shape is not None else None,
            op.io_role,
            op.func_name,
            _geometry_args_snapshot(op),
            _edge_uses_snapshot(op),
        )
        for op in trace._layer_list
        if _operation_is_live(op)
    )


def _operation_is_live(op: Op) -> bool:
    """Return whether an operation has not been cleared by trace removal."""

    return isinstance(getattr(op, "label", None), str)


__all__: list[str] = []
