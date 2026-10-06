"""Profiler bridge helpers."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, cast

from .._io import _json
from ..utils.display import atomic_write_text


def execution_trace(log: Any, trace_path: str | Path) -> dict[str, Any]:
    """Export a lightweight TorchLens execution-trace JSON file.

    This writes TorchLens' own ``torchlens.execution_trace.v1`` schema (per-layer
    ``id``/``name``/``op``/``inputs``/``bytes`` nodes). It is NOT the PyTorch
    ExecutionTraceObserver / Chakra execution-trace schema (``1.1.1-chakra`` with
    ``attrs``/``ctrl_deps``/``outputs``), so Chakra/HTA consumers cannot parse it.

    Parameters
    ----------
    log:
        ``Trace`` to export.
    trace_path:
        Destination JSON path.

    Returns
    -------
    dict[str, Any]
        Trace payload written to disk.
    """

    nodes = []
    for layer in getattr(log, "layer_list", []):
        nodes.append(
            {
                "id": getattr(layer, "raw_index", None),
                "name": getattr(layer, "layer_label", None),
                "op": getattr(layer, "func_name", None),
                "inputs": list(getattr(layer, "parents", []) or []),
                "bytes": getattr(layer, "activation_memory", None),
            }
        )
    payload = {"schema": "torchlens.execution_trace.v1", "nodes": nodes}
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, indent=2))
    return payload


def join(log: Any, kineto_trace: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Join TorchLens layer records with a PyTorch Kineto trace payload.

    Each complete event is assigned to at most one layer:

    - A ``record_function`` range whose name EQUALS a layer label is that
      layer's (exact equality, never substring).
    - The op events of one type (``aten::conv2d`` for ``func_name ==
      "conv2d"``; nested calls of the same name inside one outer event are not
      counted twice) are matched in time order to that type's layers in
      execution order: the k-th event to the k-th layer. When the event count
      is a multiple ``n`` of the layer count (``n`` repeated forwards under
      one profiler), event ``k`` goes to layer ``k mod L``. Events inside a
      matched label range are left to that range.
    - Any other event is unmatched and disclosed: ``unmatched_event_counts``
      counts every unmatched complete event by name, and
      ``mismatched_op_types`` names each op type whose event count is not a
      multiple of its layer count (those events stay unmatched, with a
      ``UserWarning``).

    Durations are host-side profiler event times: a per-layer diagnostic, not
    a rate denominator.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    kineto_trace:
        Kineto/Chrome trace JSON path or already-loaded dictionary.

    Returns
    -------
    dict[str, Any]
        Merged per-operation timing view.
    """

    # D22 (F09): host-side event times fill labeled diagnostic columns only and
    # are FORBIDDEN as a rate denominator; device time enters report schemas
    # only via the correlation-ID join.
    trace = _load_trace(kineto_trace)
    events = [event for event in _trace_events(trace) if event.get("ph", "X") == "X"]
    layers = list(getattr(log, "layer_list", []))
    assigned: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(layers))}
    used: set[int] = set()
    _assign_label_ranges(layers, events, assigned, used)
    mismatched = _assign_op_types(layers, events, assigned, used)
    rows = []
    for index, layer in enumerate(layers):
        matched_events = assigned[index]
        rows.append(
            {
                "layer_label": str(getattr(layer, "layer_label", "")),
                "func_name": str(getattr(layer, "func_name", "")),
                "raw_index": getattr(layer, "raw_index", None),
                "kineto_event_count": len(matched_events),
                "kineto_duration_us": sum(_duration(event) for event in matched_events),
                "kineto_events": matched_events,
            }
        )
    unmatched: dict[str, int] = {}
    for position, event in enumerate(events):
        if position not in used:
            name = str(event.get("name", ""))
            unmatched[name] = unmatched.get(name, 0) + 1
    if mismatched:
        warnings.warn(
            "profiler.join left op types unmatched because their event count is not a "
            f"multiple of their layer count: {mismatched}",
            UserWarning,
            stacklevel=2,
        )
    return {
        "schema": "torchlens.profiler_join.v2",
        "attribution": (
            "order-matched: k-th event of an op type to k-th layer of that type; "
            "record_function ranges by exact label (host-side; not for rates)"
        ),
        "ops": rows,
        "unmatched_event_counts": unmatched,
        "mismatched_op_types": mismatched,
        "trace_metadata": _metadata(trace),
    }


def _duration(event: dict[str, Any]) -> float:
    """Return an event's duration in microseconds (0.0 when absent)."""

    return float(event.get("dur", 0.0) or 0.0)


def _start(event: dict[str, Any]) -> float:
    """Return an event's start timestamp (0.0 when absent)."""

    return float(event.get("ts", 0.0) or 0.0)


def _contains(outer: dict[str, Any], inner: dict[str, Any]) -> bool:
    """Return whether ``inner`` lies inside ``outer`` on the same thread.

    Events without timestamps never contain one another.
    """

    if "ts" not in outer or "ts" not in inner:
        return False
    if (outer.get("pid"), outer.get("tid")) != (inner.get("pid"), inner.get("tid")):
        return False
    start, inner_start = _start(outer), _start(inner)
    return start <= inner_start and inner_start + _duration(inner) <= start + _duration(outer)


def _assign_label_ranges(
    layers: list[Any],
    events: list[dict[str, Any]],
    assigned: dict[int, list[dict[str, Any]]],
    used: set[int],
) -> None:
    """Assign events whose name equals a layer label to that layer.

    Parameters
    ----------
    layers:
        Layer records in execution order.
    events:
        Complete trace events.
    assigned:
        Per-layer event lists, filled in place.
    used:
        Positions of assigned events, filled in place.
    """

    by_label: dict[str, int] = {}
    for index, layer in enumerate(layers):
        label = str(getattr(layer, "layer_label", "") or "")
        if label.strip():
            by_label.setdefault(label, index)
    for position, event in enumerate(events):
        owner = by_label.get(str(event.get("name", "")))
        if owner is not None:
            assigned[owner].append(event)
            used.add(position)


def _op_event_names(func_name: str) -> frozenset[str]:
    """Return the profiler event names one TorchLens ``func_name`` records as.

    Parameters
    ----------
    func_name:
        TorchLens function name such as ``"conv2d"`` or ``"__add__"``.

    Returns
    -------
    frozenset[str]
        Exact event names (``conv2d``, ``aten::conv2d``; dunder methods also
        map to their aten op, ``__add__`` to ``aten::add``).
    """

    names = {func_name, f"aten::{func_name}"}
    if func_name.startswith("__") and func_name.endswith("__"):
        names.add(f"aten::{func_name.strip('_')}")
    return frozenset(names)


def _assign_op_types(
    layers: list[Any],
    events: list[dict[str, Any]],
    assigned: dict[int, list[dict[str, Any]]],
    used: set[int],
) -> dict[str, dict[str, int]]:
    """Assign op events to layers of the same type by execution order.

    Parameters
    ----------
    layers:
        Layer records in execution order.
    events:
        Complete trace events.
    assigned:
        Per-layer event lists, filled in place.
    used:
        Positions of assigned events, filled in place.

    Returns
    -------
    dict[str, dict[str, int]]
        Op types left unmatched, with their event and layer counts.
    """

    ranges = [events[position] for position in used]
    by_type: dict[str, list[int]] = {}
    for index, layer in enumerate(layers):
        func_name = str(getattr(layer, "func_name", "") or "")
        if func_name.strip() and func_name != "none" and not assigned[index]:
            by_type.setdefault(func_name, []).append(index)
    mismatched: dict[str, dict[str, int]] = {}
    for func_name, layer_indices in by_type.items():
        names = _op_event_names(func_name)
        pool = _outer_events(
            [
                position
                for position, event in enumerate(events)
                if position not in used
                and str(event.get("name", "")) in names
                and not any(_contains(outer, event) for outer in ranges)
            ],
            events,
        )
        if not pool:
            continue
        if len(pool) % len(layer_indices):
            mismatched[func_name] = {"events": len(pool), "layers": len(layer_indices)}
            continue
        for k, position in enumerate(pool):
            assigned[layer_indices[k % len(layer_indices)]].append(events[position])
            used.add(position)
    return mismatched


def _outer_events(positions: list[int], events: list[dict[str, Any]]) -> list[int]:
    """Drop events nested inside another event of the same pool; order by start.

    Parameters
    ----------
    positions:
        Event positions of one op type.
    events:
        Complete trace events.

    Returns
    -------
    list[int]
        Outermost event positions in time order (trace order breaks ties).
    """

    ordered = sorted(positions, key=lambda position: (_start(events[position]), position))
    outer: list[int] = []
    for position in ordered:
        if not any(_contains(events[kept], events[position]) for kept in outer):
            outer.append(position)
    return outer


def _load_trace(kineto_trace: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load a Kineto trace dictionary.

    Parameters
    ----------
    kineto_trace:
        Trace path or dictionary.

    Returns
    -------
    dict[str, Any]
        Loaded trace dictionary.
    """

    if isinstance(kineto_trace, dict):
        return kineto_trace
    path = Path(kineto_trace)
    # A Kineto trace file is external, potentially attacker-supplied input on the
    # same footing as a ``.tlspec`` manifest, so it routes through the ONE bounded
    # reader: ``read_text`` allocated the whole file before any ceiling applied, and
    # stdlib ``json.loads`` answered a deeply nested payload with an untyped
    # ``RecursionError`` rather than a typed refusal.
    return cast(dict[str, Any], _json.read_bounded(path))


def _trace_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Return trace events from a Kineto-like payload.

    Parameters
    ----------
    trace:
        Trace dictionary.

    Returns
    -------
    list[dict[str, Any]]
        Event dictionaries.
    """

    events = trace.get("traceEvents")
    if events is None:
        events = trace.get("events")
    if events is None:
        events = []
    return [event for event in events if isinstance(event, dict)]


def _metadata(trace: dict[str, Any]) -> dict[str, Any]:
    """Return non-event trace metadata.

    Parameters
    ----------
    trace:
        Trace dictionary.

    Returns
    -------
    dict[str, Any]
        Metadata with bulky event lists removed.
    """

    return {key: value for key, value in trace.items() if key not in {"traceEvents", "events"}}


__all__ = ["execution_trace", "join"]
