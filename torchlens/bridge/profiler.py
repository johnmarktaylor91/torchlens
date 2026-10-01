"""Profiler bridge helpers."""

from __future__ import annotations

import json
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

    # D22 (F09): this bridge matches events by NAME SUBSTRING -- a
    # collision-prone approximate diagnostic. It fills labeled diagnostic
    # columns only and is FORBIDDEN as a rate denominator; device time
    # enters report schemas only via the correlation-ID join.
    trace = _load_trace(kineto_trace)
    events = _trace_events(trace)
    rows = []
    for layer in getattr(log, "layer_list", []):
        label = str(getattr(layer, "layer_label", ""))
        func_name = str(getattr(layer, "func_name", ""))
        matched_events = [
            event
            for event in events
            if _event_matches_layer(event, label=label, func_name=func_name)
        ]
        duration_us = sum(float(event.get("dur", 0.0) or 0.0) for event in matched_events)
        rows.append(
            {
                "layer_label": label,
                "func_name": func_name,
                "raw_index": getattr(layer, "raw_index", None),
                "kineto_event_count": len(matched_events),
                "kineto_duration_us": duration_us,
                "kineto_events": matched_events,
            }
        )
    return {
        "schema": "torchlens.profiler_join.v1",
        "attribution": "name-matched (approximate; not for rates)",
        "ops": rows,
        "trace_metadata": _metadata(trace),
    }


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


def _event_matches_layer(event: dict[str, Any], *, label: str, func_name: str) -> bool:
    """Return whether a Kineto event should be associated with a layer.

    Parameters
    ----------
    event:
        Kineto event dictionary.
    label:
        TorchLens layer label.
    func_name:
        TorchLens function name.

    Returns
    -------
    bool
        Whether the event name references the layer label or function.
    """

    event_name = str(event.get("name", ""))
    if not event_name:
        return False
    # A blank label/func_name would substring-match EVERY event ("" in anything is
    # True), so a layer record missing layer_label/func_name would silently absorb
    # the entire trace. Refuse blank matches. The substring/many-to-many matching
    # of NON-blank labels/funcs is the owner-reserved join contract and is left
    # unchanged here.
    label_match = bool(label.strip()) and label in event_name
    func_match = bool(func_name.strip()) and func_name != "none" and func_name in event_name
    return label_match or func_match


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
