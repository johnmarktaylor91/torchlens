"""Profiler bridge helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def execution_trace(log: Any, trace_path: str | Path) -> dict[str, Any]:
    """Export a lightweight PyTorch ExecutionTraceObserver-compatible trace.

    Parameters
    ----------
    log:
        ``ModelLog`` to export.
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
                "id": getattr(layer, "creation_order", None),
                "name": getattr(layer, "layer_label", None),
                "op": getattr(layer, "func_name", None),
                "inputs": list(getattr(layer, "parent_layers", []) or []),
                "bytes": getattr(layer, "tensor_memory", None),
            }
        )
    payload = {"schema": "torchlens.execution_trace.v1", "nodes": nodes}
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def join(log: Any, kineto_trace: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Join TorchLens layer records with a PyTorch Kineto trace payload.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog``.
    kineto_trace:
        Kineto/Chrome trace JSON path or already-loaded dictionary.

    Returns
    -------
    dict[str, Any]
        Merged per-operation timing view.
    """

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
                "creation_order": getattr(layer, "creation_order", None),
                "kineto_event_count": len(matched_events),
                "kineto_duration_us": duration_us,
                "kineto_events": matched_events,
            }
        )
    return {"schema": "torchlens.profiler_join.v1", "ops": rows, "trace_metadata": _metadata(trace)}


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
    return json.loads(path.read_text(encoding="utf-8"))


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

    events = trace.get("traceEvents", trace.get("events", []))
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
    return label in event_name or (func_name != "none" and func_name in event_name)


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
