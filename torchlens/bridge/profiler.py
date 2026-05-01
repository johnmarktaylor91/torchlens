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


__all__ = ["execution_trace"]
