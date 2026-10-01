"""Shared export helpers (subpackage promotion, C01 item 18).

Helpers consumed by more than one export-target family live here so the
per-family modules never import each other or the package facade.
"""

from __future__ import annotations

from typing import Any

__tl_layer__ = "L7"


def _iter_layers(log: Any) -> list[Any]:
    """Return layer-pass entries in export order.

    Parameters
    ----------
    log:
        Model log-like object.

    Returns
    -------
    list[Any]
        Layer entries.
    """

    return list(
        getattr(log, "layer_list", None) or getattr(log, "layer_dict_main_keys", {}).values()
    )


def _repeated_layer_labels(entries: list[Any]) -> set[str]:
    """Return rolled layer labels that occur in multiple execution passes.

    Parameters
    ----------
    entries:
        Layer-pass entries in export order.

    Returns
    -------
    set[str]
        Labels requiring pass qualification for unique export node IDs.
    """

    counts: dict[str, int] = {}
    for entry in entries:
        label = str(getattr(entry, "layer_label", ""))
        counts[label] = counts.get(label, 0) + 1
    return {label for label, count in counts.items() if count > 1}


def _export_node_id(entry: Any, repeated_labels: set[str]) -> str:
    """Return a unique static-export node ID for one layer pass.

    Parameters
    ----------
    entry:
        Layer-pass entry.
    repeated_labels:
        Rolled labels requiring pass qualification.

    Returns
    -------
    str
        Pass-qualified ID for recurrent layers, otherwise the stable rolled label.
    """

    layer_label = str(getattr(entry, "layer_label", ""))
    if layer_label in repeated_labels:
        return str(getattr(entry, "label", layer_label))
    return layer_label


def _node_type(entry: Any) -> str:
    """Return the semantic node type for an exported entry.

    Parameters
    ----------
    entry:
        Layer-pass log entry.

    Returns
    -------
    str
        Semantic node type.
    """

    if getattr(entry, "is_input", False):
        return "input"
    if getattr(entry, "is_output", False):
        return "output"
    if getattr(entry, "is_buffer", False):
        return "buffer"
    if getattr(entry, "is_terminal_bool", False):
        return "bool"
    if int(getattr(entry, "num_params", 0) or 0) > 0:
        return "parameterized"
    return "operation"


def _static_graph_data(log: Any) -> dict[str, Any]:
    """Serialize a Trace into static graph data.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to serialize.

    Returns
    -------
    dict[str, Any]
        Node and edge metadata for SVG/HTML exporters.
    """

    entries = _iter_layers(log)
    repeated_labels = _repeated_layer_labels(entries)
    node_ids = {_export_node_id(entry, repeated_labels) for entry in entries}
    nodes: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        node_id = _export_node_id(entry, repeated_labels) or f"node_{index}"
        nodes.append(
            {
                "id": node_id,
                "label": str(getattr(entry, "layer_label", node_id)),
                "type": _node_type(entry),
                "shape": "x".join(str(dim) for dim in getattr(entry, "shape", ()) or ()),
                "memory": str(getattr(entry, "activation_memory", "")),
                "x": 80 + (index % 8) * 180,
                "y": 80 + (index // 8) * 110,
            }
        )
    edges: list[dict[str, str]] = []
    for entry in entries:
        target = _export_node_id(entry, repeated_labels)
        for parent in getattr(entry, "parents", None) or []:
            if parent in node_ids:
                edges.append({"source": str(parent), "target": target})
    width = max((int(node["x"]) for node in nodes), default=0) + 160
    height = max((int(node["y"]) for node in nodes), default=0) + 100
    return {
        "title": getattr(log, "model_class_name", "TorchLens graph"),
        "nodes": nodes,
        "edges": edges,
        "width": width,
        "height": height,
    }


def _scalarize_cell(value: Any) -> Any:
    """Return a scalar-safe representation of a table cell.

    Shared body for ``_parquet_cell`` and ``_tracker_cell``, which apply
    the identical primitive-or-repr coercion for two distinct call sites
    (pyarrow/Parquet column safety and strict tracker table types respectively).

    Parameters
    ----------
    value:
        Original dataframe cell.

    Returns
    -------
    Any
        Primitive value or string representation.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    try:
        import numpy as np
        import pandas as pd

        missing = pd.isna(value)
        if isinstance(missing, bool | np.bool_) and bool(missing):
            return None
    except Exception:
        pass
    return repr(value)
