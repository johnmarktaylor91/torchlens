"""Foreign graph-viewer export targets (bridge tier; C01 item 18).

Netron and Model Explorer writers: bridge-tier members of the export-target
registry (their shapes come from foreign peers). Lanes F14/F15 grow these
files without touching the package facade.
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any

from .._capture_honesty import capture_honesty_facts
from ..utils.display import atomic_write_text
from ._common import _export_node_id, _iter_layers, _repeated_layer_labels, _static_graph_data

__tl_layer__ = "L8"


def model_explorer(log: Any, path: str | Path) -> Path:
    """Export a JSON graph using Google Model Explorer's graph schema.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination JSON path.

    Returns
    -------
    Path
        Written JSON path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _static_graph_data(log)
    incoming_edges: dict[str, list[dict[str, str]]] = {
        str(node["id"]): [] for node in data["nodes"]
    }
    for edge in data["edges"]:
        incoming_edges[str(edge["target"])].append({"sourceNodeId": str(edge["source"])})
    label = str(getattr(log, "trace_label", None) or getattr(log, "model_class_name", "model"))
    payload = {
        "schema": "torchlens.model_explorer.v2",
        "disclaimer": (
            "TorchLens graph-collection JSON for Google Model Explorer; a data export of the "
            "captured graph, not a runnable model. The top-level label/graphs shape matches "
            "Model Explorer's file-ingest contract (pinned against ai-edge-model-explorer "
            "0.1.32); acceptance by future external releases is not guaranteed."
        ),
        # Model Explorer's JSON ingest requires BOTH top-level keys label and
        # graphs to treat the file as a graph collection; without label the
        # app refuses with "Unsupported JSON format". Extra top-level keys
        # (schema, disclaimer, capture honesty) are tolerated by the pinned
        # ingest contract.
        "torchlens_capture_honesty": capture_honesty_facts(log),
        "label": label,
        "graphs": [
            {
                "id": str(
                    getattr(log, "trace_label", None) or getattr(log, "model_class_name", "model")
                ),
                "nodes": [
                    {
                        "id": node["id"],
                        "label": node["label"],
                        "namespace": node["type"],
                        "attrs": [
                            {"key": "shape", "value": node["shape"]},
                            {"key": "memory", "value": node["memory"]},
                        ],
                        "incomingEdges": incoming_edges[str(node["id"])],
                    }
                    for node in data["nodes"]
                ],
            }
        ],
    }
    atomic_write_text(destination, _json.dumps(payload, indent=2))
    return destination


#: Disclaimer embedded in the Netron export's model and graph doc strings.
NETRON_DISCLAIMER = (
    "TorchLens lossy graph export: not a runnable ONNX model; graph inspection "
    "only. Ops keep their captured TorchLens names under the ai.torchlens.lossy "
    "domain and carry no standard-ONNX execution semantics."
)


def netron(log: Any, path: str | Path) -> Path:
    """Export a lossy ONNX ``ModelProto`` JSON graph that Netron can open.

    The payload is valid ONNX protobuf JSON (camelCase field names, parseable
    into ``onnx.ModelProto``), which is the exact acceptance contract of
    Netron's ONNX JSON reader. It is intentionally NOT a runnable model: ops
    keep their captured TorchLens names under the custom
    ``ai.torchlens.lossy`` operator domain, only names, edges, and output
    shapes are preserved, and the disclaimer rides ``docString`` and
    ``metadataProps``.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination JSON path.

    Returns
    -------
    Path
        Written JSON path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = _iter_layers(log)
    repeated_labels = _repeated_layer_labels(entries)
    nodes = []
    for layer in entries:
        node_id = _export_node_id(layer, repeated_labels)
        shape = list(getattr(layer, "shape", ()) or ())
        node: dict[str, Any] = {
            "name": node_id,
            "opType": str(getattr(layer, "layer_type", None) or getattr(layer, "func_name", "")),
            "domain": "ai.torchlens.lossy",
            "input": [str(parent) for parent in (getattr(layer, "parents", []) or [])],
            "output": [node_id],
        }
        if shape and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
            node["attribute"] = [{"name": "shape", "type": "INTS", "ints": shape}]
        nodes.append(node)
    payload = {
        "irVersion": 8,
        "producerName": "torchlens",
        "docString": NETRON_DISCLAIMER,
        "opsetImport": [{"domain": "ai.torchlens.lossy", "version": 1}],
        "metadataProps": [
            {"key": "torchlens.lossy_export", "value": "true"},
            {"key": "torchlens.runnable", "value": "false"},
            # Honesty facts as a JSON string value: metadataProps is the one
            # slot valid ONNX protobuf JSON offers for free-form metadata.
            {
                "key": "torchlens.capture_honesty",
                "value": _json.dumps(capture_honesty_facts(log)),
            },
        ],
        "graph": {
            "name": str(getattr(log, "model_class_name", "TorchLens graph")),
            "docString": NETRON_DISCLAIMER,
            "node": nodes,
        },
    }
    atomic_write_text(destination, _json.dumps(payload, indent=2))
    return destination
