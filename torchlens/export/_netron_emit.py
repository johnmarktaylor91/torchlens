"""ModelProto JSON emission for the netron export (lane F14, memo s4).

Serializes a :class:`~._netron_records.NetronProjection` into camelCase
protobuf-JSON accepted by netron's ProtoReader door and by a strict
``google.protobuf.json_format.Parse`` into ``onnx.ModelProto``:

- irVersion 10 (the named-ports hook uses ``NodeProto.metadataProps`` and
  drill-down shapes use ``FunctionProto.valueInfo`` -- IR-10 fields);
- integer ``elemType``/``dimValue`` always (netron's decoder runs a bare
  ``Number()`` on elemType; a string enum name would NaN);
- compact separators above ~1 MB, indent=2 below (human-diffable goldens);
- deterministic bytes: fixed key order, no timestamps, no randomness.
"""

from __future__ import annotations

import base64
import json as _json
from typing import Any

from ._netron_fields import onnx_elem_type  # re-exported for the records layer
from ._netron_records import (
    NETRON_MODULE_DOMAIN,
    NETRON_OP_DOMAIN,
    NetronAttr,
    NetronFunction,
    NetronNode,
    NetronProjection,
    NetronValueInfo,
)

__all__ = ["emit_model_json", "onnx_elem_type"]

__tl_layer__ = "L8"

#: Declared IR version for enriched exports (memo D-03).
NETRON_IR_VERSION = 10

#: Artifact schema stamp (memo D-19); bump on breaking payload changes.
NETRON_SCHEMA_VERSION = "2"

#: Byte size above which the JSON switches to compact separators.
_COMPACT_THRESHOLD_BYTES = 1_000_000


def _encode_attr(attr: NetronAttr) -> dict[str, Any]:
    """Encode one curated attribute into protobuf-JSON AttributeProto form."""

    if attr.kind == "s":
        return {
            "name": attr.name,
            "type": "STRING",
            "s": base64.b64encode(str(attr.value).encode()).decode(),
        }
    if attr.kind == "i":
        return {"name": attr.name, "type": "INT", "i": str(int(attr.value))}
    if attr.kind == "f":
        return {"name": attr.name, "type": "FLOAT", "f": float(attr.value)}
    return {
        "name": attr.name,
        "type": "INTS",
        "ints": [str(int(value)) for value in attr.value],
    }


def _encode_type(info: NetronValueInfo) -> dict[str, Any] | None:
    """Encode a value's TypeProto; ``None`` when nothing honest can be said.

    The dtype/shape rules (memo D-05): shape is always emitted when the rank
    is known (empty ``dim`` for rank-0), integer ``dimValue`` for concrete
    axes, ``dimParam`` only for genuinely unknown axes, and an unknown dtype
    OMITS the type entirely -- elemType 0 is UNDEFINED pretending to be
    information.
    """

    if info.elem_type is None:
        return None
    tensor_type: dict[str, Any] = {"elemType": info.elem_type}
    if info.dims is not None:
        dims = []
        for dim in info.dims:
            if isinstance(dim, int) and not isinstance(dim, bool):
                dims.append({"dimValue": str(dim)})
            else:
                dims.append({"dimParam": str(dim)})
        tensor_type["shape"] = {"dim": dims} if dims else {}
    return {"tensorType": tensor_type}


def _encode_value_info(info: NetronValueInfo) -> dict[str, Any]:
    """Encode one ValueInfoProto row (typed edge label / graph I/O)."""

    row: dict[str, Any] = {"name": info.name}
    encoded = _encode_type(info)
    if encoded is not None:
        row["type"] = encoded
    if info.doc:
        row["docString"] = info.doc
    return row


def _encode_node(node: NetronNode) -> dict[str, Any]:
    """Encode one NodeProto row, including the ``input_names`` port hook."""

    row: dict[str, Any] = {
        "name": node.name,
        "opType": node.op_type,
        "domain": node.domain,
        "input": list(node.inputs),
        "output": list(node.outputs),
    }
    if node.doc:
        row["docString"] = node.doc
    if node.attrs:
        row["attribute"] = [_encode_attr(attr) for attr in node.attrs]
    if node.input_names and len(node.input_names) == len(node.inputs):
        # netron's undocumented hook (onnx.js:313): a node metadataProps key
        # ``input_names`` holding a well-formed python list literal names the
        # input ports; it fires only for op types netron cannot resolve, so
        # it works exactly because we stay in the custom domain (memo D-16).
        row["metadataProps"] = [
            {"key": "input_names", "value": repr([str(n) for n in node.input_names])}
        ]
    return row


def _encode_function(function: NetronFunction) -> dict[str, Any]:
    """Encode one FunctionProto with its own valueInfo (drill-down shapes)."""

    row: dict[str, Any] = {
        "name": function.name,
        "domain": NETRON_MODULE_DOMAIN,
        "input": list(function.formal_inputs),
        "output": list(function.outputs),
        "node": [_encode_node(node) for node in function.nodes],
        "opsetImport": _opset_imports(function.nodes),
    }
    if function.doc:
        row["docString"] = function.doc
    if function.value_infos:
        row["valueInfo"] = [_encode_value_info(info) for info in function.value_infos]
    return row


def _opset_imports(nodes: list[NetronNode]) -> list[dict[str, Any]]:
    """Import every domain the node list uses (memo s4: root AND functions)."""

    domains = {NETRON_OP_DOMAIN}
    domains.update(node.domain for node in nodes)
    return [{"domain": domain, "version": 1} for domain in sorted(domains)]


def emit_model_json(
    projection: NetronProjection,
    *,
    graph_name: str,
    disclaimer: str,
    metadata_props: list[tuple[str, str]],
) -> str:
    """Serialize one projection into deterministic ModelProto JSON text.

    Parameters
    ----------
    projection:
        Finished projection from the records layer.
    graph_name:
        Root graph name (the traced model's class name).
    disclaimer:
        Honesty disclaimer riding ``docString`` at model and graph level.
    metadata_props:
        Ordered model-level metadata rows (memo D-19).

    Returns
    -------
    str
        JSON text: indent=2 below ~1 MB, compact separators above.
    """

    graph: dict[str, Any] = {
        "name": graph_name,
        "docString": disclaimer,
        "node": [_encode_node(node) for node in projection.nodes],
        "input": [_encode_value_info(info) for info in projection.graph_inputs],
        "output": [_encode_value_info(info) for info in projection.graph_outputs],
    }
    if projection.value_infos:
        graph["valueInfo"] = [_encode_value_info(info) for info in projection.value_infos]
    all_nodes = list(projection.nodes)
    for function in projection.functions:
        all_nodes.extend(function.nodes)
    payload: dict[str, Any] = {
        "irVersion": NETRON_IR_VERSION,
        "producerName": "torchlens",
        "producerVersion": _torchlens_version(),
        "docString": disclaimer,
        "opsetImport": _opset_imports(all_nodes),
        "metadataProps": [{"key": key, "value": value} for key, value in metadata_props],
        "graph": graph,
    }
    if projection.functions:
        payload["functions"] = [_encode_function(function) for function in projection.functions]
    compact = _json.dumps(payload, separators=(",", ":"))
    if len(compact) > _COMPACT_THRESHOLD_BYTES:
        return compact
    return _json.dumps(payload, indent=2)


def _torchlens_version() -> str:
    """Return the installed TorchLens version string (stable per install)."""

    try:
        from .. import __version__

        return str(__version__)
    except (ImportError, AttributeError):
        return ""
