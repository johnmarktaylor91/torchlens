"""Netron export projection records (lane F14, netron memo B1).

The projection layer turns a finished ``Trace`` into small internal records
(nodes, values, functions, disclosures) that the emitter serializes into
ONNX protobuf-JSON. The records are netron-only in v1 but exporter-neutral
in shape: nothing here knows about camelCase field spellings or attribute
byte encodings, so a future exporter can reuse the same projection.

Granularity builders:

- :func:`project_op` -- the flat leaf-op projection (memo D-04..D-06).
- :func:`project_rolled` -- repeated passes merged by rolled identity with
  the variance rule (memo D-13); lives here beside the op builder.
- module granularity lives in :mod:`._netron_module` (memo D-11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._common import _iter_layers, _repeated_layer_labels
from ._netron_fields import (
    build_leaf_attrs,
    node_doc,
    onnx_elem_type,
    port_names,
    rolled_variance_attrs,
)

__tl_layer__ = "L8"

#: Custom op domain marking these records as non-runnable TorchLens captures.
NETRON_OP_DOMAIN = "ai.torchlens.lossy"

#: Domain for module-call FunctionProtos (memo D-02).
NETRON_MODULE_DOMAIN = "ai.torchlens.module"


@dataclass
class NetronAttr:
    """One curated node attribute (name, wire kind, python value)."""

    name: str
    kind: str  # "s" | "i" | "f" | "ints"
    value: Any


@dataclass
class NetronValueInfo:
    """Type disclosure for one produced value (netron paints it on the edge)."""

    name: str
    elem_type: int | None
    dims: tuple[Any, ...] | None  # ints or symbolic strings; None = unknown rank
    doc: str = ""


@dataclass
class NetronNode:
    """One exported node (leaf op, visible buffer, or module-call site)."""

    name: str
    op_type: str
    domain: str
    inputs: list[str]
    outputs: list[str]
    attrs: list[NetronAttr] = field(default_factory=list)
    doc: str = ""
    input_names: list[str] | None = None  # netron's undocumented port hook


@dataclass
class NetronFunction:
    """One per-call module FunctionProto (memo D-11)."""

    name: str
    formal_inputs: list[str]
    outputs: list[str]
    nodes: list[NetronNode]
    value_infos: list[NetronValueInfo]
    doc: str = ""


@dataclass
class NetronProjection:
    """The full projection an emitter serializes into one artifact."""

    granularity: str
    nodes: list[NetronNode]
    graph_inputs: list[NetronValueInfo]
    graph_outputs: list[NetronValueInfo]
    value_infos: list[NetronValueInfo]
    functions: list[NetronFunction] = field(default_factory=list)
    #: Owning-module address -> hidden buffer names (memo D-10 disclosure).
    hidden_buffers: dict[str, list[str]] = field(default_factory=dict)
    hidden_op_count: int = 0
    omitted_value_count: int = 0
    #: Longest root-graph path in boxes; proxy for netron's rank count (D-12).
    extent_ranks: int = 0
    module_depth: int | None = None
    fallback_reason: str | None = None
    #: node name -> source record (Layer entry or ModuleCall) for sidecars.
    node_sources: dict[str, Any] = field(default_factory=dict)

    @property
    def edge_count(self) -> int:
        """Total input references across root nodes and function bodies."""

        total = sum(len(node.inputs) for node in self.nodes)
        for function in self.functions:
            total += sum(len(node.inputs) for node in function.nodes)
        return total


@dataclass
class _Classified:
    """Shared per-trace classification consumed by every granularity."""

    entries: list[Any]
    repeated: set[str]
    node_ids: dict[int, str]  # id(entry) -> exported node id
    by_ref: dict[str, Any]  # any parent spelling -> entry
    input_entries: list[Any]
    output_entries: list[Any]
    hidden: set[str]  # node ids hidden by the buffer policy + chain rule
    hidden_buffers: dict[str, list[str]]
    hidden_op_count: int


#: Buffer names filtered by the ``"meaningful"`` policy. Mirrors the
#: canonical visualization constant (``_render_common._NOISE_BUFFER_NAMES``);
#: duplicated here because export leaves never import viz-private emitters.
_NOISE_BUFFER_NAMES = frozenset({"running_mean", "running_var", "num_batches_tracked"})


def _entry_node_id(entry: Any, repeated: set[str]) -> str:
    """Return the exported node id for one layer pass (pass-qualified iff repeated)."""

    layer_label = str(getattr(entry, "layer_label", ""))
    if layer_label in repeated:
        return str(getattr(entry, "label", layer_label))
    return layer_label


def _is_noise_buffer(entry: Any) -> bool:
    """Return whether the buffer's address ends in a ``meaningful``-hidden name."""

    address = getattr(entry, "address", None)
    if not address:
        return False
    return str(address).rsplit(".", 1)[-1] in _NOISE_BUFFER_NAMES


def _buffer_hidden(entry: Any, policy: str) -> bool:
    """Apply the tri-state buffer visibility policy to one buffer entry."""

    if policy == "always":
        return False
    if policy == "never":
        return True
    return _is_noise_buffer(entry)


def _owning_module(entry: Any) -> str:
    """Return the innermost containing module-call address, or ``""`` for root."""

    stack = getattr(entry, "module_call_stack", None) or ()
    return str(stack[-1]) if stack else ""


def _hide_counter_chains(classified: _Classified, policy: str) -> None:
    """Hide ops whose every dataflow neighbor is a hidden buffer or hidden op.

    The counter-update chains (the zero-dim ``add`` ops between
    ``num_batches_tracked`` reads and writes) are ops, not buffers, so the
    buffer policy alone misses them (memo D-10). The rule is a fixpoint so
    multi-op chains collapse too; an op keeping any visible neighbor stays.
    """

    if policy == "always":
        return
    changed = True
    while changed:
        changed = False
        for entry in classified.entries:
            node_id = classified.node_ids[id(entry)]
            if node_id in classified.hidden:
                continue
            if getattr(entry, "is_buffer", False):
                continue
            if getattr(entry, "is_input", False) or getattr(entry, "is_output", False):
                continue
            neighbors = list(getattr(entry, "parents", ()) or ()) + list(
                getattr(entry, "children", ()) or ()
            )
            if not neighbors:
                continue
            resolved = [classified.by_ref.get(str(ref)) for ref in neighbors]
            ids = [classified.node_ids[id(e)] for e in resolved if e is not None]
            if ids and all(node_id_ in classified.hidden for node_id_ in ids):
                classified.hidden.add(node_id)
                classified.hidden_op_count += 1
                changed = True


def classify(log: Any, show_buffers: str) -> _Classified:
    """Classify trace entries and apply the buffer policy + chain rule."""

    entries = _iter_layers(log)
    repeated = _repeated_layer_labels(entries)
    node_ids: dict[int, str] = {}
    by_ref: dict[str, Any] = {}
    for entry in entries:
        node_id = _entry_node_id(entry, repeated)
        node_ids[id(entry)] = node_id
        by_ref[str(getattr(entry, "label", node_id))] = entry
        layer_label = str(getattr(entry, "layer_label", ""))
        if layer_label not in repeated:
            by_ref[layer_label] = entry
    hidden: set[str] = set()
    hidden_buffers: dict[str, list[str]] = {}
    for entry in entries:
        if getattr(entry, "is_buffer", False) and _buffer_hidden(entry, show_buffers):
            node_id = node_ids[id(entry)]
            hidden.add(node_id)
            hidden_buffers.setdefault(_owning_module(entry), []).append(node_id)
    classified = _Classified(
        entries=entries,
        repeated=repeated,
        node_ids=node_ids,
        by_ref=by_ref,
        input_entries=[e for e in entries if getattr(e, "is_input", False)],
        output_entries=[e for e in entries if getattr(e, "is_output", False)],
        hidden=hidden,
        hidden_buffers=hidden_buffers,
        hidden_op_count=0,
    )
    _hide_counter_chains(classified, show_buffers)
    return classified


def value_info_for(entry: Any, name: str) -> NetronValueInfo:
    """Build the type disclosure for one entry's produced value (memo D-05)."""

    shape = getattr(entry, "shape", None)
    dims: tuple[Any, ...] | None = None if shape is None else tuple(shape)
    elem_type = onnx_elem_type(getattr(entry, "dtype", None))
    doc = f"{getattr(entry, 'layer_type', '')} output"
    return NetronValueInfo(name=name, elem_type=elem_type, dims=dims, doc=doc)


def _resolve_intervened_ids(log: Any, classified: _Classified) -> set[str]:
    """Return node ids of ops marked by the intervention audit (compo row C2)."""

    audit = getattr(log, "intervention_audit", None) or []
    if not audit:
        return set()
    site_keys: set[str] = set()
    for row in audit:
        sites = row.get("sites", []) if isinstance(row, dict) else []
        for site in sites:
            raw = site.get("site_key", "") if isinstance(site, dict) else ""
            # The audit spells site keys as a repr'd tuple of strings.
            for piece in str(raw).strip("()").split(","):
                cleaned = piece.strip().strip("'\"")
                if cleaned:
                    site_keys.add(cleaned)
    marked: set[str] = set()
    for entry in classified.entries:
        ops = getattr(entry, "ops", None) or [entry]
        for op in ops:
            key = getattr(op, "site_key", None)
            replaced = bool(getattr(op, "intervention_replaced", False))
            if replaced or (key is not None and str(key) in site_keys):
                marked.add(classified.node_ids[id(entry)])
                break
    return marked


def _visible_inputs(entry: Any, classified: _Classified, projection: NetronProjection) -> list[str]:
    """Map an entry's parents to visible value names, counting omissions."""

    inputs: list[str] = []
    for parent in getattr(entry, "parents", ()) or ():
        parent_entry = classified.by_ref.get(str(parent))
        if parent_entry is None:
            projection.omitted_value_count += 1
            continue
        parent_id = classified.node_ids[id(parent_entry)]
        if parent_id in classified.hidden:
            projection.omitted_value_count += 1
            continue
        if getattr(parent_entry, "is_output", False):
            projection.omitted_value_count += 1
            continue
        inputs.append(parent_id)
    return inputs


def compute_extent_ranks(nodes: list[NetronNode], io_boxes: int) -> int:
    """Estimate netron's rank count as the longest path over root boxes.

    Netron lays ranks top-to-bottom, one per topological level, at roughly
    55 px per rank (memo D-12), so the longest dataflow chain over the root
    graph's boxes -- graph I/O draw as boxes too -- is the extent proxy the
    budget warning checks.
    """

    producers: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for output in node.outputs:
            producers[output] = index
    depth: dict[int, int] = {}

    def _depth(index: int) -> int:
        """Longest-chain depth of one node, memoized with a cycle guard."""

        if index in depth:
            return depth[index]
        depth[index] = 1  # cycle guard: self-referential chains count once
        best = 0
        for parent in nodes[index].inputs:
            producer = producers.get(parent)
            if producer is not None and producer != index:
                best = max(best, _depth(producer))
        depth[index] = best + 1
        return depth[index]

    longest = max((_depth(i) for i in range(len(nodes))), default=0)
    return longest + min(io_boxes, 2)


def project_op(log: Any, show_buffers: str) -> NetronProjection:
    """Build the flat leaf-op projection (granularity ``"op"``)."""

    classified = classify(log, show_buffers)
    projection = NetronProjection(
        granularity="op",
        nodes=[],
        graph_inputs=[],
        graph_outputs=[],
        value_infos=[],
        hidden_buffers=classified.hidden_buffers,
        hidden_op_count=classified.hidden_op_count,
    )
    intervened = _resolve_intervened_ids(log, classified)
    for entry in classified.entries:
        node_id = classified.node_ids[id(entry)]
        if node_id in classified.hidden:
            continue
        if getattr(entry, "is_input", False):
            projection.graph_inputs.append(value_info_for(entry, node_id))
            projection.node_sources[node_id] = entry
            continue
        if getattr(entry, "is_output", False):
            parents = list(getattr(entry, "parents", ()) or ())
            if not parents:
                continue
            producer_entry = classified.by_ref.get(str(parents[0]))
            if producer_entry is None:
                continue
            producer_id = classified.node_ids[id(producer_entry)]
            projection.graph_outputs.append(value_info_for(producer_entry, producer_id))
            continue
        inputs = _visible_inputs(entry, classified, projection)
        node = NetronNode(
            name=node_id,
            op_type=str(getattr(entry, "layer_type", None) or getattr(entry, "func_name", "")),
            domain=NETRON_OP_DOMAIN,
            inputs=inputs,
            outputs=[node_id],
            attrs=build_leaf_attrs(entry, intervened=node_id in intervened),
            doc=node_doc(entry),
            input_names=port_names(entry, inputs),
        )
        projection.nodes.append(node)
        projection.node_sources[node_id] = entry
        projection.value_infos.append(value_info_for(entry, node_id))
    _strip_io_value_infos(projection)
    io_boxes = len(projection.graph_inputs) + len(projection.graph_outputs)
    projection.extent_ranks = compute_extent_ranks(projection.nodes, io_boxes)
    return projection


def _strip_io_value_infos(projection: NetronProjection) -> None:
    """Drop valueInfo rows duplicating graph I/O declarations (checker rule)."""

    io_names = {info.name for info in projection.graph_inputs}
    io_names.update(info.name for info in projection.graph_outputs)
    projection.value_infos = [info for info in projection.value_infos if info.name not in io_names]


def _rolled_groups(classified: _Classified) -> dict[str, list[Any]]:
    """Group op entries by rolled identity (``layer_label``), insertion-ordered."""

    groups: dict[str, list[Any]] = {}
    for entry in classified.entries:
        if getattr(entry, "is_input", False) or getattr(entry, "is_output", False):
            continue
        if classified.node_ids[id(entry)] in classified.hidden:
            continue
        groups.setdefault(str(getattr(entry, "layer_label", "")), []).append(entry)
    return groups


def _rolled_group_edges(
    rolled_id: str,
    members: list[Any],
    classified: _Classified,
    projection: NetronProjection,
    rolled_index: tuple[dict[str, str], dict[str, int]],
) -> tuple[list[str], list[str]]:
    """Merge one rolled group's edges into forward inputs + feedback sources.

    A merged edge whose producer first executes at or after this node is
    recurrent feedback: drawn literally it forms a dataflow cycle
    ``check_model(full_check=True)`` rejects (probed against onnx 1.22), so
    it is DISCLOSED via the ``recurrence`` attribute instead of an edge.
    """

    rolled_of, first_seen = rolled_index
    inputs: list[str] = []
    feedback_sources: list[str] = []
    for member in members:
        for parent in _visible_inputs(member, classified, projection):
            parent_rolled = rolled_of.get(parent, parent)
            if first_seen.get(parent_rolled, -1) >= first_seen[rolled_id]:
                if parent_rolled not in feedback_sources:
                    feedback_sources.append(parent_rolled)
                continue
            if parent_rolled not in inputs:
                inputs.append(parent_rolled)
    return inputs, feedback_sources


def project_rolled(log: Any, show_buffers: str) -> NetronProjection:
    """Build the rolled projection: passes merged, variance-gated types (D-13)."""

    classified = classify(log, show_buffers)
    projection = NetronProjection(
        granularity="rolled",
        nodes=[],
        graph_inputs=[],
        graph_outputs=[],
        value_infos=[],
        hidden_buffers=classified.hidden_buffers,
        hidden_op_count=classified.hidden_op_count,
    )
    intervened = _resolve_intervened_ids(log, classified)
    rolled_of: dict[str, str] = {}
    for entry in classified.entries:
        rolled_of[classified.node_ids[id(entry)]] = str(getattr(entry, "layer_label", ""))
    for entry in classified.input_entries:
        node_id = classified.node_ids[id(entry)]
        projection.graph_inputs.append(value_info_for(entry, node_id))
        rolled_of[node_id] = node_id
        projection.node_sources[node_id] = entry
    groups = _rolled_groups(classified)
    first_seen = {rolled_id: index for index, rolled_id in enumerate(groups)}
    for rolled_id, members in groups.items():
        inputs, feedback_sources = _rolled_group_edges(
            rolled_id, members, classified, projection, (rolled_of, first_seen)
        )
        representative = members[0]
        attrs = build_leaf_attrs(
            representative,
            intervened=any(classified.node_ids[id(member)] in intervened for member in members),
            rolled_members=members,
        )
        attrs.extend(rolled_variance_attrs(members, feedback_sources=feedback_sources))
        node = NetronNode(
            name=rolled_id,
            op_type=str(
                getattr(representative, "layer_type", None)
                or getattr(representative, "func_name", "")
            ),
            domain=NETRON_OP_DOMAIN,
            inputs=inputs,
            outputs=[rolled_id],
            attrs=attrs,
            doc=node_doc(representative),
        )
        projection.nodes.append(node)
        projection.node_sources[rolled_id] = representative
        shapes = {tuple(getattr(m, "shape", ()) or ()) for m in members}
        dtypes = {str(getattr(m, "dtype", None)) for m in members}
        if len(shapes) == 1 and len(dtypes) == 1:
            projection.value_infos.append(value_info_for(representative, rolled_id))
    for entry in classified.output_entries:
        parents = list(getattr(entry, "parents", ()) or ())
        if not parents:
            continue
        producer_entry = classified.by_ref.get(str(parents[0]))
        if producer_entry is None:
            continue
        producer_rolled = rolled_of[classified.node_ids[id(producer_entry)]]
        projection.graph_outputs.append(value_info_for(producer_entry, producer_rolled))
    _strip_io_value_infos(projection)
    io_boxes = len(projection.graph_inputs) + len(projection.graph_outputs)
    projection.extent_ranks = compute_extent_ranks(projection.nodes, io_boxes)
    return projection
