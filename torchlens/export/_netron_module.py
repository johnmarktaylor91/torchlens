"""Module-granularity netron projection (lane F14, netron memo D-11/D-12).

The model's module hierarchy exports as nested per-CALL FunctionProtos: the
root graph shows the calls at the chosen depth (interim default 1 per the
memo's DISSENT D1 ruling), every deeper call nests recursively as its own
FunctionProto, dissolved levels above the depth flatten into the root, and
single-node calls inline with their FunctionProto DELETED (an empty or
one-box drill-down room is noise, not navigation).

Per-call keying makes function-call cycles structurally impossible; the
explicit DAG check stays as the named assertion with fail-soft to the valid
op projection, because a cycle is not a degraded render, it is a dead file.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from ._netron_fields import _strip_single_pass, build_leaf_attrs, node_doc, port_names
from ._netron_records import (
    NETRON_MODULE_DOMAIN,
    NETRON_OP_DOMAIN,
    NetronAttr,
    NetronFunction,
    NetronNode,
    NetronProjection,
    _resolve_intervened_ids,
    _strip_io_value_infos,
    _visible_inputs,
    classify,
    compute_extent_ranks,
    value_info_for,
)

__tl_layer__ = "L8"

#: Root-graph scope key (module calls use their pass-qualified call label).
_ROOT_SCOPE = ""


class _CallTree(NamedTuple):
    """Shared per-projection call-tree context threaded through the phases."""

    levels: dict[str, int]
    records: dict[str, Any]
    hidden_buffers: dict[str, list[str]]


@dataclass
class _Scope:
    """One retained module call: a future FunctionProto and its call site."""

    label: str
    call: Any
    parent: str  # parent scope key (retained ancestor or root)
    body: list[NetronNode] = field(default_factory=list)
    body_sources: list[Any] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)  # outer values consumed
    exports: list[str] = field(default_factory=list)  # values leaving the call
    hidden_note: str = ""  # per-call hidden-buffer disclosure (memo D-10)


def _call_levels(module_calls: Any) -> tuple[dict[str, int], dict[str, Any]]:
    """Return call label -> tree level (root = 0) and label -> record."""

    records: dict[str, Any] = {}
    # noqa rationale: module_calls is the Trace ACCESSOR, not a dict -- its
    # __iter__ yields ModuleCall records, so .keys() is load-bearing.
    for label in module_calls.keys():  # noqa: SIM118
        records[str(label)] = module_calls[label]
    levels: dict[str, int] = {}

    def _level(label: str) -> int:
        """Memoized call-tree depth (root = 0)."""

        if label in levels:
            return levels[label]
        parent = records[label].call_parent_label
        levels[label] = 0 if parent is None else _level(str(parent)) + 1
        return levels[label]

    for label in records:
        _level(label)
    return levels, records


def _retained_scopes(
    levels: dict[str, int], records: dict[str, Any], depth: int
) -> dict[str, _Scope]:
    """Build the retained-scope table: every call at level >= depth."""

    scopes: dict[str, _Scope] = {}
    for label, level in levels.items():
        if level < depth:
            continue
        parent_label = str(records[label].call_parent_label)
        parent = parent_label if levels.get(parent_label, 0) >= depth else _ROOT_SCOPE
        scopes[label] = _Scope(label=label, call=records[label], parent=parent)
    return scopes


def _entry_scope(entry: Any, scopes: dict[str, _Scope]) -> str:
    """Return the innermost retained scope containing one op entry."""

    for call_label in reversed(list(getattr(entry, "module_call_stack", ()) or ())):
        if str(call_label) in scopes:
            return str(call_label)
    return _ROOT_SCOPE


def _display_name(label: str, records: dict[str, Any]) -> str:
    """Return the per-call display name, pass-qualified only when repeated."""

    address = label.rsplit(":", 1)[0]
    siblings = sum(1 for other in records if other.rsplit(":", 1)[0] == address)
    return label if siblings > 1 else _strip_single_pass(label)


def _inline_single_node_scopes(
    scopes: dict[str, _Scope], root_body: list[NetronNode], root_sources: list[Any]
) -> int:
    """Inline calls whose body is one node; DELETE their FunctionProto (D-11).

    Runs bottom-up to a fixpoint so wrapper chains collapse: a call whose
    only child inlined to a single op becomes single-node itself.
    """

    inlined = 0
    changed = True
    while changed:
        changed = False
        for label in list(scopes):
            scope = scopes[label]
            has_nested_call_sites = any(other.parent == label for other in scopes.values())
            if has_nested_call_sites or len(scope.body) != 1:
                continue
            target_body, target_sources = (
                (root_body, root_sources)
                if scope.parent == _ROOT_SCOPE
                else (scopes[scope.parent].body, scopes[scope.parent].body_sources)
            )
            target_body.append(scope.body[0])
            target_sources.append(scope.body_sources[0])
            del scopes[label]
            inlined += 1
            changed = True
    return inlined


def _scope_path(scope_key: str, scopes: dict[str, _Scope]) -> list[str]:
    """Return the scope chain from root (exclusive) down to ``scope_key``."""

    path: list[str] = []
    cursor = scope_key
    while cursor != _ROOT_SCOPE:
        path.append(cursor)
        cursor = scopes[cursor].parent
    path.reverse()
    return path


def _route_one_value(
    scopes: dict[str, _Scope], producer_scope: dict[str, str], scope_key: str, value: str
) -> None:
    """Route one consumption through exports up and imports down (LCA rule)."""

    source = producer_scope.get(value, _ROOT_SCOPE)
    if source == scope_key:
        return
    source_path = _scope_path(source, scopes)
    consumer_path = _scope_path(scope_key, scopes)
    common = 0
    while (
        common < len(source_path)
        and common < len(consumer_path)
        and source_path[common] == consumer_path[common]
    ):
        common += 1
    for exporter in reversed(source_path[common:]):
        exporter_scope = scopes[exporter]
        if value not in exporter_scope.exports:
            exporter_scope.exports.append(value)
    for importer in consumer_path[common:]:
        importer_scope = scopes[importer]
        if value not in importer_scope.imports:
            importer_scope.imports.append(value)


def _route_cross_scope_values(
    scopes: dict[str, _Scope],
    root_body: list[NetronNode],
    producer_scope: dict[str, str],
    graph_output_names: list[str],
) -> None:
    """Thread every cross-scope value through function imports and exports.

    For a value produced in scope ``S`` and consumed in scope ``T`` the
    lowest common ancestor rule applies: the value exports upward from ``S``
    to the common ancestor and imports downward into ``T``; a scope never
    both imports and exports the same value.
    """

    for scope in scopes.values():
        for node in scope.body:
            for value in node.inputs:
                _route_one_value(scopes, producer_scope, scope.label, value)
    for node in root_body:
        for value in node.inputs:
            _route_one_value(scopes, producer_scope, _ROOT_SCOPE, value)
    for value in graph_output_names:
        _route_one_value(scopes, producer_scope, _ROOT_SCOPE, value)


def _formal_names(scope: _Scope) -> dict[str, str]:
    """Choose signature-style formal input names for one function (D-11).

    ``forward_arg_names`` supplies real signature names when they align
    one-to-one with the imported values; the fallback is ``x``, ``x2``, ...
    -- ports are named like signatures, never like wires.
    """

    arg_names = [
        str(name)
        for name in (getattr(scope.call, "forward_arg_names", None) or [])
        if str(name) and not str(name).isdigit()
    ]
    formals: dict[str, str] = {}
    used: set[str] = set()
    for index, value in enumerate(scope.imports):
        if len(arg_names) == len(scope.imports):
            candidate = arg_names[index]
        else:
            candidate = "x" if index == 0 else f"x{index + 1}"
        while candidate in used:
            candidate = f"{candidate}_"
        used.add(candidate)
        formals[value] = candidate
    return formals


def _module_call_attrs(scope: _Scope, display: str) -> list[NetronAttr]:
    """Curated inclusive-metric attributes for one module-call node (D-17)."""

    call = scope.call
    attrs: list[NetronAttr] = []
    class_name = str(
        getattr(call, "class_name", None) or getattr(call, "class_qualname", None) or ""
    )
    if class_name:
        attrs.append(NetronAttr("module_class", "s", class_name))
    attrs.append(NetronAttr("module_path", "s", display))
    duration = getattr(call, "forward_duration", None)
    if duration is not None:
        with contextlib.suppress(TypeError, ValueError):
            attrs.append(
                NetronAttr("observed_duration_inclusive_us", "i", int(float(duration) * 1e6))
            )
    ops_inside = getattr(call, "num_ops", None)
    if ops_inside:
        attrs.append(NetronAttr("ops_inside", "i", int(ops_inside)))
    param_bytes = getattr(call, "internal_param_memory", None)
    if param_bytes is not None:
        try:
            total = int(float(param_bytes))
        except (TypeError, ValueError):
            total = 0
        if total:
            attrs.append(NetronAttr("param_bytes_inclusive", "i", total))
    return attrs


class _RootBody(NamedTuple):
    """The root graph's own nodes and their source records."""

    nodes: list[NetronNode]
    sources: list[Any]


def _place_entries(
    log: Any,
    classified: Any,
    scopes: dict[str, _Scope],
    projection: NetronProjection,
) -> tuple[_RootBody, dict[str, str], dict[str, Any]]:
    """Build every visible op node and place it in its owning scope."""

    intervened = _resolve_intervened_ids(log, classified)
    root_body: list[NetronNode] = []
    root_sources: list[Any] = []
    producer_scope: dict[str, str] = {}
    value_infos: dict[str, Any] = {}
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
            producer = classified.by_ref.get(str(parents[0])) if parents else None
            if producer is not None:
                producer_id = classified.node_ids[id(producer)]
                projection.graph_outputs.append(value_info_for(producer, producer_id))
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
        scope_key = _entry_scope(entry, scopes)
        if scope_key == _ROOT_SCOPE:
            root_body.append(node)
            root_sources.append(entry)
        else:
            scopes[scope_key].body.append(node)
            scopes[scope_key].body_sources.append(entry)
        producer_scope[node_id] = scope_key
        projection.node_sources[node_id] = entry
        value_infos[node_id] = value_info_for(entry, node_id)
    return _RootBody(root_body, root_sources), producer_scope, value_infos


def _settle_scopes(
    scopes: dict[str, _Scope],
    call_tree: _CallTree,
    root: _RootBody,
    producer_scope: dict[str, str],
    show_buffers: str,
) -> None:
    """Drop empty scopes, inline single-node calls, refresh producers/notes."""

    records = call_tree.records
    root_body, root_sources = root.nodes, root.sources
    hidden_buffers = call_tree.hidden_buffers

    # An empty FunctionProto still draws a clickable icon onto a blank graph.
    for label in [key for key, scope in scopes.items() if not scope.body]:
        if not any(other.parent == label for other in scopes.values()):
            del scopes[label]
    _inline_single_node_scopes(scopes, root_body, root_sources)
    for node in root_body:
        producer_scope[node.outputs[0]] = _ROOT_SCOPE
    hidden_by_scope: dict[str, list[str]] = {}
    for owner, buffer_names in hidden_buffers.items():
        cursor = owner
        while cursor and cursor not in scopes:
            parent = records.get(cursor)
            cursor = str(parent.call_parent_label) if parent is not None else ""
            if cursor == "None":
                cursor = ""
        hidden_by_scope.setdefault(cursor, []).extend(buffer_names)
    for scope in scopes.values():
        for node in scope.body:
            producer_scope[node.outputs[0]] = scope.label
        names = hidden_by_scope.get(scope.label, [])
        if names:
            scope.hidden_note = (
                f"{len(names)} buffer node(s) hidden by show_buffers={show_buffers!r}: "
                + ", ".join(sorted(names))
            )


def _assemble_functions(
    scopes: dict[str, _Scope],
    call_tree: _CallTree,
    root_body: list[NetronNode],
    projection: NetronProjection,
    value_infos: dict[str, Any],
) -> None:
    """Rename bodies to formals, mint call-site nodes, build FunctionProtos."""

    levels, records = call_tree.levels, call_tree.records

    call_display = {label: _display_name(label, records) for label in scopes}
    built_functions: list[tuple[str, NetronFunction]] = []
    # Children first: a nested call site must land in its parent's body
    # before the parent's FunctionProto is assembled.
    for scope in sorted(scopes.values(), key=lambda s: -levels[s.label]):
        formals = _formal_names(scope)
        renamed_body: list[NetronNode] = []
        for node in scope.body:
            node.inputs = [formals.get(value, value) for value in node.inputs]
            renamed_body.append(node)
        call_node = NetronNode(
            name=call_display[scope.label],
            op_type=call_display[scope.label],
            domain=NETRON_MODULE_DOMAIN,
            inputs=list(scope.imports),
            outputs=list(scope.exports),
            attrs=_module_call_attrs(scope, call_display[scope.label]),
            doc=node_doc(scope.call),
        )
        projection.node_sources[call_display[scope.label]] = scope.call
        if scope.parent == _ROOT_SCOPE:
            root_body.append(call_node)
        else:
            scopes[scope.parent].body.append(call_node)
        function_value_infos = [
            value_infos[name]
            for node in renamed_body
            for name in node.outputs
            if name in value_infos
        ]
        for value, formal in formals.items():
            if value in value_infos:
                source_info = value_infos[value]
                function_value_infos.append(
                    type(source_info)(
                        name=formal,
                        elem_type=source_info.elem_type,
                        dims=source_info.dims,
                        doc="function input",
                    )
                )
        built_functions.append(
            (
                scope.label,
                NetronFunction(
                    name=call_display[scope.label],
                    formal_inputs=[formals[value] for value in scope.imports],
                    outputs=list(scope.exports),
                    nodes=renamed_body,
                    value_infos=function_value_infos,
                    doc=_function_doc(scope),
                ),
            )
        )
    execution_order = {label: index for index, label in enumerate(records)}
    built_functions.sort(key=lambda pair: execution_order.get(pair[0], 0))
    projection.functions = [function for _, function in built_functions]


def project_module(log: Any, show_buffers: str, depth: int) -> NetronProjection:
    """Build the module-granularity projection (granularity ``"module"``).

    Returns a projection whose ``fallback_reason`` is set (and whose shape is
    the plain op projection) when the trace exposes no module-call table --
    the cycle fail-soft is enforced by the caller after the explicit
    call-graph DAG assertion.
    """

    from ._netron_records import project_op

    module_calls = getattr(log, "module_calls", None)
    labels = list(module_calls.keys()) if module_calls is not None else []
    if len(labels) <= 1:
        # Only the root call (or no call table): the op projection IS the
        # module view; not a fallback, just a flat model.
        projection = project_op(log, show_buffers)
        projection.granularity = "module"
        projection.module_depth = depth
        return projection

    classified = classify(log, show_buffers)
    levels, records = _call_levels(module_calls)
    scopes = _retained_scopes(levels, records, depth)
    projection = NetronProjection(
        granularity="module",
        nodes=[],
        graph_inputs=[],
        graph_outputs=[],
        value_infos=[],
        hidden_buffers=classified.hidden_buffers,
        hidden_op_count=classified.hidden_op_count,
        module_depth=depth,
    )
    call_tree = _CallTree(levels, records, classified.hidden_buffers)
    root, producer_scope, value_infos = _place_entries(log, classified, scopes, projection)
    root_body = root.nodes
    _settle_scopes(scopes, call_tree, root, producer_scope, show_buffers)
    graph_output_names = [info.name for info in projection.graph_outputs]
    _route_cross_scope_values(scopes, root_body, producer_scope, graph_output_names)
    _assemble_functions(scopes, call_tree, root_body, projection, value_infos)

    projection.nodes = _topo_sorted(root_body)
    for function in projection.functions:
        function.nodes = _topo_sorted(function.nodes)
    root_value_names = {name for node in projection.nodes for name in node.outputs}
    projection.value_infos = [
        value_infos[name] for name in sorted(root_value_names) if name in value_infos
    ]
    _strip_io_value_infos(projection)
    io_boxes = len(projection.graph_inputs) + len(projection.graph_outputs)
    projection.extent_ranks = compute_extent_ranks(projection.nodes, io_boxes)
    return projection


def _function_doc(scope: _Scope) -> str:
    """One-line function docString disclosing hidden buffers for this call."""

    hidden = getattr(scope, "hidden_note", "")
    class_name = str(
        getattr(scope.call, "class_name", None) or getattr(scope.call, "class_qualname", None) or ""
    )
    base = f"module call {scope.label}"
    if class_name:
        base = f"{class_name} -- {base}"
    return f"{base}; {hidden}" if hidden else base


def _topo_sorted(nodes: list[NetronNode]) -> list[NetronNode]:
    """Stable-topologically sort one node list (ONNX ordering requirement).

    Execution order is already topological for leaf ops; module-call sites
    are inserted at their last member's position, so this pass is the named
    safety assertion that failed both labs' LLM prototypes -- it re-sorts
    instead of asserting, keeping emission valid by construction.
    """

    produced_at: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for output in node.outputs:
            produced_at.setdefault(output, index)
    ordered: list[NetronNode] = []
    emitted: set[str] = set()
    visiting: set[int] = set()

    def _emit(index: int) -> None:
        """Emit one node after its producers (stable DFS)."""

        if index in visiting:
            return  # cycle: leave residual order; the emitter's checker catches it
        node = nodes[index]
        if node.name in emitted:
            return
        visiting.add(index)
        for value in node.inputs:
            producer = produced_at.get(value)
            if producer is not None and producer != index:
                _emit(producer)
        visiting.discard(index)
        if node.name not in emitted:
            emitted.add(node.name)
            ordered.append(node)

    for index in range(len(nodes)):
        _emit(index)
    return ordered
