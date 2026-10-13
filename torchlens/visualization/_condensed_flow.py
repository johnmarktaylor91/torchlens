"""Child condensed-flow-graph construction for smart module collapse.

Split out of ``auto_collapse.py`` under the R43 file-size ratchet: the
per-parent condensed dataflow view (owner condensation, interval flags,
junction/landmark/passthrough edge counting) that feeds the collapse
signals. ``auto_collapse`` re-exports the dataclasses; relationship
resolution stays in ``auto_collapse`` (``_resolve_relationship_op`` /
``_op_adjacency_index``) and is imported lazily so its monkeypatchable
seam keeps one home.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace

JUNCTION_FUNC_NAMES = frozenset({"__add__", "add", "cat", "concat", "concatenate"})


def _resolve_relationship_op(*args: Any, **kwargs: Any) -> Op:
    """Late-bound proxy to ``auto_collapse._resolve_relationship_op``.

    Resolved at call time through the ``auto_collapse`` module namespace so
    the adjacency-index seam (and its test monkeypatches) keeps exactly one
    authoritative home. Signature-transparent by design.
    """

    from . import auto_collapse

    return auto_collapse._resolve_relationship_op(*args, **kwargs)


@dataclass(frozen=True)
class ModuleCollapseSignals:
    """Precomputed structural signals for one module.

    Parameters
    ----------
    address:
        Primary module address.
    subtree_ops:
        Pass-qualified operation labels in module scope.
    own_func_names:
        Function names for ops directly owned by the module, in call order.
    internal_edges:
        Distinct op-graph edges with both endpoints in the module.
    input_edges:
        Distinct op-graph edges entering the module from outside.
    output_edges:
        Distinct op-graph edges leaving the module.
    landmark_edges:
        Boundary-crossing edges that enter or leave non-boundary internal
        operations and therefore hide a meaningful cross-module junction.
    passthrough_edges:
        Internal output junctions that combine module input with internal work.
    output_junctions:
        External multi-parent children fed by module outputs.
    params:
        Number of recursive parameters for the module.
    depth:
        Address-tree depth.
    num_calls:
        Number of module calls.
    structural_digest:
        Trace-local structural digest.
    peer_count:
        Number of modules in the same address-keyed peer group.
    hidden_ops:
        Rendered op count hidden by collapsing this module.
    eligible:
        Whether renderer-faithful hard gating allows collapse.
    """

    address: str
    subtree_ops: tuple[str, ...]
    own_func_names: tuple[str, ...]
    internal_edges: int
    input_edges: int
    output_edges: int
    landmark_edges: int
    passthrough_edges: int
    output_junctions: tuple[str, ...]
    params: int
    depth: int
    num_calls: int
    structural_digest: str
    peer_count: int
    hidden_ops: int
    eligible: bool


@dataclass(frozen=True)
class FlowIntervalFlags:
    """Blocker flags for an interval between flow-adjacent children.

    Parameters
    ----------
    landmark:
        Whether landmark edges cross the interval.
    passthrough:
        Whether passthrough-style parent-owned flow crosses the interval.
    """

    landmark: bool
    passthrough: bool


@dataclass(frozen=True)
class ChildCondensedFlowGraph:
    """Child-condensed flow graph for one parent module call.

    Parameters
    ----------
    parent:
        Parent module address.
    flow_children:
        Direct child module addresses ordered by first executed op.
    parent_owned_ops:
        Parent-owned operation labels in execution order.
    nodes:
        Condensed graph nodes: child subtrees plus parent-owned ops.
    edges:
        Condensed op-flow edges between nodes.
    child_external_endpoint_counts:
        Per-child ``(entries, exits)`` counts against the condensed graph.
    interval_flags:
        Flags keyed by flow-child address pairs.
    """

    parent: str
    flow_children: tuple[str, ...]
    parent_owned_ops: tuple[str, ...]
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    child_external_endpoint_counts: Mapping[str, tuple[int, int]]
    interval_flags: Mapping[tuple[str, str], FlowIntervalFlags]


def _module_address_stack(op: Op) -> tuple[str, ...]:
    """Return pass-free module addresses enclosing an op."""

    return tuple(str(module).rsplit(":", 1)[0] for module in getattr(op, "modules", ()) or ())


def _compute_child_condensed_flow_graphs(
    trace: Trace,
    signals: Mapping[str, ModuleCollapseSignals],
    revision: tuple[object, ...],
) -> dict[str, ChildCondensedFlowGraph]:
    """Compute child-condensed flow graphs for every parent module.

    Parameters
    ----------
    trace:
        Trace owning the module hierarchy.
    signals:
        Precomputed module signal skeletons.
    revision:
        Walk-level graph fingerprint threaded through relationship resolution.

    Returns
    -------
    dict[str, ChildCondensedFlowGraph]
        Flow graph artifacts keyed by parent module address.
    """

    graphs: dict[str, ChildCondensedFlowGraph] = {}
    op_order = {op.label: index for index, op in enumerate(trace.ops)}
    for module in trace.modules:
        parent = module.address
        child_addresses = tuple(
            str(child)
            for child in getattr(module, "address_children", ()) or ()
            if child in trace.modules
        )
        if not child_addresses:
            graphs[parent] = ChildCondensedFlowGraph(
                parent=parent,
                flow_children=(),
                parent_owned_ops=(),
                nodes=(),
                edges=(),
                child_external_endpoint_counts={},
                interval_flags={},
            )
            continue
        child_sets = {
            child: set(signals[child].subtree_ops) for child in child_addresses if child in signals
        }
        flow_children = tuple(
            sorted(
                child_sets,
                key=lambda child: (
                    min((op_order[label] for label in child_sets[child]), default=10**12),
                    child,
                ),
            )
        )
        owner_by_op = _condensed_owner_map(flow_children, child_sets)
        parent_ops = tuple(
            label
            for label in signals.get(parent, _empty_signal(parent)).subtree_ops
            if _condensed_owner_for_op(label, owner_by_op) == label
            and not _is_buffer_op_label(trace, label)
        )
        nodes = (*flow_children, *parent_ops)
        edges = _condensed_edges(
            trace,
            flow_children,
            child_sets,
            set(parent_ops),
            owner_by_op,
            revision,
        )
        endpoint_counts = _child_external_endpoint_counts(edges, flow_children)
        interval_flags = _flow_interval_flags(trace, flow_children, child_sets, edges)
        graphs[parent] = ChildCondensedFlowGraph(
            parent=parent,
            flow_children=flow_children,
            parent_owned_ops=parent_ops,
            nodes=nodes,
            edges=edges,
            child_external_endpoint_counts=endpoint_counts,
            interval_flags=interval_flags,
        )
    return graphs


def _empty_signal(address: str) -> ModuleCollapseSignals:
    """Return an empty signal used for missing parent bookkeeping.

    Parameters
    ----------
    address:
        Module address.

    Returns
    -------
    ModuleCollapseSignals
        Empty signal with no subtree operations.
    """

    return ModuleCollapseSignals(
        address=address,
        subtree_ops=(),
        own_func_names=(),
        internal_edges=0,
        input_edges=0,
        output_edges=0,
        landmark_edges=0,
        passthrough_edges=0,
        output_junctions=(),
        params=0,
        depth=0,
        num_calls=1,
        structural_digest="",
        peer_count=1,
        hidden_ops=0,
        eligible=False,
    )


def _first_flow_op_order(
    trace: Trace,
    op_labels: set[str],
    op_order: Mapping[str, int],
) -> int:
    """Return first non-buffer op order for a child subtree.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    op_labels:
        Operation labels in the child subtree.
    op_order:
        Deterministic operation-order index keyed by op label.

    Returns
    -------
    int
        First non-buffer operation index, falling back to any operation index
        when the subtree has no non-buffer ops.
    """

    non_buffer_orders = [
        op_order[label] for label in op_labels if not _is_buffer_op_label(trace, label)
    ]
    if non_buffer_orders:
        return min(non_buffer_orders)
    return min((op_order[label] for label in op_labels), default=10**12)


def _is_buffer_op_label(trace: Trace, op_label: str) -> bool:
    """Return whether ``op_label`` identifies a buffer/source op.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    op_label:
        Operation label to inspect.

    Returns
    -------
    bool
        True when the label exists and represents a buffer op.
    """

    if op_label not in trace.ops:
        return False
    return bool(getattr(cast("Op", trace.ops[op_label]), "is_buffer", False))


def _is_forward_dataflow_edge(trace: Trace, source_label: str, target_label: str) -> bool:
    """Return whether an op edge is real forward tensor dataflow.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    source_label:
        Source operation label.
    target_label:
        Target operation label.

    Returns
    -------
    bool
        True for non-buffer endpoint edges. Registered-buffer provenance and
        write-version edges are excluded from the child-condensed dataflow
        artifact.
    """

    return not _is_buffer_op_label(trace, source_label) and not _is_buffer_op_label(
        trace,
        target_label,
    )


def _condensed_owner_map(
    flow_children: Sequence[str],
    child_sets: Mapping[str, set[str]],
) -> dict[str, str]:
    """Invert child subtree membership into a first-wins op owner map.

    Parameters
    ----------
    flow_children:
        Direct children in flow order.
    child_sets:
        Child subtree operation labels.

    Returns
    -------
    dict[str, str]
        Child address keyed by every operation in a direct child subtree.
    """

    owner_by_op: dict[str, str] = {}
    for child in flow_children:
        for op_label in child_sets[child]:
            owner_by_op.setdefault(op_label, child)
    return owner_by_op


def _condensed_owner_for_op(op_label: str, owner_by_op: Mapping[str, str]) -> str:
    """Return the condensed node that owns an operation.

    Parameters
    ----------
    op_label:
        Operation label.
    owner_by_op:
        First-wins direct-child owner index.

    Returns
    -------
    str
        Child address when the op belongs to a child subtree; otherwise the op label.
    """

    return owner_by_op.get(op_label, op_label)


def _condensed_edges(
    trace: Trace,
    flow_children: Sequence[str],
    child_sets: Mapping[str, set[str]],
    parent_ops: set[str],
    owner_by_op: Mapping[str, str],
    revision: tuple[object, ...],
) -> tuple[tuple[str, str], ...]:
    """Return condensed edges within one parent module subtree.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    flow_children:
        Direct children in flow order.
    child_sets:
        Child subtree operation labels.
    parent_ops:
        Parent-owned operation labels.
    owner_by_op:
        First-wins direct-child owner index.
    revision:
        Walk-level graph fingerprint threaded through relationship resolution.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Deterministically sorted condensed edges.
    """

    parent_subtree = set().union(*child_sets.values()) if child_sets else set()
    parent_subtree.update(parent_ops)
    order = {node: index for index, node in enumerate((*flow_children, *sorted(parent_ops)))}
    edges: set[tuple[str, str]] = set()
    for label in sorted(
        parent_subtree, key=lambda item: int(getattr(trace.ops[item], "step_index", 0))
    ):
        op = cast("Op", trace.ops[label])
        source = _condensed_owner_for_op(label, owner_by_op)
        for parent_label in getattr(op, "parents", ()) or ():
            parent_op = _resolve_relationship_op(trace, parent_label, revision)
            normalized_parent_label = parent_op.label
            if normalized_parent_label in parent_subtree:
                continue
            if not _is_forward_dataflow_edge(trace, normalized_parent_label, label):
                continue
            edges.add((f"external_source:{normalized_parent_label}", source))
        for child_label in getattr(op, "children", ()) or ():
            child = _resolve_relationship_op(trace, child_label, revision)
            normalized_child_label = child.label
            if not _is_forward_dataflow_edge(trace, label, normalized_child_label):
                continue
            if normalized_child_label not in parent_subtree:
                edges.add((source, f"external_sink:{normalized_child_label}"))
                continue
            target = _condensed_owner_for_op(
                normalized_child_label,
                owner_by_op,
            )
            if source != target:
                edges.add((source, target))
    return tuple(
        sorted(edges, key=lambda edge: (order.get(edge[0], 10**9), order.get(edge[1], 10**9), edge))
    )


def _child_external_endpoint_counts(
    edges: Sequence[tuple[str, str]],
    flow_children: Sequence[str],
) -> dict[str, tuple[int, int]]:
    """Return per-child external entry and exit endpoint counts.

    Parameters
    ----------
    edges:
        Condensed graph edges.
    flow_children:
        Direct children in flow order.

    Returns
    -------
    dict[str, tuple[int, int]]
        Mapping from child address to ``(entries, exits)``.
    """

    child_set = set(flow_children)
    entries: dict[str, set[str]] = {child: set() for child in flow_children}
    exits: dict[str, set[str]] = {child: set() for child in flow_children}
    for source, target in edges:
        if target in child_set and source != target:
            entries[target].add(source)
        if source in child_set and source != target:
            exits[source].add(target)
    return {child: (len(entries[child]), len(exits[child])) for child in flow_children}


def _flow_interval_flags(
    trace: Trace,
    flow_children: Sequence[str],
    child_sets: Mapping[str, set[str]],
    edges: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], FlowIntervalFlags]:
    """Return landmark and passthrough flags for child-flow intervals.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    flow_children:
        Direct children in flow order.
    child_sets:
        Child subtree operation labels.
    edges:
        Condensed graph edges.

    Returns
    -------
    dict[tuple[str, str], FlowIntervalFlags]
        Flags keyed by adjacent child pairs in flow order.
    """

    if len(flow_children) < 2:
        return {}
    child_index = {child: index for index, child in enumerate(flow_children)}
    crossing_deltas = [0] * len(flow_children)
    external_touches = [False] * len(flow_children)
    for source, target in set(edges):
        source_index = child_index.get(source)
        target_index = child_index.get(target)
        if source_index is not None and target_index is not None:
            if source_index < target_index:
                crossing_deltas[source_index] += 1
                crossing_deltas[target_index] -= 1
            continue
        if source_index is not None:
            external_touches[source_index] = True
        if target_index is not None:
            external_touches[target_index] = True
    junction_children = {
        child: _child_has_junction_op(trace, child_sets.get(child, set()))
        for child in flow_children
    }
    flags: dict[tuple[str, str], FlowIntervalFlags] = {}
    crossing_count = 0
    for left_index, (left, right) in enumerate(
        zip(flow_children[:-1], flow_children[1:], strict=True)
    ):
        right_index = left_index + 1
        crossing_count += crossing_deltas[left_index]
        passthrough = external_touches[left_index] or external_touches[right_index]
        landmark = junction_children[left] or junction_children[right] or crossing_count > 0
        flags[(left, right)] = FlowIntervalFlags(landmark=landmark, passthrough=passthrough)
    return flags


def _child_has_junction_op(trace: Trace, op_labels: set[str]) -> bool:
    """Return whether a child subtree contains a junction operation.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    op_labels:
        Operation labels in the child subtree.

    Returns
    -------
    bool
        True when a known fan-in/fan-out junction op is present.
    """

    return any(
        _op_func_name(cast("Op", trace.ops[label])) in JUNCTION_FUNC_NAMES for label in op_labels
    )


def _op_func_name(op: Op) -> str:
    """Return a stable operation function name for digesting."""

    return str(getattr(op, "func_name", None) or getattr(op, "layer_type", "") or "")


def _count_landmark_edges(
    trace: Trace,
    module: Module,
    subtree_ops: tuple[str, ...],
    boundary_edges: set[tuple[str, str]],
    revision: tuple[object, ...],
) -> int:
    """Return boundary-crossing junction edges for a module.

    Parameters
    ----------
    trace:
        Trace that owns the operation graph.
    module:
        Candidate module being scored.
    subtree_ops:
        Pass-qualified operation labels in the module subtree.
    boundary_edges:
        Distinct edges crossing the module boundary.
    revision:
        Walk-level graph fingerprint threaded through relationship resolution.

    Returns
    -------
    int
        Count of boundary edges that would hide or visually skip a junction
        across the collapsed module boundary. Fully internal junctions and
        ordinary module I/O edges are intentionally not counted because they are
        safely represented by the collapsed module box.
    """

    subtree = set(subtree_ops)
    input_layers = {_base_label(label) for label in getattr(module, "input_layers", ()) or ()}
    output_layers = {_base_label(label) for label in getattr(module, "output_layers", ()) or ()}
    landmarks: set[tuple[str, str]] = set()
    for parent_label, child_label in boundary_edges:
        parent = cast("Op", trace.ops[parent_label])
        child = cast("Op", trace.ops[child_label])
        if getattr(parent, "is_buffer", False) or getattr(child, "is_buffer", False):
            continue
        parent_inside = parent.label in subtree
        child_inside = child.label in subtree
        if parent_inside == child_inside:
            continue
        parent_base = _base_label(parent.label)
        child_base = _base_label(child.label)
        if child_inside and parent_base in input_layers:
            continue
        if parent_inside and parent_base in output_layers:
            continue
        if child_inside and child_base in output_layers:
            continue
        if getattr(parent, "is_output", False) or getattr(child, "is_output", False):
            continue
        if not _boundary_edge_preserves_junction(trace, parent, child, subtree, revision):
            continue
        landmarks.add((parent.label, child.label))
    return len(landmarks)


def _boundary_edge_preserves_junction(
    trace: Trace,
    parent: Op,
    child: Op,
    subtree: set[str],
    revision: tuple[object, ...],
) -> bool:
    """Return whether a boundary edge is part of a cross-boundary junction.

    Parameters
    ----------
    trace:
        Trace that owns the operation graph.
    parent:
        Parent endpoint of the boundary edge.
    child:
        Child endpoint of the boundary edge.
    subtree:
        Pass-qualified operation labels in the candidate module subtree.
    revision:
        Walk-level graph fingerprint threaded through relationship resolution.

    Returns
    -------
    bool
        True when collapsing the subtree would obscure a junction whose visible
        endpoints span the module boundary.
    """

    parent_inside = parent.label in subtree
    child_inside = child.label in subtree
    if parent_inside == child_inside:
        return False
    internal = parent if parent_inside else child
    external = child if parent_inside else parent
    if _is_junction_op(external):
        return True
    if not _is_junction_op(internal):
        return False
    return _has_external_parent(trace, internal, subtree, revision) and _has_external_child(
        trace,
        internal,
        subtree,
        revision,
    )


def _is_junction_op(op: Op) -> bool:
    """Return whether an operation is a fan-in or fan-out junction."""

    return _op_func_name(op) in JUNCTION_FUNC_NAMES


def _has_external_parent(
    trace: Trace, op: Op, subtree: set[str], revision: tuple[object, ...]
) -> bool:
    """Return whether an operation has a non-buffer parent outside ``subtree``."""

    for parent_label in getattr(op, "parents", ()) or ():
        parent = _resolve_relationship_op(trace, parent_label, revision)
        if parent.label not in subtree and not getattr(parent, "is_buffer", False):
            return True
    return False


def _has_external_child(
    trace: Trace, op: Op, subtree: set[str], revision: tuple[object, ...]
) -> bool:
    """Return whether an operation has a non-buffer child outside ``subtree``."""

    for child_label in getattr(op, "children", ()) or ():
        child = _resolve_relationship_op(trace, child_label, revision)
        if child.label not in subtree and not getattr(child, "is_buffer", False):
            return True
    return False


def _base_label(label: str) -> str:
    """Return a pass-free operation label.

    Parameters
    ----------
    label:
        Operation label that may include a pass suffix.

    Returns
    -------
    str
        Operation label without the trailing pass suffix.
    """

    return str(label).rsplit(":", 1)[0]


def _count_passthrough_edges(
    trace: Trace,
    module: Module,
    subtree_ops: tuple[str, ...],
    revision: tuple[object, ...],
) -> int:
    """Return internal output joins fed directly by module inputs.

    Parameters
    ----------
    trace:
        Trace that owns the operation graph.
    module:
        Candidate module being scored.
    subtree_ops:
        Pass-qualified operation labels in the module subtree.
    revision:
        Walk-level graph fingerprint threaded through relationship resolution.

    Returns
    -------
    int
        Number of module-output Ops that merge an external module input with
        internal computation. These joins are useful orientation landmarks for
        ``collapse="auto"`` but may be hidden by ``collapse="max"``.
    """

    subtree = set(subtree_ops)
    input_layers = {_base_label(label) for label in getattr(module, "input_layers", ()) or ()}
    output_layers = {_base_label(label) for label in getattr(module, "output_layers", ()) or ()}
    passthrough_edges = 0
    for label in subtree_ops:
        op = cast("Op", trace.ops[label])
        if _base_label(op.label) not in output_layers:
            continue
        if _op_func_name(op) not in JUNCTION_FUNC_NAMES:
            continue
        has_internal_parent = False
        has_input_parent = False
        for parent_label in op.parents:
            parent = _resolve_relationship_op(trace, parent_label, revision)
            if parent.label in subtree:
                has_internal_parent = True
            elif _base_label(parent.label) in input_layers:
                has_input_parent = True
        if has_internal_parent and has_input_parent:
            passthrough_edges += 1
    return passthrough_edges


def _output_junctions(
    trace: Trace,
    module: Module,
    subtree_ops: tuple[str, ...],
    output_edges: set[tuple[str, str]],
) -> tuple[str, ...]:
    """Return external multi-parent junction children fed by module outputs.

    Parameters
    ----------
    trace:
        Trace that owns the operation graph.
    module:
        Candidate module being scored.
    subtree_ops:
        Pass-qualified operation labels in the module subtree.
    output_edges:
        Distinct edges leaving the module subtree.

    Returns
    -------
    tuple[str, ...]
        Pass-free labels for external multi-parent children fed by this module.
    """

    subtree = set(subtree_ops)
    output_layers = {_base_label(label) for label in getattr(module, "output_layers", ()) or ()}
    junctions: set[str] = set()
    for parent_label, child_label in output_edges:
        parent = cast("Op", trace.ops[parent_label])
        child = cast("Op", trace.ops[child_label])
        if parent.label not in subtree:
            continue
        if _base_label(parent.label) not in output_layers:
            continue
        if len(getattr(child, "parents", ()) or ()) < 2:
            continue
        junctions.add(_base_label(child.label))
    return tuple(sorted(junctions))
