"""Single-graph assembly for Model Explorer payloads (memo B1-B3, D10).

``build_graph`` assembles one unrolled (exact-execution) graph from a
layer-pass entry sequence; ``build_rolled_graph`` assembles the disclosed
rolled DAG projection whose recurrent back-edges are removed from
``incomingEdges`` (dagre silently REVERSES cyclic edges rather than
erroring) and re-emitted as a named "recurrent feedback" relation through
the edge-overlay carrier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...errors._base import TorchLensError
from ._attrs import (
    AttrContext,
    curated_attrs,
    curated_rolled_attrs,
    node_kind,
    node_label,
    output_metadata,
)
from ._edges import build_label_map, incoming_edges
from ._errors import ModelExplorerExportError
from ._grouprows import accumulate_namespace_stats, build_group_rows
from ._ids import mint_node_ids
from ._namespace import (
    DRIVER_NAMESPACE,
    INPUTS_NAMESPACE,
    OUTPUTS_NAMESPACE,
    NamespaceLevel,
    address_call_counts,
    namespace_for_entry,
)

__tl_layer__ = "L8"

#: Okabe-Ito reddish purple: the colorblind-safe accent the codebase already
#: uses for intervention marks; carries the recurrent-feedback overlay.
FEEDBACK_EDGE_COLOR = "#CC79A7"

FEEDBACK_OVERLAY_NAME = "recurrent feedback"


@dataclass
class BuildResult:
    """One built graph plus the disclosure counters its callers fold."""

    graph: dict[str, Any]
    node_ids: list[str]
    label_map: dict[str, str]
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphSpec:
    """Per-graph assembly directives consumed by ``build_graph``.

    ``extra_label_map`` and ``extra_nodes`` are the episode boundary-proxy
    doors: pre-built nodes join the graph verbatim and their ids resolve
    parent references that cross the entry set.
    """

    graph_id: str
    attr_context: AttrContext
    strict_namespace: bool = False
    strip_prefix_entry: str | None = None
    driver_labels: frozenset[str] = frozenset()
    extra_label_map: dict[str, str] | None = None
    extra_nodes: tuple[dict[str, Any], ...] = ()
    resolve_missing: str = "raise"
    root_facts: dict[str, str] | None = None


def build_graph(trace: Any, entries: list[Any], spec: GraphSpec) -> BuildResult:
    """Assemble one unrolled graph dict from layer-pass entries per ``spec``."""

    node_ids, legacy_count = mint_node_ids(entries)
    _refuse_duplicate_ids(spec.graph_id, node_ids, spec.extra_nodes)
    call_counts = address_call_counts(entries)
    namespaces, levels_per_entry, namespace_report = _derive_namespaces(
        entries,
        call_counts,
        strict_namespace=spec.strict_namespace,
        strip_prefix_entry=spec.strip_prefix_entry,
        driver_labels=spec.driver_labels,
    )
    label_map = build_label_map(entries, node_ids)
    if spec.extra_label_map:
        label_map = {**label_map, **spec.extra_label_map}
    nodes, edge_report = _build_nodes(entries, node_ids, namespaces, label_map, spec)
    nodes.extend(dict(node) for node in spec.extra_nodes)
    stats = accumulate_namespace_stats(entries, namespaces, levels_per_entry)
    facts = dict(spec.root_facts or {})
    facts.update(_disclosure_facts(len(nodes), legacy_count, namespace_report, edge_report))
    group_rows = build_group_rows(trace, stats, facts)
    _add_synthetic_namespace_rows(group_rows, namespaces, spec.extra_nodes)
    graph = {"id": spec.graph_id, "nodes": nodes, "groupNodeAttributes": group_rows}
    report = {
        "legacy_ids": legacy_count,
        "edges": edge_report["edges"],
        **namespace_report,
        **{key: value for key, value in edge_report.items() if key != "edges"},
    }
    return BuildResult(graph=graph, node_ids=node_ids, label_map=label_map, report=report)


def build_rolled_graph(
    trace: Any,
    entries: list[Any],
    *,
    graph_id: str,
    attr_context: AttrContext,
    root_facts: dict[str, str] | None = None,
) -> BuildResult:
    """Assemble the rolled DAG projection with the feedback overlay carrier."""

    # Essential complexity (CC>10 named): rolled assembly is a cohesive
    # projection (ids, namespaces, attrs, forward/feedback split, carriers)
    # over shared indices; splitting would thread five maps through helpers.
    layer_order, ops_by_layer = _rolled_layers(entries)
    rolled_entries = [ops_by_layer[label][0] for label in layer_order]
    node_ids, legacy_count = mint_node_ids(rolled_entries)
    rolled_id_by_layer = dict(zip(layer_order, node_ids, strict=True))
    layer_index = {label: index for index, label in enumerate(layer_order)}
    forward_edges, feedback_edges = _rolled_edges(entries, rolled_id_by_layer, layer_index)
    namespaces, levels_per_entry, namespace_report = _derive_namespaces(
        rolled_entries,
        {},
        strict_namespace=False,
        strip_prefix_entry=None,
        driver_labels=frozenset(),
    )
    nodes = []
    for label, node_id, namespace in zip(layer_order, node_ids, namespaces, strict=True):
        ops = ops_by_layer[label]
        layer = _layer_record(trace, label, ops)
        node: dict[str, Any] = {
            "id": node_id,
            "label": node_label(ops[0]),
            "namespace": namespace,
            "attrs": curated_rolled_attrs(layer, ops, attr_context),
        }
        edges = forward_edges.get(node_id)
        if edges:
            node["incomingEdges"] = edges
        metadata = _rolled_output_metadata(ops)
        if metadata:
            node["outputsMetadata"] = metadata
        _apply_boundary_config(node, ops[0])
        nodes.append(node)
    stats = accumulate_namespace_stats(rolled_entries, namespaces, levels_per_entry)
    facts = dict(root_facts or {})
    facts["feedback_edges"] = str(len(feedback_edges))
    facts.update(
        _disclosure_facts(
            len(nodes),
            legacy_count,
            namespace_report,
            {"edges": sum(len(rows) for rows in forward_edges.values())},
        )
    )
    group_rows = build_group_rows(trace, stats, facts)
    _add_synthetic_namespace_rows(group_rows, namespaces, ())
    graph: dict[str, Any] = {"id": graph_id, "nodes": nodes, "groupNodeAttributes": group_rows}
    if feedback_edges:
        graph["tasksData"] = {
            "edgeOverlaysDataListLeftPane": [
                {
                    "name": "TorchLens relations",
                    "type": "edge_overlays",
                    "overlays": [
                        {
                            "name": FEEDBACK_OVERLAY_NAME,
                            "edgeColor": FEEDBACK_EDGE_COLOR,
                            "edges": feedback_edges,
                        }
                    ],
                }
            ]
        }
    report = {
        "legacy_ids": legacy_count,
        "edges": sum(len(rows) for rows in forward_edges.values()),
        "feedback_edges": feedback_edges,
        **namespace_report,
    }
    return BuildResult(
        graph=graph,
        node_ids=node_ids,
        label_map={label: rolled_id_by_layer[label] for label in layer_order},
        report=report,
    )


def rolling_changes_identity(entries: list[Any]) -> bool:
    """Return whether a rolled projection differs from the unrolled graph.

    Feed-forward models have ZERO multi-pass ops and emit NO rolled graph at
    all (memo D10): the rolled/feedback machinery is real only for episode
    and recurrent captures.
    """

    return any(int(getattr(entry, "num_passes", 1) or 1) > 1 for entry in entries)


def _refuse_duplicate_ids(
    graph_id: str, node_ids: list[str], extra_nodes: tuple[dict[str, Any], ...]
) -> None:
    """Tripwire: a duplicate id would be SILENTLY dropped by the viewer."""

    all_ids = list(node_ids) + [str(node.get("id", "")) for node in extra_nodes]
    if len(set(all_ids)) == len(all_ids):
        return
    seen: set[str] = set()
    duplicate = ""
    for node_id in all_ids:
        if node_id in seen:
            duplicate = node_id
            break
        seen.add(node_id)
    raise ModelExplorerExportError(
        f"Graph {graph_id!r} minted duplicate node id {duplicate!r}; Model Explorer "
        "keeps the first duplicate and silently drops the rest with their edges",
        code="model_explorer_duplicate_node_id",
        remedy=(
            "this indicates a TorchLens id-minting bug, not a user error; report it "
            "with the trace's summary() and this graph id"
        ),
        graph_id=graph_id,
    )


def _derive_namespaces(
    entries: list[Any],
    call_counts: dict[str, set[int]],
    *,
    strict_namespace: bool,
    strip_prefix_entry: str | None,
    driver_labels: frozenset[str],
) -> tuple[list[str], list[list[NamespaceLevel]], dict[str, int]]:
    """Derive namespaces and boundary assignments for every entry."""

    namespaces: list[str] = []
    levels_per_entry: list[list[NamespaceLevel]] = []
    partial_count = 0
    stackless_root = 0
    for entry in entries:
        namespace, verified, levels = namespace_for_entry(
            entry,
            call_counts,
            strict=strict_namespace,
            strip_prefix_entry=strip_prefix_entry,
        )
        if not verified:
            partial_count += 1
        if not namespace and not levels:
            namespace, was_root = _boundary_namespace(entry, driver_labels)
            stackless_root += was_root
        namespaces.append(namespace)
        levels_per_entry.append(levels)
    return (
        namespaces,
        levels_per_entry,
        {"namespace_partial": partial_count, "stackless_root_ops": stackless_root},
    )


def _boundary_namespace(entry: Any, driver_labels: frozenset[str]) -> tuple[str, int]:
    """Assign the D19 boundary namespace for one stackless entry."""

    if getattr(entry, "is_input", False):
        return INPUTS_NAMESPACE, 0
    if getattr(entry, "is_output", False):
        return OUTPUTS_NAMESPACE, 0
    if str(getattr(entry, "label", "")) in driver_labels:
        return DRIVER_NAMESPACE, 0
    return "", 1


def _build_nodes(
    entries: list[Any],
    node_ids: list[str],
    namespaces: list[str],
    label_map: dict[str, str],
    spec: GraphSpec,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Assemble node dicts and fold the edge disclosure counters."""

    nodes: list[dict[str, Any]] = []
    total_edges = 0
    structural_edges = 0
    skipped_parents = 0
    for entry, node_id, namespace in zip(entries, node_ids, namespaces, strict=True):
        edges, metadata, skipped = incoming_edges(
            entry, label_map, resolve_missing=spec.resolve_missing
        )
        skipped_parents += skipped
        total_edges += len(edges)
        structural_edges += sum(1 for edge in edges if "targetNodeInputId" not in edge)
        node: dict[str, Any] = {
            "id": node_id,
            "label": node_label(entry),
            "namespace": namespace,
            "attrs": curated_attrs(entry, spec.attr_context),
        }
        if edges:
            node["incomingEdges"] = edges
        if metadata:
            node["inputsMetadata"] = metadata
        outputs = output_metadata(entry)
        if outputs:
            node["outputsMetadata"] = outputs
        _apply_boundary_config(node, entry)
        nodes.append(node)
    return nodes, {
        "edges": total_edges,
        "structural_edges": structural_edges,
        "skipped_parents": skipped_parents,
    }


def _apply_boundary_config(node: dict[str, Any], entry: Any) -> None:
    """Pin true graph inputs to the top of their boundary group (D19)."""

    if getattr(entry, "is_input", False):
        node["config"] = {"pinToGroupTop": True}


def _disclosure_facts(
    node_count: int,
    legacy_count: int,
    namespace_report: dict[str, int],
    edge_report: dict[str, int],
) -> dict[str, str]:
    """Format the per-graph disclosure facts for the ``""`` group row."""

    facts = {
        "nodes": str(node_count),
        "edges": str(edge_report.get("edges", 0)),
        "id_fidelity": f"legacy({legacy_count})" if legacy_count else "site_key",
        "namespace_fidelity": (
            f"partial({namespace_report['namespace_partial']})"
            if namespace_report.get("namespace_partial")
            else "full"
        ),
    }
    for key in ("structural_edges", "skipped_parents"):
        if edge_report.get(key):
            facts[key] = str(edge_report[key])
    if namespace_report.get("stackless_root_ops"):
        facts["stackless_root_ops"] = str(namespace_report["stackless_root_ops"])
    return facts


def _add_synthetic_namespace_rows(
    group_rows: dict[str, dict[str, str]],
    namespaces: list[str],
    extra_nodes: tuple[dict[str, Any], ...],
) -> None:
    """Add rows for boundary/driver namespaces so EVERY namespace has one."""

    synthetic: dict[str, int] = {}
    for namespace in namespaces:
        if namespace and namespace not in group_rows:
            synthetic[namespace] = synthetic.get(namespace, 0) + 1
    for node in extra_nodes:
        namespace = str(node.get("namespace", "") or "")
        if namespace and namespace not in group_rows:
            synthetic[namespace] = synthetic.get(namespace, 0) + 1
    for namespace, count in synthetic.items():
        group_rows[namespace] = {"ops": str(count)}


def _rolled_layers(entries: list[Any]) -> tuple[list[str], dict[str, list[Any]]]:
    """Group entries by layer label in first-occurrence order."""

    layer_order: list[str] = []
    ops_by_layer: dict[str, list[Any]] = {}
    for entry in entries:
        label = str(getattr(entry, "layer_label", "") or getattr(entry, "label", ""))
        if label not in ops_by_layer:
            ops_by_layer[label] = []
            layer_order.append(label)
        ops_by_layer[label].append(entry)
    return layer_order, ops_by_layer


def _rolled_edges(
    entries: list[Any],
    rolled_id_by_layer: dict[str, str],
    layer_index: dict[str, int],
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    """Project pass-level parent refs onto rolled nodes, splitting feedback.

    An edge whose parent layer does not strictly precede its child layer in
    first-occurrence order is a recurrent back-edge: removed from
    ``incomingEdges`` and re-emitted through the feedback overlay.
    """

    # Essential complexity (CC>10 named): forward/feedback classification and
    # two dedup sets are one pass over pass-level refs; the state is shared.
    layer_of_ref: dict[str, str] = {}
    for entry in entries:
        layer_label = str(getattr(entry, "layer_label", "") or "")
        layer_of_ref[str(getattr(entry, "label", ""))] = layer_label
        layer_of_ref.setdefault(layer_label, layer_label)
    forward: dict[str, list[dict[str, str]]] = {}
    forward_seen: set[tuple[str, str]] = set()
    feedback: list[dict[str, str]] = []
    feedback_seen: set[tuple[str, str]] = set()
    for entry in entries:
        child_layer = str(getattr(entry, "layer_label", "") or "")
        for parent_ref in _parent_refs(entry):
            parent_layer = layer_of_ref.get(parent_ref)
            if parent_layer is None or parent_layer not in layer_index:
                continue
            pair = (rolled_id_by_layer[parent_layer], rolled_id_by_layer[child_layer])
            if layer_index[parent_layer] < layer_index[child_layer]:
                if pair not in forward_seen:
                    forward_seen.add(pair)
                    forward.setdefault(pair[1], []).append(
                        {"sourceNodeId": pair[0], "sourceNodeOutputId": "0"}
                    )
            elif pair not in feedback_seen:
                feedback_seen.add(pair)
                feedback.append(
                    {
                        "sourceNodeId": pair[0],
                        "targetNodeId": pair[1],
                        "label": FEEDBACK_OVERLAY_NAME,
                    }
                )
    return forward, feedback


def _parent_refs(entry: Any) -> list[str]:
    """Return every recorded parent reference for one entry, ports intact."""

    positions = getattr(entry, "parent_arg_positions", None) or {}
    refs = [str(ref) for ref in (positions.get("args") or {}).values()]
    refs.extend(str(ref) for ref in (positions.get("kwargs") or {}).values())
    if refs:
        return refs
    return [str(ref) for ref in (getattr(entry, "parents", ()) or ())]


def _layer_record(trace: Any, layer_label: str, ops: list[Any]) -> Any:
    """Return the aggregate Layer record, falling back to the first op."""

    try:
        return trace[layer_label]
    except (LookupError, TorchLensError):
        return ops[0]


def _rolled_output_metadata(ops: list[Any]) -> list[dict[str, Any]]:
    """Emit rolled output metadata only for provably single-valued facts."""

    metadata_per_op = [output_metadata(op) for op in ops]
    first = metadata_per_op[0]
    if first and all(item == first for item in metadata_per_op[1:]):
        return first
    return []


# ``node_kind`` is re-exported for the episode proxy builder.
__all__ = [
    "BuildResult",
    "FEEDBACK_EDGE_COLOR",
    "FEEDBACK_OVERLAY_NAME",
    "GraphSpec",
    "build_graph",
    "build_rolled_graph",
    "node_kind",
    "rolling_changes_identity",
]
