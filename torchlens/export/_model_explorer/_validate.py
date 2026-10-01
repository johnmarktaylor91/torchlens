"""Semantic validator for Model Explorer payloads (memo B0).

Model Explorer's worker silently drops duplicate-id nodes and dangling
edges, so a payload that parses cleanly can still be silently lossy. This
validator asserts the semantic floor BEFORE a payload ships: unique ids
within EVERY emitted graph, every ``incomingEdges.sourceNodeId`` resolving
in-graph, per-graph edge counts, slot referential integrity, namespace-
prefix group rows, the stackless-non-boundary counter, and unique graph ids
(the app silently RENAMES duplicates, disconnecting graph-keyed node data).

``validate_model_explorer_payload`` returns a structured report (the
downstream catalog ``validate_only`` seam); ``strict=True`` raises the typed refusal
for the first failure instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._errors import ModelExplorerExportError
from ._ids import PROXY_ID_PREFIX
from ._namespace import DRIVER_NAMESPACE, INPUTS_NAMESPACE, OUTPUTS_NAMESPACE

__tl_layer__ = "L8"


@dataclass
class ValidationReport:
    """Structured semantic-validation outcome for one payload."""

    ok: bool = True
    failures: list[dict[str, str]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    def add_failure(self, code: str, graph_id: str, detail: str) -> None:
        """Record one failure row and flip the verdict."""

        self.ok = False
        self.failures.append({"code": code, "graph_id": graph_id, "detail": detail})


def validate_model_explorer_payload(
    payload: dict[str, Any], *, strict: bool = False
) -> ValidationReport:
    """Validate one graph-collection payload's semantic floor.

    Parameters
    ----------
    payload:
        A ``{label, graphs, ...}`` collection dict (the exporter's output).
    strict:
        Raise the typed refusal on the first failure instead of reporting.

    Returns
    -------
    ValidationReport
        Verdict, failure rows, and disclosure counters.
    """

    report = ValidationReport()
    graphs = payload.get("graphs") or []
    _check_graph_ids(graphs, report)
    for graph in graphs:
        _check_graph(graph, report)
    if strict and not report.ok:
        first = report.failures[0]
        raise ModelExplorerExportError(
            f"Model Explorer payload failed semantic validation: {first['detail']} "
            f"(graph {first['graph_id']!r}, first of {len(report.failures)} failures)",
            code="model_explorer_payload_invalid",
            remedy=(
                "this payload would be silently lossy in the viewer; inspect the "
                "structured validation report (fields['failures']) and report the bug"
            ),
            failures=tuple(f"{row['code']}:{row['graph_id']}" for row in report.failures),
        )
    return report


def _check_graph_ids(graphs: list[dict[str, Any]], report: ValidationReport) -> None:
    """Refuse duplicate graph ids (silently renamed by the app)."""

    seen: set[str] = set()
    for graph in graphs:
        graph_id = str(graph.get("id", ""))
        if graph_id in seen:
            report.add_failure("model_explorer_duplicate_graph_id", graph_id, "duplicate graph id")
        seen.add(graph_id)


def _check_graph(graph: dict[str, Any], report: ValidationReport) -> None:
    """Run every per-graph semantic check."""

    graph_id = str(graph.get("id", ""))
    nodes = graph.get("nodes") or []
    node_ids = [str(node.get("id", "")) for node in nodes]
    _check_unique_node_ids(graph_id, node_ids, report)
    edge_count = _check_edges(graph_id, nodes, set(node_ids), report)
    if len(nodes) > 1 and edge_count == 0:
        report.add_failure(
            "model_explorer_zero_edges", graph_id, f"{len(nodes)} nodes but zero edges"
        )
    _check_group_rows(graph_id, graph, nodes, report)
    _count_root_ops(nodes, report)
    report.counters[f"{graph_id}:nodes"] = len(nodes)
    report.counters[f"{graph_id}:edges"] = edge_count


def _check_unique_node_ids(graph_id: str, node_ids: list[str], report: ValidationReport) -> None:
    """Assert ids unique within the graph (the worker drops duplicates)."""

    seen: set[str] = set()
    for node_id in node_ids:
        if node_id in seen:
            report.add_failure(
                "model_explorer_duplicate_node_id",
                graph_id,
                f"duplicate node id {node_id!r}",
            )
        seen.add(node_id)


def _check_edges(
    graph_id: str,
    nodes: list[dict[str, Any]],
    node_ids: set[str],
    report: ValidationReport,
) -> int:
    """Assert every edge resolves in-graph and slots are referenced."""

    edge_count = 0
    for node in nodes:
        edges = node.get("incomingEdges") or []
        edge_count += len(edges)
        slots = {str(edge.get("targetNodeInputId", "0")) for edge in edges}
        for edge in edges:
            source_id = str(edge.get("sourceNodeId", ""))
            if source_id not in node_ids:
                report.add_failure(
                    "model_explorer_edge_unresolved",
                    graph_id,
                    f"edge source {source_id!r} of node {node.get('id')!r} "
                    "does not resolve in-graph",
                )
        for metadata in node.get("inputsMetadata") or []:
            slot = str(metadata.get("id", ""))
            if slot not in slots:
                report.add_failure(
                    "model_explorer_slot_unreferenced",
                    graph_id,
                    f"inputsMetadata slot {slot!r} of node {node.get('id')!r} "
                    "matches no incoming edge",
                )
    return edge_count


def _check_group_rows(
    graph_id: str,
    graph: dict[str, Any],
    nodes: list[dict[str, Any]],
    report: ValidationReport,
) -> None:
    """Assert a group row exists for every namespace prefix in use."""

    group_rows = graph.get("groupNodeAttributes")
    if group_rows is None:
        report.add_failure(
            "model_explorer_group_rows_missing", graph_id, "groupNodeAttributes absent"
        )
        return
    if "" not in group_rows:
        report.add_failure(
            "model_explorer_group_rows_missing",
            graph_id,
            "the '' provenance/disclosure row is absent",
        )
    for node in nodes:
        namespace = str(node.get("namespace", "") or "")
        if not namespace:
            continue
        components = namespace.split("/")
        for depth in range(1, len(components) + 1):
            prefix = "/".join(components[:depth])
            if prefix not in group_rows:
                report.add_failure(
                    "model_explorer_group_row_missing_namespace",
                    graph_id,
                    f"no group row for namespace {prefix!r}",
                )


def _count_root_ops(nodes: list[dict[str, Any]], report: ValidationReport) -> None:
    """Count stackless non-boundary nodes (memo D19 re-open tripwire)."""

    count = 0
    for node in nodes:
        namespace = str(node.get("namespace", "") or "")
        if namespace in (INPUTS_NAMESPACE, OUTPUTS_NAMESPACE, DRIVER_NAMESPACE):
            continue
        if not namespace and not str(node.get("id", "")).startswith(PROXY_ID_PREFIX):
            count += 1
    report.counters["stackless_root_ops"] = report.counters.get("stackless_root_ops", 0) + count
