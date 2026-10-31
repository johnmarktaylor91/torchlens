"""Per-node diff facts for the netron baseline slot.

``tl.export.netron(subject, path, baseline=reference)`` fills the reserved
``torchlens.baseline`` slot (netron #1544) from TorchLens' own guarded
differential, :func:`torchlens.report.diff`, instead of
re-deriving a join here. Netron has no diff view, so the facts ride the two
channels it does render:

- every exported node whose source op settled into a diff row carries
  ``torchlens.diff.*`` rows in its ``NodeProto.metadataProps``, and the same
  facts land in the ``netron:attachment`` sidecar as node metadata;
- the model-level ``torchlens.baseline`` row holds the JSON summary: the
  direction, the closed-vocabulary row counts, and the reference-only
  (``removed``) labels a subject graph cannot show as nodes.

A node built from several ops (a module call, a rolled layer) carries the
row counts over its ops and the largest value delta among them; a node with
no settled row carries no diff facts (absent, never a fabricated zero).
Every key spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import json as _json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ._netron_records import NetronNode, NetronProjection

__tl_layer__ = "L8"

#: Prefix of every per-node diff fact key (node metadataProps and sidecar).
DIFF_KEY_PREFIX = "torchlens.diff."

#: Model-level summary schema identifier for the ``torchlens.baseline`` row.
BASELINE_SUMMARY_SCHEMA = "torchlens.netron_baseline.v1"

#: Reference-only labels listed in the summary before the list is cut.
_REMOVED_LABEL_LIMIT = 50


@dataclass
class NetronBaselineDiff:
    """The settled diff facts one netron artifact carries."""

    summary: dict[str, Any]
    #: node name -> ordered ``(key, value)`` diff fact rows.
    node_facts: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def summary_json(self) -> str:
        """Return the deterministic JSON text for the ``torchlens.baseline`` row."""

        return _json.dumps(self.summary, sort_keys=True)


def _source_op_labels(source: Any) -> list[str]:
    """Return the subject op labels one exported node stands for."""

    call_ops = getattr(source, "ops", None)
    if isinstance(call_ops, list) and not hasattr(source, "layer_label"):
        # A module-call node stands for every op executed inside the call.
        return [str(label) for label in call_ops]
    label = getattr(source, "label", None)
    if label is None:
        return []
    return [str(label)]


def _rows_for_labels(
    labels: list[str],
    rows_by_label: dict[str, Any],
    rows_by_layer: dict[str, list[Any]],
    *,
    whole_layer: bool,
) -> list[Any]:
    """Collect the diff rows for a node's op labels, each row at most once."""

    selected: dict[int, Any] = {}
    for label in labels:
        if whole_layer:
            for row in rows_by_layer.get(label.split(":", 1)[0], ()):
                selected[id(row)] = row
            continue
        row = rows_by_label.get(label)
        if row is not None:
            selected[id(row)] = row
            continue
        # A layer-level label (no pass suffix) stands for all of its passes.
        for row in rows_by_layer.get(label, ()):
            selected[id(row)] = row
    return list(selected.values())


def _single_row_facts(row: Any) -> list[tuple[str, str]]:
    """Facts for a node backed by exactly one settled diff row."""

    facts = [(f"{DIFF_KEY_PREFIX}status", str(row.status))]
    if row.max_abs_delta is not None:
        facts.append((f"{DIFF_KEY_PREFIX}max_abs_delta", repr(float(row.max_abs_delta))))
    if row.reference_label is not None:
        facts.append((f"{DIFF_KEY_PREFIX}reference_label", str(row.reference_label)))
    if row.verdict is not None:
        facts.append((f"{DIFF_KEY_PREFIX}verdict", str(row.verdict)))
    if row.detail:
        facts.append((f"{DIFF_KEY_PREFIX}detail", str(row.detail)))
    return facts


def _aggregate_facts(rows: list[Any]) -> list[tuple[str, str]]:
    """Facts for a node backed by several settled rows (module call, rolled)."""

    counts = Counter(str(row.status) for row in rows)
    facts = [
        (f"{DIFF_KEY_PREFIX}row_counts", _json.dumps(dict(sorted(counts.items())))),
    ]
    deltas = [float(row.max_abs_delta) for row in rows if row.max_abs_delta is not None]
    if deltas:
        facts.append((f"{DIFF_KEY_PREFIX}max_abs_delta", repr(max(deltas))))
    return facts


def _projection_nodes(projection: NetronProjection) -> list[NetronNode]:
    """Every exported node: root graph plus every function body."""

    nodes = list(projection.nodes)
    for function in projection.functions:
        nodes.extend(function.nodes)
    return nodes


def build_baseline_diff(
    subject: Any, reference: Any, projection: NetronProjection
) -> NetronBaselineDiff:
    """Settle ``subject - reference`` and map the rows onto the projection.

    Parameters
    ----------
    subject:
        The exported trace (the delta's perspective).
    reference:
        The baseline trace.
    projection:
        The projection the artifact is emitted from.

    Returns
    -------
    NetronBaselineDiff
        The model-level summary and the per-node fact rows.
    """

    from torchlens.differential import diff

    report = diff(subject, reference)
    rows_by_label: dict[str, Any] = {}
    rows_by_layer: dict[str, list[Any]] = {}
    for row in report.rows:
        if row.subject_label is None:
            continue
        rows_by_label[row.subject_label] = row
        rows_by_layer.setdefault(row.subject_label.split(":", 1)[0], []).append(row)
    whole_layer = projection.granularity == "rolled"
    node_facts: dict[str, list[tuple[str, str]]] = {}
    for name, source in projection.node_sources.items():
        labels = _source_op_labels(source)
        rows = _rows_for_labels(labels, rows_by_label, rows_by_layer, whole_layer=whole_layer)
        if not rows:
            continue
        aggregate = len(rows) > 1 or len(labels) > 1
        node_facts[name] = _aggregate_facts(rows) if aggregate else _single_row_facts(rows[0])
    removed = [row.reference_label for row in report.rows if row.status == "removed"]
    summary: dict[str, Any] = {
        "schema": BASELINE_SUMMARY_SCHEMA,
        "source": "torchlens.report.diff",
        "direction": "subject - reference",
        "reference_model_class": str(getattr(reference, "_source_class_name", "")),
        "row_counts": dict(sorted(report.counts.items())),
        "nodes_with_diff_facts": len(node_facts),
        "removed_count": len(removed),
        "removed_labels": [str(label) for label in removed[:_REMOVED_LABEL_LIMIT]],
    }
    return NetronBaselineDiff(summary=summary, node_facts=node_facts)


def apply_node_facts(projection: NetronProjection, diff: NetronBaselineDiff) -> None:
    """Attach each node's diff fact rows to its ``NodeProto.metadataProps``."""

    for node in _projection_nodes(projection):
        facts = diff.node_facts.get(node.name)
        if facts:
            node.metadata = [*node.metadata, *facts]


def sidecar_rows(diff: NetronBaselineDiff) -> list[dict[str, Any]]:
    """Return the diff facts as ``netron:attachment`` node metadata rows."""

    rows: list[dict[str, Any]] = []
    for name, facts in diff.node_facts.items():
        for key, value in facts:
            rows.append(
                {
                    "kind": "node",
                    "target": name,
                    "name": key.removeprefix("torchlens."),
                    "value": value,
                }
            )
    return rows
