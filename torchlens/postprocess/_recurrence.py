"""Shared preview-backend recurrence grouping feed.

Single-pass preview finalization historically never grouped recurrent calls:
every op became a one-pass layer and ``trace.recurrence_detection`` was stored
as ``False``. This module ports the torch/JAX Step-7 neutral grouper
(:mod:`torchlens.postprocess.loop_grouping_adapter`) to any preview backend
whose materialized raw op logs carry structural call metadata, producing
:class:`RecurrenceAssignment` values that
:func:`torchlens.backends._finalize.finalize_single_pass_trace` applies with an
atomic relabel of every label-keyed graph structure.

Equivalence keys are computed here at finalize time (capture-time preview keys
are the bare layer type, far too coarse to claim structural correspondence).
The key mirrors the torch capture-time fingerprint: function name, output
shape/dtype and slot, the non-tensor structural argument signature, and the
module ADDRESS stack; parameterized calls key on their sorted parameter
barcodes plus the same site axes. Pseudo-ops (inputs, outputs, internal
sources) get label-unique keys and the grouper's pseudo ``func_name`` so they
can never be claimed as recurrent passes of anything.
"""

from __future__ import annotations

__tl_layer__ = "L3"

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace as _dataclass_replace
from typing import TYPE_CHECKING, Any

from ..ir.events import is_control_edge_use
from .loop_detection import _module_site, _structural_arg_signature
from .loop_grouping_adapter import (
    _PSEUDO_FUNC_NAME,
    RecurrenceAssignment,
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


def compute_preview_recurrence_assignments(
    trace: Trace,
    *,
    backend_name: str,
) -> dict[str, RecurrenceAssignment]:
    """Compute recurrence assignments for a materialized preview raw graph.

    Parameters
    ----------
    trace:
        Trace whose ``_raw_graph_ws.raw_layer_dict`` contains materialized
        preview ops (labels are raw single-pass labels; ``parents``/``children``
        hold raw labels).
    backend_name:
        Canonical backend name, namespacing the computed equivalence keys.

    Returns
    -------
    dict[str, RecurrenceAssignment]
        One assignment per raw label. Ungrouped ops receive singleton
        assignments preserving their own label.
    """

    graph = _build_preview_recurrence_graph(trace, backend_name=backend_name)
    assignments = group_recurrent_nodes(graph)
    return {
        label: assignments.get(
            label,
            RecurrenceAssignment(
                layer_label=label,
                recurrent_labels=(label,),
                pass_index=1,
                num_passes=1,
                equivalence_key=graph.nodes[label].equivalence_key,
                site_key=graph.nodes[label].site_key,
            ),
        )
        for label in graph.raw_labels
    }


def _build_preview_recurrence_graph(
    trace: Trace,
    *,
    backend_name: str,
) -> RecurrenceGroupingGraph:
    """Build the neutral recurrence graph from materialized preview raw logs.

    Parameters
    ----------
    trace:
        Trace containing materialized preview raw ops.
    backend_name:
        Canonical backend name used to namespace equivalence keys.

    Returns
    -------
    RecurrenceGroupingGraph
        Backend-neutral graph with data edges only.
    """

    raw_dict = trace._raw_graph_ws.raw_layer_dict
    raw_labels = tuple(trace._raw_graph_ws.raw_layer_labels_list)
    raw_label_set = set(raw_labels)

    data_parents_by_label: dict[str, tuple[str, ...]] = {}
    data_children_by_label: dict[str, list[str]] = {label: [] for label in raw_labels}
    for label in raw_labels:
        op_log = raw_dict[label]
        control_parents = {
            parent_label
            for edge in getattr(op_log, "_edge_uses", ()) or ()
            if is_control_edge_use(edge)
            and isinstance(parent_label := getattr(edge, "parent_label", None), str)
        }
        ordered: list[str] = []
        seen: set[str] = set()
        for parent in getattr(op_log, "parents", ()) or ():
            if (
                isinstance(parent, str)
                and parent in raw_label_set
                and parent not in seen
                and parent not in control_parents
            ):
                seen.add(parent)
                ordered.append(parent)
        data_parents_by_label[label] = tuple(ordered)
    for label, parents in data_parents_by_label.items():
        for parent in parents:
            data_children_by_label[parent].append(label)

    equivalence_keys = {
        label: _preview_equivalence_key(raw_dict[label], label, backend_name)
        for label in raw_labels
    }
    labels_by_key: dict[str, list[str]] = defaultdict(list)
    for label in raw_labels:
        labels_by_key[equivalence_keys[label]].append(label)

    nodes: dict[str, RecurrenceNode] = {}
    eligible_labels: list[str] = []
    source_labels: list[str] = []
    for raw_order, label in enumerate(raw_labels):
        op_log = raw_dict[label]
        pruned = bool(getattr(op_log, "is_orphan", False))
        retain = not pruned
        if retain:
            eligible_labels.append(label)
        if (
            getattr(op_log, "is_input", False)
            or getattr(op_log, "is_internal_source", False)
            or not data_parents_by_label[label]
        ):
            source_labels.append(label)
        uses_params = bool(getattr(op_log, "uses_params", False))
        nodes[label] = RecurrenceNode(
            label=label,
            raw_order=raw_order,
            equivalence_key=equivalence_keys[label],
            equivalent_labels=tuple(labels_by_key[equivalence_keys[label]]),
            data_parents=data_parents_by_label[label],
            data_children=tuple(data_children_by_label[label]),
            layer_label=label,
            recurrent_labels=(label,),
            uses_params=uses_params,
            func_name=(
                _PSEUDO_FUNC_NAME
                if _is_pseudo_op(op_log)
                else str(getattr(op_log, "func_name", "") or _PSEUDO_FUNC_NAME)
            ),
            param_barcodes=tuple(getattr(op_log, "_param_barcodes", ()) or ()),
            retain=retain,
            pruned=pruned,
            output_slot=(
                getattr(op_log, "multi_output_index", None)
                if getattr(op_log, "in_multi_output", False)
                else None
            ),
            module_site=_module_site(op_log),
            arg_signature=_structural_arg_signature(op_log) if uses_params else None,
            recurrence_anchored=(
                bool(getattr(op_log, "modules", None)) or bool(getattr(op_log, "is_buffer", False))
            ),
            site_key=getattr(op_log, "site_key", None),
        )

    return RecurrenceGroupingGraph(
        nodes=nodes,
        raw_labels=raw_labels,
        source_labels=tuple(source_labels),
        eligible_labels=tuple(eligible_labels),
    )


def _is_pseudo_op(op_log: Any) -> bool:
    """Return whether one preview op is a pseudo-op that can never be recurrent.

    Parameters
    ----------
    op_log:
        Materialized preview op log.

    Returns
    -------
    bool
        ``True`` for model inputs, model outputs, internal source events, and
        parentless buffer source events -- each executes once by definition.
    """

    if getattr(op_log, "is_input", False) or getattr(op_log, "is_output", False):
        return True
    if getattr(op_log, "is_internal_source", False):
        return True
    return bool(getattr(op_log, "is_buffer", False)) and not (getattr(op_log, "parents", ()) or ())


def _preview_equivalence_key(op_log: Any, label: str, backend_name: str) -> str:
    """Return the structural equivalence key for one materialized preview op.

    Parameters
    ----------
    op_log:
        Materialized preview op log.
    label:
        Raw single-pass label.
    backend_name:
        Canonical backend name namespacing the key.

    Returns
    -------
    str
        Stable structural key. Pseudo-ops get label-unique keys; parameterized
        calls key on parameter identity plus site axes; bare functional calls
        key on function name, output shape/dtype/slot, non-tensor argument
        structure, and module address stack.
    """

    if _is_pseudo_op(op_log):
        return f"{backend_name}:pseudo:{label}"
    func_name = str(getattr(op_log, "func_name", "") or "unknown")
    module_suffix = "_".join(_module_site(op_log))
    slot = (
        getattr(op_log, "multi_output_index", None)
        if getattr(op_log, "in_multi_output", False)
        else None
    )
    arg_signature = _structural_arg_signature(op_log)
    param_barcodes = tuple(getattr(op_log, "_param_barcodes", ()) or ())
    if getattr(op_log, "uses_params", False) and param_barcodes:
        param_key = "_".join(sorted(param_barcodes))
        return (
            f"{backend_name}:param:{func_name}:{param_key}:slot{slot}"
            f":argsig{arg_signature}:{module_suffix}"
        )
    shape = getattr(op_log, "shape", None)
    dtype = getattr(op_log, "dtype", None)
    return (
        f"{backend_name}:func:{func_name}:out{tuple(shape) if shape is not None else None}"
        f":{dtype}:slot{slot}:argsig{arg_signature}:{module_suffix}"
    )


def relabel_edge_metadata(
    op_log: Any,
    raw_to_final: Mapping[str, str],
) -> None:
    """Replace raw graph edge labels on one op with final pass-qualified labels.

    Parameters
    ----------
    op_log:
        Finalized preview op log.
    raw_to_final:
        Mapping from CHANGED raw labels to final pass-qualified labels. Labels
        absent from the mapping are left untouched.

    Returns
    -------
    None
        ``parents``, ``children``, ``parent_arg_positions``, and ``_edge_uses``
        are updated in place.
    """

    op_log.parents = [
        raw_to_final.get(parent, parent) if isinstance(parent, str) else parent
        for parent in op_log.parents
    ]
    op_log.children = [
        raw_to_final.get(child, child) if isinstance(child, str) else child
        for child in op_log.children
    ]
    parent_arg_positions = getattr(op_log, "parent_arg_positions", None)
    if parent_arg_positions:
        op_log.parent_arg_positions = {
            section: {
                position: (raw_to_final.get(value, value) if isinstance(value, str) else value)
                for position, value in positions.items()
            }
            for section, positions in parent_arg_positions.items()
        }
    edge_uses = getattr(op_log, "_edge_uses", None)
    if edge_uses:
        op_log._internal_set(
            "_edge_uses",
            tuple(_relabel_edge_use(edge, raw_to_final) for edge in edge_uses),
        )


def _relabel_edge_use(edge: Any, raw_to_final: Mapping[str, str]) -> Any:
    """Return one edge-use record with raw endpoint labels replaced.

    Mirrors the JAX relabel helper: materialized edge-use records carry
    ``parent_label``/``child_label`` fields; legacy tuples lead with the parent
    label. Records without a relabelable endpoint are returned unchanged.

    Parameters
    ----------
    edge:
        Edge-use record or legacy tuple.
    raw_to_final:
        Mapping from changed raw labels to final labels.

    Returns
    -------
    Any
        Relabeled edge-use record, or the original when no relabel applies.
    """

    parent_label = getattr(edge, "parent_label", None)
    child_label = getattr(edge, "child_label", None)
    if isinstance(parent_label, str) and isinstance(child_label, str):
        if parent_label not in raw_to_final and child_label not in raw_to_final:
            return edge
        return _dataclass_replace(
            edge,
            parent_label=raw_to_final.get(parent_label, parent_label),
            child_label=raw_to_final.get(child_label, child_label),
        )
    if isinstance(edge, tuple) and len(edge) >= 3 and isinstance(edge[0], str):
        return (raw_to_final.get(edge[0], edge[0]), *edge[1:])
    return edge
