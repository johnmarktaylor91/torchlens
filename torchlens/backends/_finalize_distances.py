"""Input and output hop distances for preview-backend traces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import op_attr

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def compute_preview_input_output_distances(trace: Trace) -> None:
    """Run torch's Step-4 depth flood over a finalized preview graph.

    Parameters
    ----------
    trace:
        Finalized preview trace whose ops carry label-based ``parents``/
        ``children`` edges and populated ``input_layers``/``output_layers``.

    Returns
    -------
    None
        ``min/max_distance_from_input``, ``min/max_distance_to_output``,
        ``input_ancestors``, and ``output_descendants`` are populated in place.

    Notes
    -----
    This mirrors ``postprocess._layer_depths._mark_layer_depths`` but
    resolves labels through an explicit op index instead of
    ``Trace.__getitem__`` (whose finished-mode lookup returns ``Layer``
    objects, not the ops the flood must mutate). Lineage sets are normalized
    first because backends may emit immutable placeholders (e.g. TF's
    ``input_ancestors=()``).
    """

    ops_by_label = _index_ops_by_label(trace)
    ordered_ops = [
        ops_by_label[label]
        for label in trace._raw_graph_ws.raw_layer_labels_list
        if label in ops_by_label
    ]
    for mode, starting_labels, min_field, max_field, marker_field, lineage_field, edge_field in (
        (
            "input",
            trace.input_layers,
            op_attr("min_distance_from_input"),
            op_attr("max_distance_from_input"),
            op_attr("has_input_ancestor"),
            "input_ancestors",
            "children",
        ),
        (
            "output",
            trace.output_layers,
            op_attr("min_distance_to_output"),
            op_attr("max_distance_to_output"),
            op_attr("has_output_descendant"),
            "output_descendants",
            "parents",
        ),
    ):
        traversal = ordered_ops if mode == "input" else list(reversed(ordered_ops))
        fields = (min_field, max_field, marker_field, lineage_field)
        _seed_starting_ops(ops_by_label, starting_labels, fields)
        _flood_edges(traversal, ops_by_label, fields, edge_field)


def _index_ops_by_label(trace: Trace) -> dict[str, Any]:
    """Index a preview trace's ops by every label spelling, normalizing lineage sets.

    Parameters
    ----------
    trace:
        Finalized preview trace.

    Returns
    -------
    dict[str, Any]
        Ops keyed by raw label, label and layer label (first op wins). Each op's
        ``input_ancestors``/``output_descendants`` is made a mutable set in place.
    """

    ops_by_label: dict[str, Any] = {}
    for op_log in trace._layer_list:
        for field_name in ("input_ancestors", "output_descendants"):
            value = getattr(op_log, field_name, None)
            if not isinstance(value, set):
                setattr(op_log, field_name, set(value or ()))
        for key in (
            getattr(op_log, "_label_raw", None),
            getattr(op_log, "label", None),
            getattr(op_log, "layer_label", None),
        ):
            if key is not None:
                ops_by_label.setdefault(key, op_log)
    return ops_by_label


def _seed_starting_ops(
    ops_by_label: dict[str, Any], starting_labels: Any, fields: tuple[str, str, str, str]
) -> None:
    """Mark the flood's starting ops at distance zero with themselves as lineage.

    Parameters
    ----------
    ops_by_label:
        Ops keyed by label.
    starting_labels:
        Input or output layer labels the flood starts from.
    fields:
        ``(min_field, max_field, marker_field, lineage_field)`` of the flood direction.

    Returns
    -------
    None
        Starting ops are updated in place; unknown labels are skipped.
    """

    min_field, max_field, marker_field, lineage_field = fields
    for starting_label in starting_labels:
        starting_op = ops_by_label.get(starting_label)
        if starting_op is None:
            continue
        _update_distance(starting_op, min_field, max_field, 0)
        setattr(starting_op, marker_field, True)
        getattr(starting_op, lineage_field).add(starting_label)


def _flood_edges(
    traversal: list[Any],
    ops_by_label: dict[str, Any],
    fields: tuple[str, str, str, str],
    edge_field: str,
) -> None:
    """Propagate hop distances and lineage along one direction's edges.

    Parameters
    ----------
    traversal:
        Ops in flood order (forward for input distances, reversed for output).
    ops_by_label:
        Ops keyed by label.
    fields:
        ``(min_field, max_field, marker_field, lineage_field)`` of the flood direction.
    edge_field:
        ``"children"`` or ``"parents"``: the edge list followed from each op.

    Returns
    -------
    None
        Reached ops are updated in place.
    """

    min_field, max_field, marker_field, lineage_field = fields
    for op_log in traversal:
        current_min = getattr(op_log, min_field, None)
        current_max = getattr(op_log, max_field, None)
        if current_min is None or current_max is None:
            continue
        lineage = getattr(op_log, lineage_field)
        for next_label in getattr(op_log, edge_field, ()) or ():
            next_op = ops_by_label.get(next_label)
            if next_op is None:
                continue
            _update_distance(next_op, min_field, max_field, current_min + 1)
            _update_distance(next_op, min_field, max_field, current_max + 1)
            setattr(next_op, marker_field, True)
            getattr(next_op, lineage_field).update(lineage)


def _update_distance(op_log: Any, min_field: str, max_field: str, hops: int) -> None:
    """Fold one candidate hop count into an op's min/max distance fields.

    Parameters
    ----------
    op_log:
        Op receiving the update.
    min_field, max_field:
        Distance attribute names for the active flood direction.
    hops:
        Candidate hop count from the flood frontier.

    Returns
    -------
    None
        The op's distance fields are widened in place.
    """

    current_min = getattr(op_log, min_field, None)
    current_max = getattr(op_log, max_field, None)
    setattr(op_log, min_field, hops if current_min is None else min(current_min, hops))
    setattr(op_log, max_field, hops if current_max is None else max(current_max, hops))
