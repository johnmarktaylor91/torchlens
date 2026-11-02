"""Layer view functions: per-pass aggregation and graph unions moved off ``Layer``.

Every function takes the ``Layer`` record as its first argument and reads
only the record, its passes and the owning trace; the record classes are
imported under ``TYPE_CHECKING`` only, so this module stays out of the
record import cycle. The thin ``Layer`` properties and methods forward here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

from .._constants import LAYER_LOG_FIELD_ORDER
from .._quantities import Macs
from ._layer_mirror import _layer_mirror_read, _layer_rep_op
from ._layer_spec import _LAYER_MIRROR_SPEC

if TYPE_CHECKING:
    from .layer import Layer

__all__: list[str] = []

# Per-pass fields that ``Layer`` resolves via ``_single_pass_or_error``: they
# describe ONE op/pass and deliberately raise ``ValueError`` on a multi-pass
# (recurrent) Layer, directing the user to a specific ``layer.ops[i]`` pass.
# The field-order-driven ``to_pandas`` row builders cannot let that raise, so
# they report these as None for multi-pass Layers -- mirroring
# ``_MULTI_CALL_PER_PASS_MODULE_FIELDS`` one level up in module.py. No
# information is lost: the per-pass value remains reachable via
# ``layer.ops[i].transformed_out`` / ``.transformed_grad``.
_MULTI_PASS_PER_CALL_LAYER_FIELDS: frozenset[str] = frozenset(
    {
        "transformed_out",
        "transformed_grad",
    }
)


def materialize_layer_mirrors(layer_log: Layer) -> None:
    """Materialize every unmaterialized mirror field into ``__dict__``.

    Called before the backing ops are husked (``Trace.cleanup()``, log-entry
    removal): a user-held Layer keeps exactly the readable state the dict-era
    copies would have kept. Tombstoned and already-shadowed fields are left
    untouched; unreadable mirrors (an already-scrubbed op cell) stay absent,
    matching a field the dict era had already deleted.
    """

    instance_dict = layer_log.__dict__
    for field_name in _LAYER_MIRROR_SPEC:
        if field_name in instance_dict:
            continue
        try:
            instance_dict[field_name] = _layer_mirror_read(layer_log, field_name)
        except AttributeError:
            continue
    if "equivalent_ops" not in instance_dict:
        rep = _layer_rep_op(layer_log)
        if rep is not None:
            try:
                instance_dict["equivalent_ops"] = rep.equivalent_ops
            except AttributeError:
                pass


def _layer_log_to_row(layer_log: Layer) -> dict[str, Any]:
    """Convert a Layer into one DataFrame row.

    Parameters
    ----------
    layer_log:
        Layer metadata entry to export.

    Returns
    -------
    Dict[str, Any]
        Mapping from canonical field name to exported value, ordered by
        ``LAYER_LOG_FIELD_ORDER``. Per-pass fields
        (``_MULTI_PASS_PER_CALL_LAYER_FIELDS``) are reported as ``None`` for
        multi-pass Layers so the table never raises.
    """

    multi_pass = layer_log.num_passes > 1
    row: dict[str, Any] = {}
    for field_name in LAYER_LOG_FIELD_ORDER:
        if multi_pass and field_name in _MULTI_PASS_PER_CALL_LAYER_FIELDS:
            row[field_name] = None
            continue
        row[field_name] = getattr(layer_log, field_name)
    return row


def layer_total_macs_forward(layer: Layer) -> Macs | None:
    """Sum TRUE forward MACs across all Ops in the layer (``None`` on any unknown split)."""

    per_op = [op.macs_forward for op in layer.ops.values()]
    if any(value is None for value in per_op):
        return None
    return Macs(sum(int(value) for value in per_op if value is not None))


# ********************************************
# ***** Aggregate graph properties ***********
# ********************************************
# Graph-edge views compute the union across all ops, returning no-pass
# labels (i.e. Layer-level identifiers).  This gives a complete picture of
# which layers are connected across all recurrent iterations.  Order is
# preserved (first-seen insertion order).


def relation_no_pass_label(layer: Layer, label: str) -> str | None:
    """Resolve one per-op relation label to its no-pass layer label.

    Mirrors the op-level orphan tolerance (r3 b3-fable R05-N1): the
    per-op ``siblings``/``co_parents`` properties deliberately resolve
    relation labels through the ``orphans`` fallback on
    ``keep_orphans=True`` traces, so the Layer aggregate must fold the
    same labels instead of crashing on a bare mainline lookup. A label
    resolvable through neither surface is skipped, matching the op-level
    behavior.
    """

    trace = layer.source_trace
    try:
        return trace[label].layer_label
    except (KeyError, ValueError):
        try:
            return trace.graph.orphans[label].layer_label
        except KeyError:
            return None


def layer_children(layer: Layer) -> tuple[str, ...]:
    """Union of child layers (no-pass labels) across all ops, as an immutable tuple."""

    result = []
    seen = set()
    for pass_log in layer.ops.values():
        for label in pass_log.children:
            no_pass = relation_no_pass_label(layer, label)
            if no_pass is not None and no_pass not in seen:
                seen.add(no_pass)
                result.append(no_pass)
    return tuple(result)


def layer_parents(layer: Layer) -> tuple[str, ...]:
    """Union of parent layers (no-pass labels) across all ops, as an immutable tuple."""

    result = []
    seen = set()
    for pass_log in layer.ops.values():
        for label in pass_log.parents:
            no_pass = relation_no_pass_label(layer, label)
            if no_pass is not None and no_pass not in seen:
                seen.add(no_pass)
                result.append(no_pass)
    return tuple(result)


def layer_siblings(layer: Layer) -> list[str]:
    """Union of sibling layers (no-pass labels) across all ops, excluding the layer itself."""

    result = []
    seen = {layer.layer_label}
    for pass_log in layer.ops.values():
        for label in pass_log.siblings:
            no_pass = relation_no_pass_label(layer, label)
            if no_pass is not None and no_pass not in seen:
                seen.add(no_pass)
                result.append(no_pass)
    return result


def layer_co_parents(layer: Layer) -> list[str]:
    """Union of spouse layers (no-pass labels) across all ops, excluding the layer itself."""

    result = []
    seen = {layer.layer_label}
    for pass_log in layer.ops.values():
        for label in pass_log.co_parents:
            no_pass = relation_no_pass_label(layer, label)
            if no_pass is not None and no_pass not in seen:
                seen.add(no_pass)
                result.append(no_pass)
    return result


# ********************************************
# **** Rolled-vis computed properties ********
# ********************************************
# These provide per-pass edge tracking for rolled (recurrence-aware)
# graph visualization.  Computed on-the-fly from the ops dict.
# Used by the visualization renderer to draw pass-annotated edges.


def layer_children_per_pass(layer: Layer) -> dict[int, list[str]]:
    """Dict[int, List[str]]: child layer labels (no-pass) for each pass."""

    result = {}
    for call_index, pass_log in layer.ops.items():
        children = []
        for label in pass_log.children:
            no_pass = relation_no_pass_label(layer, label)
            if no_pass is not None and no_pass not in children:
                children.append(no_pass)
        result[call_index] = children
    return result


def layer_parents_per_pass(layer: Layer) -> dict[int, list[str]]:
    """Dict[int, List[str]]: parent layer labels (no-pass) for each pass."""

    result = {}
    for call_index, pass_log in layer.ops.items():
        parents = []
        for label in pass_log.parents:
            no_pass = relation_no_pass_label(layer, label)
            if no_pass is not None and no_pass not in parents:
                parents.append(no_pass)
        result[call_index] = parents
    return result


def layer_child_ops_per_layer(layer: Layer) -> dict[str, list[int]]:
    """Dict[str, List[int]]: for each child layer, which ops connect to it."""

    result: defaultdict[str, list[int]] = defaultdict(list)
    for call_index, pass_log in layer.ops.items():
        for label in pass_log.children:
            no_pass = relation_no_pass_label(layer, label)
            if no_pass is not None and call_index not in result[no_pass]:
                result[no_pass].append(call_index)
    return dict(result)


def layer_parent_ops_per_layer(layer: Layer) -> dict[str, list[int]]:
    """Dict[str, List[int]]: for each parent layer, which ops connect from it."""

    result: defaultdict[str, list[int]] = defaultdict(list)
    for call_index, pass_log in layer.ops.items():
        for label in pass_log.parents:
            no_pass = relation_no_pass_label(layer, label)
            if no_pass is not None and call_index not in result[no_pass]:
                result[no_pass].append(call_index)
    return dict(result)


def layer_edges_vary_across_ops(layer: Layer) -> bool:
    """Whether graph edges differ across ops."""

    if layer.num_passes <= 1:
        return False
    all_pass_lists = list(layer.child_ops_per_layer.values()) + list(
        layer.parent_ops_per_layer.values()
    )
    return any(len(ops) < layer.num_passes for ops in all_pass_lists)


def layer_leaf_module_ops(layer: Layer) -> set[Any]:
    """Set of module ops exited across all ops."""

    result = set()
    for pass_log in layer.ops.values():
        if pass_log.is_atomic_module:
            result.add(pass_log.atomic_module_call)
    return result


def layer_parent_arg_positions(layer: Layer) -> dict[str, dict[Any, str]]:
    """Merged parent_arg_positions across ops (set-union).

    For single-pass layers, delegates to ops[0].
    For multi-pass, merges arg locs using set-union of no-pass labels.
    """

    if layer.num_passes == 1:
        return cast(dict[str, dict[Any, str]], layer.ops[0].parent_arg_positions)

    result: dict[str, dict[Any, str]] = {"args": {}, "kwargs": {}}
    for pass_log in layer.ops.values():
        for arg_type in ["args", "kwargs"]:
            for arg_key, layer_label in pass_log.parent_arg_positions[arg_type].items():
                no_pass = layer.source_trace[layer_label].layer_label
                if arg_key not in result[arg_type]:
                    result[arg_type][arg_key] = no_pass
    return result


# ********************************************
# ******* Relation record resolution *********
# ********************************************


def resolve_relation_record(layer: Layer, label: str) -> Layer | None:
    """Resolve one relation label through the mainline, then orphans.

    Mirrors the op-level orphan tolerance (b3 R05-N2): the aggregates on
    a ``keep_orphans=True`` trace include orphan relation labels, so the
    object-resolving surfaces must fold the ``orphans`` fallback instead
    of crashing on the bare mainline lookup. An orphan label resolves to
    its retained orphan Op record (the only record that exists for it),
    presented through the shared record surface.
    """

    trace = layer.source_trace
    try:
        return cast("Layer", trace[label])
    except (KeyError, ValueError):
        try:
            return cast("Layer", trace.graph.orphans[label])
        except KeyError:
            return None


def layer_get_children(layer: Layer) -> list[Layer]:
    """Child Layer objects resolved through the owning trace (unresolvable skipped)."""

    resolved = (resolve_relation_record(layer, label) for label in layer.children)
    return [record for record in resolved if record is not None]


def layer_get_parents(layer: Layer) -> list[Layer]:
    """Parent Layer objects resolved through the owning trace (unresolvable skipped)."""

    resolved = (resolve_relation_record(layer, label) for label in layer.parents)
    return [record for record in resolved if record is not None]
