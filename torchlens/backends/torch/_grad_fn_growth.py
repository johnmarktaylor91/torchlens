"""Append newly discovered ``GradFn`` records to a backward projection in place.

Each ``torch.autograd.grad`` or ``backward`` call on a live trace usually
discovers a few new nodes (the indexing or reduction that built its root) in
front of a graph the projection already holds. ``_plan_grad_fn_growth`` builds
those records exactly as a scratch rebuild would and returns ``None`` whenever
appending them could change an existing record, so the caller rebuilds from
scratch instead; ``_commit_grad_fn_growth`` appends a plan. The scratch
rebuild builds its records with the same ``_grad_fn_record_from_discovery``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ..._data_classes._grad_fn_naming import _grad_fn_label_parts, _grad_fn_type_from_class_name
from ..._data_classes.grad_fn import GradFn
from ._grad_fn_graph import _extend_grad_fn_graph_relations
from ._grad_fn_membership import (
    MembershipSweeps,
    _op_module_address,
    _param_module_address,
    _replay_membership_for_new_grad_fns,
)
from ._grad_fn_orders import _assign_new_grad_fn_orders

if TYPE_CHECKING:
    from ..._ir.events import GradFnDiscovered


@dataclass
class _GradFnGrowth:
    """New records planned for one fold, in discovery order."""

    records: OrderedDict[int, GradFn]
    events: dict[int, GradFnDiscovered]
    type_counter: dict[str, int]
    membership_sweeps: MembershipSweeps


def _grad_fn_record_from_discovery(
    trace: Any,
    object_id: int,
    event: GradFnDiscovered,
    topology: Sequence[int],
    ordinal_index: int,
    type_counter: dict[str, int],
) -> GradFn:
    """Build the GradFn record a discovery projects to; advances ``type_counter``.

    Parameters
    ----------
    trace:
        Trace that owns the record.
    object_id:
        Autograd identity of the node.
    event:
        The node's merged discovery.
    topology:
        The node's latest next-function ids.
    ordinal_index:
        Zero-based discovery index among the trace's records.
    type_counter:
        Running per-type label counter.

    Returns
    -------
    GradFn
        The record, with its paired module (if any) and no order or relations.
    """

    grad_fn_type = _grad_fn_type_from_class_name(event.class_name)
    type_index, step_index, label = _grad_fn_label_parts(
        trace, grad_fn_type, event.op_label, type_counter, ordinal_index + 1
    )
    module_address = _op_module_address(trace, event.op_label, object_id)
    if module_address is None:
        module_address = _param_module_address(trace, event.param_ref)
    grad_fn_record = GradFn(
        grad_fn_object_id=object_id,
        class_name=event.class_name,
        class_qualname=event.class_qualname,
        label=label,
        type=grad_fn_type,
        type_index=type_index,
        ordinal_index=ordinal_index,
        step_index=step_index,
        is_custom=event.is_custom,
        has_op=event.op_label is not None,
        op_label=event.op_label,
        order=None,
        origin_backward_pass=event.created_in_pass,
        creator_object_id=event.creator_object_id,
        modules=[] if module_address is None else [module_address],
        module_address=module_address,
        module_membership_source=None if module_address is None else "paired",
        next_grad_fn_ids=list(topology),
        **cast(dict[str, Any], dict(event.source)),
    )
    grad_fn_record.trace = trace
    return grad_fn_record


def _layer_grad_fn_ids(trace: Any) -> frozenset[int]:
    """Return every autograd identity a forward op or layer records."""

    return frozenset(
        grad_fn_object_id
        for layer in (*trace._layer_list, *trace._layer_logs.values())
        if (grad_fn_object_id := getattr(layer, "grad_fn_object_id", None)) is not None
    )


def _merge_fields(event: GradFnDiscovered) -> tuple[Any, ...]:
    """Return the discovery fields a scratch rebuild merges across repeat events."""

    return (
        event.op_label,
        event.param_ref,
        event.created_in_pass,
        event.creator_object_id,
        tuple(event.topology),
    )


def _plan_grad_fn_growth(
    trace: Any, state: Any, new_events: Sequence[GradFnDiscovered]
) -> _GradFnGrowth | None:
    """Plan the records ``new_events`` add, or return ``None`` when only a rebuild is exact.

    Appending is exact when each new node is an unpaired, creator-less node
    that no forward op records and no existing record feeds or names as its
    creator, the projection has no preserved pre-detach records, and the
    replayed orders and module sweeps leave every existing record as it is.

    Parameters
    ----------
    trace:
        Trace whose projection is folding.
    state:
        The projection's ``_BackwardFoldState``.
    new_events:
        Tail discoveries of ids the projection does not hold, in order.

    Returns
    -------
    _GradFnGrowth | None
        The planned records, or ``None``.
    """

    if not new_events:
        return _GradFnGrowth(OrderedDict(), {}, dict(state.type_counter), {})
    grad_fn_logs = trace._grad_fn_logs
    if not state.grows_in_place or not state.orders_converged:
        return None
    events: dict[int, GradFnDiscovered] = {}
    for event in new_events:
        first = events.setdefault(event.object_id, event)
        if _merge_fields(first) != _merge_fields(event):
            return None
    for object_id, event in events.items():
        if (
            event.op_label is not None
            or event.param_ref is not None
            or event.creator_object_id is not None
            or object_id in grad_fn_logs
            or object_id in state.referenced_ids
            or object_id in state.layer_grad_fn_ids
        ):
            return None
    type_counter = dict(state.type_counter)
    records: OrderedDict[int, GradFn] = OrderedDict()
    for ordinal_index, (object_id, event) in enumerate(events.items(), start=len(grad_fn_logs)):
        records[object_id] = _grad_fn_record_from_discovery(
            trace, object_id, event, event.topology, ordinal_index, type_counter
        )
    if not _assign_new_grad_fn_orders(grad_fn_logs, records):
        return None
    membership_sweeps = _replay_membership_for_new_grad_fns(
        trace,
        grad_fn_logs,
        records,
        _new_record_neighbors(grad_fn_logs, records),
        state.membership_sweeps,
    )
    if membership_sweeps is None:
        return None
    return _GradFnGrowth(records, events, type_counter, membership_sweeps)


def _new_record_neighbors(
    grad_fn_logs: Mapping[int, GradFn], records: Mapping[int, GradFn]
) -> dict[int, list[int]]:
    """Return each new record's parent ids then child ids, as the scratch sweep reads them."""

    children = {
        object_id: [
            child_id
            for child_id in record.next_grad_fn_ids
            if child_id in records or child_id in grad_fn_logs
        ]
        for object_id, record in records.items()
    }
    parents: dict[int, list[int]] = {object_id: [] for object_id in records}
    for object_id, child_ids in children.items():
        for child_id in child_ids:
            if child_id in parents and object_id not in parents[child_id]:
                parents[child_id].append(object_id)
    return {object_id: [*parents[object_id], *children[object_id]] for object_id in records}


def _commit_grad_fn_growth(trace: Any, state: Any, growth: _GradFnGrowth) -> None:
    """Append a planned growth to the trace's records and the fold state."""

    if not growth.records:
        return
    grad_fn_logs = trace._grad_fn_logs
    for object_id, grad_fn_record in growth.records.items():
        grad_fn_logs[object_id] = grad_fn_record
        event = growth.events[object_id]
        state.discovered[object_id] = event
        state.latest_topology[object_id] = event.topology
        state.referenced_ids.update(grad_fn_record.next_grad_fn_ids)
        state.label_to_id[grad_fn_record.label] = object_id
    trace._autograd_grad_fn_order = list(grad_fn_logs)
    state.type_counter = growth.type_counter
    state.membership_sweeps.update(growth.membership_sweeps)
    _extend_grad_fn_graph_relations(grad_fn_logs, list(growth.records), state.label_to_id)
