"""Append newly discovered ``GradFn`` records to a backward projection in place.

Each ``torch.autograd.grad`` or ``backward`` call on a live trace usually
discovers a few new nodes (the indexing or reduction that built its root) in
front of a graph the projection already holds. ``_plan_backward_tail_fold``
first checks that the events past the projection's watermark change no
existing record or built pass (``_backward_tail_is_foldable``), collecting the
new-node discoveries; ``_plan_grad_fn_growth`` builds
those records exactly as a scratch rebuild would and returns ``None`` whenever
appending them could change an existing record, so the caller rebuilds from
scratch instead; ``_commit_grad_fn_growth`` appends a plan. The scratch
rebuild builds its records with the same ``_grad_fn_record_from_discovery``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from ..._data_classes._grad_fn_naming import _grad_fn_label_parts, _grad_fn_type_from_class_name
from ..._data_classes.grad_fn import GradFn
from ..._ir.events import (
    BackwardPassEnd,
    BackwardPassStart,
    CheckpointInvocationObserved,
    GradFnDiscovered,
    GradFnFired,
    OpGradObserved,
    ParamGradObserved,
)
from ..._utils._record_group_table import record_member
from ._grad_fn_graph import _extend_grad_fn_graph_relations, _rename_appended_labels
from ._grad_fn_membership import (
    MembershipSweeps,
    _op_module_address,
    _param_module_address,
    _replay_membership_for_new_grad_fns,
)
from ._grad_fn_orders import _assign_new_grad_fn_orders


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
    module_address = _discovery_module_address(trace, object_id, event)
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


def _discovery_module_address(trace: Any, object_id: int, event: GradFnDiscovered) -> str | None:
    """Return the module a discovery pairs with: its op's, else its parameter's."""

    module_address = _op_module_address(trace, event.op_label, object_id)
    if module_address is None:
        module_address = _param_module_address(trace, event.param_ref)
    return module_address


def _layer_grad_fn_ids(trace: Any) -> frozenset[int]:
    """Return every autograd identity a forward op or layer records."""

    return frozenset(
        grad_fn_object_id
        for layer in (*trace._layer_list, *trace._layer_logs.values())
        if (grad_fn_object_id := record_member(layer, "grad_fn_object_id", None)) is not None
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


def _rediscovery_changes_node(
    existing: Any, event: Any, latest_topology: dict[int, tuple[int, ...]]
) -> bool:
    """Return whether a rediscovery fills a field or changes the topology of a known node."""

    for field in ("op_label", "param_ref", "created_in_pass", "creator_object_id"):
        if getattr(existing, field) is None and getattr(event, field) is not None:
            return True
    return tuple(event.topology) != tuple(latest_topology.get(event.object_id, ()))


def _backward_tail_is_foldable(
    state: Any, tail: Sequence[Any], new_events: list[Any] | None = None
) -> bool:
    """Return whether an appended event tail can fold without a full rebuild.

    Foldable means the tail introduces no new grad-fn node (with
    ``new_events``, new-node discoveries are collected there instead, for
    ``_plan_grad_fn_growth`` to accept or refuse), no merge that
    would retroactively change an already-assigned label or field, no
    topology change (which would require re-running the global order and
    module-inference passes), and no event for an already-built pass. Any
    other tail falls back to the byte-identical full rebuild.
    """

    max_built = max(state.built_pass_indices, default=0)
    for event in tail:
        if isinstance(event, BackwardPassStart):
            if event.pass_index <= max_built or event.pass_index in state.starts:
                return False
        elif isinstance(event, BackwardPassEnd):
            if event.pass_index <= max_built or event.pass_index in state.ends:
                return False
        elif isinstance(event, GradFnDiscovered):
            existing = state.discovered.get(event.object_id)
            if existing is None:
                if new_events is None:
                    return False
                new_events.append(event)
                continue
            if _rediscovery_changes_node(existing, event, state.latest_topology):
                return False
        elif isinstance(event, (GradFnFired, OpGradObserved, ParamGradObserved)):
            if event.pass_index <= max_built:
                return False
        elif isinstance(event, CheckpointInvocationObserved):
            # Projection-neutral: token evidence lives in runtime state and
            # the witness field, never in the folded records.
            continue
        else:
            return False
    return True


def _plan_backward_tail_fold(
    trace: Any, state: Any, stream_events: Sequence[Any], watermark: Any
) -> tuple[list[Any], _GradFnGrowth] | None:
    """Return the events past ``watermark`` and their growth plan, or ``None`` to rebuild.

    Parameters
    ----------
    trace:
        Trace whose projection is folding.
    state:
        The projection's ``_BackwardFoldState``.
    stream_events:
        The live backward event list; only its tail is read.
    watermark:
        Event count the projection already holds.

    Returns
    -------
    tuple[list[Any], _GradFnGrowth] | None
        The tail and the records it adds, or ``None`` when only a rebuild is exact.
    """

    if not isinstance(watermark, int) or not 0 < watermark < len(stream_events):
        return None
    tail = list(stream_events[watermark:])
    new_events: list[GradFnDiscovered] = []
    if not _backward_tail_is_foldable(state, tail, new_events):
        return None
    growth = _plan_grad_fn_growth(trace, state, new_events)
    return None if growth is None else (tail, growth)


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
    events = _appendable_discoveries(new_events, grad_fn_logs, state)
    if events is None:
        return None
    type_counter = dict(state.type_counter)
    records: OrderedDict[int, GradFn] = OrderedDict()
    for ordinal_index, (object_id, event) in enumerate(events.items(), len(state.discovered)):
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


def _appendable_discoveries(
    new_events: Sequence[GradFnDiscovered], grad_fn_logs: Mapping[int, GradFn], state: Any
) -> dict[int, GradFnDiscovered] | None:
    """Return the first discovery of each new node, or ``None`` if one cannot be appended.

    A node is appendable when its repeat discoveries agree, it is unpaired and
    creator-less, and no existing record, forward op or layer names it. The
    live discovery walk may already hold a runtime record for a new node after
    the projection's records; any other record outside the projection refuses.
    """

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
            or object_id in state.discovered
            or object_id in state.referenced_ids
            or object_id in state.layer_grad_fn_ids
        ):
            return None
    runtime_records = sum(object_id in grad_fn_logs for object_id in events)
    if len(grad_fn_logs) != len(state.discovered) + runtime_records:
        return None
    return events


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
    """Append a planned growth to the trace's records and the fold state.

    A runtime record the live walk made for a new node is replaced in place,
    and its label is replaced in the relations the walk's sync gave it.
    """

    if not growth.records:
        return
    grad_fn_logs = trace._grad_fn_logs
    renames = {
        runtime.label: grad_fn_record.label
        for object_id, grad_fn_record in growth.records.items()
        if (runtime := grad_fn_logs.get(object_id)) is not None
        and runtime.label != grad_fn_record.label
    }
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
    _rename_appended_labels(grad_fn_logs, list(growth.records), renames, state.label_to_id)
    _extend_grad_fn_graph_relations(grad_fn_logs, list(growth.records), state.label_to_id)


def _fold_created_records(
    trace: Any,
    growth: _GradFnGrowth,
    pass_to_calls: dict[int, list[Any]],
    new_pass_indices: list[int],
) -> tuple[list[Any], ...]:
    """Return the grad-fn, call and pass records one tail fold created."""

    new_calls = [call for calls in pass_to_calls.values() for call in calls]
    new_passes = [trace._backward_pass_logs[pass_index] for pass_index in new_pass_indices]
    return list(growth.records.values()), new_calls, new_passes
