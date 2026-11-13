"""Backward records built from backend-neutral backward events.

A native backward lane (``backends/_backward_native/<name>.py``) observes one
executed backward pass in its framework and journals what it saw as the same
events PyTorch's backward capture emits (``torchlens._ir.events``):
``BackwardPassStart``; one ``GradFnDiscovered`` per backward node (its
framework's own rule name in ``class_name``, the forward op it differentiates
in ``op_label``, its next nodes in ``topology``); one ``GradFnFired`` per
executed call; ``OpGradObserved`` and ``ParamGradObserved`` for gradient
values; ``BackwardPassEnd``. This module folds those events into ``GradFn``,
``GradFnCall``, ``BackwardPass``, ``Op.grad`` and the per-op and per-param
gradient records, with PyTorch's meanings: the fold below is the scratch
rebuild of ``backends/torch/backward.py`` without its torch-only parts (live
autograd handles, storage pointers, activation checkpoint tokens).

Nothing is filled that was not observed. A ``GradFnCall`` exists only for a
journaled fire, so it always describes a call that ran; timing fields, peak
memory and engine flags come from the events and stay ``None`` when the event
carries ``None``.

The derived parts (labels, orders, graph relations, module membership) reuse
the torch backend's torch-free helpers, so the two backends cannot drift.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .._data_classes.backward_pass import BackwardPass
from .._data_classes.grad_fn import GradFn
from .._data_classes.grad_fn_call import GradFnCall
from .._ir.events import (
    BackwardPassEnd,
    BackwardPassStart,
    GradFnDiscovered,
    GradFnFired,
    OpGradObserved,
    ParamGradObserved,
)
from .._quantities import Bytes, Duration
from .._trace_core.record_rows import BackwardEpoch, adopt_rows
from .._utils._record_group_table import record_member
from ..errors import RunCapabilityUnavailableError
from .torch._grad_fn_graph import _sync_grad_fn_graph_relations
from .torch._grad_fn_growth import _grad_fn_record_from_discovery
from .torch._grad_fn_membership import (
    _infer_grad_fn_module_membership,
    _prefer_direct_pairing_source_for_paired_ops,
)
from .torch._grad_fn_orders import (
    _assign_grad_fn_orders,
    _max_call_order,
    _order_attribution_coverage,
    _pass_order_from_roots,
)
from .torch._grad_fn_pairing import _pair_discovered_with_layers

__all__ = (
    "NEUTRAL_PAYLOADS",
    "PayloadPolicy",
    "build_backward_records",
    "journal_backward_event",
    "materialize_neutral_backward",
    "next_backward_pass_index",
    "set_pass_roots",
)


def _attr_shape(value: Any) -> tuple[int, ...] | None:
    """Return ``tuple(value.shape)`` for an array, else ``None``."""

    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(int(dim) for dim in shape)


def _attr_memory(value: Any) -> int | None:
    """Return ``value.nbytes`` for an array that reports it, else ``None``."""

    nbytes = getattr(value, "nbytes", None)
    return None if nbytes is None else int(nbytes)


def _neutral_retained_refs(
    raw: Any, transformed: Any, *, raw_memory: int | None
) -> list[tuple[Any, int]]:
    """Charge each retained payload once, keyed by identity."""

    refs: list[tuple[Any, int]] = []
    if raw is not None:
        refs.append((("id", id(raw)), int(raw_memory or _attr_memory(raw) or 0)))
    if transformed is not None and (memory := _attr_memory(transformed)) is not None:
        refs.append((("id", id(transformed)), memory))
    return refs


@dataclass(frozen=True)
class PayloadPolicy:
    """How a backend's gradient payloads are read; one instance per backend.

    Parameters
    ----------
    is_payload:
        Whether an event's ``payload_ref`` is a retained gradient value.
    dtype_from_string:
        Maps an event's dtype string to the value stored on ``Op``.
    shape_of, dtype_of, memory_of:
        Metadata of a transformed payload, ``None`` when not an array.
    retained_refs:
        ``(dedupe key, bytes)`` per retained payload, for the trace's memory totals.
    """

    is_payload: Callable[[Any], bool] = lambda value: value is not None
    dtype_from_string: Callable[[str], Any] = lambda name: name
    shape_of: Callable[[Any], Any] = _attr_shape
    dtype_of: Callable[[Any], Any] = lambda value: getattr(value, "dtype", None)
    memory_of: Callable[[Any], Any] = _attr_memory
    retained_refs: Callable[..., list[tuple[Any, int]]] = _neutral_retained_refs


NEUTRAL_PAYLOADS = PayloadPolicy()
"""Policy for array types exposing ``shape``, ``dtype`` and ``nbytes`` (NumPy, JAX, MLX)."""


def _event_stream(trace: Any) -> Any:
    """Return the trace's capture event bundle, refusing typed when it was released."""

    for name in ("event_stream", "_capture_events", "capture_events"):
        stream = getattr(trace, name, None)
        if stream is not None:
            return stream
    raise RunCapabilityUnavailableError(
        "This trace no longer owns a capture event stream, so backward capture cannot "
        "record events. Capture a fresh trace before calling backward-capture APIs.",
        code="run_capability_unavailable",
        detection_stage="native_backward_capture",
    )


def journal_backward_event(trace: Any, event: Any) -> None:
    """Append one observed backward event to ``trace``'s event stream (stamps ``seq``)."""

    _event_stream(trace).append_backward(event)


def next_backward_pass_index(trace: Any) -> int:
    """Return the one-based number the next backward pass on ``trace`` takes."""

    stream = _event_stream(trace)
    seen = [getattr(event, "pass_index", 0) or 0 for event in stream.backward_events]
    floor = max(int(stream.pass_index_base or 0), int(trace._num_backward_passes or 0))
    return max([floor, *seen]) + 1


def set_pass_roots(trace: Any, pass_index: int, root_ids: Iterable[int]) -> None:
    """Record the backward nodes a pass started from (session-only, as on PyTorch)."""

    trace.__dict__.setdefault("_backward_roots_by_pass", {})[pass_index] = list(root_ids)


@dataclass
class _Scan:
    """Events of one stream, sorted by kind; ``discovered`` holds merged discoveries."""

    starts: dict[int, BackwardPassStart] = field(default_factory=dict)
    ends: dict[int, BackwardPassEnd] = field(default_factory=dict)
    discovered: OrderedDict[int, GradFnDiscovered] = field(default_factory=OrderedDict)
    topology: dict[int, tuple[int, ...]] = field(default_factory=dict)
    fired: list[GradFnFired] = field(default_factory=list)
    op_grads: list[OpGradObserved] = field(default_factory=list)
    param_grads: list[ParamGradObserved] = field(default_factory=list)


_MERGED_FIELDS = ("op_label", "param_ref", "created_in_pass", "creator_object_id")


def _merge_discovery(existing: GradFnDiscovered, event: GradFnDiscovered) -> GradFnDiscovered:
    """Fill fields a rediscovery knows and the first discovery did not."""

    updates = {
        name: getattr(event, name)
        for name in _MERGED_FIELDS
        if getattr(existing, name) is None and getattr(event, name) is not None
    }
    return replace(existing, **updates) if updates else existing


def _scan_events(events: Iterable[Any]) -> _Scan:
    """Sort ``events`` by kind; rediscoveries merge into the first discovery."""

    scan = _Scan()
    for event in events:
        if isinstance(event, BackwardPassStart):
            scan.starts[event.pass_index] = event
        elif isinstance(event, BackwardPassEnd):
            scan.ends[event.pass_index] = event
        elif isinstance(event, GradFnDiscovered):
            prior = scan.discovered.get(event.object_id)
            scan.discovered[event.object_id] = (
                event if prior is None else _merge_discovery(prior, event)
            )
            scan.topology[event.object_id] = event.topology
        elif isinstance(event, GradFnFired):
            scan.fired.append(event)
        elif isinstance(event, OpGradObserved):
            scan.op_grads.append(event)
        elif isinstance(event, ParamGradObserved):
            scan.param_grads.append(event)
    return scan


def _event_order(event: Any) -> tuple[int, float, int]:
    """Sort key PyTorch's fold uses: pass, then timestamp, then journal order."""

    return (event.pass_index, event.timestamp, event.seq)


def _build_grad_fn_records(trace: Any, scan: _Scan) -> OrderedDict[int, GradFn]:
    """Project discoveries to GradFn records with labels, orders and ``differentiates``."""

    _pair_discovered_with_layers(trace, scan.discovered)
    records: OrderedDict[int, GradFn] = OrderedDict()
    type_counter: dict[str, int] = {}
    for ordinal_index, (object_id, event) in enumerate(scan.discovered.items()):
        topology = scan.topology.get(object_id, event.topology)
        records[object_id] = _grad_fn_record_from_discovery(
            trace, object_id, event, topology, ordinal_index, type_counter
        )
    _assign_grad_fn_orders(records)
    labels = {object_id: record.label for object_id, record in records.items()}
    for record in records.values():
        if record.creator_object_id is not None:
            record.differentiates = labels.get(record.creator_object_id)
    return records


def _fold_fired(trace: Any, fired: list[GradFnFired]) -> dict[int, list[GradFnCall]]:
    """One ``GradFnCall`` per journaled fire; returns the calls of each pass."""

    ordinals: dict[int, int] = {}
    pass_to_calls: dict[int, list[GradFnCall]] = {}
    for event in sorted(fired, key=_event_order):
        record = trace._grad_fn_logs.get(event.object_id)
        if record is None:
            continue
        ordinal = ordinals.get(event.object_id, 0) + 1
        ordinals[event.object_id] = ordinal
        call = GradFnCall(
            call_index=ordinal,
            ordinal=ordinal,
            backward_pass_number=event.pass_index,
            grad_fn_label=record.label,
            grad_inputs=event.grad_input_refs,
            grad_outputs=event.grad_output_refs,
            fire_record=event.fire_record,
            timestamp=event.timestamp,
            _time_started=event.fire_started_monotonic,
            _time_finished=event.fire_finished_monotonic,
        )
        call._source_trace = trace
        record.calls[ordinal] = call
        pass_to_calls.setdefault(event.pass_index, []).append(call)
    return pass_to_calls


@dataclass
class _GradTotals:
    """Memory totals of the retained op gradients."""

    saved_labels: set[str] = field(default_factory=set)
    unique_keys: set[Any] = field(default_factory=set)
    gradient_memory: int = 0
    backward_memory: int = 0


def _apply_op_grad(op: Any, event: OpGradObserved, payloads: PayloadPolicy) -> tuple[Any, Any]:
    """Record one op-gradient event on ``op``; returns the retained (raw, transformed)."""

    payload = event.payload_ref if payloads.is_payload(event.payload_ref) else None
    transformed = event.transformed_payload_ref
    op._record_gradient(
        backward_pass_number=event.pass_index,
        grad=payload,
        transformed_grad=transformed,
        shape=event.shape,
        dtype=event.dtype,
        memory=event.memory,
        timestamp=event.timestamp,
    )
    if payload is None and transformed is None:
        return None, None
    op.has_grad = True
    if event.shape is not None:
        op._autograd_grad_shape = tuple(event.shape)
    if event.dtype is not None:
        op._autograd_grad_dtype = payloads.dtype_from_string(event.dtype)
    op.gradient_memory = Bytes(event.memory or 0)
    op._internal_set("grad", payload)
    op._internal_set("transformed_grad", transformed)
    op._payload_transformed_grad_shape = payloads.shape_of(transformed)
    op._payload_transformed_grad_dtype = payloads.dtype_of(transformed)
    op.transformed_gradient_memory = payloads.memory_of(transformed)
    return payload, transformed


def _final_op_label(trace: Any, op_label: str) -> str:
    """Map a raw op label to its final label when postprocessing renamed it."""

    for name in ("_raw_to_final_layer_labels", "_raw_to_final_op_labels"):
        mapping = getattr(trace, name, {})
        if isinstance(mapping, dict) and op_label in mapping:
            return str(mapping[op_label])
    return op_label


def _fold_op_grads(trace: Any, events: list[OpGradObserved], payloads: PayloadPolicy) -> None:
    """Fold op-gradient events into Op records and the trace's gradient memory totals."""

    for op in trace._layer_list:
        op._clear_gradient_records()
    totals = _GradTotals()
    for event in sorted(events, key=_event_order):
        op = trace._layer_dict_all_keys.get(_final_op_label(trace, event.op_label))
        if op is None:
            continue
        raw, transformed = _apply_op_grad(op, event, payloads)
        if raw is None and transformed is None:
            continue
        totals.saved_labels.add(op.layer_label)
        for key, memory in payloads.retained_refs(raw, transformed, raw_memory=event.memory):
            if key not in totals.unique_keys:
                totals.unique_keys.add(key)
                totals.backward_memory += memory
        totals.gradient_memory += int(event.memory or 0)
    trace._saved_grad_labels = set(totals.saved_labels)
    trace._cost_saved_grad_memory = Bytes(totals.gradient_memory)
    trace._cost_total_grad_memory = Bytes(totals.gradient_memory)
    trace._cost_total_backward_memory = Bytes(totals.backward_memory)


def _fold_param_grads(trace: Any, events: list[ParamGradObserved]) -> None:
    """Fold parameter-gradient events into per-Param accumulating records."""

    param_logs = trace._param_logs
    if param_logs is None:
        return
    for param_log in param_logs.values():
        param_log._clear_gradient_records()
    for event in sorted(events, key=_event_order):
        if event.param_address in param_logs:
            param_logs[event.param_address]._append_gradient_record(
                backward_pass_number=event.pass_index,
                grad=event.payload_ref,
                shape=event.shape,
                dtype=event.dtype,
                memory=event.memory,
                timestamp=event.timestamp,
            )


def _pass_order(
    trace: Any,
    start: BackwardPassStart | None,
    root_ids: list[int],
    calls: list[GradFnCall],
    resolved: dict[int, int],
) -> int | None:
    """The pass's order: from its roots, else its start event, raised to its latest call."""

    logs = trace._grad_fn_logs
    order = _pass_order_from_roots(root_ids, logs, resolved) if root_ids else None
    if order is None and start is not None:
        order = start.order
    max_call_order = _max_call_order(calls, logs)
    if max_call_order is not None and (order is None or max_call_order > order):
        order = max_call_order
    return order


def _start_fields(start: BackwardPassStart | None) -> dict[str, Any]:
    """BackwardPass fields read from the start event; an unstarted pass is implicit."""

    if start is None:
        return {"trigger": "implicit", "implicit": True}
    return {
        "trigger": start.trigger,
        "implicit": start.implicit,
        "outer_context": start.outer_context,
        "backward_call_context": start.call_context_ref,
        "root_meta": tuple(start.root_meta),
        "root_grad_arguments": start.root_grad_arguments,
        "inputs_subset": tuple(start.inputs_subset),
        "origin_backward_pass": start.origin_backward_pass,
        "engine_flags": start.engine_flags,
        "save_grads_policy": start.save_grads_policy_repr,
    }


def _end_fields(end: BackwardPassEnd | None) -> dict[str, Any]:
    """BackwardPass fields read from the end event; nothing unobserved is filled."""

    if end is None:
        return {}
    return {
        "duration": None if end.duration is None else Duration(end.duration),
        "peak_memory": end.peak_memory,
        "status": end.status,
    }


def _build_pass_records(
    trace: Any, scan: _Scan, pass_to_calls: dict[int, list[GradFnCall]]
) -> OrderedDict[int, BackwardPass]:
    """One BackwardPass per pass index any event names."""

    roots_by_pass = getattr(trace, "_backward_roots_by_pass", {})
    pass_indices = sorted(
        set(scan.starts)
        | set(scan.ends)
        | set(pass_to_calls)
        | {event.pass_index for event in scan.op_grads}
        | {event.pass_index for event in scan.param_grads}
    )
    resolved: dict[int, int] = {}
    records: OrderedDict[int, BackwardPass] = OrderedDict()
    for pass_index in pass_indices:
        start, end = scan.starts.get(pass_index), scan.ends.get(pass_index)
        root_ids = list(roots_by_pass.get(pass_index, ()))
        calls = pass_to_calls.get(pass_index, [])
        order = _pass_order(trace, start, root_ids, calls, resolved)
        coverage = None if end is None else end.order_attribution_coverage
        if coverage is None:
            coverage = _order_attribution_coverage(calls, trace._grad_fn_logs)
        record = BackwardPass(
            pass_number=pass_index,
            root_grad_fn_ids=root_ids,
            order=order,
            order_attribution_coverage=coverage,
            grad_fn_calls=calls,
            **_start_fields(start),
            **_end_fields(end),
        )
        record._source_trace = trace
        records[pass_index] = record
        if order is not None:
            resolved[pass_index] = order
    return records


def _link_ops_to_grad_fns(trace: Any) -> None:
    """Point each paired Op and its Layer at its GradFn record."""

    for record in trace._grad_fn_logs.values():
        if record.op is not None:
            record.op._autograd_grad_fn = record
            parent_layer = trace._layer_logs.get(record.op.layer_label)
            if parent_layer is not None:
                parent_layer._autograd_grad_fn = record
    for layer in (*trace._layer_list, *trace._layer_logs.values()):
        object_id = record_member(layer, "grad_fn_object_id", None)
        if object_id in trace._grad_fn_logs:
            layer._autograd_grad_fn = trace._grad_fn_logs[object_id]


def build_backward_records(
    trace: Any, events: Iterable[Any], *, payloads: PayloadPolicy = NEUTRAL_PAYLOADS
) -> None:
    """Replace ``trace``'s backward records with the projection of ``events``.

    Parameters
    ----------
    trace:
        Trace whose backward records are rebuilt in place.
    events:
        Every backward event of the trace's stream, in journal order.
    payloads:
        The backend's payload policy.
    """

    scan = _scan_events(events)
    grad_fn_logs = _build_grad_fn_records(trace, scan)
    trace._grad_fn_logs = grad_fn_logs
    trace._autograd_grad_fn_order = list(grad_fn_logs)
    pass_to_calls = _fold_fired(trace, scan.fired)
    _sync_grad_fn_graph_relations(trace)
    _infer_grad_fn_module_membership(trace)
    _prefer_direct_pairing_source_for_paired_ops(trace)
    _fold_op_grads(trace, scan.op_grads, payloads)
    _fold_param_grads(trace, scan.param_grads)
    trace._backward_pass_logs = _build_pass_records(trace, scan, pass_to_calls)
    roots_by_pass: Mapping[int, list[int]] = getattr(trace, "_backward_roots_by_pass", {})
    trace._autograd_backward_root_grad_fn_object_ids = [
        root_id for roots in roots_by_pass.values() for root_id in roots
    ]
    trace._num_backward_passes = max(trace._backward_pass_logs, default=0)
    trace._autograd_has_backward_pass = bool(trace._backward_pass_logs)
    _link_ops_to_grad_fns(trace)


def _bind_epoch(trace: Any, revision: int, watermark: int) -> None:
    """Adopt the rebuilt records into a fresh core backward epoch (atomic swap)."""

    core = trace.__dict__.get("_trace_core")
    if core is None:
        return
    grad_fns = list((trace._grad_fn_logs or {}).values())
    calls = [
        call
        for record in grad_fns
        for call in getattr(record.calls, "_dict", record.calls).values()
    ]
    epoch = BackwardEpoch()
    adopt_rows(epoch.stores, "grad_fn", grad_fns)
    adopt_rows(epoch.stores, "grad_fn_call", calls)
    adopt_rows(epoch.stores, "backward_pass", list((trace._backward_pass_logs or {}).values()))
    epoch.revision = revision
    epoch.watermark = watermark
    core.backward_epochs = [epoch]


def materialize_neutral_backward(trace: Any, *, payloads: PayloadPolicy = NEUTRAL_PAYLOADS) -> None:
    """Rebuild ``trace``'s backward records when its event stream changed.

    The read-side twin of PyTorch's projection: stamps the stream revision it
    built from, so an unchanged stream costs nothing, and poisons the stamps on
    failure so the next read rebuilds from scratch.
    """

    if getattr(trace, "_tl_materializing_backward_projection", False):
        return
    stream = _event_stream(trace)
    events = list(stream.backward_events)
    if not events or int(stream.pass_index_base or 0):
        # A detached stream (fork or reload of a projected trace) keeps the
        # records it was detached with; their events were dropped by design.
        return
    revision = stream.backward_revision
    if getattr(trace, "_backward_projection_revision", None) == revision:
        return
    trace._tl_materializing_backward_projection = True
    try:
        build_backward_records(trace, events, payloads=payloads)
        _bind_epoch(trace, revision, len(events))
        trace._backward_projection_event_count = len(events)
        trace._backward_projection_revision = revision
    except BaseException:
        trace._backward_projection_revision = None
        trace._backward_projection_event_count = None
        raise
    finally:
        trace.__dict__.pop("_tl_materializing_backward_projection", None)
