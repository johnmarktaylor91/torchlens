"""Materialize and bind the private wave-0 primitive-operation profile."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .._trace_core.record_rows import adopt_records, adopt_rows
from ..data_classes.aten_op import (
    AtenOp,
    OpRef,
    _ModePausedInteriorGap,
    _PrimitiveOpProfile,
)
from ..ir.events import _AtenCallEvent, _ModePausedInteriorEvent


def _op_refs_by_func_call_id(trace: Any) -> dict[int, tuple[OpRef, ...]]:
    """Return final redundant Op references grouped by wrapper call id.

    Parameters
    ----------
    trace
        Postprocessed Trace whose Op order and labels are final.

    Returns
    -------
    dict[int, tuple[OpRef, ...]]
        All final Op rows emitted by each captured wrapper call.
    """

    grouped: dict[int, list[OpRef]] = defaultdict(list)
    for row_index, op in enumerate(trace.ops):
        func_call_id = getattr(op, "func_call_id", None)
        label = getattr(op, "layer_label", None)
        if isinstance(func_call_id, int) and isinstance(label, str):
            grouped[func_call_id].append(
                OpRef(
                    op_row_index=row_index,
                    op_label=label,
                    func_call_id=func_call_id,
                )
            )
    return {func_call_id: tuple(refs) for func_call_id, refs in grouped.items()}


def _aten_row_from_event(event: _AtenCallEvent, *, slot: int) -> AtenOp:
    """Project one immutable dispatcher event into a primitive row.

    Parameters
    ----------
    event
        Value-free dispatcher event.
    slot
        Zero-based position within its observed owner partition.

    Returns
    -------
    AtenOp
        Detached primitive row awaiting final parent resolution.
    """

    owner_status = "unresolved" if event.owner_func_call_id is not None else "orphan"
    if event.capture_phase == "backward" and event.parent_grad_fn_call_ref is not None:
        owner_status = "unresolved"
    return AtenOp(
        label=f"aten_{event.seq}",
        sequence=event.seq,
        capture_phase=event.capture_phase,
        forward_pass_index=event.forward_pass_index,
        backward_epoch_index=event.backward_epoch_index,
        owner_func_call_id=event.owner_func_call_id,
        parent_op_refs=(),
        parent_grad_fn_call_ref=None,
        owner_status=owner_status,
        decomposition_slot=slot,
        namespace=event.namespace,
        operator=event.operator,
        overload=event.overload,
        schema=event.schema,
        schema_fingerprint=event.schema_fingerprint,
        module_call_stack=event.module_call_stack,
        input_tensor_facts=event.input_tensor_facts,
        output_tensor_facts=event.output_tensor_facts,
        mutation_kind=event.mutation_kind,
        view_copy_kind=event.view_copy_kind,
        autocast_context=event.autocast_context,
        dispatch_key_context=event.dispatch_key_context,
        grad_fn_ref=event.grad_fn_ref,
        grad_fn_link_status=event.grad_fn_link_status,
        grad_fn_link_provenance=event.grad_fn_link_provenance,
        algorithmic_flops=event.algorithmic_flops,
        flop_status=event.flop_status,
        flop_formula_source=event.flop_formula_source,
        flop_formula_version=event.flop_formula_version,
        outcome=event.outcome,
        exception_type=event.exception_type,
        execution_context=event.execution_context,
    )


def _materialize_forward_primitive_profile(
    trace: Any,
    events: Iterable[Any],
    *,
    recording_enabled: bool,
) -> None:
    """Create the present forward profile from the reserved journal lane.

    Parameters
    ----------
    trace
        Trace receiving the private profile.
    events
        Immutable primitive-call and observer-gap events.
    recording_enabled
        Whether this capture armed the private recorder. A present-empty
        profile stays distinguishable from recording absence.
    """

    if not recording_enabled:
        trace._primitive_op_profile = None
        return
    profile = _PrimitiveOpProfile()
    owner_evidence: list[tuple[int, int | None]] = []
    slots: dict[tuple[str, object], int] = defaultdict(int)
    for event in events:
        profile.aten_event_watermark = max(
            profile.aten_event_watermark, int(getattr(event, "seq", 0) or 0)
        )
        if isinstance(event, _AtenCallEvent):
            owner_evidence.append((event.seq, event.owner_func_call_id))
            owner_key: object = (
                event.owner_func_call_id
                if event.capture_phase == "forward"
                else event.parent_grad_fn_call_ref
            )
            partition = (event.capture_phase, owner_key)
            slot = slots[partition]
            slots[partition] += 1
            profile.primitive_ops.append(_aten_row_from_event(event, slot=slot))
        elif isinstance(event, _ModePausedInteriorEvent):
            profile.mode_paused_interior.append(
                _ModePausedInteriorGap(
                    kind="mode_paused_interior",
                    capture_phase=event.capture_phase,
                    sequence_before=event.sequence_before,
                    sequence_after=event.sequence_after,
                    owner_func_call_id=event.owner_func_call_id,
                    parent_op_refs=(),
                    reason=event.reason,
                )
            )
    trace._primitive_op_profile = profile
    profile._event_owner_evidence = tuple(owner_evidence)


def _finalize_forward_primitive_profile(trace: Any) -> None:
    """Resolve forward Op FKs and adopt primitive rows into the core store.

    Parameters
    ----------
    trace
        Fully postprocessed Trace whose Op labels and dense order are final.
    """

    profile = getattr(trace, "_primitive_op_profile", None)
    if profile is None:
        return
    refs_by_call = _op_refs_by_func_call_id(trace)
    for row in profile.primitive_ops:
        if row.capture_phase != "forward":
            continue
        if row.owner_func_call_id is None:
            row.owner_status = "orphan"
            continue
        refs = refs_by_call.get(row.owner_func_call_id, ())
        row.parent_op_refs = refs
        row.owner_status = "forward_op" if refs else "orphan"
    for gap in profile.mode_paused_interior:
        if gap.capture_phase != "forward" or gap.owner_func_call_id is None:
            continue
        object.__setattr__(gap, "parent_op_refs", refs_by_call.get(gap.owner_func_call_id, ()))
    core = trace.__dict__.get("_trace_core")
    if core is not None:
        adopt_records(core, "primitive_op", profile.primitive_ops)


def _resolve_grad_fn_call_ref(
    trace: Any,
    event_ref: tuple[int, int, int] | None,
) -> tuple[str, int, int] | None:
    """Resolve an event-side GradFn object witness to its portable call key.

    Parameters
    ----------
    trace
        Trace with materialized backward projections.
    event_ref
        GradFn object id, predicted call index, and backward pass index.

    Returns
    -------
    tuple[str, int, int] | None
        GradFn label, call index, and backward pass index when exact.
    """

    if event_ref is None:
        return None
    object_id, call_index, pass_index = event_ref
    grad_fn = getattr(trace, "grad_fn_logs", {}).get(object_id)
    if grad_fn is None:
        return None
    try:
        call = grad_fn.calls[call_index - 1]
    except (IndexError, KeyError, TypeError):
        return None
    if int(getattr(call, "backward_pass_index", 0) or 0) != pass_index:
        return None
    return (str(grad_fn.label), call_index, pass_index)


def _normalized_grad_name(value: str) -> str:
    """Return a comparison key for an operator or GradFn class name.

    Parameters
    ----------
    value
        ATen operator or autograd class name.

    Returns
    -------
    str
        Lowercase alphanumeric stem with autograd suffixes removed.
    """

    lowered = value.lower().removesuffix("_")
    lowered = re.sub(r"backward\d*$", "", lowered)
    return re.sub(r"[^a-z0-9]", "", lowered)


def _link_forward_grad_fn_rows(trace: Any, profile: _PrimitiveOpProfile) -> None:
    """Settle delayed forward-row links through parent Op GradFn witnesses.

    Parameters
    ----------
    trace
        Trace with a materialized GradFn table.
    profile
        Primitive profile whose forward partitions should be linked.

    Notes
    -----
    Python dispatch observes tensors below autograd, before their GradFn is
    attached. The parent Op retains the exact GradFn object id. Within one
    wrapper fire, operator/class-name agreement selects the row first; capture
    sequence is the deterministic fallback. The latter is explicitly marked
    ``heuristic`` and never presented as ``exact_via_aten`` evidence.
    """

    grad_fns = getattr(trace, "grad_fn_logs", {}) or {}
    if not grad_fns:
        return
    ops = list(trace.ops)
    partitions: dict[int, list[AtenOp]] = defaultdict(list)
    for row in profile.primitive_ops:
        if (
            row.capture_phase == "forward"
            and row.owner_status == "forward_op"
            and row.owner_func_call_id is not None
        ):
            partitions[row.owner_func_call_id].append(row)
    for rows in partitions.values():
        grad_fn_ids = {
            getattr(ops[ref.op_row_index], "grad_fn_object_id", None)
            for row in rows
            for ref in row.parent_op_refs
            if 0 <= ref.op_row_index < len(ops)
        }
        resolved_grad_fns = [
            grad_fns[object_id]
            for object_id in grad_fn_ids
            if isinstance(object_id, int) and object_id in grad_fns
        ]
        unmatched = sorted(rows, key=lambda row: row.sequence)
        for grad_fn in resolved_grad_fns:
            grad_key = _normalized_grad_name(str(grad_fn.class_name))
            matching = [
                row
                for row in unmatched
                if _normalized_grad_name(row.operator)
                and _normalized_grad_name(row.operator) in grad_key
            ]
            selected = matching[-1] if matching else (unmatched[-1] if unmatched else None)
            if selected is None:
                continue
            selected.grad_fn_ref = str(grad_fn.label)
            selected.grad_fn_link_status = "linked"
            selected.grad_fn_link_provenance = "heuristic"
            unmatched.remove(selected)


def _materialize_backward_primitive_profile(trace: Any) -> None:
    """Project newly appended backward primitive events into the active epoch.

    Parameters
    ----------
    trace
        Trace whose backward projection has just settled.
    """

    profile = getattr(trace, "_primitive_op_profile", None)
    stream = getattr(trace, "_capture_events", None)
    if profile is None or stream is None:
        return
    new_events = [
        event
        for event in stream.aten_events
        if int(getattr(event, "seq", 0) or 0) > profile.aten_event_watermark
    ]
    if not new_events:
        return
    slots: dict[tuple[str, object], int] = defaultdict(int)
    for prior in profile.primitive_ops:
        if prior.capture_phase == "backward":
            key = (prior.capture_phase, prior.parent_grad_fn_call_ref)
            slots[key] = max(slots[key], prior.decomposition_slot + 1)
    added_rows: list[AtenOp] = []
    added_owner_evidence: list[tuple[int, int | None]] = []
    for event in new_events:
        profile.aten_event_watermark = max(profile.aten_event_watermark, int(event.seq))
        if isinstance(event, _AtenCallEvent):
            added_owner_evidence.append((event.seq, event.owner_func_call_id))
            resolved = _resolve_grad_fn_call_ref(trace, event.parent_grad_fn_call_ref)
            partition = (event.capture_phase, resolved or event.parent_grad_fn_call_ref)
            slot = slots[partition]
            slots[partition] += 1
            row = _aten_row_from_event(event, slot=slot)
            row.parent_grad_fn_call_ref = resolved
            row.owner_status = (
                "backward_grad_fn_call"
                if resolved is not None
                else ("orphan" if event.parent_grad_fn_call_ref is None else "unresolved")
            )
            profile.primitive_ops.append(row)
            added_rows.append(row)
        elif isinstance(event, _ModePausedInteriorEvent):
            profile.mode_paused_interior.append(
                _ModePausedInteriorGap(
                    kind="mode_paused_interior",
                    capture_phase=event.capture_phase,
                    sequence_before=event.sequence_before,
                    sequence_after=event.sequence_after,
                    owner_func_call_id=event.owner_func_call_id,
                    parent_op_refs=(),
                    reason=event.reason,
                )
            )
    core = trace.__dict__.get("_trace_core")
    profile._event_owner_evidence = (
        *profile._event_owner_evidence,
        *added_owner_evidence,
    )
    if core is not None and core.backward_epochs and added_rows:
        adopt_rows(core.backward_epochs[-1].stores, "primitive_op", added_rows)
    _link_forward_grad_fn_rows(trace, profile)


__all__ = [
    "_finalize_forward_primitive_profile",
    "_materialize_backward_primitive_profile",
    "_materialize_forward_primitive_profile",
]
