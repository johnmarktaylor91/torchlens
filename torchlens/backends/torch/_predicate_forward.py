"""Predicate-mode module forward: the cold arm of ``decorated_forward``.

An exhaustive capture never enters these functions. Under predicate capture
each module call records an enter event, runs the original forward, logs
boundary replacements and retained outputs, then records the exit event; a
``StopSignal`` from the halt predicate pops the module frame and propagates.
Split out of ``model_prep.module_forward_decorator`` so the plain forward path
stays one function with no added call per module.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ... import _state
from ..._capture import predicates as _predicates
from ..._capture.predicates import (
    _evaluate_halt,
    _evaluate_keep_op,
    _is_halt_only_capture,
    _module_capture_spec,
)
from ..._capture.projections import (
    _build_record_context,
    _record_from_record_context,
    append_projected_event,
    get_active_recording_state,
)
from ..._ir import ModuleExitEvent
from ..._ir.op_record import amend_module_boundary_retention
from ..._ir.predicate import RetroactiveCaptureDecision
from ..._ir.selector_eval import selector_contains_kind
from ..._utils._tensor_label import get_tensor_label
from ...recording._halt import StopSignal
from ...recording.types import ActivationRecord, CaptureSpec
from . import module_stack as _mstack
from ._ops_containers import _walk_output_tensors_with_paths
from ._predicate_boundary_replacements import log_predicate_boundary_replacements
from .escape_detection import expected_original_call
from .prehook_provenance import bind_invocation

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


def _record_predicate_module_boundary_outputs(
    trace: Trace,
    state: Any,
    out: Any,
    *,
    module_address: str,
    module_call_index: int,
) -> None:
    """Log boundary replacements, then retain ``tl.module``-selected outputs.

    Parameters
    ----------
    trace:
        Active predicate-mode trace.
    state:
        Active fastlog recording state.
    out:
        Module output after live boundary interventions.
    module_address:
        Address of the exiting module.
    module_call_index:
        One-based call index of the exiting module.

    Returns
    -------
    None
        Boundary-replaced leaves become sparse replacement ops; matching output
        ops are added to the sparse recording once.
    """

    from ..._selection.selectors import BaseSelector
    from ...intervention.runtime import _peek_module_intervention_parent_labels

    log_predicate_boundary_replacements(trace, state, out)
    predicate = state.options.keep_op
    if not isinstance(predicate, BaseSelector) or (
        _predicates._plain_module_union(predicate) is None
        and not selector_contains_kind(predicate, "module")
    ):
        return
    contexts_by_label = {
        ctx.raw_label: ctx
        for ctx in state.all_contexts
        if ctx.kind == "op" and ctx.raw_label is not None
    }
    existing = {
        (record.ctx.raw_label, record.ctx.pass_number)
        for record in state.recording.records
        if record.ctx.raw_label is not None
    }
    module_call_label = f"{module_address}:{module_call_index}"
    labeled_outputs: list[tuple[torch.Tensor, tuple[Any, ...], str]] = []
    walked_leaf_count = 0
    for tensor, container_path, _container_spec in _walk_output_tensors_with_paths(out):
        walked_leaf_count += 1
        raw_label = get_tensor_label(tensor)
        if raw_label is None:
            parent_labels = _peek_module_intervention_parent_labels(tensor, trace)
            raw_label = parent_labels[0] if parent_labels else None
        if raw_label is not None:
            labeled_outputs.append((tensor, tuple(container_path), raw_label))
    trace.capture_events.append_module_exit(
        ModuleExitEvent(
            address=module_address,
            call_index=module_call_index,
            call_label=module_call_label,
            forward_duration=0.0,
            output_structure=None,
            output_tensor_labels_raw=tuple(label for _tensor, _path, label in labeled_outputs),
            output_paths=tuple(path for _tensor, path, _label in labeled_outputs),
            per_output_atomic=(),
            output_names=tuple(None for _tensor, _path, _label in labeled_outputs),
            output_tensor_leaf_count=walked_leaf_count,
        )
    )
    for tensor, container_path, raw_label in labeled_outputs:
        ctx = contexts_by_label.get(raw_label)
        if ctx is None:
            continue
        boundary_ctx = replace(
            ctx,
            output_of_module_calls=tuple(
                dict.fromkeys((*ctx.output_of_module_calls, module_call_label))
            ),
        )
        decision = _evaluate_keep_op(boundary_ctx, state.options)
        if isinstance(decision, RetroactiveCaptureDecision):
            continue
        if not decision.save_out and not decision.save_metadata:
            continue
        trace._tl_save_selector_fire_count = (
            int(getattr(trace, "_tl_save_selector_fire_count", 0)) + 1
        )
        key = (boundary_ctx.raw_label, boundary_ctx.pass_number)
        if key in existing:
            continue
        existing.add(key)
        ram_payload, disk_payload, transformed_ram, transformed_disk = state.resolve_storage(
            tensor,
            decision,
            ctx=boundary_ctx,
        )
        state.add_record(
            ActivationRecord(
                ctx=boundary_ctx,
                spec=decision,
                ram_payload=ram_payload,
                disk_payload=disk_payload,
                transformed_ram_payload=transformed_ram,
                transformed_disk_payload=transformed_disk,
            )
        )
        selected_event = _record_from_record_context(
            boundary_ctx,
            decision,
            tensor=tensor,
            ram_payload=ram_payload,
            transformed_ram_payload=transformed_ram,
            predicate_matched=True,
            container_path=tuple(container_path),
            detach_unkept=state.options.detach_saved_activations,
        )
        selected_policy = selected_event.policy
        if selected_policy is None:  # pragma: no cover - sparse freeze always stamps one
            raise RuntimeError("sparse freeze produced a record without a policy facet")
        boundary_label = boundary_ctx.raw_label or boundary_ctx.label
        boundary_event = trace.capture_events.op_event_by_label_raw.get(boundary_label)
        if boundary_event is not None:
            trace.capture_events.append_amendment(
                amend_module_boundary_retention(
                    boundary_event.seq,
                    boundary_label,
                    output=selected_event.output,
                    policy=selected_policy,
                    predicate_matched=True,
                    capture_spec=decision,
                    record_context=boundary_ctx,
                )
            )


def _predicate_module_enter(trace: Trace, state: Any, frame: Any, halt_only: bool) -> Any:
    """Record a predicate-mode module entry and evaluate the halt predicate there.

    Returns the skipped ``CaptureSpec`` the exit record reuses. A ``StopSignal``
    raised by the halt predicate pops the module frame and propagates before
    the forward runs (exactly the pre-split shape).
    """

    state.event_index += 1
    enter_ctx = _build_record_context(
        kind="module_enter",
        op_log_or_op_data={
            "label": f"{frame.address}:enter:{frame.pass_number}",
            "address": frame.address,
            "module_type": frame.module_type,
            "module_pass_number": frame.pass_number,
        },
        module_stack=state.module_stack,
        history=tuple(state.history),
        op_counts=state.op_counts,
        pass_index=state.pass_index,
        event_index=state.event_index,
        step_index=None,
        time_since_pass_start=0.0,
        include_source_events=state.options.include_source_events,
        sample_id=state.sample_id,
    )
    skipped_spec = CaptureSpec(save_out=False, save_metadata=False)
    enter_spec = skipped_spec
    try:
        if halt_only:
            _evaluate_halt(enter_ctx, state.options)
        else:
            enter_spec = _module_capture_spec(state.options)
            if enter_spec.save_out or enter_spec.save_metadata:
                if state.storage_intent.on_disk:
                    state.add_record(ActivationRecord(ctx=enter_ctx, spec=enter_spec))
            append_projected_event(
                trace,
                enter_ctx,
                enter_spec,
                predicate_matched=enter_spec.save_out or enter_spec.save_metadata,
            )
            _evaluate_halt(enter_ctx, state.options)
    except StopSignal:
        _mstack.pop_frame(state.module_stack, frame)
        raise
    except Exception as exc:
        state.handle_predicate_exception(enter_ctx, exc)
    finally:
        if not halt_only:
            if not _predicate_event_was_appended(trace, enter_ctx.raw_label or enter_ctx.label):
                append_projected_event(trace, enter_ctx, skipped_spec, predicate_matched=False)
            state.append_context(enter_ctx)
            # Echo narrator slot (snoop D1, module seam): structure lines ride
            # the events already in the stream. Duck-typed session read: the
            # hot path imports nothing.
            echo_session = trace.__dict__.get("_echo_session")
            if echo_session is not None:
                echo_session.emit_module_enter(frame.address, frame.module_type)
    return skipped_spec


def _predicate_module_exit(
    trace: Trace,
    state: Any,
    entry: tuple[Any, bool, Any],
    out: Any,
    active_model_exc: BaseException | None,
) -> None:
    """Record a predicate-mode module exit, evaluate the halt there and pop the frame.

    Runs from the forward's ``finally``: ``active_model_exc`` is the exception
    the module raised, if any; a predicate failure then attaches as a
    predicate failure instead of replacing it, and the echo frame is abandoned.
    """
    frame, halt_only, skipped_spec = entry

    state.event_index += 1
    exit_ctx = _build_record_context(
        kind="module_exit",
        op_log_or_op_data={
            "label": f"{frame.address}:exit:{frame.pass_number}",
            "address": frame.address,
            "module_type": frame.module_type,
            "module_pass_number": frame.pass_number,
        },
        module_stack=state.module_stack,
        history=tuple(state.history),
        op_counts=state.op_counts,
        pass_index=state.pass_index,
        event_index=state.event_index,
        step_index=None,
        time_since_pass_start=0.0,
        include_source_events=state.options.include_source_events,
        sample_id=state.sample_id,
    )
    exit_spec = skipped_spec
    try:
        if halt_only:
            _evaluate_halt(exit_ctx, state.options, frontier_output=out)
        else:
            exit_spec = _module_capture_spec(state.options)
            if exit_spec.save_out or exit_spec.save_metadata:
                if state.storage_intent.on_disk:
                    state.add_record(ActivationRecord(ctx=exit_ctx, spec=exit_spec))
            append_projected_event(
                trace,
                exit_ctx,
                exit_spec,
                predicate_matched=exit_spec.save_out or exit_spec.save_metadata,
            )
            _evaluate_halt(exit_ctx, state.options, frontier_output=out)
    except StopSignal:
        if active_model_exc is None:
            raise
    except Exception as exc:
        if active_model_exc is None:
            state.handle_predicate_exception(exit_ctx, exc)
        else:
            state.add_predicate_failure(exit_ctx, exc)
    finally:
        if not halt_only:
            if not _predicate_event_was_appended(trace, exit_ctx.raw_label or exit_ctx.label):
                append_projected_event(trace, exit_ctx, skipped_spec, predicate_matched=False)
            state.append_context(exit_ctx)
            # Echo module-exit line: only a COMPLETED module call claims
            # completion; a raising module abandons its frame silently (the
            # enter line without an exit is the honest crash shape).
            echo_session = trace.__dict__.get("_echo_session")
            if echo_session is not None:
                if active_model_exc is None:
                    echo_session.emit_module_exit(frame.address)
                else:
                    echo_session.abandon_module_frame()
        _mstack.pop_frame(state.module_stack, frame)


def _predicate_module_forward(
    trace: Trace,
    module: nn.Module,
    orig_forward: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run one module forward under predicate capture: enter record, forward, exit record.

    The cold arm of ``decorated_forward``: an exhaustive capture never enters it.
    """

    state = get_active_recording_state()
    frame = _mstack.push_frame(trace, state.module_stack, module)
    bind_invocation(trace, module, frame.address, frame.pass_number, (args, kwargs))
    halt_only = _is_halt_only_capture(state.options)
    skipped_spec = _predicate_module_enter(trace, state, frame, halt_only)
    out = None
    try:
        if (
            _state._escape_detector_mode == "shadow"
            or _state._completeness_witness_mode == "shadow"
        ):
            with expected_original_call(orig_forward, "module_forward:predicate"):
                out = orig_forward(*args, **kwargs)
        else:
            out = orig_forward(*args, **kwargs)
        from ...intervention.runtime import _apply_module_boundary_live_hooks

        out = _apply_module_boundary_live_hooks(
            out,
            module_address=frame.address,
            module_call_index=frame.pass_number,
            module_type=frame.module_type,
            call_args=args,
            call_kwargs=dict(kwargs),
        )
        _record_predicate_module_boundary_outputs(
            trace,
            state,
            out,
            module_address=frame.address,
            module_call_index=frame.pass_number,
        )
        return out
    finally:
        _predicate_module_exit(
            trace, state, (frame, halt_only, skipped_spec), out, sys.exc_info()[1]
        )


def _predicate_event_was_appended(trace: Trace, label_raw: str) -> bool:
    """Return whether predicate capture already appended ``label_raw``.

    Parameters
    ----------
    trace:
        Active trace whose predicate event buffer may already contain the
        projected event.
    label_raw:
        Raw label used when appending the projected event.

    Returns
    -------
    bool
        True when ``trace.capture_events.op_event_by_label_raw`` already owns
        ``label_raw``.
    """

    capture_events = getattr(trace, "capture_events", None)
    if capture_events is None:
        return False
    return label_raw in capture_events.op_event_by_label_raw
