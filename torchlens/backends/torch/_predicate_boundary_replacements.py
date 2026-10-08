"""Sparse replacement ops for module-boundary edits on the predicate path.

``tl.record`` (predicate capture) must log a live ``tl.module(...)`` edit as an
explicit ``interventionreplacement`` op, as the exhaustive path does, or the
edited value has no producer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._tl import get_tensor_label, set_tensor_label

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

__all__ = ["log_predicate_boundary_replacements"]


def log_predicate_boundary_replacements(trace: Trace, state: Any, out: Any) -> None:
    """Mint one sparse replacement op per module-boundary-replaced output leaf.

    A live ``tl.module(...)`` edit returns a fresh tensor whose copied label the
    boundary door clears on purpose (``_attach_boundary_fire_evidence``). The
    exhaustive exit path logs that value as an explicit
    ``interventionreplacement`` op (``_ensure_module_output_tensor_logged``);
    the predicate path must do the same, or the edited value has no producer:
    a downstream op loses its parent edge, and a model output that IS the
    edited value cannot be attributed (``output_attribution_failed``).

    Parameters
    ----------
    trace:
        Active predicate-mode trace.
    state:
        Active fastlog recording state; the exiting module's frame is still
        the innermost entry of ``state.module_stack``.
    out:
        Module output after live boundary interventions.

    Returns
    -------
    None
        Each replaced leaf is labeled and committed as one sparse op event
        whose parent is the op the edit replaced.
    """

    from ...capture.predicates import _evaluate_keep_op, build_op_record_context
    from ...capture.projections import append_projected_event
    from ...fastlog.types import CaptureSpec
    from ...intervention.runtime import (
        _peek_module_intervention_parent_labels,
        _peek_tensor_live_fire_results,
    )
    from ...ir.predicate import RetroactiveCaptureDecision
    from ._ops_predicates import _record_predicate_output
    from .model_prep import _live_intervention_machinery_armed, _note_replacement_event
    from .ops import _walk_output_tensors_with_paths

    layer_type = "interventionreplacement"
    # Like the exhaustive replacement op, the edited value belongs to the scope
    # that CONSUMES the exited module's output, never to the exited module.
    consumer_stack = tuple(state.module_stack)[:-1]
    consumer_frame = consumer_stack[-1] if consumer_stack else None
    for tensor, _container_path, _container_spec in _walk_output_tensors_with_paths(out):
        if get_tensor_label(tensor) is not None:
            continue
        if not any(result.replaced for result in _peek_tensor_live_fire_results(tensor)):
            continue
        trace._raw_graph_ws.layer_counter += 1
        trace._raw_graph_ws.raw_layer_type_counter[layer_type] += 1
        state.op_counts[layer_type] = state.op_counts.get(layer_type, 0) + 1
        state.step_index += 1
        state.event_index += 1
        raw_index = trace._raw_graph_ws.layer_counter
        type_index = trace._raw_graph_ws.raw_layer_type_counter[layer_type]
        raw_label = f"{layer_type}_{type_index}_{raw_index}_raw"
        parent_labels = tuple(_peek_module_intervention_parent_labels(tensor, trace))
        set_tensor_label(tensor, raw_label)
        ctx = build_op_record_context(
            kind="op",
            label=raw_label,
            raw_label=raw_label,
            raw_index=raw_index,
            layer_type=layer_type,
            type_index=type_index,
            func_name="intervention_replacement",
            parent_labels=parent_labels,
            tensor=tensor,
            output_index=0,
            is_bottom_level_func=True,
            module_stack=consumer_stack,
            history=tuple(state.history),
            op_counts=state.op_counts,
            pass_index=state.pass_index,
            event_index=state.event_index,
            step_index=state.step_index,
            capture_start_time=trace.capture_start_time,
            include_source_events=state.options.include_source_events,
            sample_id=state.sample_id,
            address=consumer_frame.address if consumer_frame else None,
            module_type=consumer_frame.module_type if consumer_frame else None,
            module_pass_index=consumer_frame.pass_index if consumer_frame else None,
        )
        spec = CaptureSpec(save_out=False, save_metadata=False)
        ram_payload = None
        transformed_ram_payload = None
        try:
            decision = _evaluate_keep_op(ctx, state.options)
            if not isinstance(decision, RetroactiveCaptureDecision):
                spec = decision
                ram_payload, transformed_ram_payload = _record_predicate_output(ctx, tensor, spec)
        # The user's save= predicate may raise anything; the recording state's
        # handler applies the capture's predicate-error policy, as every
        # predicate evaluation site does.
        except Exception as exc:  # noqa: BLE001
            state.handle_predicate_exception(ctx, exc)
        append_projected_event(
            trace,
            ctx,
            spec,
            tensor=tensor,
            ram_payload=ram_payload,
            transformed_ram_payload=transformed_ram_payload,
            predicate_matched=spec.save_out or spec.save_metadata,
        )
        state.append_context(ctx)
        if _live_intervention_machinery_armed() or (
            getattr(getattr(trace, "_predicate_save_options", None), "intervene", None) is not None
        ):
            _note_replacement_event(trace, raw_label, origin="live_fire")
