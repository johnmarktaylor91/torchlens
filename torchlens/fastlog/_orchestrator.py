"""Internal predicate-pass orchestrator for fastlog."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .. import _state
from ..backends.torch.sources import log_source_tensor
from ..data_classes.model_log import Trace
from ..backends.torch.model_prep import (
    _cleanup_model_session,
    _ensure_model_prepared,
    _prepare_model_session,
)
from ..backends.torch import module_stack as _mstack
from ..utils.introspection import get_vars_of_type_from_obj
from ..utils.rng import set_random_seed
from ._predicate import _evaluate_keep_module
from ._record_context import _build_record_context
from ._state import RecordingState, active_recording_state
from .options import RecordingOptions
from .types import ActivationRecord, ModuleStackFrame, Recording


def _empty_recording(options: RecordingOptions) -> Recording:
    """Create an empty in-memory Recording for a predicate pass."""

    return Recording(
        records=[],
        by_pass={},
        by_label={},
        by_address={},
        bundle_path=(
            None
            if options.streaming is None or options.streaming.bundle_path is None
            else Path(options.streaming.bundle_path)
        ),
        n_ops=0,
        n_records=0,
        start_times=[],
        end_times=[],
        predicate_failures=[],
        predicate_failure_overflow_count=0,
        keep_op_repr=repr(options.keep_op) if options.keep_op is not None else None,
        keep_module_repr=repr(options.keep_module) if options.keep_module is not None else None,
        history_size=options.history_size,
        _out_transform_repr=(
            repr(options.out_transform) if options.out_transform is not None else None
        ),
    )


def _normalize_input_args(input_args: Any) -> tuple[Any, ...]:
    """Normalize public-style input args into a tuple for model invocation."""

    if isinstance(input_args, tuple):
        return input_args
    if isinstance(input_args, list):
        return tuple(input_args)
    return (input_args,)


def _reset_state_for_pass(
    state: RecordingState,
    *,
    pass_index: int,
    sample_id: str | int | None,
) -> None:
    """Reset per-pass state while preserving accumulated records and storage."""

    state.history.clear()
    state.op_counts.clear()
    state.module_stack.clear()
    state.sample_id = sample_id
    state.pass_index = pass_index
    state.event_index = 0
    state.compute_index = 0


def _emit_root_module_event(
    *,
    trace: Trace,
    state: RecordingState,
    model: nn.Module,
    kind: str,
    frame: ModuleStackFrame,
) -> None:
    """Emit one synthetic root module event."""

    state.event_index += 1
    ctx = _build_record_context(
        kind="module_enter" if kind == "enter" else "module_exit",
        op_log_or_op_data={
            "label": f"root:{kind}:1",
            "address": "",
            "module_type": type(model).__name__,
            "module_pass_index": frame.pass_index,
        },
        module_stack=state.module_stack,
        history=tuple(state.history),
        op_counts=state.op_counts,
        pass_index=state.pass_index,
        event_index=state.event_index,
        compute_index=None,
        time_since_pass_start=time.time() - trace.start_time,
        include_source_events=state.options.include_source_events,
        sample_id=state.sample_id,
    )
    try:
        spec = _evaluate_keep_module(ctx, state.options)
        if spec.save_out or spec.save_metadata:
            state.add_record(ActivationRecord(ctx=ctx, spec=spec))
    except Exception as exc:
        state.handle_predicate_exception(ctx, exc)
    finally:
        state.append_context(ctx)


def _run_predicate_pass(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None,
    options: RecordingOptions,
    *,
    state: RecordingState | None = None,
    pass_index: int = 1,
    sample_id: str | int | None = None,
    finalize_storage: bool = True,
) -> tuple[Any, Recording]:
    """Run one predicate-mode forward pass and return output plus Recording."""

    if options.random_seed is not None:
        set_random_seed(options.random_seed)
    args = _normalize_input_args(input_args)
    kwargs = input_kwargs or {}
    trace = Trace(str(type(model).__name__))
    trace.capture_mode = "predicate"
    trace.start_time = time.time()
    if state is None:
        recording = _empty_recording(options)
        state = RecordingState(options=options, recording=recording)
    else:
        recording = state.recording
    _reset_state_for_pass(state, pass_index=pass_index, sample_id=sample_id)
    recording.start_times.append(trace.start_time)
    input_tensors = get_vars_of_type_from_obj([args, kwargs], torch.Tensor, [torch.nn.Parameter])
    _ensure_model_prepared(model)
    _prepare_model_session(trace, model)
    model_output = None
    root_frame = ModuleStackFrame(
        address="",
        module_type=type(model).__name__,
        module_id=id(model),
        pass_index=1,
    )
    pass_failed = False
    try:
        with active_recording_state(state), _state.active_logging(trace):
            for index, tensor in enumerate(input_tensors):
                log_source_tensor(trace, tensor, "input", f"input.{index}")
            _mstack.push_existing_frame(state.module_stack, root_frame)
            _emit_root_module_event(
                trace=trace,
                state=state,
                model=model,
                kind="enter",
                frame=root_frame,
            )
            try:
                model_output = model(*args, **kwargs)
            finally:
                _emit_root_module_event(
                    trace=trace,
                    state=state,
                    model=model,
                    kind="exit",
                    frame=root_frame,
                )
                _mstack.pop_frame(state.module_stack, root_frame)
    except Exception as exc:
        pass_failed = True
        state.abort_storage(str(exc))
        raise
    finally:
        _cleanup_model_session(model, input_tensors)
        end_time = time.time()
        recording.end_times.append(end_time)
        object.__setattr__(recording, "n_ops", max(recording.n_ops, pass_index))
        object.__setattr__(recording, "n_records", len(recording.records))
        if not pass_failed and finalize_storage:
            state.finalize_storage()
            state.raise_accumulated_predicate_error()
    return model_output, recording
