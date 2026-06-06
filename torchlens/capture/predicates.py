"""Predicate evaluation helpers for capture-time fastlog projections."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from ..fastlog.exceptions import PredicateError
from ..fastlog.types import CaptureSpec, ModuleStackFrame, RecordContext

if TYPE_CHECKING:
    from ..fastlog.options import RecordingOptions


def _coerce_default_capture_spec(default: bool | CaptureSpec) -> CaptureSpec:
    """Normalize a default capture value to a CaptureSpec."""

    if isinstance(default, CaptureSpec):
        return default
    if default is True:
        return CaptureSpec(save_out=True, save_metadata=True)
    if default is False:
        return CaptureSpec(save_out=False, save_metadata=False)
    raise PredicateError("default capture decision must be bool or CaptureSpec")


def _normalize_capture_decision(
    result: bool | CaptureSpec | None,
    ctx: RecordContext,
    default: bool | CaptureSpec,
) -> CaptureSpec:
    """Normalize one predicate return value to a CaptureSpec.

    Parameters
    ----------
    result:
        Predicate return value.
    ctx:
        Event context supplied to the predicate.
    default:
        Slot default used when ``result`` is None.

    Returns
    -------
    CaptureSpec
        Normalized capture policy.

    Raises
    ------
    PredicateError
        If the predicate returned a value outside the supported contract.
    """

    default_spec = _coerce_default_capture_spec(default)
    if result is True:
        return CaptureSpec(
            save_out=True,
            save_metadata=True,
            keep_grad=default_spec.keep_grad,
            device=default_spec.device,
            dtype=default_spec.dtype,
        )
    if result is False:
        return CaptureSpec(save_out=False, save_metadata=False)
    if result is None:
        return default_spec
    if isinstance(result, CaptureSpec):
        return result
    raise PredicateError(
        "predicate must return bool, CaptureSpec, or None",
        ctx=ctx,
        result=result,
    )


def _evaluate_keep_op(ctx: RecordContext, options: "RecordingOptions") -> CaptureSpec:
    """Evaluate the operation/source predicate slot for one event."""

    if options.keep_op is None:
        result = None
    else:
        result = options.keep_op(ctx)
        if (
            result is False
            and ctx.kind == "op"
            and ctx.layer_type is not None
            and ctx.type_index is not None
        ):
            alias_ctx = replace(ctx, label=f"{ctx.layer_type}_{ctx.type_index}")
            result = options.keep_op(alias_ctx)
            if result is not False:
                ctx = alias_ctx
    return _normalize_capture_decision(result, ctx, options.default_op)


def _evaluate_keep_module(ctx: RecordContext, options: "RecordingOptions") -> CaptureSpec:
    """Evaluate the module predicate slot for one event."""

    if options.keep_module is None:
        result = None
    else:
        result = options.keep_module(ctx)
    return _normalize_capture_decision(result, ctx, options.default_module)


def build_op_record_context(
    *,
    kind: str,
    label: str,
    raw_label: str,
    raw_index: int,
    layer_type: str,
    type_index: int,
    func_name: str | None,
    parent_labels: Sequence[str],
    tensor: torch.Tensor,
    output_index: int | None,
    is_bottom_level_func: bool | None,
    module_stack: Sequence[ModuleStackFrame | Mapping[str, Any]],
    history: Sequence[RecordContext],
    op_counts: Mapping[str, int],
    pass_index: int,
    event_index: int,
    step_index: int | None,
    capture_start_time: float,
    include_source_events: bool,
    sample_id: str | int | None,
    address: str | None = None,
    module_type: str | None = None,
    module_pass_index: int | None = None,
) -> RecordContext:
    """Build the unified operation ``RecordContext`` used by all capture paths.

    Parameters
    ----------
    kind:
        Event kind, usually ``"op"``.
    label, raw_label:
        Public-in-flight and raw labels for the operation.
    raw_index:
        Global raw operation index.
    layer_type, type_index:
        Normalized TorchLens operation type and per-type counter.
    func_name:
        Original function name, when known.
    parent_labels:
        Raw parent labels visible at forward time.
    tensor:
        Output tensor being considered.
    output_index:
        Index within a multi-output operation, when applicable.
    is_bottom_level_func:
        Whether the decorated call is a bottom-level function.
    module_stack:
        Active module-stack frames.
    history:
        Bounded recent ``RecordContext`` window.
    op_counts:
        Per-operation-type counts visible to predicate code.
    pass_index:
        Forward pass index for the active capture session.
    event_index:
        Chronological event index.
    step_index:
        Operation step index.
    capture_start_time:
        Wall-clock start time for the current capture.
    include_source_events:
        Whether source events should appear in ``recent_ops``.
    sample_id:
        Optional sample id for batched predicate runs.
    address, module_type, module_pass_index:
        Nearest module context fields, when known.

    Returns
    -------
    RecordContext
        Frozen predicate context.
    """

    from .projections import _build_record_context

    return _build_record_context(
        kind=kind,
        op_log_or_op_data={
            "label": label,
            "raw_label": raw_label,
            "_label_raw": raw_label,
            "raw_index": raw_index,
            "type": layer_type,
            "type_index": type_index,
            "func_name": func_name,
            "parent_labels": tuple(parent_labels),
            "tensor": tensor,
            "output_index": output_index,
            "is_bottom_level_func": is_bottom_level_func,
            "address": address,
            "module_type": module_type,
            "module_pass_index": module_pass_index,
        },
        module_stack=module_stack,
        history=tuple(history),
        op_counts=op_counts,
        pass_index=pass_index,
        event_index=event_index,
        step_index=step_index,
        time_since_pass_start=time.time() - capture_start_time,
        include_source_events=include_source_events,
        sample_id=sample_id,
    )
