"""Predicate evaluation helpers for fastlog capture decisions."""

from __future__ import annotations

from .exceptions import PredicateError
from .options import RecordingOptions
from .types import CaptureSpec, RecordContext


def _coerce_default_capture_spec(default: bool | CaptureSpec) -> CaptureSpec:
    """Normalize a default capture value to a CaptureSpec."""

    if isinstance(default, CaptureSpec):
        return default
    if default is True:
        return CaptureSpec(save_activation=True, save_metadata=True)
    if default is False:
        return CaptureSpec(save_activation=False, save_metadata=False)
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
            save_activation=True,
            save_metadata=True,
            keep_grad=default_spec.keep_grad,
            device=default_spec.device,
            dtype=default_spec.dtype,
        )
    if result is False:
        return CaptureSpec(save_activation=False, save_metadata=False)
    if result is None:
        return default_spec
    if isinstance(result, CaptureSpec):
        return result
    raise PredicateError(
        "predicate must return bool, CaptureSpec, or None",
        ctx=ctx,
        result=result,
    )


def _evaluate_keep_op(ctx: RecordContext, options: RecordingOptions) -> CaptureSpec:
    """Evaluate the operation/source predicate slot for one event."""

    if options.keep_op is None:
        result = None
    else:
        result = options.keep_op(ctx)
    return _normalize_capture_decision(result, ctx, options.default_op)


def _evaluate_keep_module(ctx: RecordContext, options: RecordingOptions) -> CaptureSpec:
    """Evaluate the module predicate slot for one event."""

    if options.keep_module is None:
        result = None
    else:
        result = options.keep_module(ctx)
    return _normalize_capture_decision(result, ctx, options.default_module)
