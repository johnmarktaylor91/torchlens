"""Resolved stop and capture configuration for one predicate recording pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._capture.stages import InternalCaptureConfig
from .._deprecations import MISSING, MissingType
from .._ir.predicate import CaptureSpec
from .._training_validation import TrainingModeConfigError

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .options import _RecordingConfig


def _configure_predicate_capture(trace: Trace, options: _RecordingConfig) -> None:
    """Install the capture configuration using the trace's resolved stop directive.

    Parameters
    ----------
    trace
        Predicate-mode trace prepared for the next recorded pass.
    options
        Resolved recorder settings supplying the capture seed.
    """

    trace._capture_config = InternalCaptureConfig(
        capture_mode="predicate",
        layers_to_save=[],
        grad_layers_to_save=[],
        random_seed=options.seed,
        postprocess=False,
        stop=trace._stop_directive,
    )


def _resolve_train_mode_default(
    *,
    field_name: str,
    value: bool | CaptureSpec | MissingType,
    backward_ready: bool,
) -> bool | CaptureSpec | MissingType:
    """Resolve one default capture option for train-mode sugar."""

    if backward_ready and value is MISSING:
        return CaptureSpec(save_grad=True, save_out=True, save_metadata=True)
    if not backward_ready or value is MISSING or value is False:
        return value
    if value is True:
        raise TrainingModeConfigError(
            f"backward_ready=True conflicts with {field_name}=True because True uses "
            "save_grad=False; use CaptureSpec(save_grad=True) or omit the default. "
            "Remedy: pass CaptureSpec(save_grad=True) or omit the default.",
            code="backward_ready_conflict",
        )
    if isinstance(value, CaptureSpec) and not value.save_grad:
        raise TrainingModeConfigError(
            f"backward_ready=True conflicts with {field_name}=CaptureSpec(save_grad=False). "
            "Remedy: pass CaptureSpec(save_grad=True) or omit the default.",
            code="backward_ready_conflict",
        )
    return value


def _resolve_train_mode_defaults(
    default_op: bool | CaptureSpec | MissingType,
    default_module: bool | CaptureSpec | MissingType,
    backward_ready: bool,
) -> tuple[bool | CaptureSpec | MissingType, bool | CaptureSpec | MissingType]:
    """Resolve both train-mode default capture options (op, then module)."""

    return (
        _resolve_train_mode_default(
            field_name="default_op", value=default_op, backward_ready=backward_ready
        ),
        _resolve_train_mode_default(
            field_name="default_module", value=default_module, backward_ready=backward_ready
        ),
    )
