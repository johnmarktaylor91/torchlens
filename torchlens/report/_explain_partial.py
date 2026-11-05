"""Partial-trace detection and diagnosis sections for explain()."""

from __future__ import annotations

import traceback
from typing import Any

import torch


def _partial_sections(diagnosis: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Return the section structure of a partial-capture report.

    Parameters
    ----------
    diagnosis:
        Evidence fields from :func:`_partial_diagnosis`.

    Returns
    -------
    list[tuple[str, list[str]]]
        Ordered (title, bullet lines) sections; all are budget-undroppable
        because every line is failure evidence.
    """

    return [
        (
            "Capture status",
            [
                "- This is a partial capture; only operations completed"
                " before the failure are known.",
            ],
        ),
        (
            "Failure diagnosis",
            [
                (
                    "- Last completed op: "
                    f"{diagnosis['last_completed_op_label']} "
                    f"(shape={diagnosis['last_completed_op_shape']}, "
                    f"dtype={diagnosis['last_completed_op_dtype']}, "
                    f"device={diagnosis['last_completed_op_device']})."
                ),
                f"- Failing boundary: {diagnosis['failing_boundary']}.",
                (
                    "- Captured exception: "
                    f"{diagnosis['exception_type']}: {diagnosis['exception_message']}"
                ),
                f"- First non-finite evidence: {diagnosis['first_nonfinite']}",
            ],
        ),
    ]


def _is_partial_trace(log: Any) -> bool:
    """Return whether ``log`` is a :class:`PartialTrace` instance.

    Parameters
    ----------
    log:
        Candidate capture object.

    Returns
    -------
    bool
        Whether the object is a failed partial capture wrapper.
    """

    from .._partial import PartialTrace

    return isinstance(log, PartialTrace)


def _last_partial_op(log: Any) -> Any | None:
    """Return the last recorded completed operation in a partial capture.

    Parameters
    ----------
    log:
        Partial capture wrapper.

    Returns
    -------
    Any | None
        Last raw operation, or ``None`` when none completed.
    """

    raw_layers = tuple(getattr(log, "raw_layers", ()))
    return raw_layers[-1] if raw_layers else None


def _partial_boundary(log: Any) -> str:
    """Return the nearest evidence-backed failure boundary description.

    Parameters
    ----------
    log:
        Partial capture wrapper.

    Returns
    -------
    str
        Recorded exception field or traceback location, otherwise ``"unknown"``.
    """

    exception = log.original_exception
    fields = getattr(exception, "fields", {})
    if isinstance(fields, dict) and (fields.get("layer") or fields.get("op")):
        return f"layer={fields.get('layer', 'unknown')}, op={fields.get('op', 'unknown')}"
    extracted = traceback.extract_tb(exception.__traceback__)
    forward_frames = [frame for frame in extracted if frame.name == "forward"]
    if forward_frames:
        frame = forward_frames[-1]
        return f"forward at {frame.filename}:{frame.lineno}"
    if extracted:
        frame = extracted[-1]
        return f"{frame.name} at {frame.filename}:{frame.lineno}"
    return "unknown"


def _partial_diagnosis(log: Any) -> dict[str, Any]:
    """Collect evidence-backed fields for a failed partial capture.

    Parameters
    ----------
    log:
        Partial capture wrapper.

    Returns
    -------
    dict[str, Any]
        Last-op, boundary, exception, and non-finite evidence.
    """

    op = _last_partial_op(log)
    output = getattr(op, "out", None) if op is not None else None
    label = (
        str(getattr(op, "_label_raw", getattr(op, "_layer_label_raw", "unknown")))
        if op is not None
        else "unknown"
    )
    shape = getattr(op, "shape", None) if op is not None else None
    dtype = getattr(op, "dtype", None) if op is not None else None
    device = getattr(op, "device", None) if op is not None else None
    if isinstance(output, torch.Tensor):
        shape = tuple(output.shape) if shape is None else shape
        dtype = output.dtype if dtype is None else dtype
        device = output.device if device is None else device
    exception = log.original_exception
    return {
        "last_completed_op_label": label,
        "last_completed_op_shape": tuple(shape) if shape is not None else "unknown",
        "last_completed_op_dtype": str(dtype) if dtype is not None else "unknown",
        "last_completed_op_device": str(device) if device is not None else "unknown",
        "failing_boundary": _partial_boundary(log),
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "first_nonfinite": str(log.first_nonfinite()),
    }
