"""Op-level intervention verbs: ``do``, ``set``, ``grad_for``, facets.

Reached from ``Op`` through ``_data_classes._doors.intervention()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._errors import PayloadUnavailableError

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


__all__ = [
    "derived_grad",
    "grad_for",
    "do",
    "set",
]


def derived_grad(op: Op) -> Any | None:
    """Return this op's derived intermediate gradient payload, if available.

    Returns
    -------
    Any | None
        Backend-owned gradient payload from ``trace.autograd.intermediate_derived_grads``,
        or ``None`` when this op has no exact intermediate-derived record.
    """

    trace = op._source_trace
    records = getattr(getattr(trace, "autograd", None), "intermediate_derived_grads", None)
    if records is None:
        return None
    if op.label not in records:
        return None
    record = records[op.label]
    if record.provenance.get("status") != "exact":
        return None
    return record.grad


def grad_for(op: Op, *, bwd: int) -> torch.Tensor:
    """Return the saved gradient tensor for one backward pass.

    Parameters
    ----------
    op:
        The op record this function reads and writes.
    bwd:
        One-based backward pass number.

    Returns
    -------
    torch.Tensor
        Saved raw gradient tensor for the requested pass.
    """

    record = op.autograd.grads.for_pass(bwd)
    if record.grad is None:
        raise PayloadUnavailableError(
            f"op {op.label} has no saved gradient payload for backward pass {bwd}",
            code="gradient_not_saved",
            remedy="capture with gradient saving enabled for this op and pass",
            label=str(op.label),
            backward_pass=bwd,
        )
    return record.grad


def do(
    op: Op,
    input_transform: Any,
    *,
    model: Any = None,
    inputs: Any = None,
    intervention: Any = None,
) -> Trace:
    """Apply an intervention to this op through the owning Trace.

    Parameters
    ----------
    op:
        The op record this function reads and writes.
    input_transform:
        Transform or hook to apply to this operation's output.
    model:
        Model required when ``intervention.engine="rerun"``.
    inputs:
        Input required when ``intervention.engine="rerun"``.
    intervention:
        Grouped intervention options (``InterventionOptions``).

    Returns
    -------
    Trace
        Source Trace after applying the intervention.
    """

    return op._source_trace_or_error().do(
        op.layer_label,
        input_transform,
        model=model,
        inputs=inputs,
        intervention=intervention,
    )
