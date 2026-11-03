"""Parentless-op exemption evidence for the graph-connectivity invariant."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def _op_follows_recorded_backward_trigger(ml: Trace, layer: Op) -> bool:
    """Return whether an op ran after a RECORDED mid-forward backward trigger.

    ``torch.autograd.grad`` / ``loss.backward()`` fired mid-forward produce
    gradient tensors whose construction is legitimately untraceable, and a
    downstream op consuming one is a genuine parentless consumer of an
    unattributed tensor (e.g. MAML-style ``weight_grad.mean()``) -- NOT a
    silent capture drop. The trigger itself is positively recorded in the
    capture's backward events, so the exemption is anchored to trace-level
    evidence, not the op's self-claims: a plain capture with no recorded
    backward trigger can never use it.

    Parameters
    ----------
    ml:
        Trace being validated.
    layer:
        Parentless op under the connectivity check.

    Returns
    -------
    bool
        True when a backward trigger was recorded at a forward-op position
        before this op ran.
    """

    capture_events = getattr(ml, "_capture_events", None) or getattr(ml, "capture_events", None)
    backward_events = getattr(capture_events, "backward_events", None) or ()
    trigger_positions = [
        event.forward_op_count_at_trigger
        for event in backward_events
        if type(event).__name__ == "BackwardPassStart"
        and getattr(event, "forward_op_count_at_trigger", None) is not None
    ]
    if not trigger_positions:
        return False
    step_index = getattr(layer, "step_index", None)
    return step_index is not None and step_index > min(trigger_positions)


def _consumed_unattributed_data_operand(layer: Op) -> bool:
    """Return whether the capture witness flagged a DATA-operand consumption.

    Parameters
    ----------
    layer:
        Layer pass whose ``unattributed_tensor_args`` witness is inspected.

    Returns
    -------
    bool
        True when the capture-side witness recorded any unattributed tensor
        argument. A runtime tensor at ANY input slot is a data dependency
        (round-31 H2) -- including schema-typed ``int``/``Scalar`` control
        slots -- so every witness position counts; the former ATen-schema
        metadata-slot exclusion is gone.
    """

    return bool(tuple(getattr(layer, "unattributed_tensor_args", ()) or ()))
