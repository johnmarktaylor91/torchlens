"""Shared saved-output summary helpers."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


def count_saved_module_calls_from_output_ops(
    trace: "Trace",
    saved_op_labels: Iterable[str] | None = None,
) -> int:
    """Count saved ModuleCalls using the Trace.saved_module_calls output-op rule.

    Parameters
    ----------
    trace:
        Trace whose module calls should be counted.
    saved_op_labels:
        Optional saved Op labels. When omitted, labels are derived from saved
        activations on ``trace.layer_list``.

    Returns
    -------
    int
        Number of ModuleCalls whose output Op labels include at least one saved
        activation Op.
    """

    if saved_op_labels is None:
        saved_labels = {
            op.label
            for op in getattr(trace, "layer_list", ())
            if getattr(op, "has_saved_activation", False) and not getattr(op, "is_orphan", False)
        }
    else:
        saved_labels = set(saved_op_labels)
    return sum(
        1
        for module_call in getattr(trace, "module_calls", ())
        if any(label in saved_labels for label in getattr(module_call, "output_ops", ()))
    )


def refresh_saved_module_call_count(
    trace: "Trace",
    saved_op_labels: Iterable[str] | None = None,
) -> int:
    """Refresh and return ``trace.num_saved_module_calls``.

    Parameters
    ----------
    trace:
        Trace whose stored counter should be refreshed.
    saved_op_labels:
        Optional saved Op labels forwarded to
        ``count_saved_module_calls_from_output_ops``.

    Returns
    -------
    int
        Refreshed saved ModuleCall count.
    """

    count = count_saved_module_calls_from_output_ops(trace, saved_op_labels)
    trace.num_saved_module_calls = count
    return count


def refresh_pinned_autograd_memory(trace: "Trace", saved_ops: Iterable[object]) -> None:
    """Disclose the autograd bytes the trace's attached saved activations keep alive.

    Parameters
    ----------
    trace:
        Trace whose saved-output summary is being refreshed.
    saved_ops:
        Ops that retain an activation payload.

    Notes
    -----
    Sets ``trace.pinned_autograd_memory`` (R2-37): ``Bytes(0)`` when every
    payload is detached, ``None`` on a non-torch backend, where the walk does
    not apply.
    """

    from ..quantities import Bytes

    if str(getattr(trace, "backend", "torch") or "torch") != "torch":
        trace.pinned_autograd_memory = None
        return
    from ..backends.torch._pinned_graph import pinned_autograd_bytes

    payloads = [
        payload
        for op in saved_ops
        for payload in (getattr(op, "out", None), getattr(op, "transformed_out", None))
    ]
    trace.pinned_autograd_memory = Bytes(pinned_autograd_bytes(payloads))
