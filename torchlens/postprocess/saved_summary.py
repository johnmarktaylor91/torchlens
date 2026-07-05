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
