"""Shared saved-output summary helpers."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


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
        activations on ``trace._layer_list``.

    Returns
    -------
    int
        Number of ModuleCalls whose output Op labels include at least one saved
        activation Op.
    """

    if saved_op_labels is None:
        saved_labels = {
            op.label
            for op in getattr(trace, "_layer_list", ())
            if getattr(op, "has_saved_activation", False) and not getattr(op, "is_orphan", False)
        }
    else:
        saved_labels = set(saved_op_labels)
    aliases = op_label_aliases(trace)
    return sum(
        1
        for module_call in getattr(trace, "module_calls", ())
        if any(
            aliases.get(label, label) in saved_labels
            for label in getattr(module_call, "output_ops", ())
        )
    )


def op_label_aliases(trace: "Trace") -> dict[str, str]:
    """Map each single-pass Layer label to its one pass-qualified Op label.

    The root ModuleCall stores its ``output_ops`` as layer-space labels
    (``output_1``) while every other call stores pass-qualified Op labels
    (``linear_1_1:1``), so a membership test against Op labels must resolve
    the layer-space form first or it never counts the root call. Only a Layer
    with exactly one pass names one Op; a multi-pass Layer label stays as is.

    Parameters
    ----------
    trace:
        Trace whose Op labels are mapped.

    Returns
    -------
    dict[str, str]
        Pass-qualified Op label keyed by single-pass Layer label.
    """

    by_layer: dict[str, list[str]] = {}
    for op in getattr(trace, "_layer_list", ()):
        by_layer.setdefault(op.layer_label, []).append(op.label)
    return {layer: labels[0] for layer, labels in by_layer.items() if len(labels) == 1}


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
    Sets ``trace.pinned_autograd_memory``: ``Bytes(0)`` when every
    payload is detached, ``None`` on a non-torch backend, where the walk does
    not apply.
    """

    from .._quantities import Bytes
    from ..backends import TORCH_BACKEND_NAME

    backend_name = str(getattr(trace, "backend", TORCH_BACKEND_NAME) or TORCH_BACKEND_NAME)
    if backend_name != TORCH_BACKEND_NAME:
        trace.pinned_autograd_memory = None
        return
    from ..backends.torch._pinned_graph import pinned_autograd_bytes

    payloads = [
        payload
        for op in saved_ops
        for payload in (getattr(op, "out", None), getattr(op, "transformed_out", None))
    ]
    trace.pinned_autograd_memory = Bytes(pinned_autograd_bytes(payloads))
