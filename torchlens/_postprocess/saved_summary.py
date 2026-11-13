"""Shared saved-output summary helpers."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .._quantities import Bytes
from .._utils._record_group_table import record_member
from ..backends import TORCH_BACKEND_NAME
from ..backends.registry import get_backend_spec
from ..backends.torch._pinned_graph import pinned_autograd_bytes

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
            for op in trace._layer_list
            if getattr(op, "has_saved_activation", False)
            and not record_member(op, "is_orphan", False)
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
    for op in trace._layer_list:
        by_layer.setdefault(op.layer_label, []).append(op.label)
    return {layer: labels[0] for layer, labels in by_layer.items() if len(labels) == 1}


def refresh_saved_module_call_count(
    trace: "Trace",
    saved_op_labels: Iterable[str] | None = None,
) -> int:
    """Return the saved ModuleCall count.

    Parameters
    ----------
    trace:
        Trace whose saved ModuleCalls are counted.
    saved_op_labels:
        Optional saved Op labels forwarded to
        ``count_saved_module_calls_from_output_ops``.

    Returns
    -------
    int
        Saved ModuleCall count.
    """

    count = count_saved_module_calls_from_output_ops(trace, saved_op_labels)
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
    Sets ``trace.cost.pinned_autograd_memory``: ``Bytes(0)`` when every
    payload is detached, ``None`` on a non-torch backend, where the walk does
    not apply.
    """

    backend_name = str(trace._provenance_backend or TORCH_BACKEND_NAME)
    if backend_name != TORCH_BACKEND_NAME:
        trace._cost_pinned_autograd_memory = None
        return

    payloads = [
        payload
        for op in saved_ops
        for payload in (getattr(op, "out", None), record_member(op, "transformed_out", None))
    ]
    trace._cost_pinned_autograd_memory = Bytes(pinned_autograd_bytes(payloads))


def _has_retained_gradient_payloads(self: "Trace") -> bool:
    """Return whether any completed backward pass retained gradient payloads.

    Parameters
    ----------
    self:
        Trace whose projected operation and parameter gradients should be
        inspected.

    Returns
    -------
    bool
        Whether portable scrubbing must include gradient payloads.

    Notes
    -----
    The projection is the authority here, rather than ``Trace.options.save_grads``:
    ``backward(save_grads=...)`` may override that capture-time default on
    each pass.
    """

    backend = str(self._provenance_backend)
    if (
        backend != TORCH_BACKEND_NAME
        and not get_backend_spec(backend).capabilities.backward_capture
    ):
        return False  # no true-backward grads exist; the grads accessor refuses here
    for op in self._layer_list:
        for record in record_member(op, "grads", ()):
            if record.grad is not None or record.transformed_grad is not None:
                return True
    for param in self._param_logs.values():
        if any(record.grad is not None for record in record_member(param, "grads", ())):
            return True
    return False


def _refresh_fast_saved_summary(self: "Trace") -> None:
    """Refresh saved-output counters after retained layers are finalized.

    Args:
        self: Trace whose final retained layer entries were updated.

    Returns:
        None. Mutates aggregate saved-output fields on ``self``.
    """

    saved_layers = [
        layer_entry
        for layer_entry in self._layer_list
        if getattr(layer_entry, "has_saved_activation", False)
        and not record_member(layer_entry, "is_orphan", False)
    ]
    self._num_saved_ops = len(saved_layers)
    # Read the ONE byte model -- physically retained
    # alias-aware bytes -- never a sum of raw activation_memory fields. The
    # explorer memory truth (reduce-only captures count the retained
    # TRANSFORMED bytes, raw-retaining captures the raw bytes) is served by
    # this same aggregate: it counts the payloads capture actually kept.
    from .._save_budget import retained_activation_bytes

    self._cost_saved_activation_memory = Bytes(retained_activation_bytes(saved_layers))
    refresh_saved_module_call_count(self, {layer_entry.label for layer_entry in saved_layers})
    refresh_pinned_autograd_memory(self, saved_layers)
