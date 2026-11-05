"""Module-build helpers for the root call's labels, uncalled modules and Layer roles.

Split out of ``finalization.py`` (step 16 module build and step 15.5
conditional records), which sits at the package's file-size cap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace

__all__ = ("bind_layer_conditional_roles", "op_space_labels", "uncalled_module_addresses")


def op_space_labels(trace: Trace, labels: Sequence[str]) -> list[str]:
    """Map trace-side boundary labels to pass-qualified Op labels.

    ``ModuleCall.output_ops`` speaks Op space on every call (``output_1:1``),
    while ``trace.output_layers`` may hold bare labels.

    Parameters
    ----------
    trace:
        Trace owning the label lookup tables.
    labels:
        Boundary labels (raw, pass-qualified, or layer labels).

    Returns
    -------
    list[str]
        Pass-qualified Op labels, preserving order; unresolvable labels pass through.
    """

    op_labels: list[str] = []
    for label in labels:
        op = trace._layer_dict_all_keys.get(label)
        resolved = getattr(op, "label", None) if op is not None else None
        op_labels.append(resolved if isinstance(resolved, str) else label)
    return op_labels


def uncalled_module_addresses(trace: Trace) -> tuple[str, ...] | None:
    """Return the registered module addresses this capture never called.

    Parameters
    ----------
    trace:
        Trace whose module records were just built.

    Returns
    -------
    tuple[str, ...] | None
        Sorted addresses (the root is ``"self"``), or ``None`` when the source
        model is already gone and the set cannot be known.
    """

    source_ref = trace._source_model_ref
    model = source_ref() if source_ref is not None else None
    if model is None or not hasattr(model, "named_modules"):
        return None
    registered = {address or "self" for address, _module in model.named_modules()}
    module_logs = trace._module_logs
    called = set(getattr(module_logs, "_dict", {}).keys())
    called.update(getattr(module_logs, "_alias_dict", {}).keys())
    return tuple(sorted(registered - called))


def bind_layer_conditional_roles(trace: Trace, roles_by_label: Mapping[str, Sequence[Any]]) -> None:
    """Give each aggregate Layer the conditional roles its Ops carry.

    The aggregate Layers are built before the conditional records exist, so
    their ``in_conditionals`` snapshot is empty; the roles are keyed by Layer
    label, so every pass of a Layer carries the same list.

    Parameters
    ----------
    trace:
        Trace whose conditional records were just built.
    roles_by_label:
        Conditional role refs keyed by Layer label.
    """

    for layer_label, layer in (trace._layer_logs or {}).items():
        layer._control_flow_in_conditionals = list(roles_by_label.get(layer_label, []))


def _call_io_layers(
    self: Trace,
    call_label: str,
    pass_layers: list[str],
    pass_input_layers_by_call: dict[str, list[str]] | None,
) -> tuple[list[str], list[str], list[str], list[Op]]:
    """Derive one module call's input and output layers from its Op fields.

    Parameters
    ----------
    self:
        Trace being finalized.
    call_label:
        Module call label (``address:call_index``).
    pass_layers:
        Op labels executed inside the call.
    pass_input_layers_by_call:
        Pre-collected input layers keyed by call label, if any.

    Returns
    -------
    tuple[list[str], list[str], list[str], list[Op]]
        Input layer labels, output layer labels, output op labels and the
        output Op entries.
    """

    pass_input_layers = list((pass_input_layers_by_call or {}).get(call_label, []))
    pass_output_layers = []
    pass_output_ops = []
    pass_outputs = []
    for op_label in pass_layers:
        try:
            te = self.ops[op_label]
        except (KeyError, TypeError):
            continue
        # B3R7-R05-1: ``input_to_module_calls`` is the authoritative
        # fed-call fact. The old ``len(module_call_stack) > 0`` companion
        # test was a proxy for the same thing only while the enter lane
        # wrote both together; under containment semantics it would drop
        # top-level-created inputs (e.g. the model input) from the call's
        # input roster.
        if (
            call_label in te.module_context.input_to_module_calls
            and te.layer_label not in pass_input_layers
        ):
            pass_input_layers.append(te.layer_label)
        if (
            te.module_context.is_module_output
            and call_label in te.module_context.output_of_module_calls
        ):
            pass_output_layers.append(te.layer_label)
            pass_output_ops.append(te.label)
            pass_outputs.append(te)
    return pass_input_layers, pass_output_layers, pass_output_ops, pass_outputs
