"""Module-build helpers for the root call's labels, uncalled modules and Layer roles.

Split out of ``finalization.py`` (step 16 module build and step 15.5
conditional records), which sits at the package's file-size cap.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .._utils._model_walk import model_named_modules

if TYPE_CHECKING:
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


def uncalled_module_addresses(trace: Trace, called: set[str]) -> tuple[str, ...] | None:
    """Return the registered module addresses this capture never called.

    Parameters
    ----------
    trace:
        Trace whose source model is walked.
    called:
        Every address and alias of the modules the capture recorded.

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
    registered = {address or "self" for address, _module in model_named_modules(model)}
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


#: ``Module`` constructor arguments copied from the captured metadata, ``None`` when absent.
_SOURCE_AND_HOOK_NAMES = (
    "class_source_file",
    "class_source_line",
    "init_source_file",
    "init_source_line",
    "forward_source_file",
    "forward_source_line",
    "class_docstring",
    "init_signature",
    "init_docstring",
    "forward_signature",
    "forward_docstring",
    "forward_pre_hooks",
    "forward_hooks",
    "backward_pre_hooks",
    "backward_hooks",
    "full_backward_pre_hooks",
    "full_backward_hooks",
)


def module_source_and_hooks(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Return a module's source and hook facts from its captured metadata.

    Parameters
    ----------
    meta:
        Metadata captured for the module during model preparation.

    Returns
    -------
    dict[str, Any]
        One ``Module`` constructor argument per source or hook fact, ``None``
        for a fact the metadata does not hold.
    """

    facts: dict[str, Any] = dict.fromkeys(_SOURCE_AND_HOOK_NAMES)
    for name in _SOURCE_AND_HOOK_NAMES:
        if name in meta:
            facts[name] = meta[name]
    return facts


def param_totals(params: Iterable[Any]) -> tuple[int, int, int, int]:
    """Return the parameter count, trainable count, frozen count and memory of ``params``.

    One pass that reads each parameter's count, trainable flag and memory once.
    """

    total = trainable = memory = 0
    for param in params:
        count = param.num_params
        total += count
        if param.is_trainable:
            trainable += count
        memory += int(param.param_memory)
    return total, trainable, total - trainable, memory


def _call_io_layers(
    call_label: str,
    pass_layers: list[str],
    input_layers_by_call: dict[str, list[str]],
    outputs_by_call: dict[str, dict[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    """Derive one module call's input and output layers from the trace-wide rosters.

    Both rosters were collected over every op of the trace, so no op is read here.

    Parameters
    ----------
    call_label:
        Module call label (``address:call_index``).
    pass_layers:
        Op labels executed inside the call.
    input_layers_by_call:
        Layer labels of the ops that feed each call (``input_to_module_calls``
        is the authoritative fed-call fact), keyed by call label.
    outputs_by_call:
        Each call's output ops as ``{op label: layer label}``, keyed by call label.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Input layer labels, output layer labels and output op labels; the
        outputs are the call's own ops that are its outputs, in call order.
    """

    pass_input_layers = list(input_layers_by_call.get(call_label, []))
    outputs = outputs_by_call.get(call_label)
    if not outputs:
        return pass_input_layers, [], []
    pass_output_ops = [op_label for op_label in pass_layers if op_label in outputs]
    return pass_input_layers, [outputs[op_label] for op_label in pass_output_ops], pass_output_ops
