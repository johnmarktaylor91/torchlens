"""Stable module-containment snapshot serialization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _stringify(value: Any) -> Any:
    """Convert nested values into JSON-stable primitive containers.

    Parameters
    ----------
    value:
        Arbitrary value from a TorchLens log field.

    Returns
    -------
    Any
        JSON-stable representation with tuples and sets normalized.
    """

    if isinstance(value, tuple):
        return ":".join(str(item) for item in value)
    if isinstance(value, set):
        return sorted(_stringify(item) for item in value)
    if isinstance(value, list):
        return [_stringify(item) for item in value]
    if isinstance(value, Mapping):
        return {str(_stringify(key)): _stringify(val) for key, val in sorted(value.items())}
    return value


def _sorted_strings(values: Iterable[Any]) -> list[str]:
    """Return values normalized to a sorted list of strings.

    Parameters
    ----------
    values:
        Values to stringify and sort.

    Returns
    -------
    list[str]
        Sorted string values.
    """

    return sorted(str(_stringify(value)) for value in values)


def _module_call_string(value: Any) -> str | None:
    """Normalize a module call label to ``address:pass`` text.

    Parameters
    ----------
    value:
        Module call tuple, string, or ``None``.

    Returns
    -------
    str | None
        Normalized module call label.
    """

    if value is None:
        return None
    if isinstance(value, tuple):
        return ":".join(str(item) for item in value)
    return str(value)


def _module_call_list(values: Iterable[Any]) -> list[str]:
    """Normalize module call labels to a list of strings.

    Parameters
    ----------
    values:
        Iterable of module call labels.

    Returns
    -------
    list[str]
        Normalized call labels.
    """

    return [value for value in (_module_call_string(item) for item in values) if value is not None]


def _normalize_argnames(value: Mapping[Any, Any]) -> dict[str, Any]:
    """Normalize module-entry argument names for JSON comparison.

    Parameters
    ----------
    value:
        Raw module-entry argnames mapping.

    Returns
    -------
    dict[str, Any]
        Mapping keyed by normalized module call labels.
    """

    return {
        _module_call_string(key) or "None": _stringify(val)
        for key, val in sorted(value.items(), key=lambda item: str(_stringify(item[0])))
    }


def _op_label(op: Any) -> str:
    """Return a stable pass-qualified label for one op.

    Parameters
    ----------
    op:
        TorchLens op log.

    Returns
    -------
    str
        Pass-qualified layer label.
    """

    return str(getattr(op, "label", getattr(op, "layer_label_w_pass")))


def _op_dict(op: Any) -> dict[str, Any]:
    """Serialize one OpLog for module-containment equality.

    Parameters
    ----------
    op:
        TorchLens op log.

    Returns
    -------
    dict[str, Any]
        Stable op snapshot record.
    """

    func_name = (
        "interventionreplacement" if op.func_name == "intervention_replacement" else op.func_name
    )
    return {
        "atomic_module_call": _module_call_string(op.atomic_module_call),
        "func_name": func_name,
        "fx_call_index": op.fx_call_index,
        "fx_qualpath": op.fx_qualpath,
        "intervention_replaced": getattr(op, "intervention_replaced", False),
        "is_atomic_module_op": op.is_atomic_module_op,
        "is_internal_sink": op.is_internal_sink,
        "is_internal_source": op.is_internal_source,
        "is_orphan": getattr(op, "is_orphan", False),
        "is_submodule_output": op.is_submodule_output,
        "label": _op_label(op),
        "lookup_keys": _sorted_strings(op.lookup_keys),
        "module": _module_call_string(op.module),
        "module_entry_argnames": _normalize_argnames(op.module_entry_argnames),
        "module_ops_entered": _module_call_list(op.module_ops_entered),
        "modules": _module_call_list(op.modules),
        "output_of_module_calls": _module_call_list(op.output_of_module_calls),
        "output_of_modules": [str(value) for value in op.output_of_modules],
        "trace_index": op.trace_index,
    }


def _layer_dict(layer: Any) -> dict[str, Any]:
    """Serialize one LayerLog for module-containment equality.

    Parameters
    ----------
    layer:
        TorchLens layer log.

    Returns
    -------
    dict[str, Any]
        Stable layer snapshot record.
    """

    lookup_keys = {
        str(_stringify(key)) for op in layer.ops.values() for key in getattr(op, "lookup_keys", [])
    }
    return {
        "in_submodule": layer.in_submodule,
        "label": layer.layer_label,
        "lookup_keys": sorted(lookup_keys),
        "module": _module_call_string(layer.module),
        "module_call_depth": layer.module_call_depth,
        "modules": _module_call_list(layer.modules),
        "output_of_module_calls": _module_call_list(layer.output_of_module_calls),
        "output_of_modules": [str(value) for value in layer.output_of_modules],
        "trace_index": layer.trace_index,
    }


def _module_dict(module: Any) -> dict[str, Any]:
    """Serialize one ModuleLog for module-containment equality.

    Parameters
    ----------
    module:
        TorchLens module log.

    Returns
    -------
    dict[str, Any]
        Stable module snapshot record.
    """

    return {
        "address": module.address,
        "all_addresses": sorted(str(value) for value in module.all_addresses),
        "buffer_layers": list(module.buffer_layers),
        "call_labels": list(module.call_labels),
        "input_layers": list(module.input_layers),
        "is_train_mode": module.is_train_mode,
        "layers": list(module.layers),
        "num_calls": module.num_calls,
        "num_param_tensors": len(module.params),
        "num_params": module.num_params,
        "num_params_frozen": module.num_params_frozen,
        "num_params_trainable": module.num_params_trainable,
        "output_layers": list(module.output_layers),
    }


def _call_dict(call: Any) -> dict[str, Any]:
    """Serialize one ModuleCallLog for module-containment equality.

    Parameters
    ----------
    call:
        TorchLens module-call log.

    Returns
    -------
    dict[str, Any]
        Stable module-call snapshot record.
    """

    return {
        "address": call.address,
        "call_label": call.call_label,
        "forward_args_summary": call.forward_args_summary,
        "forward_kwargs_summary": call.forward_kwargs_summary,
        "input_layers": list(call.input_layers),
        "layers": list(call.layers),
        "output_layers": list(call.output_layers),
    }


def _equivalence_partition(trace: Any) -> list[list[str]]:
    """Serialize Trace.equivalent_ops as a deterministic partition.

    Parameters
    ----------
    trace:
        TorchLens trace.

    Returns
    -------
    list[list[str]]
        Sorted list of sorted equivalence-class member labels.
    """

    partition = [
        sorted(str(label) for label in members) for members in trace.equivalent_ops.values()
    ]
    return sorted(partition)


def build_snapshot(trace: Any, fixture_name: str) -> dict[str, Any]:
    """Build a stable JSON-serializable module-containment snapshot.

    Parameters
    ----------
    trace:
        TorchLens trace to serialize.
    fixture_name:
        Fixture name for the snapshot.

    Returns
    -------
    dict[str, Any]
        Stable snapshot dictionary.
    """

    return {
        "equivalence_partition": _equivalence_partition(trace),
        "fixture_name": fixture_name,
        "layers": [_layer_dict(layer) for layer in trace.layers],
        "module_calls": [
            _call_dict(call) for module in trace.modules for call in module.ops.values()
        ],
        "modules": [_module_dict(module) for module in trace.modules],
        "ops": [_op_dict(op) for op in trace.layer_list],
        "trace_summary": {
            "total_flops_forward": trace.total_flops_forward,
            "uncalled_modules": sorted(str(value) for value in trace.uncalled_modules),
        },
    }
