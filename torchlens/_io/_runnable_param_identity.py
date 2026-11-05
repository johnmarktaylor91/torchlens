"""Live-reference identity checks between template tensors and cooked parameters."""

from __future__ import annotations

from typing import Any


def _same_tensor_identity(tensor: Any, param: Any) -> bool:
    """Return whether ``tensor`` is the exact captured parameter object/storage."""

    return any(_same_tensor_ref(tensor, param_ref) for param_ref in _parameter_refs(param))


def _parameter_refs(param: Any) -> tuple[Any, ...]:
    """Return live parameter objects associated with one cooked ``Param`` record."""

    refs: list[Any] = []
    param_ref = getattr(param, "_param_ref", None)
    if param_ref is not None:
        refs.append(param_ref)
    source_trace_ref = getattr(param, "_source_trace_ref", None)
    trace = source_trace_ref() if callable(source_trace_ref) else None
    source_model_ref = getattr(trace, "_source_model_ref", None)
    model = source_model_ref() if callable(source_model_ref) else None
    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        addresses = {str(getattr(param, "address", ""))}
        addresses.update(str(address) for address in getattr(param, "all_addresses", ()) or ())
        try:
            for name, value in named_parameters(remove_duplicate=False):
                if str(name) in addresses:
                    refs.append(value)
        except TypeError:
            for name, value in named_parameters():
                if str(name) in addresses:
                    refs.append(value)
    return tuple(dict.fromkeys(refs))


def _same_tensor_ref(tensor: Any, reference: Any) -> bool:
    """Return whether two tensors are the same object or share the same data pointer."""

    if tensor is reference or id(tensor) == id(reference):
        return True
    tensor_ptr = _tensor_data_ptr(tensor)
    reference_ptr = _tensor_data_ptr(reference)
    return tensor_ptr is not None and tensor_ptr == reference_ptr


def _tensor_data_ptr(tensor: Any) -> int | None:
    """Return a tensor data pointer without raising for non-tensors."""

    data_ptr = getattr(tensor, "data_ptr", None)
    if not callable(data_ptr):
        return None
    try:
        return int(data_ptr())
    except RuntimeError:
        return None
