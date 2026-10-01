"""Torch alias and mutation contract detection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from ... import _state
from ...ir.intervention import FunctionEventInput
from ...ir.semantics import BackendSemantics
from ...utils.collections import index_nested
from ...utils.tensor_utils import tensor_nanequal


def detect_torch_alias_contract(
    func_event_input: FunctionEventInput,
    *,
    backend_grad_handle: object | None = None,
    grad_fn_class_name: str | None = None,
    autograd_memory: int | None = None,
    num_autograd_tensors: int | None = None,
    bytes_delta_at_call: int | None = None,
    bytes_peak_at_call: int | None = None,
) -> BackendSemantics:
    """Detect torch input mutation and output aliasing semantics.

    Parameters
    ----------
    func_event_input
        Function call bundle containing live inputs, pre-call copies, and raw output.
    backend_grad_handle
        Backend autograd handle for the output, when applicable.
    grad_fn_class_name
        Backend autograd class name for the output, when applicable.
    autograd_memory
        Bytes retained by autograd saved tensors, when known.
    num_autograd_tensors
        Number of autograd saved tensors, when known.
    bytes_delta_at_call
        Backend memory delta at call time, when known.
    bytes_peak_at_call
        Backend peak memory at call time, when known.

    Returns
    -------
    BackendSemantics
        Normalized semantics with explicit alias contract fields.
    """

    input_tensors = _flatten_input_tensors(func_event_input.args, func_event_input.kwargs)
    copied_inputs = _flatten_copied_input_tensors(
        func_event_input.arg_copies,
        func_event_input.kwarg_copies,
    )
    mutated_positions = tuple(
        position
        for position, tensor in input_tensors
        if _position_was_mutated(position, tensor, copied_inputs)
    )
    aliased_positions = tuple(
        position
        for position, tensor in input_tensors
        if _output_aliases_input(func_event_input.raw_output, tensor)
    )
    return BackendSemantics(
        backend_grad_handle=backend_grad_handle,
        grad_fn_class_name=grad_fn_class_name,
        autograd_memory=autograd_memory,
        num_autograd_tensors=num_autograd_tensors,
        mutated_input_positions=mutated_positions,
        aliased_output_inputs=aliased_positions,
        unknown_aliasing=False,
        bytes_delta_at_call=bytes_delta_at_call,
        bytes_peak_at_call=bytes_peak_at_call,
    )


def detect_torch_output_alias_contract(
    func_event_input: FunctionEventInput,
    *,
    backend_grad_handle: object | None = None,
    grad_fn_class_name: str | None = None,
    autograd_memory: int | None = None,
    num_autograd_tensors: int | None = None,
    bytes_delta_at_call: int | None = None,
    bytes_peak_at_call: int | None = None,
) -> BackendSemantics:
    """Detect cheap output-to-input aliasing without mutation comparisons.

    Parameters
    ----------
    func_event_input
        Function call bundle containing live inputs and raw output.
    backend_grad_handle
        Backend autograd handle for the output, when applicable.
    grad_fn_class_name
        Backend autograd class name for the output, when applicable.
    autograd_memory
        Bytes retained by autograd saved tensors, when known.
    num_autograd_tensors
        Number of autograd saved tensors, when known.
    bytes_delta_at_call
        Backend memory delta at call time, when known.
    bytes_peak_at_call
        Backend peak memory at call time, when known.

    Returns
    -------
    BackendSemantics
        Normalized semantics with mutation positions intentionally empty.
    """

    input_tensors = _flatten_input_tensors(func_event_input.args, func_event_input.kwargs)
    aliased_positions = tuple(
        position
        for position, tensor in input_tensors
        if _output_aliases_input(func_event_input.raw_output, tensor)
    )
    return BackendSemantics(
        backend_grad_handle=backend_grad_handle,
        grad_fn_class_name=grad_fn_class_name,
        autograd_memory=autograd_memory,
        num_autograd_tensors=num_autograd_tensors,
        mutated_input_positions=(),
        aliased_output_inputs=aliased_positions,
        unknown_aliasing=False,
        bytes_delta_at_call=bytes_delta_at_call,
        bytes_peak_at_call=bytes_peak_at_call,
    )


def _flatten_input_tensors(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[tuple[object, torch.Tensor], ...]:
    """Return tensor input positions.

    Parameters
    ----------
    args
        Function positional arguments.
    kwargs
        Function keyword arguments.

    Returns
    -------
    tuple[tuple[object, torch.Tensor], ...]
        Integer, keyword, or nested positions paired with tensor values.
    """

    positions: list[tuple[object, torch.Tensor]] = []
    for index, value in enumerate(args):
        positions.extend(_iter_tensor_positions(index, value))
    for key, value in kwargs.items():
        positions.extend(_iter_tensor_positions(key, value))
    return tuple(positions)


def _flatten_copied_input_tensors(
    arg_copies: tuple[Any, ...] | None,
    kwarg_copies: Mapping[str, Any] | None,
) -> dict[object, torch.Tensor]:
    """Return copied tensor inputs keyed by position.

    Parameters
    ----------
    arg_copies
        Pre-call positional argument copies.
    kwarg_copies
        Pre-call keyword argument copies.

    Returns
    -------
    dict[object, torch.Tensor]
        Tensor copies keyed by integer, keyword, or nested position.
    """

    copied: dict[object, torch.Tensor] = {}
    if arg_copies is not None:
        for index, value in enumerate(arg_copies):
            copied.update(_iter_tensor_positions(index, value))
    if kwarg_copies is not None:
        for key, value in kwarg_copies.items():
            copied.update(_iter_tensor_positions(key, value))
    return copied


def _iter_tensor_positions(
    position: object,
    value: object,
) -> tuple[tuple[object, torch.Tensor], ...]:
    """Return tensor leaves under an argument position.

    Parameters
    ----------
    position
        Top-level or nested argument position.
    value
        Argument value at ``position``.

    Returns
    -------
    tuple[tuple[object, torch.Tensor], ...]
        Tensor leaves keyed by scalar top-level positions or tuple nested paths.
    """

    if isinstance(value, torch.Tensor):
        return ((position, value),)
    if isinstance(value, Mapping):
        entries: list[tuple[object, torch.Tensor]] = []
        for key, item in value.items():
            entries.extend(_iter_tensor_positions(_extend_position(position, key), item))
        return tuple(entries)
    if isinstance(value, (list, tuple)):
        entries = []
        for index, item in enumerate(value):
            entries.extend(_iter_tensor_positions(_extend_position(position, index), item))
        return tuple(entries)
    return ()


def _extend_position(position: object, key: object) -> object:
    """Append one nested component to an argument position.

    Parameters
    ----------
    position
        Existing top-level or nested position.
    key
        Nested container key.

    Returns
    -------
    object
        Nested tuple position compatible with ``index_nested``.
    """

    if isinstance(position, tuple):
        return (*position, key)
    return (position, key)


def _position_was_mutated(
    position: object,
    tensor: torch.Tensor,
    copied_inputs: Mapping[object, torch.Tensor],
) -> bool:
    """Return whether an input position changed during the function call.

    Parameters
    ----------
    position
        Top-level argument position.
    tensor
        Post-call live tensor.
    copied_inputs
        Pre-call tensor copies keyed by top-level position.

    Returns
    -------
    bool
        True when the pre-call copy differs from the post-call tensor.
    """

    copied = copied_inputs.get(position)
    if copied is None:
        return False
    if copied is tensor:
        return False
    if copied.is_meta or tensor.is_meta:
        # Meta tensors carry no data (e.g. factory outputs under a
        # ``torch.device("meta")`` context), so mutation cannot be detected
        # by content comparison — report "not mutated".
        return False
    return not tensor_nanequal(copied, tensor)


def _output_aliases_input(raw_output: object, input_tensor: torch.Tensor) -> bool:
    """Return whether any raw output tensor aliases an input tensor.

    Parameters
    ----------
    raw_output
        Function raw output object.
    input_tensor
        Candidate input tensor.

    Returns
    -------
    bool
        True when an output tensor shares storage with ``input_tensor``.
    """

    for output_tensor in _iter_output_tensors(raw_output):
        if _tensors_alias(output_tensor, input_tensor):
            return True
    return False


def _iter_output_tensors(raw_output: object) -> Iterable[torch.Tensor]:
    """Yield tensor leaves from a function output object.

    Parameters
    ----------
    raw_output
        Function raw output.

    Yields
    ------
    torch.Tensor
        Tensor outputs.
    """

    for _, tensor in _iter_tensor_positions((), raw_output):
        yield tensor


def _tensors_alias(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two tensors share the same storage.

    Parameters
    ----------
    left
        First tensor.
    right
        Second tensor.

    Returns
    -------
    bool
        True when both tensors point at the same storage allocation.
    """

    if id(left) == id(right):
        return True
    # W8C internal-caller dispatch bypass: ``untyped_storage`` is a wrapped tensor
    # method, so an unpaused read here pays the full per-op logging dispatch
    # (barcode/autocast/container+intervention snapshots) yet emits no op. Every
    # caller holds ``internal_scalar_read``, so the storage-bridge belt already
    # excludes these reads; pausing only skips the op-capture dispatch. Each
    # bypassed wrapped call still consumes its ``next_func_call_id()`` so the
    # session id sequence stamped on real ops stays identical to the unbypassed
    # path (``op.func_call_id`` persists into runnable descriptors).
    logging_enabled = _state._logging_enabled
    try:
        with _state.pause_logging():
            if logging_enabled:
                _state.next_func_call_id()
            left_ptr = left.untyped_storage().data_ptr()
            if logging_enabled:
                _state.next_func_call_id()
            right_ptr = right.untyped_storage().data_ptr()
        return left_ptr == right_ptr
    except RuntimeError:
        return False


def parent_label_has_alias_contract(
    parent_label: str,
    parent_arg_positions: dict[str, dict[Any, str]],
    contract_positions: tuple[object, ...],
) -> bool:
    """Return whether a parent label is covered by contract positions.

    Parameters
    ----------
    parent_label
        Raw parent label to test.
    parent_arg_positions
        Mapping from argument positions to raw parent labels.
    contract_positions
        Positions reported by the alias contract.

    Returns
    -------
    bool
        True when ``parent_label`` appears at a contract-covered position.
    """

    # Iterate the (few) contract positions, not the (possibly huge) argument
    # map: the old full scan cost O(fan_in) per parent — O(fan_in^2) per op
    # for variadic ops like a 4k-arg ``stack``, with an empty contract.
    if not contract_positions:
        return False
    args_positions = parent_arg_positions["args"]
    kwargs_positions = parent_arg_positions["kwargs"]
    for position in contract_positions:
        if (
            args_positions.get(position) == parent_label
            or kwargs_positions.get(position) == parent_label
        ):
            return True
    return False


def get_parent_contents_for_contract_position(
    parent_label: str,
    arg_copies: tuple[Any, ...],
    kwarg_copies: Mapping[str, Any],
    parent_arg_positions: dict[str, dict[Any, str]],
) -> Any:
    """Return a parent's pre-call value for alias-contract snapshotting.

    Parameters
    ----------
    parent_label
        Raw parent label.
    arg_copies
        Pre-call positional argument copies.
    kwarg_copies
        Pre-call keyword argument copies.
    parent_arg_positions
        Mapping from argument positions to raw parent labels.

    Returns
    -------
    Any
        Pre-call parent value.
    """

    for position, label in parent_arg_positions["args"].items():
        if label == parent_label:
            return index_nested(arg_copies, position)
    for position, label in parent_arg_positions["kwargs"].items():
        if label == parent_label:
            return index_nested(kwarg_copies, position)
    raise ValueError("Parent layer not found in function arguments.")
