"""Predicate output replacement and backend semantics."""

import dataclasses
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch

from ...capture.projections import (
    get_active_recording_state,
)
from ...fastlog.types import (
    ActivationRecord,
    CaptureSpec,
)
from ...ir.container import (
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
)
from ...ir.events import (
    FunctionCallRef,
)
from ...ir.intervention import FunctionEventInput
from ...ir.semantics import BackendSemantics
from ...utils.tensor_utils import (
    safe_copy,
)
from .aliasing import (
    detect_torch_alias_contract,
    detect_torch_output_alias_contract,
)
from .completeness_witness import internal_scalar_read

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

if TYPE_CHECKING:
    from .ops import (
        _BULK_ALIAS_FREE_TORCH_FUNCTIONS,
        _is_hf_model_output,
        _should_keep_alias_mutation_contract,
    )

__all__ = (
    "_replace_output_value",
    "_record_predicate_output",
    "_is_default_ram_payload",
    "_predicate_function_ref",
    "_predicate_backend_semantics",
    "_has_proven_alias_free_output",
    "_has_only_builtin_tensor_leaves",
    "_alias_free_backend_semantics",
)


def _replace_output_value(
    value: Any,
    path: tuple[OutputPathComponent, ...],
    replacements: dict[tuple[OutputPathComponent, ...], torch.Tensor],
) -> Any:
    """Recursively replace tensors in supported output containers.

    Parameters
    ----------
    value
        Current container value.
    path
        Path to the current value.
    replacements
        Replacement tensors keyed by full output path.

    Returns
    -------
    Any
        Rebuilt value.
    """

    if path in replacements:
        return replacements[path]
    if _is_hf_model_output(value):
        key_values = {
            key: _replace_output_value(item, (*path, HFKey(key)), replacements)
            for key, item in value.items()
        }
        return type(value)(**key_values)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        replaced_items = [
            _replace_output_value(item, (*path, NamedField(field)), replacements)
            for field, item in zip(value._fields, value)
        ]
        return type(value)(*replaced_items)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {
            field.name: _replace_output_value(
                getattr(value, field.name), (*path, DataclassField(field.name)), replacements
            )
            for field in dataclasses.fields(value)
        }
        return dataclasses.replace(value, **changes)
    if isinstance(value, tuple):
        return type(value)(
            _replace_output_value(item, (*path, TupleIndex(index)), replacements)
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _replace_output_value(item, (*path, TupleIndex(index)), replacements)
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _replace_output_value(item, (*path, DictKey(key)), replacements)
            for key, item in value.items()
        }
    return value


def _record_predicate_output(
    ctx: Any,
    out: torch.Tensor,
    spec: CaptureSpec,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Store a predicate-selected operation output."""

    if not spec.save_out and not spec.save_metadata:
        return None, None
    state = get_active_recording_state()
    ram_payload = None
    disk_payload = None
    transformed_ram_payload = None
    transformed_disk_payload = None
    if spec.save_out:
        if _is_default_ram_payload(state, spec):
            ram_payload = safe_copy(out, detach_tensor=True)
        else:
            (
                ram_payload,
                disk_payload,
                transformed_ram_payload,
                transformed_disk_payload,
            ) = state.resolve_storage(out, spec, ctx=ctx)
    if state.storage_intent.on_disk:
        state.add_record(
            ActivationRecord(
                ctx=ctx,
                spec=spec,
                ram_payload=ram_payload,
                disk_payload=disk_payload,
                transformed_ram_payload=transformed_ram_payload,
                transformed_disk_payload=transformed_disk_payload,
            )
        )
    return ram_payload, transformed_ram_payload


def _is_default_ram_payload(state: Any, spec: CaptureSpec) -> bool:
    """Return whether a payload can use the fused default RAM projection.

    Parameters
    ----------
    state
        Active recording state and its resolved storage options.
    spec
        Capture decision for the current tensor.

    Returns
    -------
    bool
        ``True`` when generic storage resolution would only perform one detached
        in-memory copy with no transform, dtype, device, or save-mode work.
    """

    return bool(
        not state.no_tensor_capture
        and state.storage_intent.in_ram
        and not state.storage_intent.on_disk
        and state.options.activation_transform is None
        and not spec.keep_grad
        and spec.device is None
        and spec.dtype is None
        and spec.save_mode == "copy"
    )


def _predicate_function_ref(
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    func_call_id: int,
) -> FunctionCallRef:
    """Build the immutable function-call summary for predicate capture.

    Parameters
    ----------
    func:
        Decorated torch callable being recorded.
    func_name:
        Normalized function name recorded for the operation.
    args:
        Positional call arguments.
    kwargs:
        Keyword call arguments.
    func_call_id:
        Stable per-capture function-call identifier.

    Returns
    -------
    FunctionCallRef
        Immutable function summary stored on the projected event.
    """

    return FunctionCallRef(
        func=func,
        func_name=func_name,
        func_qualname=getattr(func, "__qualname__", None),
        func_call_id=func_call_id,
        code_context=(),
        func_duration=None,
        flops_forward=None,
        flops_backward=None,
        func_rng_states=None,
        func_autocast_state=None,
        arg_names=(),
        num_args_total=len(args) + len(kwargs),
        num_pos_args=len(args),
        num_kwargs=len(kwargs),
        non_tensor_pos_args=(),
        non_tensor_kwargs=tuple(
            (key, value) for key, value in kwargs.items() if not isinstance(value, torch.Tensor)
        ),
        func_non_tensor_args=(),
        is_inplace=False,
        func_config=(),
    )


def _predicate_backend_semantics(
    trace: "Trace",
    out: torch.Tensor,
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out_orig: Any,
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
    is_bottom_level_func: bool,
    func_call_id: int,
    expected_output_count: int,
    *,
    bulk_default_ram: bool = False,
) -> BackendSemantics:
    """Compute demanded backend semantics for one predicate output.

    Parameters
    ----------
    trace
        Active predicate capture trace.
    out
        Live tensor output being recorded.
    func
        Decorated torch callable.
    func_name
        Recorded callable name.
    args
        Live positional inputs.
    kwargs
        Live keyword inputs.
    out_orig
        Complete original call output.
    arg_copies
        Pre-call positional snapshots.
    kwarg_copies
        Pre-call keyword snapshots.
    is_bottom_level_func
        Whether this is a bottom-level decorated call.
    func_call_id
        Stable capture-time function-call identifier.
    expected_output_count
        Number of loggable outputs from the call.
    bulk_default_ram
        Whether the save-all default-RAM projection may use proven eager
        semantics shortcuts.

    Returns
    -------
    BackendSemantics
        Alias and mutation semantics for the projected event.
    """

    grad_fn_handle = out.grad_fn
    keep_alias_mutation_contract = _should_keep_alias_mutation_contract(trace)
    if bulk_default_ram and _has_proven_alias_free_output(
        func_name,
        args,
        kwargs,
        out_orig,
        arg_copies=arg_copies,
        kwarg_copies=kwarg_copies,
    ):
        return _alias_free_backend_semantics(grad_fn_handle)
    func_event_input = FunctionEventInput(
        func=func,
        func_name=func_name,
        func_qualname=getattr(func, "__qualname__", None),
        args=args,
        kwargs=kwargs,
        raw_output=out_orig,
        arg_copies=arg_copies,
        kwarg_copies=kwarg_copies,
        module_stack=(),
        is_bottom_level_func=is_bottom_level_func,
        func_call_id=func_call_id,
        expected_output_count=expected_output_count,
    )
    detect_backend_semantics = (
        detect_torch_alias_contract
        if keep_alias_mutation_contract
        else detect_torch_output_alias_contract
    )
    # Mutation/alias detection compares input copies via ``tensor_nanequal``
    # (``torch.equal`` / ``torch.allclose`` -> Python ``bool``); mark it as a
    # capture-internal read so the completeness witness does not record it as
    # a user host escape.
    with internal_scalar_read():
        return detect_backend_semantics(
            func_event_input,
            backend_grad_handle=grad_fn_handle,
            grad_fn_class_name=(
                type(grad_fn_handle).__name__ if grad_fn_handle is not None else None
            ),
            autograd_memory=None,
            num_autograd_tensors=None,
        )


def _has_proven_alias_free_output(
    func_name: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    raw_output: Any,
    *,
    arg_copies: tuple[Any, ...],
    kwarg_copies: Mapping[str, Any],
) -> bool:
    """Return whether a built-in call is guaranteed to allocate its output.

    Parameters
    ----------
    func_name
        Recorded callable name.
    args
        Live positional inputs.
    kwargs
        Live keyword inputs.
    raw_output
        Complete original call output.
    arg_copies
        Pre-call positional snapshots or the live argument tuple.
    kwarg_copies
        Pre-call keyword snapshots or the live keyword mapping.

    Returns
    -------
    bool
        ``True`` only for an allowlisted allocating operator on ordinary torch
        tensors, with no subclass dispatch or ``out=`` destination.
    """

    if (
        func_name not in _BULK_ALIAS_FREE_TORCH_FUNCTIONS
        or "out" in kwargs
        or type(raw_output) is not torch.Tensor
    ):
        return False
    if func_name == "batch_norm":
        training = kwargs.get("training", args[5] if len(args) > 5 else False)
        if training:
            return False
    if func_name == "embedding":
        max_norm = kwargs.get("max_norm", args[3] if len(args) > 3 else None)
        if max_norm is not None:
            return False
    if func_name in {"relu_", "__iadd__"} and (
        arg_copies is not args or kwarg_copies is not kwargs
    ):
        return False
    return all(_has_only_builtin_tensor_leaves(value) for value in (*args, *kwargs.values()))


def _has_only_builtin_tensor_leaves(value: Any) -> bool:
    """Return whether tensor leaves cannot override built-in alias semantics.

    Parameters
    ----------
    value
        One positional or keyword argument.

    Returns
    -------
    bool
        ``False`` when any tensor leaf is a user-defined subclass.
    """

    if isinstance(value, torch.Tensor):
        return type(value) in {torch.Tensor, torch.nn.Parameter}
    if isinstance(value, Mapping):
        return all(_has_only_builtin_tensor_leaves(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_has_only_builtin_tensor_leaves(item) for item in value)
    return True


def _alias_free_backend_semantics(grad_fn_handle: Any) -> BackendSemantics:
    """Build exact semantics for a proven allocating torch operation.

    Parameters
    ----------
    grad_fn_handle
        Backend autograd handle for the output, when applicable.

    Returns
    -------
    BackendSemantics
        Empty mutation and alias contracts with live autograd metadata.
    """

    return BackendSemantics(
        backend_grad_handle=grad_fn_handle,
        grad_fn_class_name=(type(grad_fn_handle).__name__ if grad_fn_handle is not None else None),
        autograd_memory=None,
        num_autograd_tensors=None,
        mutated_input_positions=(),
        aliased_output_inputs=(),
        unknown_aliasing=False,
        bytes_delta_at_call=None,
        bytes_peak_at_call=None,
    )
