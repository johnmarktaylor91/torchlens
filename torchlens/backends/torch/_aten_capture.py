"""Private wave-0 ATen-profile recorder and observer-gap bridge."""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch

from ... import _state
from ..._ir.events import (
    _AtenCallEvent,
    _AtenExecutionContext,
    _AtenTensorFact,
    _ModePausedInteriorEvent,
)
from ..._utils._torch_compat import get_current_autograd_node_fn
from ._backward_markers import _ACTIVE_BACKWARD_GRAD_FN_REFS, executing_grad_fn_ref

_RECORD_ATEN_FOR_TESTS: ContextVar[bool] = ContextVar(
    "torchlens_record_aten_for_tests", default=False
)


@dataclass(frozen=True, slots=True)
class _PendingAtenCall:
    """Pre-dispatch value-free facts awaiting an outcome and output facts."""

    capture_phase: str
    forward_pass_index: int | None
    backward_epoch_index: int | None
    owner_func_call_id: int | None
    parent_grad_fn_call_ref: tuple[int, int, int] | None
    namespace: str
    operator: str
    overload: str
    schema: str | None
    schema_fingerprint: str | None
    module_call_stack: tuple[tuple[str, int], ...]
    input_tensor_facts: tuple[_AtenTensorFact, ...]
    mutation_kind: str
    autocast_context: tuple[tuple[str, bool, str], ...]
    dispatch_key_context: str | None
    execution_context: _AtenExecutionContext


@contextmanager
def _activate_aten_recording_for_tests() -> Iterator[None]:
    """Arm the private wave-0 ATen recorder for one pytest scope.

    Yields
    ------
    None
        The recorder is enabled only for captures entered in this scope.

    Raises
    ------
    RuntimeError
        If called outside pytest.
    """

    if "PYTEST_CURRENT_TEST" not in os.environ:
        raise RuntimeError("the wave-0 ATen recorder activation is test-only")
    token = _RECORD_ATEN_FOR_TESTS.set(True)
    try:
        yield
    finally:
        _RECORD_ATEN_FOR_TESTS.reset(token)


def _aten_recording_requested() -> bool:
    """Return whether the private wave-0 recorder is armed in this context."""

    return _RECORD_ATEN_FOR_TESTS.get()


def _module_call_stack(trace: Any) -> tuple[tuple[str, int], ...]:
    """Snapshot the active exhaustive module stack as immutable value facts.

    Parameters
    ----------
    trace
        Active torch Trace.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Module address and one-based pass-index pairs.
    """

    workspace = getattr(trace, "_module_capture_ws", None)
    frames = getattr(workspace, "exhaustive_module_stack", ()) or ()
    rows: list[tuple[str, int]] = []
    for frame in frames:
        address = getattr(frame, "address", None)
        pass_index = getattr(frame, "pass_index", None)
        if isinstance(address, str) and isinstance(pass_index, int):
            rows.append((address, pass_index))
    return tuple(rows)


def _walk_tensor_facts(
    value: Any,
    state: Any,
    *,
    path: tuple[object, ...] = (),
    depth: int = 0,
) -> tuple[_AtenTensorFact, ...]:
    """Return value-free tensor facts from a bounded Python container walk.

    Parameters
    ----------
    value
        Dispatcher argument or result tree.
    state
        Active witness state carrying the capture-local alias registry.
    path
        Current structural container path.
    depth
        Current recursion depth.

    Returns
    -------
    tuple[_AtenTensorFact, ...]
        Tensor facts in deterministic container order.
    """

    if depth > 64:
        return ()
    if isinstance(value, torch.Tensor):
        alias_group = _storage_alias_group(value, state)
        try:
            logical_version = int(value._version)
        except (RuntimeError, TypeError):
            logical_version = None
        try:
            stride = tuple(int(item) for item in value.stride())
        except (RuntimeError, TypeError):
            stride = ()
        return (
            _AtenTensorFact(
                container_path=path,
                tensor_impl_capability=f"{type(value).__module__}.{type(value).__qualname__}",
                logical_version=logical_version,
                storage_alias_group=alias_group,
                shape=tuple(int(item) for item in value.shape),
                stride=stride,
                dtype=str(value.dtype),
                device=str(value.device),
                layout=str(value.layout),
                requires_grad=bool(value.requires_grad),
            ),
        )
    if isinstance(value, tuple | list):
        return tuple(
            fact
            for index, item in enumerate(value)
            for fact in _walk_tensor_facts(
                item,
                state,
                path=(*path, index),
                depth=depth + 1,
            )
        )
    if isinstance(value, Mapping):
        return tuple(
            fact
            for key in sorted(value, key=lambda item: repr(item))
            for fact in _walk_tensor_facts(
                value[key],
                state,
                path=(*path, str(key)),
                depth=depth + 1,
            )
        )
    return ()


def _storage_alias_group(tensor: torch.Tensor, state: Any) -> int | None:
    """Return a capture-local storage alias-group number for ``tensor``.

    Parameters
    ----------
    tensor
        Tensor whose storage identity should be classified.
    state
        Active witness state carrying the alias registry.

    Returns
    -------
    int | None
        Positive capture-local group number, or ``None`` when storage is unreadable.
    """

    try:
        storage_key = (str(tensor.device), int(tensor.untyped_storage()._cdata))
    except (AttributeError, RuntimeError, TypeError):
        return None
    registry = state.aten_storage_alias_groups
    group = registry.get(storage_key)
    if group is None:
        group = len(registry) + 1
        registry[storage_key] = group
    return group


def _operator_parts(func: Any) -> tuple[str, str, str, str | None, str | None]:
    """Return normalized operator identity and schema facts.

    Parameters
    ----------
    func
        Dispatcher operator overload.

    Returns
    -------
    tuple[str, str, str, str | None, str | None]
        Namespace, operator, overload, schema, and SHA-256 schema fingerprint.
    """

    qualified = str(func)
    parts = qualified.split(".")
    namespace = parts[0] if parts else "unknown"
    operator = parts[1] if len(parts) >= 2 else qualified
    overload = ".".join(parts[2:]) if len(parts) >= 3 else "default"
    try:
        schema = str(func._schema)
    except (AttributeError, RuntimeError, TypeError):
        schema = None
    fingerprint = None if schema is None else hashlib.sha256(schema.encode("utf-8")).hexdigest()
    return namespace, operator, overload, schema, fingerprint


def _mutation_kind(operator: str, overload: str, *, mutates: bool) -> str:
    """Classify a schema-mutable operator without overstating its write kind.

    Parameters
    ----------
    operator
        Unqualified dispatcher operator name.
    overload
        Dispatcher overload name.
    mutates
        Whether the dispatcher schema declares a mutable argument.

    Returns
    -------
    str
        One closed mutation-kind token.
    """

    if not mutates:
        return "none"
    if overload == "out" or overload.startswith("out_"):
        return "out_variant"
    metadata_writes = {
        "as_strided_",
        "resize_",
        "resize_as_",
        "set_",
        "sparse_resize_",
        "squeeze_",
        "t_",
        "transpose_",
        "unsqueeze_",
    }
    if operator in metadata_writes:
        return "metadata_only"
    if operator.endswith("_"):
        return "in_place"
    return "unknown"


def _autocast_context() -> tuple[tuple[str, bool, str], ...]:
    """Return an immutable snapshot of public autocast state."""

    rows: list[tuple[str, bool, str]] = []
    for device_type in ("cpu", "cuda"):
        try:
            enabled = bool(torch.is_autocast_enabled(device_type))
            dtype = str(torch.get_autocast_dtype(device_type))
        except (RuntimeError, TypeError):
            enabled = False
            dtype = "unknown"
        rows.append((device_type, enabled, dtype))
    return tuple(rows)


def _execution_context(
    trace: Any, autocast: tuple[tuple[str, bool, str], ...]
) -> _AtenExecutionContext:
    """Build the immutable execution stamp shared by one observed call.

    Parameters
    ----------
    trace
        Active torch Trace.
    autocast
        Already captured autocast facts.

    Returns
    -------
    _AtenExecutionContext
        Value-free environment stamp.
    """

    workspace = getattr(trace, "_module_capture_ws", None)
    module_build_data = getattr(workspace, "module_build_data", {}) or {}
    training_modes = module_build_data.get("module_training_modes", {}) or {}
    training_summary = tuple(
        sorted((str(address), bool(training)) for address, training in training_modes.items())
    )
    try:
        tf32_policy = bool(torch.backends.cuda.matmul.allow_tf32)
    except (AttributeError, RuntimeError):
        tf32_policy = None
    sdpa_policy: list[tuple[str, bool]] = []
    for accessor_name in (
        "flash_sdp_enabled",
        "math_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "cudnn_sdp_enabled",
    ):
        accessor = getattr(torch.backends.cuda, accessor_name, None)
        if callable(accessor):
            try:
                sdpa_policy.append((accessor_name, bool(accessor())))
            except RuntimeError:
                continue
    return _AtenExecutionContext(
        pytorch_version=str(torch.__version__),
        backend="torch",
        device_model=None,
        device_capability=None,
        grad_mode=bool(torch.is_grad_enabled()),
        inference_mode=bool(torch.is_inference_mode_enabled()),
        module_training_summary=training_summary,
        autocast=autocast,
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        tf32_matmul_policy=tf32_policy,
        sdpa_policy=tuple(sdpa_policy),
        compile_stance="forced_eager",
        owner_thread_coverage=(threading.get_ident(),),
        completeness_witness_mode=str(getattr(trace, "_health_completeness_mode", "off")),
    )


def _prepare_aten_call(
    state: Any,
    func: Any,
    input_tree: tuple[tuple[Any, ...], dict[str, Any]],
    owner_func_call_id: int | None,
    *,
    mutates: bool,
) -> _PendingAtenCall:
    """Capture pre-dispatch primitive-call facts without retaining runtime objects.

    Parameters
    ----------
    state
        Active witness/ATen recorder state.
    func
        Dispatcher operator overload.
    input_tree
        The ``(args, kwargs)`` dispatcher input pair.
    owner_func_call_id
        Exact active wrapper call id when present.
    mutates
        Whether the operator schema is mutation-capable.

    Returns
    -------
    _PendingAtenCall
        Immutable pre-dispatch facts.
    """

    namespace, operator, overload, schema, fingerprint = _operator_parts(func)
    autocast = _autocast_context()
    grad_ref = executing_grad_fn_ref(id(state.trace), resolver=get_current_autograd_node_fn())
    return _PendingAtenCall(
        capture_phase=state.capture_phase,
        forward_pass_index=1 if state.capture_phase == "forward" else None,
        backward_epoch_index=state.backward_epoch_index,
        owner_func_call_id=owner_func_call_id,
        parent_grad_fn_call_ref=grad_ref,
        namespace=namespace,
        operator=operator,
        overload=overload,
        schema=schema,
        schema_fingerprint=fingerprint,
        module_call_stack=_module_call_stack(state.trace),
        input_tensor_facts=_walk_tensor_facts(input_tree, state),
        mutation_kind=_mutation_kind(operator, overload, mutates=mutates),
        autocast_context=autocast,
        dispatch_key_context=None,
        execution_context=_execution_context(state.trace, autocast),
    )


def _view_copy_kind(
    pending: _PendingAtenCall,
    output_facts: tuple[_AtenTensorFact, ...],
) -> str:
    """Classify output storage relation using only captured alias-group facts.

    Parameters
    ----------
    pending
        Pre-dispatch input facts.
    output_facts
        Post-dispatch output facts.

    Returns
    -------
    str
        ``view``, ``copy``, ``alias``, or ``unknown``.
    """

    input_groups = {
        fact.storage_alias_group
        for fact in pending.input_tensor_facts
        if fact.storage_alias_group is not None
    }
    output_groups = {
        fact.storage_alias_group for fact in output_facts if fact.storage_alias_group is not None
    }
    if not output_facts:
        return "unknown"
    if pending.mutation_kind != "none" and input_groups & output_groups:
        return "alias"
    if input_groups & output_groups:
        return "view"
    if input_groups and output_groups:
        return "copy"
    return "unknown"


def _finish_aten_call(
    state: Any,
    pending: _PendingAtenCall,
    *,
    result: Any = None,
    exception: BaseException | None = None,
) -> None:
    """Append a completed value-free primitive event to the active journal.

    Parameters
    ----------
    state
        Active witness/ATen recorder state.
    pending
        Pre-dispatch immutable facts.
    result
        Dispatcher result for a successful call.
    exception
        Raised exception for a failed call.
    """

    output_facts = () if exception is not None else _walk_tensor_facts(result, state)
    # Autograd wraps the dispatcher above TorchDispatchMode, so the tensors visible
    # at this redispatch return seam do not yet carry their final GradFn. The
    # post-backward projector links the row through its parent Op's captured
    # ``grad_fn_object_id`` and the materialized GradFn table instead. Keeping this
    # event value-free avoids retaining a tensor or live autograd node meanwhile.
    grad_fn_ref = None
    link_status = "unlinked"
    provenance = None
    state.aten_events.append_aten(
        _AtenCallEvent(
            capture_phase=pending.capture_phase,
            forward_pass_index=pending.forward_pass_index,
            backward_epoch_index=pending.backward_epoch_index,
            owner_func_call_id=pending.owner_func_call_id,
            parent_grad_fn_call_ref=pending.parent_grad_fn_call_ref,
            namespace=pending.namespace,
            operator=pending.operator,
            overload=pending.overload,
            schema=pending.schema,
            schema_fingerprint=pending.schema_fingerprint,
            module_call_stack=pending.module_call_stack,
            input_tensor_facts=pending.input_tensor_facts,
            output_tensor_facts=output_facts,
            mutation_kind=pending.mutation_kind,
            view_copy_kind=_view_copy_kind(pending, output_facts),
            autocast_context=pending.autocast_context,
            dispatch_key_context=pending.dispatch_key_context,
            grad_fn_ref=grad_fn_ref,
            grad_fn_link_status=link_status,
            grad_fn_link_provenance=provenance,
            algorithmic_flops=None,
            flop_status="unsupported",
            flop_formula_source=None,
            flop_formula_version=None,
            outcome="raised" if exception is not None else "returned",
            exception_type=(
                None
                if exception is None
                else f"{type(exception).__module__}.{type(exception).__qualname__}"
            ),
            execution_context=pending.execution_context,
        )
    )


def _iter_tensors(value: Any, *, depth: int = 0) -> Iterator[torch.Tensor]:
    """Yield tensors from a bounded result container.

    Parameters
    ----------
    value
        Result tree to traverse.
    depth
        Current recursion depth.

    Yields
    ------
    torch.Tensor
        Tensor leaves in deterministic order.
    """

    if depth > 64:
        return
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, tuple | list):
        for item in value:
            yield from _iter_tensors(item, depth=depth + 1)
    elif isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: repr(item)):
            yield from _iter_tensors(value[key], depth=depth + 1)


def _record_mode_paused_interior(trace: Any, *, owner_func_call_id: int | None) -> None:
    """Append one lower-bound disclosure for an owned-mode pause.

    Parameters
    ----------
    trace
        Active torch Trace.
    owner_func_call_id
        Strict-constructor wrapper call id, when available.
    """

    stream = getattr(trace, "capture_events", None)
    if stream is None:
        stream = getattr(trace, "_capture_events", None)
    if stream is None or not getattr(stream, "aten_recording_enabled", False):
        return
    before = int(stream.event_seq)
    stream.append_aten(
        _ModePausedInteriorEvent(
            capture_phase="backward" if _ACTIVE_BACKWARD_GRAD_FN_REFS.get() else "forward",
            sequence_before=before,
            sequence_after=before + 1,
            owner_func_call_id=owner_func_call_id,
        )
    )


@contextmanager
def _capture_backward_aten(trace: Any, pass_index: int) -> Iterator[None]:
    """Install the existing dispatch observer around an armed backward pass.

    Parameters
    ----------
    trace
        Trace receiving backward primitive events.
    pass_index
        Active one-based backward epoch index.

    Yields
    ------
    None
        Backward executes with the shared TorchLens dispatch observer installed.
    """

    stream = getattr(trace, "_capture_events", None)
    if stream is None or not getattr(stream, "aten_recording_enabled", False):
        with nullcontext():
            yield
        return
    from ._completeness_types import _WitnessState
    from .completeness_witness import _CompletenessDispatchMode

    state = _WitnessState(
        trace=trace,
        owner_thread_id=threading.get_ident(),
        guard_pass_index=1,
        census=False,
        record_escapes=False,
        ledger=False,
        record_aten=True,
        aten_events=stream,
        capture_phase="backward",
        backward_epoch_index=pass_index,
    )
    marker_token = _ACTIVE_BACKWARD_GRAD_FN_REFS.set(())
    try:
        with _state.aten_recording(), _CompletenessDispatchMode(state):
            yield
    finally:
        _ACTIVE_BACKWARD_GRAD_FN_REFS.reset(marker_token)
