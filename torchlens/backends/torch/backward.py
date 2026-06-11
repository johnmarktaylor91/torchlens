"""Backward-pass graph walking, grad_fn_handle hooks, and Trace APIs."""

from __future__ import annotations

import contextlib
import inspect
import re
import time
import weakref
from collections import OrderedDict, deque
from collections.abc import Sequence
from typing import Any, Callable, Iterator, Literal, cast

import torch

from ..._deprecations import MISSING, MissingType
from ...quantities import Bytes, Duration
from ..._state import pause_logging
from ...data_classes.op import _dtype_or_none, _memory_or_none, _shape_or_none
from ...data_classes.grad_fn import GradFn
from ...data_classes.grad_fn_call import GradFnCall
from ...data_classes.backward_pass import BackwardPass
from ...ir.events import (
    BackwardPassEnd,
    BackwardPassStart,
    GradFnDiscovered,
    GradFnFired,
    OpGradObserved,
)
from .tensor_tracking import _ensure_backward_event_stream

_BACKWARD_GRAD_FN_REGISTRY: dict[int, weakref.ReferenceType[Any]] = {}
_ORIGINAL_AUTOGRAD_BACKWARD: Callable[..., Any] | None = None
_ORIGINAL_AUTOGRAD_GRAD: Callable[..., Any] | None = None
_AUTOGRAD_WRAPPERS_INSTALLED = False


def _strong_grad_fn_refs(trace: Any) -> list[Any]:
    """Return the trace-owned grad-fn strong-reference list.

    Parameters
    ----------
    trace:
        Trace that owns runtime autograd node references.

    Returns
    -------
    list[Any]
        Mutable list holding grad-fn wrapper objects for the trace lifetime.
    """

    return trace.__dict__.setdefault("_backward_gradfn_refs", [])


def _register_forward_grad_fn(trace: Any, grad_fn_handle: Any, _raw_label: str | None) -> None:
    """Register a pinned forward grad-fn object for later autograd triggers.

    Parameters
    ----------
    trace:
        Trace that recorded the forward operation.
    grad_fn_handle:
        Live PyTorch autograd node wrapper from the operation output.
    _raw_label:
        Raw TorchLens op label associated with the output tensor, if known.

    Returns
    -------
    None
        The process registry and trace strong-ref set are updated in place.
    """

    if grad_fn_handle is None:
        return
    refs = _strong_grad_fn_refs(trace)
    refs.append(grad_fn_handle)
    grad_fn_object_id = id(grad_fn_handle)
    _BACKWARD_GRAD_FN_REGISTRY[grad_fn_object_id] = weakref.ref(trace)


def _purge_trace_from_backward_registry(trace: Any) -> None:
    """Remove every registry key currently owned by ``trace``.

    Parameters
    ----------
    trace:
        Trace being cleaned up or disarmed.

    Returns
    -------
    None
        Matching process-registry entries are removed.
    """

    stale_ids = [
        grad_fn_object_id
        for grad_fn_object_id, trace_ref in _BACKWARD_GRAD_FN_REGISTRY.items()
        if trace_ref() is trace or trace_ref() is None
    ]
    for grad_fn_object_id in stale_ids:
        _BACKWARD_GRAD_FN_REGISTRY.pop(grad_fn_object_id, None)


def _close_implicit_backward_pass_if_open(trace: Any) -> None:
    """Close an implicit backward bracket at a synchronization point.

    Parameters
    ----------
    trace:
        Trace that may have an orphan tensor-hook bracket open.

    Returns
    -------
    None
        An implicit ``BackwardPassEnd`` is appended when needed.
    """

    if not getattr(trace, "_implicit_backward_pass_open", False):
        return
    pass_index = getattr(trace, "_active_backward_pass_index", None)
    if pass_index is None:
        return
    events = _ensure_backward_event_stream(trace)
    events.append_backward(
        BackwardPassEnd(
            pass_index=int(pass_index),
            duration=None,
            peak_memory=None,
            status="ok",
            order_attribution_coverage=None,
        )
    )
    trace.num_backward_passes = max(int(getattr(trace, "num_backward_passes", 0)), int(pass_index))
    trace.__dict__.pop("_active_backward_pass_index", None)
    trace._implicit_backward_pass_open = False
    _materialize_backward_projections(trace)


def _root_tensors(value: Any) -> tuple[torch.Tensor, ...]:
    """Flatten autograd root arguments into tensors.

    Parameters
    ----------
    value:
        Tensor or nested sequence passed to an autograd engine entry.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Tensor roots in left-to-right order.
    """

    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(_root_tensors(item))
        return tuple(tensors)
    return ()


def _traces_for_roots(roots: Any) -> tuple[Any, ...]:
    """Find live traces whose pinned grad-fn ids appear under ``roots``.

    Parameters
    ----------
    roots:
        Tensor or nested tensor sequence passed to an autograd entry point.

    Returns
    -------
    tuple[Any, ...]
        Matched traces in discovery order, without duplicates.
    """

    if not _BACKWARD_GRAD_FN_REGISTRY:
        return ()
    matched: list[Any] = []
    matched_ids: set[int] = set()
    stale_ids: list[int] = []
    queue: deque[Any] = deque(
        root.grad_fn for root in _root_tensors(roots) if root.grad_fn is not None
    )
    seen: set[int] = set()
    while queue:
        grad_fn_handle = queue.popleft()
        grad_fn_object_id = id(grad_fn_handle)
        if grad_fn_object_id in seen:
            continue
        seen.add(grad_fn_object_id)
        trace_ref = _BACKWARD_GRAD_FN_REGISTRY.get(grad_fn_object_id)
        if trace_ref is not None:
            trace = trace_ref()
            if trace is None or not hasattr(trace, "layer_list"):
                stale_ids.append(grad_fn_object_id)
            elif id(trace) not in matched_ids and not getattr(
                trace, "_tl_backward_triggers_disarmed", False
            ):
                matched.append(trace)
                matched_ids.add(id(trace))
        queue.extend(_iter_next_grad_fns(grad_fn_handle))
    for stale_id in stale_ids:
        _BACKWARD_GRAD_FN_REGISTRY.pop(stale_id, None)
    return tuple(matched)


def _autograd_roots_from_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    root_kwarg: str,
) -> Any:
    """Return root tensors from an autograd wrapper call.

    Parameters
    ----------
    args:
        Positional arguments passed to the wrapped function.
    kwargs:
        Keyword arguments passed to the wrapped function.
    root_kwarg:
        Keyword name that carries roots for this autograd entry.

    Returns
    -------
    Any
        Original root argument object, or ``None`` when absent.
    """

    if args:
        return args[0]
    return kwargs.get(root_kwarg)


def _first_root_tensor(roots: Any) -> torch.Tensor | None:
    """Return the first tensor in an autograd root object.

    Parameters
    ----------
    roots:
        Tensor or nested sequence of tensors.

    Returns
    -------
    torch.Tensor | None
        First tensor root, if one exists.
    """

    tensors = _root_tensors(roots)
    return tensors[0] if tensors else None


def _grad_fn_is_custom(grad_fn_handle: Any) -> bool:
    """Return whether a grad_fn_handle appears to come from user/custom autograd code.

    Parameters
    ----------
    grad_fn_handle:
        Autograd function object.

    Returns
    -------
    bool
        True when the type's module path is outside PyTorch's built-in autograd
        and nn namespaces.
    """
    class_module = type(grad_fn_handle).__module__
    if class_module == "torch.autograd.function":
        return True
    builtin_prefixes = ("torch.autograd", "torch.nn", "torch", "builtins")
    return not class_module.startswith(builtin_prefixes)


def _safe_source_file_and_line(obj: Any) -> tuple[str | None, int | None]:
    """Return source file and first line for an object when introspection works.

    Parameters
    ----------
    obj:
        Object to inspect.

    Returns
    -------
    tuple[str | None, int | None]
        Source file and line, or ``(None, None)`` when unavailable.
    """

    try:
        source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        source_line = inspect.getsourcelines(obj)[1]
    except (OSError, TypeError):
        return None, None
    return source_file, source_line


def _safe_signature(obj: Any) -> str | None:
    """Return an inspect signature string when available.

    Parameters
    ----------
    obj:
        Callable object to inspect.

    Returns
    -------
    str | None
        Signature string, or ``None`` when unavailable.
    """

    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _grad_fn_source_metadata(grad_fn_cls: type[Any]) -> dict[str, Any]:
    """Return best-effort source metadata for a grad-fn class.

    Parameters
    ----------
    grad_fn_cls:
        Runtime class of the autograd grad-fn object.

    Returns
    -------
    dict[str, Any]
        Keyword arguments accepted by ``GradFn`` for source metadata fields.
    """

    class_source_file, class_source_line = _safe_source_file_and_line(grad_fn_cls)
    init_method = getattr(grad_fn_cls, "__init__", None)
    forward_method = getattr(grad_fn_cls, "forward", None)
    backward_method = getattr(grad_fn_cls, "backward", None)
    init_source_file, init_source_line = _safe_source_file_and_line(init_method)
    forward_source_file, forward_source_line = _safe_source_file_and_line(forward_method)
    backward_source_file, backward_source_line = _safe_source_file_and_line(backward_method)
    return {
        "class_source_file": class_source_file,
        "class_source_line": class_source_line,
        "class_docstring": grad_fn_cls.__doc__,
        "init_source_file": init_source_file,
        "init_source_line": init_source_line,
        "init_signature": _safe_signature(init_method),
        "init_docstring": getattr(init_method, "__doc__", None),
        "forward_source_file": forward_source_file,
        "forward_source_line": forward_source_line,
        "forward_signature": _safe_signature(forward_method),
        "forward_docstring": getattr(forward_method, "__doc__", None),
        "backward_source_file": backward_source_file,
        "backward_source_line": backward_source_line,
        "backward_signature": _safe_signature(backward_method),
        "backward_docstring": getattr(backward_method, "__doc__", None),
    }


def _iter_next_grad_fns(grad_fn_handle: Any) -> Iterator[Any]:
    """Yield non-null child grad_fns from ``grad_fn_handle.next_functions``.

    Parameters
    ----------
    grad_fn_handle:
        Autograd function object.

    Yields
    ------
    Any
        Reachable child grad_fn_handle object.
    """
    for next_fn, _input_num in getattr(grad_fn_handle, "next_functions", ()):
        if next_fn is not None:
            yield next_fn


def _selected_for_grad_save(trace: Any, layer_label: str | None) -> bool:
    """Return whether a forward layer's grad should be saved.

    Parameters
    ----------
    trace:
        Trace being updated.
    layer_label:
        Final layer label, or ``None`` for intervening grad_fns.

    Returns
    -------
    bool
        True if this layer is selected by ``gradients_to_save``.
    """
    if layer_label is None:
        return False
    selection = getattr(trace, "_grad_layer_nums_to_save", "all")
    if selection == "all":
        return True
    if selection in [None, "none", []]:
        return False
    return trace[layer_label].raw_index in selection


def _sync_grad_fn_graph_relations(trace: Any) -> None:
    """Populate backward-oriented GradFn graph relation labels.

    Parameters
    ----------
    trace:
        Trace whose ``grad_fn_logs`` should be synchronized.
    """

    id_to_label = {
        grad_fn_object_id: grad_fn.label
        for grad_fn_object_id, grad_fn in trace.grad_fn_logs.items()
    }
    child_map: dict[str, list[str]] = {}
    parent_map: dict[str, list[str]] = {
        grad_fn.label: [] for grad_fn in trace.grad_fn_logs.values()
    }

    for grad_fn in trace.grad_fn_logs.values():
        children = [
            id_to_label[next_grad_fn_id]
            for next_grad_fn_id in grad_fn.next_grad_fn_ids
            if next_grad_fn_id in id_to_label
        ]
        child_map[grad_fn.label] = children
        for child_label in children:
            if grad_fn.label not in parent_map[child_label]:
                parent_map[child_label].append(grad_fn.label)

    for grad_fn in trace.grad_fn_logs.values():
        grad_fn.children = child_map[grad_fn.label]
        grad_fn.parents = parent_map[grad_fn.label]
        sibling_labels: list[str] = []
        for parent_label in grad_fn.parents:
            for sibling_label in child_map.get(parent_label, []):
                if sibling_label != grad_fn.label and sibling_label not in sibling_labels:
                    sibling_labels.append(sibling_label)
        grad_fn.siblings = sibling_labels

        co_parent_labels: list[str] = []
        for child_label in grad_fn.children:
            for co_parent_label in parent_map.get(child_label, []):
                if co_parent_label != grad_fn.label and co_parent_label not in co_parent_labels:
                    co_parent_labels.append(co_parent_label)
        grad_fn.co_parents = co_parent_labels


def _normalize_grad_fn_type(grad_fn_handle: Any) -> str:
    """Normalize an autograd grad_fn_handle class name for TorchLens labels.

    Parameters
    ----------
    grad_fn_handle:
        Autograd function object.

    Returns
    -------
    str
        Lowercased class name with a trailing ``Backward<digits>`` suffix removed.
    """
    return re.sub(r"Backward\d*$", "", type(grad_fn_handle).__name__).lower()


def _grad_fn_label_parts(
    trace: Any,
    type: str,
    _layer_label: str | None,
    type_counter: dict[str, int],
    total_num: int,
) -> tuple[int, int, str]:
    """Build numeric label fields for one grad_fn_handle.

    Parameters
    ----------
    trace:
        Trace being updated.
    type:
        Normalized grad_fn_handle type.
    _layer_label:
        Matching forward layer label, or ``None`` for intervening grad_fns.
    type_counter:
        Running per-type counter for intervening grad_fns.
    total_num:
        One-based discovery index in the backward graph.

    Returns
    -------
    tuple[int, int, str]
        GradFn type index, total index, and user-facing label.
    """
    del trace, _layer_label
    type_counter[type] = type_counter.get(type, 0) + 1
    type_num = type_counter[type]
    label = f"{type}_back_{type_num}_{total_num}"
    return type_num, total_num, label


def _grad_fn_type_from_class_name(class_name: str) -> str:
    """Normalize an autograd class name for native backward labels."""

    return re.sub(r"Backward\d*$", "", class_name).lower()


def _materialize_backward_projections(trace: Any) -> None:
    """Rebuild backward-facing Trace records from sidecar events.

    Parameters
    ----------
    trace:
        Trace whose runtime backward event sidecar is authoritative.

    Returns
    -------
    None
        ``grad_fn_logs``, ``grad_fn_order``, ``backward_pass_logs``, and
        related counters are replaced in place.
    """

    if getattr(trace, "_tl_active_backward_bracket", False):
        return
    if getattr(trace, "_tl_materializing_backward_projection", False):
        return
    events = list(getattr(_ensure_backward_event_stream(trace), "backward_events", ()))
    if not events:
        return
    if getattr(trace, "_backward_projection_event_count", None) == len(events):
        return
    trace._tl_materializing_backward_projection = True
    try:
        _materialize_backward_projections_impl(trace, events)
    finally:
        trace.__dict__.pop("_tl_materializing_backward_projection", None)


def _materialize_backward_projections_impl(trace: Any, events: list[Any]) -> None:
    """Rebuild backward projections from an already-snapshotted event list."""

    starts: dict[int, BackwardPassStart] = {}
    ends: dict[int, BackwardPassEnd] = {}
    discovered: OrderedDict[int, GradFnDiscovered] = OrderedDict()
    latest_topology: dict[int, tuple[int, ...]] = {}
    fired_events: list[GradFnFired] = []
    op_grad_passes: set[int] = set()
    op_grad_events: list[OpGradObserved] = []
    for event in events:
        if isinstance(event, BackwardPassStart):
            starts[event.pass_index] = event
        elif isinstance(event, BackwardPassEnd):
            ends[event.pass_index] = event
        elif isinstance(event, GradFnDiscovered):
            if event.object_id not in discovered:
                discovered[event.object_id] = event
            latest_topology[event.object_id] = event.topology
        elif isinstance(event, GradFnFired):
            fired_events.append(event)
        elif isinstance(event, OpGradObserved):
            op_grad_passes.add(event.pass_index)
            op_grad_events.append(event)

    grad_fn_logs: OrderedDict[int, GradFn] = OrderedDict()
    type_counter: dict[str, int] = {}
    object_to_label: dict[int, str] = {}
    for ordinal_index, (object_id, event) in enumerate(discovered.items()):
        grad_fn_type = _grad_fn_type_from_class_name(event.class_name)
        step_index = ordinal_index + 1
        type_index, step_index, label = _grad_fn_label_parts(
            trace,
            grad_fn_type,
            event.op_label,
            type_counter,
            step_index,
        )
        source_fields = cast(dict[str, Any], dict(event.source))
        modules: list[str] = []
        module_address = None
        module_membership_source = None
        if event.op_label is not None and event.op_label in getattr(
            trace, "layer_dict_all_keys", {}
        ):
            op = trace[event.op_label]
            module_address = getattr(op, "module_address", None)
            if module_address is not None:
                modules = [module_address]
                module_membership_source = "paired"
        grad_fn_record = GradFn(
            grad_fn_object_id=object_id,
            class_name=event.class_name,
            class_qualname=event.class_qualname,
            label=label,
            type=grad_fn_type,
            type_index=type_index,
            ordinal_index=ordinal_index,
            step_index=step_index,
            is_custom=event.is_custom,
            has_op=event.op_label is not None,
            op_label=event.op_label,
            order=1,
            origin_backward_pass=event.created_in_pass,
            creator_object_id=event.creator_object_id,
            modules=modules,
            module_address=module_address,
            module_membership_source=module_membership_source,
            next_grad_fn_ids=list(latest_topology.get(object_id, event.topology)),
            **source_fields,
        )
        grad_fn_record.source_trace = trace
        grad_fn_logs[object_id] = grad_fn_record
        object_to_label[object_id] = label

    for object_id, grad_fn_record in grad_fn_logs.items():
        if grad_fn_record.creator_object_id is not None:
            grad_fn_record.differentiates = object_to_label.get(grad_fn_record.creator_object_id)

    trace.grad_fn_logs = grad_fn_logs
    trace.grad_fn_order = list(grad_fn_logs)

    pass_to_calls: dict[int, list[GradFnCall]] = {}
    per_object_ordinals: dict[int, int] = {}
    for event in sorted(fired_events, key=lambda item: (item.pass_index, item.timestamp, item.seq)):
        grad_fn_record = trace.grad_fn_logs.get(event.object_id)
        if grad_fn_record is None:
            continue
        ordinal = per_object_ordinals.get(event.object_id, 0) + 1
        per_object_ordinals[event.object_id] = ordinal
        call = GradFnCall(
            call_index=ordinal,
            ordinal=ordinal,
            backward_pass_index=event.pass_index,
            label=grad_fn_record.label,
            grad_inputs=event.grad_input_refs,
            grad_outputs=event.grad_output_refs,
            intervention_fire_ref=event.intervention_fire_ref,
            timestamp=event.timestamp,
            _time_started=event.timestamp,
            _time_finished=event.timestamp,
        )
        call.source_trace = trace
        grad_fn_record.calls[ordinal] = call
        pass_to_calls.setdefault(event.pass_index, []).append(call)

    _sync_grad_fn_graph_relations(trace)

    unique_payload_ids: set[int] = set()
    total_gradient_memory = 0
    total_backward_memory = 0
    saved_grad_labels: set[str] = set()
    for op in getattr(trace, "layer_list", []):
        op.__dict__.setdefault("_grad_records", []).clear()
    for event in sorted(
        op_grad_events, key=lambda item: (item.pass_index, item.timestamp, item.seq)
    ):
        if event.op_label not in getattr(trace, "layer_dict_all_keys", {}):
            continue
        op = trace[event.op_label]
        payload = event.payload_ref if isinstance(event.payload_ref, torch.Tensor) else None
        transformed_payload = event.transformed_payload_ref
        op._record_gradient(
            backward_pass_index=event.pass_index,
            grad=payload,
            transformed_grad=transformed_payload,
            shape=event.shape,
            dtype=event.dtype,
            memory=event.memory,
            timestamp=event.timestamp,
        )
        if payload is None and transformed_payload is None:
            continue
        op.has_grad = True
        if event.shape is not None:
            op.grad_shape = tuple(event.shape)
        if event.dtype is not None:
            op.grad_dtype = _torch_dtype_from_string(event.dtype)
        op.gradient_memory = Bytes(event.memory or 0)
        op._internal_set("grad", payload)
        op._internal_set("transformed_grad", transformed_payload)
        op.transformed_grad_shape = _shape_or_none(transformed_payload)
        op.transformed_grad_dtype = _dtype_or_none(transformed_payload)
        op.transformed_gradient_memory = _memory_or_none(transformed_payload)
        saved_grad_labels.add(op.layer_label)
        for payload_ref, payload_memory in _retained_grad_payload_refs(
            payload, transformed_payload, raw_memory=event.memory
        ):
            payload_id = id(payload_ref)
            if payload_id not in unique_payload_ids:
                unique_payload_ids.add(payload_id)
                total_backward_memory += payload_memory
        total_gradient_memory += int(event.memory or 0)
    trace._saved_grads_set = saved_grad_labels
    trace.saved_gradient_memory = Bytes(total_gradient_memory)
    trace.total_gradient_memory = Bytes(total_gradient_memory)
    trace.total_backward_memory = Bytes(total_backward_memory)

    roots_by_pass = getattr(trace, "_backward_roots_by_pass", {})
    backward_pass_logs: OrderedDict[int, BackwardPass] = OrderedDict()
    pass_indices = sorted(set(starts) | set(ends) | set(pass_to_calls) | op_grad_passes)
    for pass_index in pass_indices:
        start = starts.get(pass_index)
        end = ends.get(pass_index)
        pass_record = BackwardPass(
            pass_index=pass_index,
            trigger=start.trigger if start is not None else "implicit",
            implicit=start.implicit if start is not None else True,
            outer_context=start.outer_context if start is not None else None,
            call_context=start.call_context_ref if start is not None else None,
            root_grad_fn_ids=list(roots_by_pass.get(pass_index, ())),
            root_meta=tuple(start.root_meta) if start is not None else (),
            root_grad_arguments=start.root_grad_arguments if start is not None else None,
            inputs_subset=tuple(start.inputs_subset) if start is not None else (),
            order=start.order if start is not None else None,
            origin_backward_pass=start.origin_backward_pass if start is not None else None,
            engine_flags=start.engine_flags if start is not None else None,
            save_grads_policy=start.save_grads_policy_repr if start is not None else None,
            duration=None if end is None or end.duration is None else Duration(end.duration),
            peak_memory=end.peak_memory if end is not None else None,
            status=end.status if end is not None else "ok",
            order_attribution_coverage=(
                end.order_attribution_coverage if end is not None else None
            ),
            grad_fn_calls=pass_to_calls.get(pass_index, []),
        )
        pass_record.source_trace = trace
        backward_pass_logs[pass_index] = pass_record

    trace.backward_pass_logs = backward_pass_logs
    trace.backward_root_grad_fn_object_ids = [
        root_id for roots in roots_by_pass.values() for root_id in roots
    ]
    trace.num_backward_passes = max(pass_indices) if pass_indices else 0
    trace.has_backward_pass = bool(pass_indices)
    trace.num_saved_grad_fn_calls = len(trace.saved_grad_fn_calls)
    trace.num_saved_grad_fns = len(trace.saved_grad_fns)
    trace._backward_projection_event_count = len(events)

    for grad_fn_record in trace.grad_fn_logs.values():
        if grad_fn_record.op is not None:
            grad_fn_record.op.grad_fn = grad_fn_record
            parent_layer = trace.layer_logs.get(grad_fn_record.op.layer_label)
            if parent_layer is not None:
                parent_layer.grad_fn = grad_fn_record


def _torch_dtype_from_string(dtype_name: str) -> torch.dtype | str:
    """Return a ``torch.dtype`` for canonical dtype strings when possible."""

    if dtype_name.startswith("torch."):
        dtype_attr = dtype_name.removeprefix("torch.")
        dtype = getattr(torch, dtype_attr, None)
        if isinstance(dtype, torch.dtype):
            return dtype
    return dtype_name


def _retained_grad_payload_refs(
    raw_payload: torch.Tensor | None,
    transformed_payload: Any | None,
    *,
    raw_memory: int | None,
) -> list[tuple[Any, int]]:
    """Return retained gradient payload refs paired with their memory cost."""

    payloads: list[tuple[Any, int]] = []
    if raw_payload is not None:
        payloads.append((raw_payload, int(raw_memory or 0)))
    transformed_memory = _memory_or_none(transformed_payload)
    if transformed_payload is not None and transformed_memory is not None:
        payloads.append((transformed_payload, int(transformed_memory)))
    return payloads


def _make_grad_fn_hook(
    trace: Any,
    grad_fn_object_id: int,
    *,
    is_accumulate_grad: bool = False,
) -> Callable[..., tuple[torch.Tensor | None, ...] | None]:
    """Build a runtime hook for one autograd grad_fn_handle.

    Parameters
    ----------
    trace:
        Trace whose flat backward fields should receive runtime data.
    grad_fn_object_id:
        ``id()`` of the hooked grad_fn_handle.
    is_accumulate_grad:
        Whether this hook is attached to an AccumulateGrad node.

    Returns
    -------
    Callable[..., tuple[torch.Tensor | None, ...] | None]
        Hook compatible with ``grad_fn_handle.register_hook``.
    """

    trace_ref = weakref.ref(trace)

    def hook(*hook_args: Any) -> tuple[torch.Tensor | None, ...] | None:
        live_trace = trace_ref()
        if live_trace is None:
            return None
        grad_fn_handle = live_trace.grad_fn_logs.get(grad_fn_object_id)
        if grad_fn_handle is None:
            return None
        grad_inputs = hook_args[0] if len(hook_args) >= 1 else None
        grad_outputs = hook_args[1] if len(hook_args) >= 2 else None
        layer_label = grad_fn_handle.op.layer_label if grad_fn_handle.has_op else None
        stored_grad_inputs = grad_inputs
        stored_grad_outputs = grad_outputs
        if not _selected_for_grad_save(live_trace, layer_label):
            stored_grad_inputs = None
            stored_grad_outputs = None
        with pause_logging():
            grad_fn_handle._log_call(stored_grad_inputs, stored_grad_outputs, time.time())
        call_index = len(grad_fn_handle.calls)
        refs = live_trace.__dict__.setdefault("_backward_gradfn_refs", [])
        for grad_value in tuple(grad_inputs or ()):
            if isinstance(grad_value, torch.Tensor) and grad_value.grad_fn is not None:
                refs.append(grad_value.grad_fn)
        events = _ensure_backward_event_stream(live_trace)
        event_timestamp = time.time()
        pass_index = int(
            getattr(
                live_trace,
                "_active_backward_pass_index",
                getattr(live_trace, "num_backward_passes", 0) + 1,
            )
        )
        events.append_backward(
            GradFnFired(
                object_id=grad_fn_object_id,
                pass_index=pass_index,
                grad_input_refs=stored_grad_inputs,
                grad_output_refs=stored_grad_outputs,
                intervention_fire_ref=None,
                timestamp=event_timestamp,
                seq=events.next_backward_seq(),
            )
        )
        if is_accumulate_grad:
            param_address = getattr(live_trace, "_grad_fn_param_refs_by_object_id", {}).get(
                grad_fn_object_id
            )
            param_log = (
                live_trace.param_logs[param_address]
                if param_address is not None and param_address in live_trace.param_logs
                else None
            )
            picked = _first_tensor_from_hook_args(hook_args)
            if param_log is not None and picked is not None:
                param_log._record_gradient_increment(
                    backward_pass_index=pass_index,
                    grad=picked[2],
                    timestamp=event_timestamp,
                )
            return None
        from ...intervention.runtime import _apply_live_backward_hooks

        return _apply_live_backward_hooks(grad_inputs, grad_outputs, grad_fn_handle, call_index)

    return hook


def _make_grad_fn_prehook(
    trace: Any,
    grad_fn_object_id: int,
) -> Callable[..., tuple[torch.Tensor | None, ...] | None]:
    """Build an AccumulateGrad prehook for mutating incoming gradients.

    Parameters
    ----------
    trace:
        Trace whose backward hook plan should dispatch.
    grad_fn_object_id:
        ``id()`` of the hooked grad_fn_handle.

    Returns
    -------
    Callable[..., tuple[torch.Tensor | None, ...] | None]
        Hook compatible with ``grad_fn_handle.register_prehook``.
    """

    trace_ref = weakref.ref(trace)

    def prehook(*hook_args: Any) -> tuple[torch.Tensor | None, ...] | None:
        live_trace = trace_ref()
        if live_trace is None:
            return None
        grad_fn_handle = live_trace.grad_fn_logs.get(grad_fn_object_id)
        if grad_fn_handle is None:
            return None
        grad_inputs = hook_args[0] if len(hook_args) >= 1 else ()
        call_index = len(grad_fn_handle.calls) + 1
        from ...intervention.runtime import _apply_live_backward_prehooks

        return _apply_live_backward_prehooks(grad_inputs, grad_fn_handle, call_index)

    return prehook


def _memory_snapshot(device: torch.device) -> tuple[str, int]:
    """Return backend name and current allocated memory for a device.

    Parameters
    ----------
    device:
        Device associated with the backward loss tensor.

    Returns
    -------
    tuple[str, int]
        Backend label and memory snapshot in bytes.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        return "cuda", int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps" and hasattr(torch, "mps"):
        return "mps", int(torch.mps.current_allocated_memory())
    try:
        import psutil
    except ImportError:
        return "cpu", 0
    return "cpu", int(psutil.Process().memory_info().rss)


def _reset_peak_memory(device: torch.device) -> tuple[str, int]:
    """Reset peak tracking when available and return the starting snapshot.

    Parameters
    ----------
    device:
        Device associated with the backward loss tensor.

    Returns
    -------
    tuple[str, int]
        Backend label and starting memory snapshot in bytes.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    return _memory_snapshot(device)


def _layer_by_grad_fn_id(trace: Any) -> dict[int, str]:
    """Build a mapping from forward grad_fn_handle identity to final layer label.

    Parameters
    ----------
    trace:
        Trace whose layers were captured during the forward pass.

    Returns
    -------
    dict[int, str]
        Mapping from ``id(grad_fn_handle)`` to final layer label.
    """
    mapping = {}
    for layer in trace.layer_list:
        grad_fn_object_id = getattr(layer, "grad_fn_object_id", None)
        if grad_fn_object_id is not None:
            mapping[grad_fn_object_id] = layer.layer_label
    return mapping


def _walk_and_hook_backward_graph(trace: Any, loss: torch.Tensor) -> list[Any]:
    """Walk ``loss.grad_fn`` and register hooks on every reachable grad_fn_handle.

    Parameters
    ----------
    trace:
        Trace that owns the flat backward fields.
    loss:
        Scalar or tensor loss whose backward graph should be captured.

    Returns
    -------
    list[Any]
        Hook handles to remove after backward finishes.
    """
    if loss.grad_fn is None:
        raise ValueError("log_backward requires a loss tensor with a grad_fn_handle.")

    layer_lookup = _layer_by_grad_fn_id(trace)
    queue: deque[Any] = deque([loss.grad_fn])
    seen: set[int] = set()
    handles: list[Any] = []
    type_counter: dict[str, int] = {}
    # Keep strong refs to every discovered grad_fn_handle for the trace's lifetime so
    # Python cannot recycle their memory addresses. ``id()`` is used as the
    # primary key for ``grad_fn_logs`` and ``next_grad_fn_ids``; if leaf nodes
    # like AccumulateGrad were gc'd, their ids could be reused by later-created
    # grad_fns (e.g. the output clone wrapper), creating phantom cycles.
    strong_refs = trace.__dict__.setdefault("_backward_gradfn_refs", [])
    strong_refs.append(loss.grad_fn)
    root_grad_fn_id = id(loss.grad_fn)
    if root_grad_fn_id not in trace.backward_root_grad_fn_object_ids:
        trace.backward_root_grad_fn_object_ids.append(root_grad_fn_id)
    pass_index = int(
        getattr(trace, "_active_backward_pass_index", getattr(trace, "num_backward_passes", 0) + 1)
    )
    roots_by_pass = trace.__dict__.setdefault("_backward_roots_by_pass", {})
    roots_by_pass.setdefault(pass_index, [])
    if root_grad_fn_id not in roots_by_pass[pass_index]:
        roots_by_pass[pass_index].append(root_grad_fn_id)
    trace.has_backward_pass = True

    while queue:
        grad_fn_handle = queue.popleft()
        grad_fn_object_id = id(grad_fn_handle)
        if grad_fn_object_id in seen:
            continue
        seen.add(grad_fn_object_id)
        next_grad_fns = list(_iter_next_grad_fns(grad_fn_handle))
        strong_refs.extend(next_grad_fns)
        queue.extend(next_grad_fns)
        layer_label = layer_lookup.get(grad_fn_object_id)
        grad_fn_record = trace.grad_fn_logs.get(grad_fn_object_id)
        if grad_fn_record is None:
            grad_fn_type = _normalize_grad_fn_type(grad_fn_handle)
            step_index = len(trace.grad_fn_order) + 1
            type_index, step_index, label = _grad_fn_label_parts(
                trace,
                grad_fn_type,
                layer_label,
                type_counter,
                step_index,
            )
            grad_fn_cls = type(grad_fn_handle)
            grad_fn_record = GradFn(
                grad_fn_object_id=grad_fn_object_id,
                class_name=grad_fn_cls.__name__,
                class_qualname=f"{grad_fn_cls.__module__}.{grad_fn_cls.__qualname__}",
                label=label,
                type=grad_fn_type,
                type_index=type_index,
                ordinal_index=len(trace.grad_fn_order),
                step_index=step_index,
                is_custom=_grad_fn_is_custom(grad_fn_handle),
                has_op=layer_label is not None,
                op_label=layer_label,
                next_grad_fn_ids=[id(next_fn) for next_fn in next_grad_fns],
                **_grad_fn_source_metadata(grad_fn_cls),
            )
            grad_fn_record.source_trace = trace
            trace.grad_fn_logs[grad_fn_object_id] = grad_fn_record
            trace.grad_fn_order.append(grad_fn_object_id)
        else:
            grad_fn_record.next_grad_fn_ids = [id(next_fn) for next_fn in next_grad_fns]
            grad_fn_record.source_trace = trace
        is_accumulate_grad = _is_accumulate_grad(grad_fn_handle)
        if is_accumulate_grad:
            param = getattr(grad_fn_handle, "variable", None)
            param_address = trace._param_log_by_pid.get(id(param)) if param is not None else None
            if param_address is not None:
                trace._grad_fn_param_refs[grad_fn_record.label] = param_address
                trace.__dict__.setdefault("_grad_fn_param_refs_by_object_id", {})[
                    grad_fn_object_id
                ] = param_address
        source_fields = {
            key: getattr(grad_fn_record, key)
            for key in (
                "class_source_file",
                "class_source_line",
                "init_source_file",
                "init_source_line",
                "forward_source_file",
                "forward_source_line",
                "backward_source_file",
                "backward_source_line",
            )
        }
        _ensure_backward_event_stream(trace).append_backward(
            GradFnDiscovered(
                object_id=grad_fn_object_id,
                class_name=grad_fn_record.class_name,
                class_qualname=grad_fn_record.class_qualname,
                is_custom=grad_fn_record.is_custom,
                op_label=layer_label,
                param_ref=trace._grad_fn_param_refs.get(grad_fn_record.label),
                created_in_pass=None,
                creator_object_id=None,
                source=source_fields,
                topology=tuple(id(next_fn) for next_fn in next_grad_fns),
            )
        )
        if layer_label is not None:
            layer = trace[layer_label]
            layer.grad_fn = grad_fn_record
            parent_layer = trace.layer_logs.get(layer.layer_label)
            if parent_layer is not None:
                parent_layer.grad_fn = grad_fn_record
        try:
            handles.append(
                grad_fn_handle.register_hook(
                    _make_grad_fn_hook(
                        trace,
                        grad_fn_object_id,
                        is_accumulate_grad=is_accumulate_grad,
                    )
                )
            )
            if is_accumulate_grad:
                handles.append(
                    grad_fn_handle.register_prehook(_make_grad_fn_prehook(trace, grad_fn_object_id))
                )
        except RuntimeError:
            continue
    _sync_grad_fn_graph_relations(trace)
    return handles


def _is_static_truthy_keep_grad(value: Any) -> bool:
    """Return whether a gradient capture decision is statically keep-grad true."""

    from ...fastlog.types import CaptureSpec

    return value is True or (isinstance(value, CaptureSpec) and value.keep_grad)


def _normalize_grad_capture_decision(value: Any) -> Any:
    """Normalize a fastlog gradient capture decision."""

    from ...fastlog.types import CaptureSpec

    if value is None or value is False:
        return CaptureSpec(save_out=False, save_metadata=False)
    if value is True:
        return CaptureSpec(save_out=True, save_metadata=True)
    if isinstance(value, CaptureSpec):
        return value
    raise TypeError("keep_grad predicates must return bool, CaptureSpec, or None")


def _recording_grad_fn_label(
    grad_fn_handle: Any,
    layer_label: str | None,
    type_counter: dict[str, int],
    total_num: int,
) -> str:
    """Build a fastlog grad_fn_handle label for one autograd node."""

    grad_fn_type = _normalize_grad_fn_type(grad_fn_handle)
    if layer_label is not None:
        return f"{grad_fn_type}_back_{layer_label}"
    type_counter[grad_fn_type] = type_counter.get(grad_fn_type, 0) + 1
    return f"{grad_fn_type}_back_{type_counter[grad_fn_type]}_{total_num}"


def _iter_backward_nodes(loss: torch.Tensor) -> list[Any]:
    """Return reachable autograd nodes from a loss tensor in BFS order."""

    if loss.grad_fn is None:
        raise ValueError("log_backward requires a loss tensor with a grad_fn_handle.")
    queue: deque[Any] = deque([loss.grad_fn])
    seen: set[int] = set()
    nodes: list[Any] = []
    while queue:
        grad_fn_handle = queue.popleft()
        grad_fn_object_id = id(grad_fn_handle)
        if grad_fn_object_id in seen:
            continue
        seen.add(grad_fn_object_id)
        nodes.append(grad_fn_handle)
        queue.extend(_iter_next_grad_fns(grad_fn_handle))
    return nodes


def _selected_fastlog_forward_labels(recording: Any) -> set[str]:
    """Return labels for predicate-selected forward records."""

    from ...capture.predicates import _evaluate_keep_op
    from ...ir.predicate import RetroactiveCaptureDecision

    recording._ensure_records()  # noqa: SLF001
    labels: set[str] = set()
    for record in recording.records:
        labels.add(record.ctx.label)
        if (
            record.ctx.kind == "op"
            and record.ctx.layer_type is not None
            and record.ctx.type_index is not None
        ):
            labels.add(f"{record.ctx.layer_type}_{record.ctx.type_index}")
        if record.ctx.raw_label is not None:
            labels.add(record.ctx.raw_label)
    recording_state = getattr(recording, "_recording_state", None)
    if recording_state is None:
        return labels
    for ctx in recording_state.all_contexts:
        spec = _evaluate_keep_op(ctx, recording_state.options)
        if isinstance(spec, RetroactiveCaptureDecision):
            continue
        if not spec.save_out and not spec.save_metadata:
            continue
        labels.add(ctx.label)
        if ctx.kind == "op" and ctx.layer_type is not None and ctx.type_index is not None:
            labels.add(f"{ctx.layer_type}_{ctx.type_index}")
        if ctx.raw_label is not None:
            labels.add(ctx.raw_label)
    return labels


def _effective_grad_spec(
    ctx: Any,
    *,
    keep_grad: Any,
    default_grad: Any,
    selected_forward_labels: set[str],
) -> Any:
    """Resolve the capture spec for one gradient event."""

    if callable(keep_grad):
        return _normalize_grad_capture_decision(keep_grad(ctx))
    if keep_grad is None:
        return _normalize_grad_capture_decision(default_grad)
    spec = _normalize_grad_capture_decision(keep_grad)
    if keep_grad is True and ctx.layer_label not in selected_forward_labels:
        return _normalize_grad_capture_decision(False)
    return spec


def _first_tensor_from_hook_args(
    hook_args: tuple[Any, ...],
) -> tuple[str, int | None, torch.Tensor] | None:
    """Pick the first tensor gradient from grad output or input hook arguments."""

    grad_outputs = hook_args[1] if len(hook_args) >= 2 else ()
    if grad_outputs is not None:
        for index, grad in enumerate(grad_outputs):
            if isinstance(grad, torch.Tensor):
                return "grad_output", index, grad
    grad_inputs = hook_args[0] if len(hook_args) >= 1 else ()
    if grad_inputs is not None:
        for index, grad in enumerate(grad_inputs):
            if isinstance(grad, torch.Tensor):
                return "grad_input", index, grad
    return None


def _make_recording_grad_hook(
    recording_state: Any,
    recording: Any,
    grad_fn_handle: Any,
    label: str,
    *,
    keep_grad: Any,
    default_grad: Any,
    selected_forward_labels: set[str],
    backward_call_index: int,
) -> Callable[..., None]:
    """Build a fastlog hook that records a selected gradient payload."""

    from ...fastlog.types import GradientRecord, build_grad_record_context

    def hook(*hook_args: Any) -> None:
        picked = _first_tensor_from_hook_args(hook_args)
        if picked is None:
            return None
        grad_kind, grad_index, grad = picked
        grad_kind_literal = cast(Literal["grad_input", "grad_output"], grad_kind)
        ctx = build_grad_record_context(
            recording_state,
            grad_fn_handle,
            grad,
            label=label,
            grad_kind=grad_kind_literal,
            backward_call_index=backward_call_index,
            grad_input_index=grad_index if grad_kind == "grad_input" else None,
            grad_output_index=grad_index if grad_kind == "grad_output" else None,
        )
        spec = _effective_grad_spec(
            ctx,
            keep_grad=keep_grad,
            default_grad=default_grad,
            selected_forward_labels=selected_forward_labels,
        )
        if not spec.save_out and not spec.save_metadata:
            return None
        ram_payload = None
        disk_payload = None
        transformed_ram_payload = None
        transformed_disk_payload = None
        if spec.save_out:
            (
                ram_payload,
                disk_payload,
                transformed_ram_payload,
                transformed_disk_payload,
            ) = recording_state.resolve_storage(grad, spec, ctx=ctx, kind="grad")
        recording.add_grad_record(
            GradientRecord(
                ctx=ctx,
                spec=spec,
                ram_payload=ram_payload,
                disk_payload=disk_payload,
                transformed_ram_payload=transformed_ram_payload,
                transformed_disk_payload=transformed_disk_payload,
            )
        )
        return None

    return hook


def log_recording_backward(
    recording: Any,
    loss: torch.Tensor,
    *,
    keep_grad: Any = None,
    default_grad: Any = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
) -> Any:
    """Run backward while capturing gradients for a fastlog ``Recording``."""

    from ...fastlog.exceptions import InvalidStorageError, RecorderStateError

    recording_state = getattr(recording, "_recording_state", None)
    if recording_state is None:
        raise RecorderStateError("Recording.log_backward() requires a live recording state")
    effective_keep_grad = keep_grad if keep_grad is not None else recording_state.options.keep_grad
    effective_default_grad = (
        default_grad if default_grad is not None else recording_state.options.default_grad
    )
    if (
        recording_state.storage_intent.on_disk
        and not recording_state.storage_intent.in_ram
        and _is_static_truthy_keep_grad(effective_keep_grad)
    ):
        raise InvalidStorageError(
            "keep_grad=True is incompatible with disk-only fastlog gradient storage"
        )
    nodes = _iter_backward_nodes(loss)
    selected_forward_labels = _selected_fastlog_forward_labels(recording)
    handles: list[Any] = []
    type_counter: dict[str, int] = {}
    backward_call_index = len(recording.grad_records) + 1
    try:
        for index, grad_fn_handle in enumerate(nodes, start=1):
            forward_ctx = recording_state.grad_fn_to_context.get(grad_fn_handle)
            layer_label = forward_ctx.label if forward_ctx is not None else None
            label = _recording_grad_fn_label(grad_fn_handle, layer_label, type_counter, index)
            handles.append(
                grad_fn_handle.register_hook(
                    _make_recording_grad_hook(
                        recording_state,
                        recording,
                        grad_fn_handle,
                        label,
                        keep_grad=effective_keep_grad,
                        default_grad=effective_default_grad,
                        selected_forward_labels=selected_forward_labels,
                        backward_call_index=backward_call_index,
                    )
                )
            )
        backward_kwargs: dict[str, Any] = {"create_graph": create_graph}
        if retain_graph is not None:
            backward_kwargs["retain_graph"] = retain_graph
        loss.backward(**backward_kwargs)
    finally:
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.remove()
    return recording


def _is_accumulate_grad(grad_fn_handle: Any) -> bool:
    """Return whether an autograd node is an AccumulateGrad leaf.

    Parameters
    ----------
    grad_fn_handle
        Autograd node to inspect.

    Returns
    -------
    bool
        True when ``grad_fn_handle`` is an AccumulateGrad node.
    """

    accumulate_grad_cls = getattr(getattr(torch._C, "_functions", object()), "AccumulateGrad", ())
    return type(grad_fn_handle).__name__ == "AccumulateGrad" or isinstance(
        grad_fn_handle, accumulate_grad_cls
    )


def _clear_forward_grad_fn_refs(trace: Any) -> None:
    """Clear strong forward references to autograd nodes after hook registration.

    Parameters
    ----------
    trace:
        Trace whose Op and Layer grad_fn_handle object refs should be
        released.
    """
    for layer in trace.layer_list:
        layer.grad_fn_handle = None
    for layer_log in trace.layer_logs.values():
        layer_log.grad_fn_handle = None


def _run_backward_with_capture(
    trace: Any,
    loss: torch.Tensor,
    backward_callable: Callable[[], Any],
    *,
    trigger: str = "backward",
    outer_context: str | None = None,
    engine_flags: dict[str, object] | None = None,
    save_grads: Any | MissingType = MISSING,
) -> Any:
    """Capture a backward graph, run backward, and record memory delta.

    Parameters
    ----------
    trace:
        Trace to mutate.
    loss:
        Loss tensor that roots the backward graph.
    backward_callable:
        Zero-argument callable that performs the actual backward pass.

    Returns
    -------
    Any
        Return value from ``backward_callable``.
    """
    from ... import _state
    from ...intervention.hooks import normalize_hooks_from_spec

    previous_trace = _state._active_trace
    previous_plan = _state._active_hook_plan
    previous_spec = _state._active_intervention_spec
    if getattr(trace, "_tl_active_backward_bracket", False):
        return backward_callable()
    _close_implicit_backward_pass_if_open(trace)
    previous_save_grads_policy = getattr(trace, "_active_save_grads_policy", None)
    previous_had_save_grads_policy = "_active_save_grads_policy" in trace.__dict__
    active_save_grads_policy = (
        getattr(trace, "save_grads", None) if save_grads is MISSING else save_grads
    )
    trace._active_save_grads_policy = active_save_grads_policy
    _state._active_trace = trace
    _state._active_intervention_spec = getattr(trace, "_intervention_spec", None)
    _state._active_hook_plan = [
        *normalize_hooks_from_spec(_state._active_intervention_spec),
        *getattr(trace, "_initial_hook_plan", ()),
    ]
    pass_index = int(getattr(trace, "num_backward_passes", 0)) + 1
    events = _ensure_backward_event_stream(trace)
    trace._active_backward_pass_index = pass_index
    trace._implicit_backward_pass_open = False
    events.append_backward(
        BackwardPassStart(
            pass_index=pass_index,
            trigger=cast(Any, trigger),
            implicit=False,
            outer_context=outer_context,
            call_context_ref=None,
            root_meta=(
                {
                    "shape": tuple(loss.shape),
                    "dtype": str(loss.dtype),
                    "device": str(loss.device),
                },
            ),
            root_grad_arguments=None,
            inputs_subset=(),
            order=None,
            origin_backward_pass=None,
            save_grads_policy_repr=repr(active_save_grads_policy),
            engine_flags=engine_flags,
            timestamp=time.time(),
        )
    )
    handles = _walk_and_hook_backward_graph(trace, loss)
    backend, before = _reset_peak_memory(loss.device)
    backward_start_time = time.time()
    status = "ok"
    result = None
    trace._tl_active_backward_bracket = True
    try:
        result = backward_callable()
    except Exception:
        status = "error"
        raise
    finally:
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.remove()
        _clear_forward_grad_fn_refs(trace)
        _state._active_trace = previous_trace
        _state._active_hook_plan = previous_plan
        _state._active_intervention_spec = previous_spec
        trace.__dict__.pop("_tl_active_backward_bracket", None)
        _backend, after = _memory_snapshot(loss.device)
        peak_delta = max(0, after - before)
        duration = time.time() - backward_start_time
        trace.backward_memory_backend = backend
        trace.backward_peak_memory += Bytes(peak_delta)
        trace.backward_durations.append(Duration(duration))
        trace.total_param_gradient_memory = Bytes(
            sum(int(param_log.gradient_memory) for param_log in getattr(trace, "param_logs", []))
        )
        trace.num_backward_passes = max(int(getattr(trace, "num_backward_passes", 0)), pass_index)
        events.append_backward(
            BackwardPassEnd(
                pass_index=pass_index,
                duration=duration,
                peak_memory=peak_delta,
                status=cast(Any, status),
                order_attribution_coverage=None,
            )
        )
        trace.__dict__.pop("_active_backward_pass_index", None)
        if previous_had_save_grads_policy:
            trace._active_save_grads_policy = previous_save_grads_policy
        else:
            trace.__dict__.pop("_active_save_grads_policy", None)
        _materialize_backward_projections(trace)
    return result


def disarm_triggers(trace: Any) -> None:
    """Detach a trace from global autograd trigger interception.

    Parameters
    ----------
    trace:
        Trace whose future tensor-hook and autograd-wrapper capture should be
        disabled.

    Returns
    -------
    None
        Registry state is updated in place.
    """

    trace._tl_backward_triggers_disarmed = True
    _purge_trace_from_backward_registry(trace)


def _capture_autograd_engine_call(
    roots: Any,
    engine_callable: Callable[[], Any],
    *,
    trigger: Literal["autograd_backward", "autograd_grad"],
    engine_flags: dict[str, object] | None,
) -> Any:
    """Run a global autograd entry through TorchLens capture when roots match.

    Parameters
    ----------
    roots:
        Tensor roots passed to the autograd engine.
    engine_callable:
        Zero-argument callable invoking the original PyTorch autograd function.
    trigger:
        Trigger label to store on ``BackwardPassStart``.
    engine_flags:
        Best-effort keyword metadata from the engine invocation.

    Returns
    -------
    Any
        Return value from ``engine_callable``.
    """

    matched_traces = tuple(
        trace
        for trace in _traces_for_roots(roots)
        if not getattr(trace, "_tl_active_backward_bracket", False)
    )
    loss = _first_root_tensor(roots)
    if loss is None or not matched_traces:
        return engine_callable()
    if len(matched_traces) == 1:
        return _run_backward_with_capture(
            matched_traces[0],
            loss,
            engine_callable,
            trigger=trigger,
            engine_flags=engine_flags,
        )

    # P1 supports multi-trace bracket opening by nesting setup and running the
    # engine once through the innermost capture. Later projection phases will
    # refine per-trace outer_context metadata.
    def nested_call(index: int) -> Any:
        """Run nested capture setup for all matched traces."""

        if index == len(matched_traces):
            return engine_callable()
        return _run_backward_with_capture(
            matched_traces[index],
            loss,
            lambda: nested_call(index + 1),
            trigger=trigger,
            outer_context=None if index == 0 else f"{trigger}:multi_trace",
            engine_flags=engine_flags,
        )

    return nested_call(0)


def install_autograd_wrappers() -> None:
    """Install global autograd trigger wrappers idempotently.

    Returns
    -------
    None
        ``torch.autograd.backward`` and ``torch.autograd.grad`` are patched once.
    """

    global _AUTOGRAD_WRAPPERS_INSTALLED, _ORIGINAL_AUTOGRAD_BACKWARD, _ORIGINAL_AUTOGRAD_GRAD
    if _AUTOGRAD_WRAPPERS_INSTALLED:
        return
    _ORIGINAL_AUTOGRAD_BACKWARD = torch.autograd.backward
    _ORIGINAL_AUTOGRAD_GRAD = torch.autograd.grad

    def wrapped_backward(*args: Any, **kwargs: Any) -> Any:
        """Route ``torch.autograd.backward`` through TorchLens when roots match."""

        original = cast(Callable[..., Any], _ORIGINAL_AUTOGRAD_BACKWARD)
        roots = _autograd_roots_from_call(args, kwargs, "tensors")

        def run() -> Any:
            """Invoke the original autograd backward function."""

            return original(*args, **kwargs)

        return _capture_autograd_engine_call(
            roots,
            run,
            trigger="autograd_backward",
            engine_flags=dict(kwargs),
        )

    def wrapped_grad(*args: Any, **kwargs: Any) -> Any:
        """Route ``torch.autograd.grad`` through TorchLens when roots match."""

        original = cast(Callable[..., Any], _ORIGINAL_AUTOGRAD_GRAD)
        roots = _autograd_roots_from_call(args, kwargs, "outputs")

        def run() -> Any:
            """Invoke the original autograd grad function."""

            return original(*args, **kwargs)

        return _capture_autograd_engine_call(
            roots,
            run,
            trigger="autograd_grad",
            engine_flags=dict(kwargs),
        )

    torch.autograd.backward = wrapped_backward
    torch.autograd.grad = wrapped_grad
    _AUTOGRAD_WRAPPERS_INSTALLED = True


def uninstall_autograd_wrappers() -> None:
    """Restore original global autograd entry points when installed.

    Returns
    -------
    None
        PyTorch autograd functions are restored in place.
    """

    global _AUTOGRAD_WRAPPERS_INSTALLED
    if not _AUTOGRAD_WRAPPERS_INSTALLED:
        return
    if _ORIGINAL_AUTOGRAD_BACKWARD is not None:
        torch.autograd.backward = _ORIGINAL_AUTOGRAD_BACKWARD
    if _ORIGINAL_AUTOGRAD_GRAD is not None:
        torch.autograd.grad = _ORIGINAL_AUTOGRAD_GRAD
    _AUTOGRAD_WRAPPERS_INSTALLED = False


def _ensure_layer_grad_hooks(trace: Any) -> None:
    """Enable legacy gradient retention after op-record-time hook installation.

    Parameters
    ----------
    trace:
        Trace whose saved outs should receive grad hooks.
    """
    if getattr(trace, "_grad_layer_nums_to_save", None) in [None, [], "none"]:
        trace._grad_layer_nums_to_save = "all"


def _finalize_grad_streaming(trace: Any) -> None:
    """Finalize a deferred grad-streaming bundle after backward capture."""

    writer = getattr(trace, "_out_writer", None)
    if writer is None or not getattr(trace, "_defer_streaming_bundle_finalization", False):
        return

    from ...postprocess.finalization import (
        _evict_streamed_outs,
        _evict_streamed_grads,
        _finalize_streamed_bundle,
    )

    _finalize_streamed_bundle(trace)
    if not getattr(trace, "_keep_outs_in_memory", True):
        _evict_streamed_outs(trace)
    if not getattr(trace, "_grad_stream_retain_in_memory", True):
        _evict_streamed_grads(trace)
    for field_name in (
        "_out_writer",
        "_out_sink",
        "_keep_outs_in_memory",
        "_grad_stream_retain_in_memory",
        "_defer_streaming_bundle_finalization",
    ):
        trace.__dict__.pop(field_name, None)


def log_backward(
    self: Any,
    loss: torch.Tensor,
    *,
    save_grads: Any | MissingType = MISSING,
    **backward_kwargs: Any,
) -> Any:
    """Run ``loss.backward`` while capturing the backward graph.

    Parameters
    ----------
    loss:
        Loss tensor whose ``grad_fn_handle`` roots the backward graph.
    save_grads:
        Optional per-call gradient retention override. ``True`` captures all
        observed op gradients, ``False``/``None`` disables retention for this
        call, and selectors/callables are evaluated at hook fire time.
    **backward_kwargs:
        Keyword arguments forwarded to ``torch.Tensor.backward``.

    Returns
    -------
    Any
        The same Trace, for chaining.
    """
    _ensure_layer_grad_hooks(self)

    def run() -> Any:
        """Run the user's requested backward call."""
        return loss.backward(**backward_kwargs)  # type: ignore[no-untyped-call]

    _run_backward_with_capture(
        self,
        loss,
        run,
        trigger="backward",
        engine_flags=dict(backward_kwargs),
        save_grads=save_grads,
    )
    _finalize_grad_streaming(self)
    return self


class RecordingBackward:
    """Context manager that captures every ``Tensor.backward`` call inside it."""

    def __init__(
        self,
        trace: Any,
        *,
        save_grads: Any | MissingType = MISSING,
    ) -> None:
        self.trace = trace
        self.save_grads = save_grads
        self._original_backward: Callable[..., Any] | None = None

    def __enter__(self) -> "RecordingBackward":
        """Patch ``torch.Tensor.backward`` and return this context object."""
        _ensure_layer_grad_hooks(self.trace)
        self._original_backward = torch.Tensor.backward
        trace = self.trace
        original_backward = self._original_backward

        def wrapped_backward(tensor_self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            """Capture the graph rooted at ``tensor_self`` before backward runs."""

            def run() -> Any:
                """Run the original Tensor.backward implementation."""
                return original_backward(tensor_self, *args, **kwargs)  # type: ignore[no-untyped-call]

            return _run_backward_with_capture(
                trace,
                tensor_self,
                run,
                trigger="recording_backward",
                engine_flags=dict(kwargs),
                save_grads=self.save_grads,
            )

        torch.Tensor.backward = wrapped_backward  # type: ignore[assignment, method-assign]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Restore ``torch.Tensor.backward``."""
        if self._original_backward is not None:
            torch.Tensor.backward = self._original_backward  # type: ignore[method-assign]
        if exc_type is None:
            _finalize_grad_streaming(self.trace)


def recording_backward(
    self: Any,
    *,
    save_grads: Any | MissingType = MISSING,
) -> RecordingBackward:
    """Return a context manager that records user-managed backward calls.

    Returns
    -------
    RecordingBackward
        Context manager that patches ``Tensor.backward`` inside the block.
    """
    return RecordingBackward(
        self,
        save_grads=save_grads,
    )
