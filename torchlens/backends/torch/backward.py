"""Backward-pass graph walking, grad_fn_handle hooks, and Trace APIs."""

from __future__ import annotations

import contextlib
import inspect
import re
import time
import weakref
from collections import deque
from typing import Any, Callable, Iterator, Literal, cast

import torch

from ..._io.streaming import BundleStreamWriter
from ...quantities import Bytes, Duration
from ..._state import pause_logging
from ...data_classes.grad_fn_log import GradFn
from .tensor_tracking import _add_tensor_backward_hook


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
    layer_label: str | None,
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
    layer_label:
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
    if layer_label is not None:
        layer = trace.layers[layer_label]
        type_num = layer.type_index
        total_num = layer.step_index
    else:
        type_counter[type] = type_counter.get(type, 0) + 1
        type_num = type_counter[type]
    label = f"{type}_back_{type_num}_{total_num}"
    return type_num, total_num, label


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
        if is_accumulate_grad:
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
    _state._active_trace = trace
    _state._active_intervention_spec = getattr(trace, "_intervention_spec", None)
    _state._active_hook_plan = [
        *normalize_hooks_from_spec(_state._active_intervention_spec),
        *getattr(trace, "_initial_hook_plan", ()),
    ]
    handles = _walk_and_hook_backward_graph(trace, loss)
    backend, before = _reset_peak_memory(loss.device)
    backward_start_time = time.time()
    try:
        result = backward_callable()
    finally:
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.remove()
        _clear_forward_grad_fn_refs(trace)
        _state._active_trace = previous_trace
        _state._active_hook_plan = previous_plan
        _state._active_intervention_spec = previous_spec
    _backend, after = _memory_snapshot(loss.device)
    trace.backward_memory_backend = backend
    trace.backward_peak_memory += Bytes(max(0, after - before))
    trace.backward_durations.append(Duration(time.time() - backward_start_time))
    trace.total_param_gradient_memory = Bytes(
        sum(int(param_log.gradient_memory) for param_log in getattr(trace, "param_logs", []))
    )
    trace.num_backward_passes += 1
    return result


def _ensure_layer_grad_hooks(trace: Any) -> None:
    """Register tensor grad hooks lazily for saved graph-connected outs.

    Parameters
    ----------
    trace:
        Trace whose saved outs should receive grad hooks.
    """
    if getattr(trace, "save_gradients", False):
        return
    trace.save_gradients = True
    if getattr(trace, "_grad_layer_nums_to_save", None) in [None, [], "none"]:
        trace._grad_layer_nums_to_save = "all"
    for layer in getattr(trace, "layer_list", []):
        out = getattr(layer, "out", None)
        if isinstance(out, torch.Tensor):
            _add_tensor_backward_hook(trace, out, layer._label_raw)


def _configure_grad_streaming(
    trace: Any,
    *,
    save_grads_to: str | None,
    keep_grads_in_memory: bool | None,
) -> None:
    """Configure deferred bundle finalization for streamed grads.

    Parameters
    ----------
    trace:
        Trace receiving streamed grad blobs.
    save_grads_to:
        Optional bundle path for grad streaming.
    keep_grads_in_memory:
        Whether grads should remain in memory after finalization.
    """

    if keep_grads_in_memory is not None:
        trace._keep_grads_in_memory = keep_grads_in_memory
    if save_grads_to is not None:
        if getattr(trace, "_out_writer", None) is not None:
            raise ValueError("Cannot set save_grads_to after a streaming writer exists.")
        trace._out_writer = BundleStreamWriter(save_grads_to)
        trace._defer_streaming_bundle_finalization = True
        trace.save_gradients = True


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
    if not getattr(trace, "_keep_grads_in_memory", True):
        _evict_streamed_grads(trace)
    for field_name in (
        "_out_writer",
        "_out_sink",
        "_keep_outs_in_memory",
        "_keep_grads_in_memory",
        "_defer_streaming_bundle_finalization",
    ):
        trace.__dict__.pop(field_name, None)


def log_backward(
    self: Any,
    loss: torch.Tensor,
    *,
    save_grads_to: str | None = None,
    keep_grads_in_memory: bool | None = None,
    **backward_kwargs: Any,
) -> Any:
    """Run ``loss.backward`` while capturing the backward graph.

    Parameters
    ----------
    loss:
        Loss tensor whose ``grad_fn_handle`` roots the backward graph.
    **backward_kwargs:
        Keyword arguments forwarded to ``torch.Tensor.backward``.

    Returns
    -------
    Any
        The same Trace, for chaining.
    """
    _configure_grad_streaming(
        self,
        save_grads_to=save_grads_to,
        keep_grads_in_memory=keep_grads_in_memory,
    )
    _ensure_layer_grad_hooks(self)

    def run() -> Any:
        """Run the user's requested backward call."""
        return loss.backward(**backward_kwargs)  # type: ignore[no-untyped-call]

    _run_backward_with_capture(self, loss, run)
    _finalize_grad_streaming(self)
    return self


class RecordingBackward:
    """Context manager that captures every ``Tensor.backward`` call inside it."""

    def __init__(
        self,
        trace: Any,
        *,
        save_grads_to: str | None = None,
        keep_grads_in_memory: bool | None = None,
    ) -> None:
        self.trace = trace
        self.save_grads_to = save_grads_to
        self.keep_grads_in_memory = keep_grads_in_memory
        self._original_backward: Callable[..., Any] | None = None

    def __enter__(self) -> "RecordingBackward":
        """Patch ``torch.Tensor.backward`` and return this context object."""
        _configure_grad_streaming(
            self.trace,
            save_grads_to=self.save_grads_to,
            keep_grads_in_memory=self.keep_grads_in_memory,
        )
        _ensure_layer_grad_hooks(self.trace)
        self._original_backward = torch.Tensor.backward
        trace = self.trace
        original_backward = self._original_backward

        def wrapped_backward(tensor_self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            """Capture the graph rooted at ``tensor_self`` before backward runs."""

            def run() -> Any:
                """Run the original Tensor.backward implementation."""
                return original_backward(tensor_self, *args, **kwargs)  # type: ignore[no-untyped-call]

            return _run_backward_with_capture(trace, tensor_self, run)

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
    save_grads_to: str | None = None,
    keep_grads_in_memory: bool | None = None,
) -> RecordingBackward:
    """Return a context manager that records user-managed backward calls.

    Returns
    -------
    RecordingBackward
        Context manager that patches ``Tensor.backward`` inside the block.
    """
    return RecordingBackward(
        self,
        save_grads_to=save_grads_to,
        keep_grads_in_memory=keep_grads_in_memory,
    )
