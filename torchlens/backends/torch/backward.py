"""Backward-pass graph walking, grad_fn hooks, and Trace APIs."""

from __future__ import annotations

import contextlib
import re
import time
from collections import deque
from typing import Any, Callable, Iterator

import torch

from ..._io.streaming import BundleStreamWriter
from ..._state import pause_logging
from ...data_classes.grad_fn_log import GradFnLog
from .tensor_tracking import _add_backward_hook


def _grad_fn_is_custom(grad_fn: Any) -> bool:
    """Return whether a grad_fn appears to come from user/custom autograd code.

    Parameters
    ----------
    grad_fn:
        Autograd function object.

    Returns
    -------
    bool
        True when the type's module path is outside PyTorch's built-in autograd
        and nn namespaces.
    """
    module_path = type(grad_fn).__module__
    if module_path == "torch.autograd.function":
        return True
    builtin_prefixes = ("torch.autograd", "torch.nn", "torch", "builtins")
    return not module_path.startswith(builtin_prefixes)


def _iter_next_grad_fns(grad_fn: Any) -> Iterator[Any]:
    """Yield non-null child grad_fns from ``grad_fn.next_functions``.

    Parameters
    ----------
    grad_fn:
        Autograd function object.

    Yields
    ------
    Any
        Reachable child grad_fn object.
    """
    for next_fn, _input_num in getattr(grad_fn, "next_functions", ()):
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
        True if this layer is selected by ``grads_to_save``.
    """
    if layer_label is None:
        return False
    selection = getattr(trace, "_grad_layer_nums_to_save", "all")
    if selection == "all":
        return True
    if selection in [None, "none", []]:
        return False
    return trace[layer_label].capture_index in selection


def _normalize_grad_fn_type(grad_fn: Any) -> str:
    """Normalize an autograd grad_fn class name for TorchLens labels.

    Parameters
    ----------
    grad_fn:
        Autograd function object.

    Returns
    -------
    str
        Lowercased class name with a trailing ``Backward<digits>`` suffix removed.
    """
    return re.sub(r"Backward\d*$", "", type(grad_fn).__name__).lower()


def _grad_fn_label_parts(
    trace: Any,
    grad_fn_type: str,
    layer_label: str | None,
    type_counter: dict[str, int],
    total_num: int,
) -> tuple[int, int, str]:
    """Build numeric label fields for one grad_fn.

    Parameters
    ----------
    trace:
        Trace being updated.
    grad_fn_type:
        Normalized grad_fn type.
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
        total_num = layer.trace_index
    else:
        type_counter[grad_fn_type] = type_counter.get(grad_fn_type, 0) + 1
        type_num = type_counter[grad_fn_type]
    label = f"{grad_fn_type}_back_{type_num}_{total_num}"
    return type_num, total_num, label


def _make_grad_fn_hook(trace: Any, grad_fn_id: int) -> Callable[..., None]:
    """Build a runtime hook for one autograd grad_fn.

    Parameters
    ----------
    trace:
        Trace whose flat backward fields should receive runtime data.
    grad_fn_id:
        ``id()`` of the hooked grad_fn.

    Returns
    -------
    Callable[..., None]
        Hook compatible with ``grad_fn.register_hook``.
    """

    def hook(*hook_args: Any) -> None:
        grad_fn_log = trace.grad_fn_logs.get(grad_fn_id)
        if grad_fn_log is None:
            return
        grad_inputs = hook_args[0] if len(hook_args) >= 1 else None
        grad_outputs = hook_args[1] if len(hook_args) >= 2 else None
        layer_label = grad_fn_log.op.layer_label if grad_fn_log.op is not None else None
        if not _selected_for_grad_save(trace, layer_label):
            grad_inputs = None
            grad_outputs = None
        with pause_logging():
            grad_fn_log._log_call(grad_inputs, grad_outputs, time.time())

    return hook


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
    """Build a mapping from forward grad_fn identity to final layer label.

    Parameters
    ----------
    trace:
        Trace whose layers were captured during the forward pass.

    Returns
    -------
    dict[int, str]
        Mapping from ``id(grad_fn)`` to final layer label.
    """
    mapping = {}
    for layer in trace.layer_list:
        grad_fn_id = getattr(layer, "grad_fn_id", None)
        if grad_fn_id is not None:
            mapping[grad_fn_id] = layer.layer_label
    return mapping


def _walk_and_hook_backward_graph(trace: Any, loss: torch.Tensor) -> list[Any]:
    """Walk ``loss.grad_fn`` and register hooks on every reachable grad_fn.

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
        raise ValueError("log_backward requires a loss tensor with a grad_fn.")

    layer_lookup = _layer_by_grad_fn_id(trace)
    queue: deque[Any] = deque([loss.grad_fn])
    seen: set[int] = set()
    handles: list[Any] = []
    type_counter: dict[str, int] = {}
    # Keep strong refs to every discovered grad_fn for the trace's lifetime so
    # Python cannot recycle their memory addresses. ``id()`` is used as the
    # primary key for ``grad_fn_logs`` and ``next_grad_fn_ids``; if leaf nodes
    # like AccumulateGrad were gc'd, their ids could be reused by later-created
    # grad_fns (e.g. the output clone wrapper), creating phantom cycles.
    trace._grad_fn_strong_refs.append(loss.grad_fn)
    if trace.backward_root_grad_fn_id is None:
        trace.backward_root_grad_fn_id = id(loss.grad_fn)
    trace.has_backward_pass = True

    while queue:
        grad_fn = queue.popleft()
        grad_fn_id = id(grad_fn)
        if grad_fn_id in seen:
            continue
        seen.add(grad_fn_id)
        next_grad_fns = list(_iter_next_grad_fns(grad_fn))
        trace._grad_fn_strong_refs.extend(next_grad_fns)
        queue.extend(next_grad_fns)
        layer_label = layer_lookup.get(grad_fn_id)
        grad_fn_log = trace.grad_fn_logs.get(grad_fn_id)
        if grad_fn_log is None:
            grad_fn_type = _normalize_grad_fn_type(grad_fn)
            grad_fn_total_num = len(trace.grad_fn_order) + 1
            grad_fn_type_num, grad_fn_total_num, label = _grad_fn_label_parts(
                trace,
                grad_fn_type,
                layer_label,
                type_counter,
                grad_fn_total_num,
            )
            op = trace.layers[layer_label] if layer_label is not None else None
            grad_fn_log = GradFnLog(
                grad_fn_id=grad_fn_id,
                name=type(grad_fn).__name__,
                label=label,
                grad_fn_type=grad_fn_type,
                grad_fn_type_num=grad_fn_type_num,
                grad_fn_total_num=grad_fn_total_num,
                module_path=type(grad_fn).__module__,
                is_custom=_grad_fn_is_custom(grad_fn),
                has_op=layer_label is None,
                op=op,
                next_grad_fn_ids=[id(next_fn) for next_fn in next_grad_fns],
            )
            trace.grad_fn_logs[grad_fn_id] = grad_fn_log
            trace.grad_fn_order.append(grad_fn_id)
        else:
            grad_fn_log.next_grad_fn_ids = [id(next_fn) for next_fn in next_grad_fns]
        if layer_label is not None:
            layer = trace[layer_label]
            layer.grad_fn_log = grad_fn_log
            if hasattr(layer, "parent_layer_log"):
                layer.parent_layer_log.grad_fn_log = grad_fn_log
        try:
            handles.append(grad_fn.register_hook(_make_grad_fn_hook(trace, grad_fn_id)))
        except RuntimeError:
            continue
    return handles


def _clear_forward_grad_fn_refs(trace: Any) -> None:
    """Clear strong forward references to autograd nodes after hook registration.

    Parameters
    ----------
    trace:
        Trace whose OpLog and LayerLog grad_fn object refs should be
        released.
    """
    for layer in trace.layer_list:
        layer.grad_fn = None
    for layer_log in trace.layer_logs.values():
        layer_log.grad_fn = None


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
    handles = _walk_and_hook_backward_graph(trace, loss)
    backend, before = _reset_peak_memory(loss.device)
    try:
        result = backward_callable()
    finally:
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.remove()
        _clear_forward_grad_fn_refs(trace)
    _backend, after = _memory_snapshot(loss.device)
    trace.backward_memory_backend = backend
    trace.backward_peak_memory += max(0, after - before)
    trace.total_param_gradient_memory = sum(
        param_log.grad_memory for param_log in getattr(trace, "param_logs", [])
    )
    trace.backward_num_calls += 1
    return result


def _ensure_layer_grad_hooks(trace: Any) -> None:
    """Register tensor grad hooks lazily for saved graph-connected outs.

    Parameters
    ----------
    trace:
        Trace whose saved outs should receive grad hooks.
    """
    if getattr(trace, "save_grads", False):
        return
    trace.save_grads = True
    if getattr(trace, "_grad_layer_nums_to_save", None) in [None, [], "none"]:
        trace._grad_layer_nums_to_save = "all"
    for layer in getattr(trace, "layer_list", []):
        out = getattr(layer, "out", None)
        if isinstance(out, torch.Tensor):
            _add_backward_hook(trace, out, layer._label_raw)


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
        trace.save_grads = True


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
        Loss tensor whose ``grad_fn`` roots the backward graph.
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
