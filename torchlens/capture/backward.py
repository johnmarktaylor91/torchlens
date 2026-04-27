"""Backward-pass graph walking, grad_fn hooks, and ModelLog APIs."""

from __future__ import annotations

import contextlib
import time
from collections import deque
from typing import Any, Callable, Iterator

import torch

from .._state import pause_logging
from ..data_classes.backward_log import BackwardLog, GradFnLog
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


def _selected_for_gradient_save(model_log: Any, layer_label: str | None) -> bool:
    """Return whether a forward layer's gradient should be saved.

    Parameters
    ----------
    model_log:
        ModelLog being updated.
    layer_label:
        Final layer label, or ``None`` for intervening grad_fns.

    Returns
    -------
    bool
        True if this layer is selected by ``gradients_to_save``.
    """
    if layer_label is None:
        return False
    selection = getattr(model_log, "_gradient_layer_nums_to_save", "all")
    if selection == "all":
        return True
    if selection in [None, "none", []]:
        return False
    return model_log[layer_label].creation_order in selection


def _make_grad_fn_hook(model_log: Any, grad_fn_id: int) -> Callable[..., None]:
    """Build a runtime hook for one autograd grad_fn.

    Parameters
    ----------
    model_log:
        ModelLog whose BackwardLog should receive runtime data.
    grad_fn_id:
        ``id()`` of the hooked grad_fn.

    Returns
    -------
    Callable[..., None]
        Hook compatible with ``grad_fn.register_hook``.
    """

    def hook(*hook_args: Any) -> None:
        grad_fn_log = model_log.backward.grad_fn_logs.get(grad_fn_id)
        if grad_fn_log is None:
            return
        grad_inputs = hook_args[0] if len(hook_args) >= 1 else None
        grad_outputs = hook_args[1] if len(hook_args) >= 2 else None
        if not _selected_for_gradient_save(model_log, grad_fn_log.corresponding_layer):
            grad_inputs = None
            grad_outputs = None
        with pause_logging():
            grad_fn_log.log_pass(grad_inputs, grad_outputs, time.time())

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


def _layer_by_grad_fn_id(model_log: Any) -> dict[int, str]:
    """Build a mapping from forward grad_fn identity to final layer label.

    Parameters
    ----------
    model_log:
        ModelLog whose layers were captured during the forward pass.

    Returns
    -------
    dict[int, str]
        Mapping from ``id(grad_fn)`` to final layer label.
    """
    mapping = {}
    for layer in model_log.layer_list:
        grad_fn_id = getattr(layer, "grad_fn_id", None)
        if grad_fn_id is not None:
            mapping[grad_fn_id] = layer.layer_label
    return mapping


def _walk_and_hook_backward_graph(model_log: Any, loss: torch.Tensor) -> list[Any]:
    """Walk ``loss.grad_fn`` and register hooks on every reachable grad_fn.

    Parameters
    ----------
    model_log:
        ModelLog that owns the BackwardLog.
    loss:
        Scalar or tensor loss whose backward graph should be captured.

    Returns
    -------
    list[Any]
        Hook handles to remove after backward finishes.
    """
    if loss.grad_fn is None:
        raise ValueError("log_backward requires a loss tensor with a grad_fn.")

    if not isinstance(getattr(model_log, "backward", None), BackwardLog):
        model_log.backward = BackwardLog()

    layer_lookup = _layer_by_grad_fn_id(model_log)
    queue: deque[Any] = deque([loss.grad_fn])
    seen: set[int] = set()
    handles: list[Any] = []
    if model_log.backward.root_grad_fn_id is None:
        model_log.backward.root_grad_fn_id = id(loss.grad_fn)

    while queue:
        grad_fn = queue.popleft()
        grad_fn_id = id(grad_fn)
        if grad_fn_id in seen:
            continue
        seen.add(grad_fn_id)
        next_grad_fns = list(_iter_next_grad_fns(grad_fn))
        queue.extend(next_grad_fns)
        layer_label = layer_lookup.get(grad_fn_id)
        grad_fn_log = model_log.backward.grad_fn_logs.get(grad_fn_id)
        if grad_fn_log is None:
            grad_fn_log = GradFnLog(
                grad_fn_id=grad_fn_id,
                name=type(grad_fn).__name__,
                module_path=type(grad_fn).__module__,
                is_custom=_grad_fn_is_custom(grad_fn),
                is_intervening=layer_label is None,
                corresponding_layer=layer_label,
                next_grad_fn_ids=[id(next_fn) for next_fn in next_grad_fns],
            )
            model_log.backward.grad_fn_logs[grad_fn_id] = grad_fn_log
            model_log.backward.grad_fn_order.append(grad_fn_id)
        else:
            grad_fn_log.next_grad_fn_ids = [id(next_fn) for next_fn in next_grad_fns]
        if layer_label is not None:
            layer = model_log[layer_label]
            layer.corresponding_grad_fn = grad_fn_log
            if hasattr(layer, "parent_layer_log"):
                layer.parent_layer_log.corresponding_grad_fn = grad_fn_log
        try:
            handles.append(grad_fn.register_hook(_make_grad_fn_hook(model_log, grad_fn_id)))
        except RuntimeError:
            continue
    return handles


def _clear_forward_grad_fn_refs(model_log: Any) -> None:
    """Clear strong forward references to autograd nodes after hook registration.

    Parameters
    ----------
    model_log:
        ModelLog whose LayerPassLog and LayerLog grad_fn object refs should be
        released.
    """
    for layer in model_log.layer_list:
        layer.grad_fn_object = None
    for layer_log in model_log.layer_logs.values():
        layer_log.grad_fn_object = None


def _run_backward_with_capture(
    model_log: Any,
    loss: torch.Tensor,
    backward_callable: Callable[[], Any],
) -> Any:
    """Capture a backward graph, run backward, and record memory delta.

    Parameters
    ----------
    model_log:
        ModelLog to mutate.
    loss:
        Loss tensor that roots the backward graph.
    backward_callable:
        Zero-argument callable that performs the actual backward pass.

    Returns
    -------
    Any
        Return value from ``backward_callable``.
    """
    handles = _walk_and_hook_backward_graph(model_log, loss)
    backend, before = _reset_peak_memory(loss.device)
    try:
        result = backward_callable()
    finally:
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.remove()
        _clear_forward_grad_fn_refs(model_log)
    _backend, after = _memory_snapshot(loss.device)
    model_log.backward.memory_backend = backend
    model_log.backward.peak_memory_bytes += max(0, after - before)
    model_log.backward.num_passes += 1
    return result


def _ensure_layer_gradient_hooks(model_log: Any) -> None:
    """Register tensor gradient hooks lazily for saved graph-connected activations.

    Parameters
    ----------
    model_log:
        ModelLog whose saved activations should receive gradient hooks.
    """
    if getattr(model_log, "save_gradients", False):
        return
    model_log.save_gradients = True
    if getattr(model_log, "_gradient_layer_nums_to_save", None) in [None, [], "none"]:
        model_log._gradient_layer_nums_to_save = "all"
    for layer in getattr(model_log, "layer_list", []):
        activation = getattr(layer, "activation", None)
        if isinstance(activation, torch.Tensor):
            _add_backward_hook(model_log, activation, layer.tensor_label_raw)


def log_backward(self: Any, loss: torch.Tensor, **backward_kwargs: Any) -> Any:
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
        The same ModelLog, for chaining.
    """
    _ensure_layer_gradient_hooks(self)

    def run() -> Any:
        """Run the user's requested backward call."""
        return loss.backward(**backward_kwargs)

    _run_backward_with_capture(self, loss, run)
    return self


class RecordingBackward:
    """Context manager that captures every ``Tensor.backward`` call inside it."""

    def __init__(self, model_log: Any) -> None:
        self.model_log = model_log
        self._original_backward: Callable[..., Any] | None = None

    def __enter__(self) -> "RecordingBackward":
        """Patch ``torch.Tensor.backward`` and return this context object."""
        _ensure_layer_gradient_hooks(self.model_log)
        self._original_backward = torch.Tensor.backward
        model_log = self.model_log
        original_backward = self._original_backward

        def wrapped_backward(tensor_self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            """Capture the graph rooted at ``tensor_self`` before backward runs."""

            def run() -> Any:
                """Run the original Tensor.backward implementation."""
                return original_backward(tensor_self, *args, **kwargs)

            return _run_backward_with_capture(model_log, tensor_self, run)

        torch.Tensor.backward = wrapped_backward  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Restore ``torch.Tensor.backward``."""
        if self._original_backward is not None:
            torch.Tensor.backward = self._original_backward  # type: ignore[assignment]


def recording_backward(self: Any) -> RecordingBackward:
    """Return a context manager that records user-managed backward calls.

    Returns
    -------
    RecordingBackward
        Context manager that patches ``Tensor.backward`` inside the block.
    """
    return RecordingBackward(self)
