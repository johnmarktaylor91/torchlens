"""Executed-backward dispatch census: the ATen calls of a backward that ran.

A backward that runs through a capture that requested it (``backward_ready``,
``save_grads=`` or ``default_grad=``) executes inside :func:`backward_window`,
which records every ATen call dispatched during that backward as a value-free
:class:`DispatchedOp` (operator, shapes, dtypes, the GradFn call it ran under)
and counts it with the executed-backward rules in
:mod:`torchlens.capture.flops` (fma2, matmul and convolution families). A
capture that requests no backward never enters the window and pays nothing.

The records are session-time facts held weakly per trace, one tuple per
backward pass; a saved and reloaded trace carries none of them, only each
pass's counted total and attribution (``BackwardPass.executed_flops_record``).
Each record names the GradFn call of the node autograd reports as executing
when the call dispatched (``_backward_markers.executing_grad_fn_ref``, the
resolver the ATen recorder and the checkpoint unpack evidence share). Calls TorchLens makes
inside its own GradFn posthooks (gradient logging and the backward edits they
apply) are not recorded: they are not the model's backward. The copies its
tensor hooks make when saving gradients are recorded, as zero-arithmetic
calls that never change the count.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

from ...capture.flops import compute_dispatched_op_flops, dispatch_op_family
from ...utils._torch_compat import get_current_autograd_node_fn
from ._aten_capture import _capture_backward_aten
from ._backward_markers import _ACTIVE_BACKWARD_GRAD_FN_REFS, executing_grad_fn_ref
from ._modes import _TorchLensDispatchMode

_T = TypeVar("_T")

_CENSUS_PAUSED: ContextVar[bool] = ContextVar("torchlens_backward_census_paused", default=False)

_BACKWARD_DISPATCH: weakref.WeakKeyDictionary[Any, dict[int, tuple[DispatchedOp, ...]]]
_BACKWARD_DISPATCH = weakref.WeakKeyDictionary()
"""Per-trace ``pass_index -> recorded ops`` (session-time; dies with the trace)."""

_WindowRef = tuple["weakref.ReferenceType[Any]", int]

_ACTIVE_WINDOWS: ContextVar[tuple[_WindowRef, ...]] = ContextVar(
    "torchlens_backward_census_windows", default=()
)

_BACKWARD_PARTNERS: weakref.WeakKeyDictionary[Any, dict[int, list[_WindowRef]]]
_BACKWARD_PARTNERS = weakref.WeakKeyDictionary()
"""Per-trace ``pass_index -> (other trace, its pass)`` sharing one engine run.

Session-time and weakly held on both sides: when one backward reaches the
nodes of several captures, each capture's census window is open at once,
and each records the others so attribution can leave their nodes to them.
"""

#: Plain positional values kept on a record (conv flags, masks); never tensors.
_PLAIN_TYPES = (bool, int, float, str, type(None))


@dataclass(frozen=True, slots=True)
class DispatchedOp:
    """One ATen call dispatched during a recorded backward pass (value-free).

    Attributes
    ----------
    operator:
        Full dispatcher spelling, for example ``"aten.mm.default"``.
    input_shapes, input_dtypes:
        One entry per positional argument: the tensor's shape and dtype, or
        ``None`` where the argument is not a tensor.
    output_shapes, output_dtypes:
        The returned tensors' shapes and dtypes, in order.
    grad_fn_call_ref:
        ``(grad_fn_object_id, call_index, pass_index)`` of the recorded GradFn
        call the op ran under, or ``None`` when it ran outside every recorded
        node (the engine's seed gradient, for example).
    family:
        ``"matmul"``, ``"convolution"``, ``"zero_arithmetic"`` or
        ``"uncounted"`` under the counting convention.
    flops:
        fma2 FLOPs for a counted call, ``0`` for a layout or copy call,
        ``None`` for a call outside the counted families.
    grad_enabled:
        Whether autograd grad mode was on when the call dispatched. The
        engine runs backward kernels with grad mode off unless the pass
        builds a graph (``create_graph=True``); activation checkpointing
        replays forward kernels with grad mode on, which is how a replayed
        forward kernel is told apart from the backward around it.
    """

    operator: str
    input_shapes: tuple[tuple[int, ...] | None, ...]
    input_dtypes: tuple[str | None, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[str, ...]
    grad_fn_call_ref: tuple[int, int, int] | None
    family: str
    flops: int | None
    grad_enabled: bool = False

    @property
    def base_name(self) -> str:
        """Operator name without namespace or overload (``"mm"``)."""

        return _base_name(self.operator)


def _base_name(operator: str) -> str:
    """Strip the namespace and overload from a dispatcher spelling."""

    parts = operator.split(".")
    return parts[1] if len(parts) >= 2 else operator


def _plain_value(value: Any) -> tuple[bool, Any]:
    """Return ``(True, value)`` for a small plain value worth keeping."""

    if isinstance(value, _PLAIN_TYPES):
        return True, value
    if (
        isinstance(value, (list, tuple))
        and len(value) <= 8
        and all(isinstance(item, _PLAIN_TYPES) for item in value)
    ):
        return True, tuple(value)
    return False, None


def _output_tensors(result: Any) -> tuple[torch.Tensor, ...]:
    """Flatten the tensors a dispatcher call returned (one level deep)."""

    if isinstance(result, torch.Tensor):
        return (result,)
    if isinstance(result, (list, tuple)):
        return tuple(item for item in result if isinstance(item, torch.Tensor))
    return ()


def _make_record(
    func: Any,
    args: tuple[Any, ...],
    result: Any,
    ref: tuple[int, int, int] | None,
) -> DispatchedOp:
    """Build the value-free record of one dispatched call."""

    operator = str(func)
    base = _base_name(operator)
    shapes: list[tuple[int, ...] | None] = []
    dtypes: list[str | None] = []
    plain: list[tuple[int, object]] = []
    for position, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            shapes.append(tuple(int(size) for size in arg.shape))
            dtypes.append(str(arg.dtype))
            continue
        shapes.append(None)
        dtypes.append(None)
        keep, value = _plain_value(arg)
        if keep:
            plain.append((position, value))
    outputs = _output_tensors(result)
    output_shapes = tuple(tuple(int(size) for size in out.shape) for out in outputs)
    return DispatchedOp(
        operator=operator,
        input_shapes=tuple(shapes),
        input_dtypes=tuple(dtypes),
        output_shapes=output_shapes,
        output_dtypes=tuple(str(out.dtype) for out in outputs),
        grad_fn_call_ref=ref,
        family=dispatch_op_family(base),
        flops=compute_dispatched_op_flops(base, shapes, output_shapes, plain),
        grad_enabled=torch.is_grad_enabled(),
    )


class _BackwardCensusMode(_TorchLensDispatchMode):
    """Record every ATen call dispatched while one backward pass runs."""

    def __init__(self, sink: list[DispatchedOp], owner_id: int) -> None:
        """Store the per-pass record list and the ``id()`` of the capture it records for."""

        super().__init__()
        self._sink = sink
        self._owner_id = owner_id

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Run the call unchanged, then record it unless TorchLens's own hook made it."""

        result = func(*args, **(kwargs or {}))
        if not _CENSUS_PAUSED.get():
            ref = executing_grad_fn_ref(
                self._owner_id, resolver=get_current_autograd_node_fn(), other_captures=True
            )
            self._sink.append(_make_record(func, args, result, ref))
        return result


@contextmanager
def backward_window(trace: Any, pass_index: int) -> Iterator[None]:
    """Run one captured backward pass with its dispatch census installed.

    Enters the existing private ATen recorder bracket first (a no-op unless a
    test armed it), then the census; the census's records for ``pass_index``
    are published when the pass ends, including a pass that raised (the ops
    that did run are evidence).

    Parameters
    ----------
    trace
        Trace whose backward pass is running.
    pass_index
        One-based backward pass index.
    """

    sink: list[DispatchedOp] = []
    window = (weakref.ref(trace), int(pass_index))
    open_windows = _ACTIVE_WINDOWS.get()
    _record_partners(trace, window, open_windows)
    windows_token = _ACTIVE_WINDOWS.set((*open_windows, window))
    marker_token = _ACTIVE_BACKWARD_GRAD_FN_REFS.set(())
    try:
        with _capture_backward_aten(trace, pass_index), _BackwardCensusMode(sink, id(trace)):
            yield
    finally:
        _ACTIVE_BACKWARD_GRAD_FN_REFS.reset(marker_token)
        _ACTIVE_WINDOWS.reset(windows_token)
        _BACKWARD_DISPATCH.setdefault(trace, {})[pass_index] = tuple(sink)


def _record_partners(trace: Any, window: _WindowRef, open_windows: tuple[_WindowRef, ...]) -> None:
    """Pair this pass with every other trace's pass already open in this engine run."""

    for other_ref, other_pass in open_windows:
        other = other_ref()
        if other is None or other is trace:
            continue
        _BACKWARD_PARTNERS.setdefault(trace, {}).setdefault(window[1], []).append(
            (other_ref, other_pass)
        )
        _BACKWARD_PARTNERS.setdefault(other, {}).setdefault(other_pass, []).append(window)


def shared_backward_partners(trace: Any, pass_index: int) -> tuple[tuple[Any, int], ...]:
    """Live ``(trace, pass_index)`` pairs that shared this pass's engine run."""

    pairs = (_BACKWARD_PARTNERS.get(trace) or {}).get(int(pass_index), ())
    live = ((ref(), other_pass) for ref, other_pass in pairs)
    return tuple((other, other_pass) for other, other_pass in live if other is not None)


def outside_census(fn: Callable[..., _T], *args: Any) -> _T:
    """Call one TorchLens hook body with the census paused.

    Parameters
    ----------
    fn
        Hook body to run.
    *args
        Its positional arguments.

    Returns
    -------
    Any
        What ``fn`` returned.
    """

    token = _CENSUS_PAUSED.set(True)
    try:
        return fn(*args)
    finally:
        _CENSUS_PAUSED.reset(token)


def recorded_ops(trace: Any, pass_index: int) -> tuple[DispatchedOp, ...] | None:
    """Return one pass's recorded backward ops, or ``None`` when none were recorded.

    ``None`` means this object holds no census for the pass (a loaded trace,
    an implicit pass opened by tensor hooks, another backend); an empty tuple
    means the census ran and saw no ATen call.
    """

    by_pass = _BACKWARD_DISPATCH.get(trace)
    if by_pass is None:
        return None
    return by_pass.get(int(pass_index))


__all__ = [
    "DispatchedOp",
    "backward_window",
    "outside_census",
    "recorded_ops",
    "shared_backward_partners",
]
