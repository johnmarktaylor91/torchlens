"""Record direct ``torch.ops.*`` calls made by user code as ordinary logged ops.

A model that calls an operator through ``torch.ops`` (``torch.ops.aten.tanh(x)``, a
``torch.library.custom_op``, an operator a C++ extension registered with
``TORCH_LIBRARY``) bypasses every namespace wrapper TorchLens installs: the call goes
straight through the ``__call__`` of ``torch._ops.OpOverloadPacket`` /
``torch._ops.OpOverload`` into the dispatcher. Its outputs therefore carried no label, and
the forward graph lost the edge from the operator's tensor arguments to its outputs (a
module-boundary adoption or a source-less argument downstream).

This module patches those two ``__call__`` methods for the wrapped epoch (installed by
``wrap_torch``, restored by ``unwrap_torch``). While a capture is logging, a call routes
through :func:`~torchlens.backends.torch.wrappers.torch_func_decorator` with the operator
object itself as the replay callable, so the op is recorded with parents from its tensor
arguments exactly like any wrapped torch function, and validation replays it by calling
the operator again. Outside a logging window the patch is one bool read.

Calls that TorchLens or torch itself makes are never recorded: a ``torch.ops`` call inside
a wrapped torch function's original (a decomposition, a custom op's body) is detected by a
frame walk, the dispatcher-census handler re-executes the operator it observed inside
:func:`enter_suppressed_region` / :func:`exit_suppressed_region`, and the namespaces in
``_UNRECORDED_NAMESPACES`` (collectives, quantized kernels, profiler markers) keep their
dedicated paths.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from types import CodeType, FrameType
from typing import Any

import torch._ops as _torch_ops

from ... import _state

_CALL_CLASS_NAMES: tuple[str, ...] = ("OpOverloadPacket", "OpOverload")
"""``torch._ops`` classes whose ``__call__`` user ``torch.ops.*`` calls flow through.

Higher-order operators (``torch.cond``, ``while_loop``) and TorchBind overloads are not
patched: they run user callables or script objects and are not plain tensor operators.
"""

_UNRECORDED_NAMESPACES: frozenset[str] = frozenset(
    {
        # Collectives: replaying one during validation would re-enter the process group;
        # funcol keeps its own armed boundary path (``funcol.py``).
        "_c10d_functional",
        "_c10d_functional_autograd",
        "c10d",
        "c10d_functional",
        "_dtensor",
        # Quantized kernels: model preparation adopts quantized modules as
        # ``quantized_*`` boundaries with their own FLOP estimates.
        "quantized",
        # Profiler range markers return script objects, never tensors.
        "profiler",
    }
)
"""Operator namespaces whose direct calls keep their dedicated (unrecorded) path."""

_CALL_ATTR = "__call__"
_ORIGINAL_CALLS: dict[type, Callable[..., Any]] = {}
_DECORATED_BY_OP: dict[int, tuple[Any, Callable[..., Any]]] = {}
_WRAPPED_FUNC_CODE: list[CodeType] = []
_SUPPRESS_DEPTH = 0


def enter_suppressed_region() -> None:
    """Begin a region whose ``torch.ops`` calls TorchLens itself issues (census redispatch).

    Paired with :func:`exit_suppressed_region` in the caller's own ``try``/``finally`` so the
    redispatch frame stays the innermost TorchLens frame for failure classification.
    """

    global _SUPPRESS_DEPTH
    _SUPPRESS_DEPTH += 1


def exit_suppressed_region() -> None:
    """End a region opened by :func:`enter_suppressed_region`."""

    global _SUPPRESS_DEPTH
    _SUPPRESS_DEPTH -= 1


def _recorded_op_name(op: Any) -> str:
    """Return the TorchLens function name for a ``torch.ops`` operator object.

    Parameters
    ----------
    op:
        ``OpOverloadPacket`` or ``OpOverload`` being called.

    Returns
    -------
    str
        The operator's unqualified name (``"tanh"`` for ``aten::tanh.default``), the same
        name its namespace twin records.
    """

    packet = getattr(op, "_overloadpacket", op)
    name = getattr(packet, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return _qualified_name(op).rsplit("::", 1)[-1]


def _qualified_name(op: Any) -> str:
    """Return ``namespace::name`` for an operator object (empty when unknown)."""

    packet = getattr(op, "_overloadpacket", op)
    return str(getattr(packet, "_qualified_op_name", ""))


def _keeps_dedicated_path(op: Any) -> bool:
    """Return whether this operator must not be recorded by the ``torch.ops`` recorder.

    Torchvision packets whose ``_op`` is already a TorchLens wrapper keep that path, and
    the namespaces in ``_UNRECORDED_NAMESPACES`` keep their dedicated handling.
    """

    decorated_to_orig, _ = _state.wrap_epoch_ledgers()
    if id(getattr(op, "_op", None)) in decorated_to_orig:
        return True
    return _qualified_name(op).split("::", 1)[0] in _UNRECORDED_NAMESPACES


def _inside_wrapped_torch_call() -> bool:
    """Return whether the caller runs inside a wrapped torch function's original call.

    A wrapped torch function keeps logging enabled while its original runs (nested calls
    are resolved by barcode), so torch's own Python code calling ``torch.ops`` there (a
    decomposition, ``broadcast_in_dim``, a custom op's body) must not become a recorded op.
    Every ``torch_func_decorator`` wrapper shares one code object, so one frame walk
    answers it; only direct ``torch.ops`` calls made while a capture is logging pay it.
    """

    if not _WRAPPED_FUNC_CODE:
        return False
    wrapped_code = _WRAPPED_FUNC_CODE[0]
    frame: FrameType | None = sys._getframe(2)
    while frame is not None:
        if frame.f_code is wrapped_code:
            return True
        frame = frame.f_back
    return False


def _decorated_for(op: Any, original: Callable[..., Any]) -> Callable[..., Any]:
    """Return the cached logging wrapper for one operator object.

    Parameters
    ----------
    op:
        ``OpOverloadPacket`` or ``OpOverload`` being called.
    original:
        The class's original ``__call__``.

    Returns
    -------
    Callable[..., Any]
        ``torch_func_decorator`` wrapper whose callable runs ``op`` through the unpatched
        ``__call__``. It is a C ``functools.partial`` (no TorchLens frame between the
        wrapper's trampoline and the operator), so a failing operator classifies as the
        user's op, and validation replays the op through it.
    """

    cached = _DECORATED_BY_OP.get(id(op))
    if cached is not None and cached[0] is op:
        return cached[1]
    from .wrappers import torch_func_decorator

    name = _recorded_op_name(op)
    call_operator = functools.partial(original, op)
    call_operator.__name__ = name  # type: ignore[attr-defined]
    call_operator.__qualname__ = name  # type: ignore[attr-defined]
    decorated = torch_func_decorator(call_operator, name)
    _DECORATED_BY_OP[id(op)] = (op, decorated)
    return decorated


def _make_recording_call(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a ``torch._ops`` class ``__call__`` so logged user calls are recorded.

    Parameters
    ----------
    original:
        The class's own ``__call__``.

    Returns
    -------
    Callable[..., Any]
        Patched ``__call__``.
    """

    @functools.wraps(original)
    def _recording_call(self: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Record a user ``torch.ops`` call during capture, else call through."""
        trace, enabled = _state.active_capture()
        if (
            not enabled
            or trace is None
            or _SUPPRESS_DEPTH
            or _keeps_dedicated_path(self)
            or _inside_wrapped_torch_call()
        ):
            return original(self, *args, **kwargs)
        return _decorated_for(self, original)(*args, **kwargs)

    _recording_call.__tl_torch_ops_recorder__ = True  # type: ignore[attr-defined]
    return _recording_call


def install_torch_ops_call_recorders() -> None:
    """Patch the ``torch._ops`` call classes for the wrapped epoch (idempotent).

    Returns
    -------
    None
        Class ``__call__`` attributes are replaced in place.
    """

    if not _WRAPPED_FUNC_CODE:
        from .wrappers import torch_func_decorator

        _WRAPPED_FUNC_CODE.append(torch_func_decorator(len, "tl_torch_ops_probe").__code__)
    for class_name in _CALL_CLASS_NAMES:
        cls = getattr(_torch_ops, class_name, None)
        if not isinstance(cls, type) or cls in _ORIGINAL_CALLS:
            continue
        current = cls.__dict__.get(_CALL_ATTR)
        if current is None or getattr(current, "__tl_torch_ops_recorder__", False):
            continue
        _ORIGINAL_CALLS[cls] = current
        setattr(cls, _CALL_ATTR, _make_recording_call(current))


def uninstall_torch_ops_call_recorders() -> None:
    """Restore the original ``torch._ops`` class ``__call__`` methods.

    Returns
    -------
    None
        A class whose ``__call__`` was replaced by someone else since install is left
        as it is (only our own recorder is removed).
    """

    for cls, original in list(_ORIGINAL_CALLS.items()):
        current = cls.__dict__.get(_CALL_ATTR)
        if getattr(current, "__tl_torch_ops_recorder__", False):
            setattr(cls, _CALL_ATTR, original)
        del _ORIGINAL_CALLS[cls]
    _DECORATED_BY_OP.clear()
