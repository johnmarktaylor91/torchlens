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

Calls that TorchLens itself makes are never recorded: every wrapped torch function pauses
logging before it reaches the dispatcher, and the dispatcher-census handler re-executes the
operator it observed inside :func:`suppress_torch_ops_call_logging`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch._ops as _torch_ops

from ... import _state

_CALL_CLASS_NAMES: tuple[str, ...] = ("OpOverloadPacket", "OpOverload")
"""``torch._ops`` classes whose ``__call__`` user ``torch.ops.*`` calls flow through.

Higher-order operators (``torch.cond``, ``while_loop``) and TorchBind overloads are not
patched: they run user callables or script objects and are not plain tensor operators.
"""

_ORIGINAL_CALLS: dict[type, Callable[..., Any]] = {}
_DECORATED_BY_OP: dict[int, tuple[Any, Callable[..., Any]]] = {}
_SUPPRESS_DEPTH = 0


@contextmanager
def suppress_torch_ops_call_logging() -> Iterator[None]:
    """Run an operator call that TorchLens itself issues without recording it.

    Yields
    ------
    None
        ``torch.ops.*`` calls inside the block pass straight to the operator.
    """

    global _SUPPRESS_DEPTH
    _SUPPRESS_DEPTH += 1
    try:
        yield
    finally:
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
    qualified = str(getattr(packet, "_qualified_op_name", "op"))
    return qualified.rsplit("::", 1)[-1]


def _is_torchlens_decorated_op(op: Any) -> bool:
    """Return whether the operator's inner callable is already a TorchLens wrapper.

    Torchvision's ``torch.ops.torchvision.*`` packets get their ``_op`` replaced by a
    decorated wrapper at wrap time; those keep their existing recording path.
    """

    return id(getattr(op, "_op", None)) in _state._decorated_to_orig


def _operator_callable(op: Any, original: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Return a plain function that runs ``op`` through the class's original ``__call__``.

    The logging wrapper keeps logging enabled while it runs its callable (nested wrapped
    calls are detected by barcode), so the replay callable must bypass the patched
    ``__call__``; validation replays the op through this same function.
    """

    def call_operator(*args: Any, **kwargs: Any) -> Any:
        """Invoke the operator with the pristine ``torch._ops`` call path."""
        return original(op, *args, **kwargs)

    call_operator.__name__ = name
    call_operator.__qualname__ = name
    return call_operator


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
        ``torch_func_decorator`` wrapper whose replay callable runs ``op`` unpatched.
    """

    cached = _DECORATED_BY_OP.get(id(op))
    if cached is not None and cached[0] is op:
        return cached[1]
    from .wrappers import torch_func_decorator

    name = _recorded_op_name(op)
    decorated = torch_func_decorator(_operator_callable(op, original, name), name)
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
        if (
            not _state._logging_enabled
            or _SUPPRESS_DEPTH
            or _state._active_trace is None
            or _is_torchlens_decorated_op(self)
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

    for class_name in _CALL_CLASS_NAMES:
        cls = getattr(_torch_ops, class_name, None)
        if not isinstance(cls, type) or cls in _ORIGINAL_CALLS:
            continue
        current = cls.__dict__.get("__call__")
        if current is None or getattr(current, "__tl_torch_ops_recorder__", False):
            continue
        _ORIGINAL_CALLS[cls] = current
        setattr(cls, "__call__", _make_recording_call(current))


def uninstall_torch_ops_call_recorders() -> None:
    """Restore the original ``torch._ops`` class ``__call__`` methods.

    Returns
    -------
    None
        A class whose ``__call__`` was replaced by someone else since install is left
        as it is (only our own recorder is removed).
    """

    for cls, original in list(_ORIGINAL_CALLS.items()):
        current = cls.__dict__.get("__call__")
        if getattr(current, "__tl_torch_ops_recorder__", False):
            setattr(cls, "__call__", original)
        del _ORIGINAL_CALLS[cls]
    _DECORATED_BY_OP.clear()
