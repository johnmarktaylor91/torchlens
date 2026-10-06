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

An operator whose schema writes its first argument and returns nothing (a
``torch.library.custom_op`` with ``mutates_args``) is recorded as an in-place op on that
argument: the replay callable returns the mutated argument, so later reads of it come
from the op. A mutating operator returning nothing that cannot be recorded that way
(another argument written, a list receiver, an unreadable schema) is disclosed as a
``source_provenance`` gap instead of vanishing from the graph.

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
import warnings
from collections.abc import Callable
from types import CodeType, FrameType
from typing import Any

import torch
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
_DECORATED_BY_OP: dict[int, tuple[Any, Callable[..., Any], str]] = {}

# Schema mutation classes (``_mutation_kind``).
_MUTATION_NONE = "none"
"""Writes no argument, or returns its outputs (in-place and ``out=`` ops return them)."""
_MUTATION_RECEIVER = "receiver"
"""Writes exactly its first argument and returns nothing: recorded in place on it."""
_MUTATION_UNRECORDABLE = "unrecordable"
"""Writes some other argument (or a mix across overloads) and returns nothing."""
_MUTATION_UNKNOWN = "unknown"
"""No readable schema: a call that returns nothing is disclosed, never trusted."""
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


def _overload_schemas(op: Any) -> list[Any] | None:
    """Return the ``FunctionSchema`` of an overload, or of every overload of a packet.

    Parameters
    ----------
    op:
        ``OpOverload`` or ``OpOverloadPacket``.

    Returns
    -------
    list[Any] | None
        The schemas, or ``None`` when any of them cannot be read.
    """

    schema = getattr(op, "_schema", None)
    if schema is not None:
        return [schema]
    overloads = getattr(op, "overloads", None)
    if not callable(overloads):
        return None
    schemas = [getattr(getattr(op, name, None), "_schema", None) for name in overloads()]
    if not schemas or any(schema is None for schema in schemas):
        return None
    return schemas


def _schema_mutation(schema: Any) -> str:
    """Classify one schema's argument writes (``alias_info.is_write``, torch ground truth)."""

    if getattr(schema, "returns", None):
        return _MUTATION_NONE
    written = [
        index
        for index, argument in enumerate(getattr(schema, "arguments", None) or ())
        if getattr(getattr(argument, "alias_info", None), "is_write", False)
    ]
    if not written:
        return _MUTATION_NONE
    return _MUTATION_RECEIVER if written == [0] else _MUTATION_UNRECORDABLE


def _mutation_kind(op: Any) -> str:
    """Return how an operator's schema writes its arguments (one of the ``_MUTATION_*``).

    Parameters
    ----------
    op:
        ``OpOverload`` or ``OpOverloadPacket`` being called.

    Returns
    -------
    str
        The shared class of every overload; overloads that disagree are unrecordable.
    """

    schemas = _overload_schemas(op)
    if schemas is None:
        return _MUTATION_UNKNOWN
    kinds = {_schema_mutation(schema) for schema in schemas}
    return kinds.pop() if len(kinds) == 1 else _MUTATION_UNRECORDABLE


def _returning_mutated_receiver(call_operator: Callable[..., Any]) -> Callable[..., Any]:
    """Return a replay callable that runs the operator and returns its mutated first argument.

    Parameters
    ----------
    call_operator:
        The operator call (``functools.partial`` of the original ``__call__``).

    Returns
    -------
    Callable[..., Any]
        Callable whose output is ``args[0]``, the tensor the operator wrote, so the
        wrapper records it as an in-place op and validation replays it the same way.
    """

    def _call_returning_receiver(*args: Any, **kwargs: Any) -> Any:
        """Run the operator, then return the argument it mutated."""
        call_operator(*args, **kwargs)
        return args[0]

    _call_returning_receiver.__name__ = call_operator.__name__  # type: ignore[attr-defined]
    _call_returning_receiver.__qualname__ = call_operator.__name__  # type: ignore[attr-defined]
    return _call_returning_receiver


def _disclose_unrecorded_mutation(trace: Any, op: Any) -> None:
    """Persist a ``source_provenance`` gap for a mutating operator call left out of the graph.

    Parameters
    ----------
    trace:
        Active capture trace.
    op:
        The operator that wrote its arguments and returned nothing.
    """

    from ..._capture_honesty import (
        ADVISORY_UNRECORDED_OPERATOR_MUTATION,
        append_capture_advisory,
    )

    entry = (
        f"{_qualified_name(op) or _recorded_op_name(op)} wrote its arguments and returned "
        "no tensor; the mutation is not in the graph"
    )
    append_capture_advisory(trace, ADVISORY_UNRECORDED_OPERATOR_MUTATION, [entry])
    warnings.warn(f"TorchLens could not record a mutating operator call: {entry}.", stacklevel=3)


def _decorated_for(op: Any, original: Callable[..., Any]) -> tuple[Callable[..., Any], str]:
    """Return the cached logging wrapper for one operator object.

    Parameters
    ----------
    op:
        ``OpOverloadPacket`` or ``OpOverload`` being called.
    original:
        The class's original ``__call__``.

    Returns
    -------
    tuple[Callable[..., Any], str]
        ``torch_func_decorator`` wrapper whose callable runs ``op`` through the unpatched
        ``__call__``, and the operator's ``_MUTATION_*`` class. The callable is a C
        ``functools.partial`` (no TorchLens frame between the wrapper's trampoline and the
        operator), so a failing operator classifies as the user's op, and validation
        replays the op through it. For a receiver-mutating operator it is
        :func:`_returning_mutated_receiver` around that partial, recorded in place.
    """

    cached = _DECORATED_BY_OP.get(id(op))
    if cached is not None and cached[0] is op:
        return cached[1], cached[2]
    from .wrappers import torch_func_decorator

    name = _recorded_op_name(op)
    call_operator = functools.partial(original, op)
    call_operator.__name__ = name  # type: ignore[attr-defined]
    call_operator.__qualname__ = name  # type: ignore[attr-defined]
    mutation = _mutation_kind(op)
    if mutation == _MUTATION_RECEIVER:
        decorated = torch_func_decorator(
            _returning_mutated_receiver(call_operator), name, mutates_first_arg=True
        )
    else:
        decorated = torch_func_decorator(call_operator, name)
    _DECORATED_BY_OP[id(op)] = (op, decorated, mutation)
    return decorated, mutation


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
        decorated, mutation = _decorated_for(self, original)
        if mutation == _MUTATION_NONE:
            return decorated(*args, **kwargs)
        if mutation == _MUTATION_RECEIVER and args and isinstance(args[0], torch.Tensor):
            decorated(*args, **kwargs)
            return None
        if mutation == _MUTATION_UNKNOWN:
            result = decorated(*args, **kwargs)
            if result is None:
                _disclose_unrecorded_mutation(trace, self)
            return result
        # Writes another argument, a list receiver, or a receiver passed by keyword.
        _disclose_unrecorded_mutation(trace, self)
        return original(self, *args, **kwargs)

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
