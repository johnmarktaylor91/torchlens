"""Foreign-kernel boundary for ``torch.autograd.Function`` forwards that leave torch.

A custom ``autograd.Function`` whose forward hands its tensors to another library
(DGL's message passing through ``dlpack.to_dlpack`` / ``from_dlpack``, a NumPy
round trip through ``.numpy()`` / ``torch.from_numpy``) computes outside torch: the
captured graph only sees the data leave and come back, so the result tensor has no
torch provenance and the completeness census correctly reports the hole. This module
records such a call as ONE opaque boundary op (type ``autogradfunction``,
``transform_kind="autograd.Function"``): its tensor inputs are the parents, the
returned tensors its outputs, and its callable runs the Function's own ``apply``
again so forward replay and perturbation validate through it.

Detection is dynamic and never changes an ordinary Function: the forward runs with
logging on, exactly as today, and the call collapses only when a host re-entry op
(``from_numpy`` / ``from_dlpack`` / ``frombuffer``, or a parentless ``as_tensor`` /
``asarray``) was logged inside its dynamic extent. The interior op events stay in
the append-only journal; their raw labels are removed in postprocess step 3
(``postprocess.graph_traversal._remove_foreign_kernel_interiors``) and their aten
dispatches, including an unowned one such as ``to_dlpack``'s ``as_strided``, are
re-owned to the boundary token ``autograd_function:foreign`` (an audited row of the
completeness witness), so the census counts the boundary once on each side and
nothing is exempted.
"""

from __future__ import annotations

import functools
import threading
import time
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import torch

from ... import _state
from ...data_classes.internal_types import FuncExecutionContext
from ...utils.rng import log_current_autocast_state, log_current_rng_states

FOREIGN_FUNCTION_FUNC_NAME = "autogradfunction"
"""Op type / func name of the boundary op."""

FOREIGN_FUNCTION_TRANSFORM_KIND = "autograd.Function"
"""``Op.transform_kind`` of the boundary op."""

FOREIGN_FUNCTION_WRAPPER_NAME = "autograd_function:foreign"
"""Completeness-witness owner token of every dispatch inside a collapsed Function."""

FOREIGN_FUNCTION_RECORDER_ATTR = "__tl_foreign_function_recorder__"
"""Marker set on the boundary op's replay callable."""

FOREIGN_FUNCTION_INTERIOR_ATTR = "_foreign_kernel_interior_labels"
"""Session-time Trace attribute: raw labels of interior ops to drop in postprocess."""

FOREIGN_FUNCTION_WARNING_CODE = "foreign_kernel_boundary"
"""``TorchLensWarning`` code of the once-per-Function disclosure."""

HOST_ENTRY_FUNC_NAMES = frozenset({"from_numpy", "from_dlpack", "frombuffer"})
"""Wrapped constructors that always build a tensor from host memory."""

PARENTLESS_HOST_ENTRY_FUNC_NAMES = frozenset({"as_tensor", "asarray"})
"""Constructors that build from host memory only when no captured tensor is an argument."""

_INSTALLED_APPLY: classmethod[Any, Any, Any] | None = None
_ORIGINAL_APPLY: classmethod[Any, Any, Any] | None = None


def is_foreign_function_recorder(func: Any) -> bool:
    """Return whether ``func`` is a foreign-kernel boundary replay callable.

    Parameters
    ----------
    func:
        An op's recorded callable.

    Returns
    -------
    bool
        ``True`` only for the callable built by :func:`record_function_apply`.
    """

    return getattr(func, FOREIGN_FUNCTION_RECORDER_ATTR, False) is True


def install_function_apply_wrapper() -> None:
    """Wrap ``torch.autograd.Function.apply`` idempotently.

    Returns
    -------
    None
        The base-class classmethod is replaced; subclasses overriding ``apply`` are
        untouched, and a foreign patch layered over ours is never clobbered.
    """

    global _INSTALLED_APPLY, _ORIGINAL_APPLY
    if _INSTALLED_APPLY is not None:
        return
    original = torch.autograd.Function.__dict__.get("apply")
    if not isinstance(original, classmethod):
        return
    original_function = original.__func__

    def wrapped_apply(cls: type[Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``Function.apply``, collapsing a forward that left torch into one op."""

        active_trace, logging_enabled = _state.active_capture()
        if not logging_enabled or active_trace is None:
            return original_function(cls, *args, **kwargs)
        return record_function_apply(active_trace, cls, original_function, args, kwargs)

    functools.update_wrapper(wrapped_apply, original_function)
    installed = classmethod(wrapped_apply)
    torch.autograd.Function.apply = installed  # type: ignore[method-assign]
    _INSTALLED_APPLY = installed
    _ORIGINAL_APPLY = original


def uninstall_function_apply_wrapper() -> None:
    """Restore ``torch.autograd.Function.apply`` when the slot still holds our wrapper.

    Returns
    -------
    None
        A foreign patch layered over ours is left in place.
    """

    global _INSTALLED_APPLY, _ORIGINAL_APPLY
    if _INSTALLED_APPLY is None or _ORIGINAL_APPLY is None:
        return
    if torch.autograd.Function.__dict__.get("apply") is _INSTALLED_APPLY:
        torch.autograd.Function.apply = _ORIGINAL_APPLY  # type: ignore[method-assign]
    _INSTALLED_APPLY = None
    _ORIGINAL_APPLY = None


def _active_witness_state(trace: Any) -> Any | None:
    """Return the armed completeness-witness state of ``trace``, if any."""

    from . import completeness_witness

    state = getattr(completeness_witness, "_ACTIVE_WITNESS_STATE", None)
    if state is None or state.trace is not trace:
        return None
    return state


def _host_entry_ops(events: Iterable[Any], already_collapsed: set[str]) -> tuple[str, ...]:
    """Return the distinct host re-entry func names among a frame's op events.

    Parameters
    ----------
    events:
        Op events logged inside the ``apply`` frame.
    already_collapsed:
        Raw labels owned by an inner, already collapsed Function; never counted again.

    Returns
    -------
    tuple[str, ...]
        Host re-entry func names in first-seen order; empty for an ordinary Function.
    """

    names: list[str] = []
    for event in events:
        if getattr(event, "label_raw", None) in already_collapsed:
            continue
        func_name = getattr(getattr(event, "function", None), "func_name", None)
        if not isinstance(func_name, str):
            continue
        parentless = not getattr(event, "parents", ())
        if func_name in HOST_ENTRY_FUNC_NAMES or (
            func_name in PARENTLESS_HOST_ENTRY_FUNC_NAMES and parentless
        ):
            if func_name not in names:
                names.append(func_name)
    return tuple(names)


def _make_replay_callable(
    cls: type[Any], original_function: Callable[..., Any], host_entry_ops: tuple[str, ...]
) -> Callable[..., Any]:
    """Build the boundary op's callable: the Function's own ``apply``."""

    from .wrappers import _set_transform_metadata

    def autogradfunction(*call_args: Any, **call_kwargs: Any) -> Any:
        """Run the Function through its original ``apply`` (the boundary's interior)."""

        return original_function(cls, *call_args, **call_kwargs)

    setattr(autogradfunction, FOREIGN_FUNCTION_RECORDER_ATTR, True)
    _set_transform_metadata(
        autogradfunction,
        transform_kind=FOREIGN_FUNCTION_TRANSFORM_KIND,
        tags=(FOREIGN_FUNCTION_TRANSFORM_KIND,),
        transform_config={
            "function": cls.__qualname__,
            "function_module": cls.__module__,
            "host_entry_ops": list(host_entry_ops),
        },
        inner_fn=cls.forward,
    )
    return autogradfunction


def _reown_witness_events(
    state: Any,
    start: int,
    stop: int,
    original_function: Callable[..., Any],
    func_call_id: int,
) -> None:
    """Hand every dispatch event of the frame to one accounted boundary token."""

    from .escape_detection import ExpectedOriginalToken

    token = ExpectedOriginalToken(
        raw_id=id(original_function),
        original=original_function,
        wrapper_name=FOREIGN_FUNCTION_WRAPPER_NAME,
        wrapper_frame_id=0,
        owner_thread_id=threading.get_ident(),
        pending=False,
        func_name=FOREIGN_FUNCTION_FUNC_NAME,
        func_call_id=func_call_id,
        census_scope="expected_opaque",
        capture_accounted=True,
    )
    for event in state.events[start:stop]:
        event.owner = token


def _log_boundary_op(
    trace: Any,
    replay: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out: Any,
    exec_ctx: FuncExecutionContext,
    func_call_id: int,
) -> None:
    """Log the returned tensors as the one ``autogradfunction`` op."""

    from .ops import log_function_output_tensors

    def emit() -> None:
        """Record the boundary op through the ordinary output logger."""

        log_function_output_tensors(
            trace,
            replay,
            FOREIGN_FUNCTION_FUNC_NAME,
            args,
            kwargs,
            args,
            kwargs,
            out,
            exec_ctx,
            True,
            func_call_id,
        )

    # Hide TorchLens bookkeeping dispatches only from the opt-in user-op census,
    # exactly as the torch.func transform boundary does.
    if _state._completeness_witness_mode == "shadow":
        with _state.pause_logging():
            emit()
    else:
        emit()


def _warn_foreign_boundary(trace: Any, cls: type[Any], host_entry_ops: tuple[str, ...]) -> None:
    """Disclose one collapsed Function class per capture."""

    from ..._errors import TorchLensWarning

    function_name = f"{cls.__module__}.{cls.__qualname__}"
    warned = trace.__dict__.setdefault("_foreign_kernel_warned", set())
    if function_name in warned:
        return
    warned.add(function_name)
    warnings.warn(
        TorchLensWarning(
            f"TorchLens captured the autograd.Function {function_name} as one foreign-kernel "
            f"boundary op: its forward left torch ({', '.join(host_entry_ops)}). The "
            "operations inside it are not logged; the boundary op replays by calling the "
            "Function's apply again.",
            code=FOREIGN_FUNCTION_WARNING_CODE,
            function=function_name,
        ),
        stacklevel=5,
    )


def record_function_apply(
    trace: Any,
    cls: type[Any],
    original_function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run one ``Function.apply`` under capture, collapsing it if the forward left torch.

    Parameters
    ----------
    trace:
        Active capture trace.
    cls:
        The ``autograd.Function`` subclass being applied.
    original_function:
        The unwrapped ``Function.apply`` classmethod body.
    args:
        Positional ``apply`` arguments.
    kwargs:
        Keyword ``apply`` arguments.

    Returns
    -------
    Any
        Whatever ``apply`` returned; the ordinary case records nothing extra.
    """

    op_events = getattr(getattr(trace, "capture_events", None), "op_events", None)
    journal_mark = len(op_events) if isinstance(op_events, list) else None
    witness_state = _active_witness_state(trace)
    witness_mark = len(witness_state.events) if witness_state is not None else 0
    capture_start_time = time.time()
    save_rng = getattr(trace, "save_rng_states", False)
    rng_states = log_current_rng_states(torch_only=True) if save_rng else {}
    autocast_state = log_current_autocast_state()

    out = original_function(cls, *args, **kwargs)

    if journal_mark is None or op_events is None:
        return out
    interior_events = list(op_events[journal_mark:])
    collapsed: set[str] = trace.__dict__.get(FOREIGN_FUNCTION_INTERIOR_ATTR, set())
    host_entry_ops = _host_entry_ops(interior_events, collapsed)
    if not host_entry_ops:
        return out

    from .wrappers import _collect_output_tensors

    if not _collect_output_tensors(out):
        return out
    witness_stop = len(witness_state.events) if witness_state is not None else 0
    func_call_id = _state.next_func_call_id()
    exec_ctx = FuncExecutionContext(
        time_elapsed=time.time() - capture_start_time,
        rng_states=rng_states,
        autocast_state=autocast_state,
    )
    replay = _make_replay_callable(cls, original_function, host_entry_ops)
    _log_boundary_op(trace, replay, args, kwargs, out, exec_ctx, func_call_id)
    interior_labels = {
        label
        for label in (getattr(event, "label_raw", None) for event in interior_events)
        if isinstance(label, str)
    }
    trace.__dict__.setdefault(FOREIGN_FUNCTION_INTERIOR_ATTR, set()).update(interior_labels)
    if witness_state is not None:
        _reown_witness_events(
            witness_state, witness_mark, witness_stop, original_function, func_call_id
        )
    _warn_foreign_boundary(trace, cls, host_entry_ops)
    return out
