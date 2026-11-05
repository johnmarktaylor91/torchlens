"""Recognise torch's own fake-tensor tracing passes so the wrappers stay out of them.

With grad-requiring operands, the Autograd kernel of a higher-order operator
(``cond``, ``while_loop``, ``map``, ``scan``) first traces its branch or body
with data-free ``FakeTensor`` inputs (``materialize_as_graph`` under
``make_fx``) to build the backward schema; the eager run that follows is what
actually executes. The decorated torch callables are called during that pass:
once to build each fake input (``Tensor._make_subclass(FakeTensor, ...)``,
before the instance has its ``fake_mode``) and once per branch op under an
active ``FakeTensorMode``. Logging those calls records data-free activations,
or crashes reading ``.device`` of a half-built ``FakeTensor``. The capture
wrapper therefore passes such calls through unlogged; only the eager run is
recorded, exactly as when no operand requires grad.

``map`` is the exception. Capture runs under the ``force_eager`` stance, which
skips the compile a grad-requiring ``map`` normally runs, and torch's eager
``map`` autograd then returns one tensor per row instead of the stacked tensor.
Passing its tracing pass through would turn the old crash into a silently
different result, so that pass is left to fail as before.

Every torch-private lookup routes through ``_torch_compat`` (the tracing
tensor classes, the dispatch-mode stack and the ``FakeTensorMode`` class); an
absent probe answers "not a tracing pass", which leaves capture unchanged.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import FrameType
from typing import Any

from ... import _state
from ..._utils._torch_compat import (
    get_current_dispatch_mode_stack,
    get_fake_tensor_mode_class,
    get_higher_order_operator,
    get_tracing_tensor_types,
)

__all__ = ("call_inside_tracing_pass", "inside_tracing_tensor_pass")


def _fake_tensor_mode_active(stack: list[Any]) -> bool:
    """Return whether a ``FakeTensorMode`` is on the current dispatch-mode stack.

    Parameters
    ----------
    stack:
        The non-empty current dispatch-mode stack.

    Returns
    -------
    bool
        ``True`` while torch runs code under fake-tensor tracing, which covers
        factory calls (``torch.zeros``) that take no tensor argument.
    """

    fake_mode_cls = get_fake_tensor_mode_class()
    if fake_mode_cls is None:
        return False
    return any(isinstance(mode, fake_mode_cls) for mode in stack)


def _inside_unsupported_grad_operator() -> bool:
    """Return whether the running tracing pass belongs to a grad-requiring ``map``.

    Returns
    -------
    bool
        ``True`` when a frame of the module defining ``map_impl`` is on the
        Python stack: the ``map`` entry point compiling its body, the operator's
        autograd kernel, or its eager kernel. Its eager autograd result differs
        from the untraced call under the capture stance (see the module
        docstring), so that pass is not passed through.
    """

    map_impl = get_higher_order_operator("map", "map_impl")
    if map_impl is None:
        return False
    map_module = type(map_impl).__module__
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        if frame.f_globals.get("__name__") == map_module:
            return True
        frame = frame.f_back
    return False


def inside_tracing_tensor_pass(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Return whether a decorated call runs inside torch's fake-tensor tracing.

    Parameters
    ----------
    args:
        Positional arguments of the decorated call.
    kwargs:
        Keyword arguments of the decorated call.

    Returns
    -------
    bool
        ``True`` when an argument is a tracing tensor or tracing tensor class,
        or a ``FakeTensorMode`` is active, unless the pass belongs to a
        grad-requiring ``map``. ``False`` whenever the tracing classes were
        never imported in this process, so ordinary capture pays one cached
        probe per logged call.
    """

    tracing_types = get_tracing_tensor_types()
    if not tracing_types:
        return False
    # Hot path (CC above 10 on purpose): this runs on every decorated call, so the
    # argument scan is inline (no helper call or generator step per argument): a
    # tracing tensor, a tracing tensor class (the ``cls`` argument of
    # ``Tensor._make_subclass``), or a list or tuple holding a tracing tensor. The
    # dispatch-mode stack, empty in ordinary capture, is read before the
    # FakeTensorMode class.
    for value in (*args, *kwargs.values()):
        if isinstance(value, tracing_types):
            return not _inside_unsupported_grad_operator()
        if isinstance(value, type):
            if issubclass(value, tracing_types):
                return not _inside_unsupported_grad_operator()
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, tracing_types):
                    return not _inside_unsupported_grad_operator()
    stack = get_current_dispatch_mode_stack()
    if not stack or not _fake_tensor_mode_active(stack):
        return False
    return not _inside_unsupported_grad_operator()


def call_inside_tracing_pass(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Run a call of torch's fake-tensor tracing pass through unlogged.

    Parameters
    ----------
    func:
        The original torch callable.
    args:
        Positional arguments of the call.
    kwargs:
        Keyword arguments of the call.

    Returns
    -------
    Any
        Whatever ``func`` returns. Under the shadow completeness witness the
        call runs paused, as the Dynamo-region pass-through does, so the
        witness does not count the tracing pass's dispatches as dropped ops.
    """

    if _state._completeness_witness_mode == "shadow":
        with _state.pause_logging():
            return func(*args, **kwargs)
    return func(*args, **kwargs)
