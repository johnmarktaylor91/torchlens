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

Every torch-private lookup routes through ``_torch_compat`` (the tracing
tensor classes, the dispatch-mode stack and the ``FakeTensorMode`` class); an
absent probe answers "not a tracing pass", which leaves capture unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ... import _state
from ..._utils._torch_compat import (
    get_current_dispatch_mode_stack,
    get_fake_tensor_mode_class,
    get_tracing_tensor_types,
)

__all__ = ["call_inside_tracing_pass", "inside_tracing_tensor_pass"]


def _is_tracing_value(value: Any, tracing_types: tuple[type[Any], ...]) -> bool:
    """Return whether one call argument belongs to a fake-tensor tracing pass.

    Parameters
    ----------
    value:
        A positional or keyword argument of the decorated call.
    tracing_types:
        The data-free tracing tensor classes of this torch build.

    Returns
    -------
    bool
        ``True`` for a tracing tensor, a tracing tensor class (the ``cls``
        argument of ``Tensor._make_subclass``), or a list or tuple holding a
        tracing tensor.
    """

    if isinstance(value, tracing_types):
        return True
    if isinstance(value, type):
        return issubclass(value, tracing_types)
    if isinstance(value, (list, tuple)):
        return any(isinstance(item, tracing_types) for item in value)
    return False


def _fake_tensor_mode_active() -> bool:
    """Return whether a ``FakeTensorMode`` is on the current dispatch-mode stack.

    Returns
    -------
    bool
        ``True`` while torch runs code under fake-tensor tracing, which covers
        factory calls (``torch.zeros``) that take no tensor argument.
    """

    fake_mode_cls = get_fake_tensor_mode_class()
    stack = get_current_dispatch_mode_stack()
    if fake_mode_cls is None or not stack:
        return False
    return any(isinstance(mode, fake_mode_cls) for mode in stack)


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
        or a ``FakeTensorMode`` is active. ``False`` whenever the tracing
        classes were never imported in this process, so ordinary capture pays
        one cached probe per logged call.
    """

    tracing_types = get_tracing_tensor_types()
    if not tracing_types:
        return False
    for value in (*args, *kwargs.values()):
        if _is_tracing_value(value, tracing_types):
            return True
    return _fake_tensor_mode_active()


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
