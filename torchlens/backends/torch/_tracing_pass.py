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

Capture runs under the ``force_eager`` stance, which would also skip the
compile torch's operator entry points run (it lifts closure parameters into
operands and gives ``map`` its stacked output). That compile is let through
(``_hop_entry_compile.hop_compile_stance_lift_scope``) inside
:func:`hop_internal_compile_scope`; while Dynamo traces it, the wrappers call
the original unlogged (:func:`inside_hop_internal_compile`), and the compiled
wrapper's eager run is captured as usual. During a witnessed capture,
``map``'s entry with a grad operand steps TorchLens's dispatch modes aside for
Dynamo and runs its compiled graph under them (:func:`_map_grad_entry_modes`):
torch's eager ``map`` grad path would return one tensor per row.

Every torch-private lookup routes through ``_torch_compat`` (the tracing
tensor classes, the dispatch-mode stack and the ``FakeTensorMode`` class) or
its sibling ``_hop_entry_compile`` (the operator entry compile); an absent
probe answers "not a tracing pass", which leaves capture unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

import torch

from ... import _state
from ..._utils import _torch_compat
from ..._utils._hop_entry_compile import hop_compile_stance_lift_scope
from ..._utils._torch_compat import (
    dynamo_is_compiling,
    get_current_dispatch_mode_stack,
    get_fake_tensor_mode_class,
    get_higher_order_operator,
    get_tracing_tensor_types,
)
from ._modes import _TorchLensDispatchMode

__all__ = (
    "call_inside_tracing_pass",
    "hop_internal_compile_scope",
    "inside_hop_internal_compile",
    "inside_tracing_tensor_pass",
    "lift_hop_entry_compiles",
)

#: Depth of TorchLens-lifted higher-order-operator entry compiles in progress;
#: the wrappers' hot path tests it before anything else (zero outside them).
hop_compile_depth = 0


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


@contextmanager
def hop_internal_compile_scope() -> Iterator[None]:
    """Mark one higher-order-operator entry compile that TorchLens let run.

    Yields
    ------
    None
        While the scope is open :data:`hop_compile_depth` is above zero.
    """

    global hop_compile_depth
    hop_compile_depth += 1
    try:
        yield
    finally:
        hop_compile_depth -= 1


def inside_hop_internal_compile() -> bool:
    """Return whether Dynamo is tracing a TorchLens-lifted operator entry compile.

    Returns
    -------
    bool
        ``True`` only when :data:`hop_compile_depth` is above zero and Dynamo is
        tracing (not running) the frame. Calls then reach the wrappers with
        data-free values, so they go to the original unlogged; the compiled
        wrapper's real run that follows is logged as usual. A user
        ``torch.compile`` region never raises the depth, so it still meets the
        Dynamo-region guard.
    """

    return hop_compile_depth > 0 and dynamo_is_compiling()


def lift_hop_entry_compiles() -> AbstractContextManager[bool]:
    """Let torch's operator entry compiles run inside the capture's forced-eager scope.

    Returns
    -------
    AbstractContextManager[bool]
        Yields whether the entry was hooked; each lifted compile runs inside
        :func:`hop_internal_compile_scope`.
    """

    return hop_compile_stance_lift_scope(
        hop_internal_compile_scope,
        enabled=_torch_compat.HAS_SET_STANCE,
        steps_aside=_map_grad_entry_modes,
    )


def _map_grad_entry_modes(fn: Any, args: Any) -> tuple[Any, ...]:
    """Return the dispatch modes ``map``'s grad entry compile runs its graph under.

    Only ``map``'s entry, with grad on and a grad-requiring operand, and only when
    every mode on the stack is TorchLens's own: torch's eager ``map`` grad path
    unpacks the stacked output per row, so the witness must see the compiled
    graph's run. Every other entry keeps its modes in place (eager under them).

    Parameters
    ----------
    fn:
        The function the entry compiles (``map``'s is defined in the module of
        ``map_impl``'s class).
    args:
        Its positional arguments; ``map`` passes the body and two tensor lists.

    Returns
    -------
    tuple[Any, ...]
        The whole dispatch-mode stack, bottom first, or ``()``.
    """

    map_impl = get_higher_order_operator("map", "map_impl")
    if map_impl is None or getattr(fn, "__module__", None) != type(map_impl).__module__:
        return ()
    if not torch.is_grad_enabled() or not any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for group in args
        if isinstance(group, (list, tuple))
        for value in group
    ):
        return ()
    stack = get_current_dispatch_mode_stack()
    if not stack or not all(isinstance(mode, _TorchLensDispatchMode) for mode in stack):
        return ()
    return tuple(stack)


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
        or a ``FakeTensorMode`` is active. ``False`` whenever the tracing classes were
        never imported in this process, so ordinary capture pays one cached
        probe per logged call.
    """

    tracing_types = get_tracing_tensor_types()
    if not tracing_types:
        return False
    in_pass = any(_is_tracing_value(value, tracing_types) for value in (*args, *kwargs.values()))
    return in_pass or _fake_tensor_mode_active()


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
