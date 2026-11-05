"""Optimizer step hooks that still fire after torch 2.1's Dynamo strips them.

TorchLens's watch session, history collector and debug session read the
optimizer boundary through torch's own ``register_step_pre_hook`` and
``register_step_post_hook``. Torch runs those hooks from a wrapper on each
optimizer class; on torch 2.1 the first ``torch.compile`` in a process removes
that wrapper from every optimizer class, so the hooks never fire again and a
session saw no optimizer boundary (see ``stripped_step_hook_wrapper``).

:func:`register_step_hooks` registers through torch's API and, only when the
class step lost the wrapper, binds torch's own wrapper on the optimizer
instance so every registered hook (TorchLens's and the user's) fires once per
step in torch's order. The binding is shared by every TorchLens hold on one
optimizer and removed with the last one, restoring the instance as it was
found. When something else rebinds ``optimizer.step`` over the binding while
it is held (an LR scheduler created during the session wraps it), the binding
is left in place on release, because removing it would drop that wrapper too.
"""

from __future__ import annotations

import types
import weakref
from collections.abc import Callable
from typing import Any

from ._torch_compat import stripped_step_hook_wrapper

#: Optimizer -> [hold count, bound wrapper, instance ``step`` found at bind time or None].
_BINDINGS: weakref.WeakKeyDictionary[Any, list[Any]] = weakref.WeakKeyDictionary()


class _BindingHold:
    """One hold on an optimizer's instance-bound hook wrapper; ``remove()`` releases it."""

    def __init__(self, optimizer: Any) -> None:
        """Remember the optimizer weakly so a hold never keeps it alive."""

        self._optimizer: weakref.ReferenceType[Any] | None = weakref.ref(optimizer)

    def remove(self) -> None:
        """Release this hold (idempotent); the last hold removes the binding."""

        optimizer = self._optimizer() if self._optimizer is not None else None
        self._optimizer = None
        if optimizer is None or optimizer not in _BINDINGS:
            return
        binding = _BINDINGS[optimizer]
        binding[0] -= 1
        if binding[0] > 0:
            return
        del _BINDINGS[optimizer]
        if vars(optimizer).get("step") is not binding[1]:
            return
        if binding[2] is None:
            del optimizer.step
        else:
            optimizer.step = binding[2]


def _hold_binding(optimizer: Any) -> _BindingHold | None:
    """Bind torch's hook wrapper on ``optimizer`` when its class step lost it.

    Returns
    -------
    _BindingHold | None
        A hold to release on close, or ``None`` when the class step runs the
        hooks itself and nothing was bound.
    """

    if optimizer in _BINDINGS:
        _BINDINGS[optimizer][0] += 1
        return _BindingHold(optimizer)
    wrap = stripped_step_hook_wrapper(optimizer)
    if wrap is None:
        return None
    found = vars(optimizer).get("step")
    if found is None:
        target: Callable[..., Any] = type(optimizer).step
    else:

        def target(_optimizer: Any, *args: Any, **kwargs: Any) -> Any:
            """Call the instance ``step`` found at bind time (already bound)."""

            return found(*args, **kwargs)

    bound = types.MethodType(wrap(target), optimizer)
    optimizer.step = bound
    _BINDINGS[optimizer] = [1, bound, found]
    return _BindingHold(optimizer)


def register_step_hooks(
    optimizer: Any,
    pre: Callable[..., Any] | None = None,
    post: Callable[..., Any] | None = None,
) -> list[Any]:
    """Register step pre/post hooks that fire even after torch 2.1 stripped them.

    Parameters
    ----------
    optimizer:
        The ``torch.optim.Optimizer`` to hook.
    pre:
        Hook for ``register_step_pre_hook``, or ``None``.
    post:
        Hook for ``register_step_post_hook``, or ``None``.

    Returns
    -------
    list[Any]
        Handles whose ``remove()`` detaches everything this call added, in the
        order to remove them.
    """

    handles: list[Any] = []
    if pre is not None:
        handles.append(optimizer.register_step_pre_hook(pre))
    if post is not None:
        handles.append(optimizer.register_step_post_hook(post))
    hold = _hold_binding(optimizer)
    if hold is not None:
        handles.append(hold)
    return handles


def hook_around(
    optimizer: Any,
    pre: Callable[..., Any],
    attach: Callable[[], object],
    post: Callable[..., Any],
) -> list[Any]:
    """Register ``pre``, run ``attach``, then register ``post``, all or nothing.

    The hooks fire in registration order, so hooks ``attach`` registers run
    between ``pre`` and ``post``. When ``attach`` raises, ``pre`` (and its
    binding hold) is removed before the error propagates.

    Parameters
    ----------
    optimizer:
        The optimizer to hook, or ``None`` to only run ``attach``.
    pre:
        Step pre-hook registered first.
    attach:
        Callable that registers the inner hooks.
    post:
        Step post-hook registered last.

    Returns
    -------
    list[Any]
        Handles to remove on close.
    """

    handles = [] if optimizer is None else register_step_hooks(optimizer, pre)
    try:
        attach()
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
    if optimizer is not None:
        handles += register_step_hooks(optimizer, post=post)
    return handles
