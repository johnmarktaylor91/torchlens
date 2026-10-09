"""Session-scoped ``forward`` wrappers for non-root submodules.

TorchLens observes module entry and exit by replacing each non-root
submodule's ``forward`` with a toggle-gated wrapper
(:func:`torchlens.backends.torch.model_prep.module_forward_decorator`). The
wrappers live exactly as long as one capture session: installed at session
start, put back by identity at session cleanup. A model between captures
therefore holds no TorchLens callable and deep-copies, pickles and saves like a
model TorchLens never touched.
"""

from __future__ import annotations

import copy
import functools
import inspect
import operator
from collections.abc import Callable
from types import MethodType
from typing import TYPE_CHECKING, Any

from torch import nn

from ._tl import is_forward_call_decorated, mark_forward_call_decorated

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


def restore_undecorated_forward(module: nn.Module) -> None:
    """Undo a TorchLens ``forward`` decoration left on ``module``.

    Session wrappers are normally removed by the session cleanup, so this only
    acts on a leftover: a cleanup cut short, or a shallow copy taken while a
    capture held the source module's wrappers. A root's ``forward`` must be
    UNDECORATED (``trace`` frames the root itself; the wrapper would call
    ``push_frame`` for a module the session never registered, raising
    ``KeyError``), and a submodule is re-wrapped from its clean ``forward``.
    The original ``forward`` is recovered from ``functools.wraps``'
    ``__wrapped__`` reference and rebound to ``module`` when it was bound to
    another instance; if it is absent, the instance-level override is dropped
    so lookup falls back to the (undecorated) class ``forward``.

    Parameters
    ----------
    module:
        Module to clean: a root about to be prepared, a submodule about to be
        wrapped, or a module being released.

    Returns
    -------
    None
        The module's ``forward`` is restored in place when it was decorated;
        otherwise this is a no-op.
    """
    current_forward = module.__dict__.get("forward", None)
    if current_forward is None or not is_forward_call_decorated(current_forward):
        return
    original_forward = getattr(current_forward, "__wrapped__", None)
    if original_forward is None:
        module.__dict__.pop("forward", None)
        return
    # A wrapper copied from ANOTHER module instance (a shallow copy taken while
    # a capture held the source's wrappers) wraps the source's bound method;
    # pinning it here would make this module run the source's weights. Rebind
    # the underlying function to this module, as deepcopy rebinds methods.
    bound_self = getattr(original_forward, "__self__", module)
    original_func = getattr(original_forward, "__func__", None)
    if bound_self is not module and original_func is not None:
        original_forward = MethodType(original_func, module)
    # When the recovered forward is just the module's own class method, drop
    # the instance override instead of pinning the bound method as an instance
    # attribute: an instance-level forward churns the implementation
    # fingerprint (`_fingerprint_model_implementation` folds it), so the
    # documented trace -> release_model -> trace(cache=True) workflow missed
    # the cache on every released model.
    original_func = getattr(original_forward, "__func__", None)
    if original_func is not None and original_func is inspect.getattr_static(
        type(module), "forward", None
    ):
        module.__dict__.pop("forward", None)
    else:
        module.forward = original_forward


# Marks a module that had no instance-level ``forward`` before the session.
_NO_INSTANCE_FORWARD = object()


class _SessionForward:
    """The ``forward`` a capture session installs on one non-root submodule.

    Calls delegate to the toggle-gated wrapper built by
    :func:`torchlens.backends.torch.model_prep.module_forward_decorator`. The
    object exists so that a ``copy.deepcopy`` or ``pickle`` of the module taken
    WHILE the session runs (from the model's own ``forward`` or a hook) never
    carries TorchLens along: ``copy.deepcopy`` treats a bare function closure
    as atomic, so a copy used to keep a wrapper closing over the ORIGINAL
    module and ran the original's weights forever after, and ``pickle`` failed.
    Both protocols now yield what the copy would hold had TorchLens never
    touched the module: its prior instance ``forward``, or none at all.

    ``functools.update_wrapper`` gives it the original's ``__name__``,
    ``__qualname__``, ``__doc__``, ``__module__`` and ``__wrapped__`` (the
    original bound ``forward``), which ``inspect.signature`` and the restore
    paths read; the ``_tl`` decoration tag marks it as TorchLens's own.

    Parameters
    ----------
    wrapper:
        The toggle-gated closure to delegate calls to.
    module:
        Module whose ``forward`` this object replaces.
    original:
        The module's ``forward`` when the session started (the bound class
        method or the user's instance ``forward``).
    prior:
        The module's prior instance ``forward``, or ``_NO_INSTANCE_FORWARD``.
    """

    def __init__(
        self,
        wrapper: Callable[..., Any],
        module: nn.Module,
        original: Callable[..., Any],
        prior: Any,
    ) -> None:
        """Hold the delegate and what a copy or pickle reduces to."""

        functools.update_wrapper(self, original)
        self._delegate = wrapper
        self._module = module
        self._prior = prior

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the module forward through the session wrapper."""

        return self._delegate(*args, **kwargs)

    def __copy__(self) -> _SessionForward:
        """Return ``self``: a shallow copy of a callable shares it, as for functions."""

        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        """Return the copied module's own ``forward``, never a TorchLens object.

        A user instance ``forward`` is deep-copied (a bound method rebinds to
        the copied module). With none, the copied module's class ``forward``
        is bound to it: ``memo`` maps the module to its copy while the module
        state is copied, and a pinned bound class method behaves exactly like
        the class lookup.
        """

        if self._prior is not _NO_INSTANCE_FORWARD:
            return copy.deepcopy(self._prior, memo)
        module_copy = memo.get(id(self._module))
        if module_copy is None:
            module_copy = copy.deepcopy(self._module, memo)
        module_type = type(module_copy)
        return inspect.getattr_static(module_type, "forward").__get__(module_copy, module_type)

    def __reduce_ex__(self, protocol: Any) -> tuple[Any, ...]:
        """Pickle as the prior ``forward`` through stdlib callables only.

        With no prior instance ``forward``, unpickling reads ``forward`` off the
        half-built module, whose instance dict is still empty, so it yields the
        class ``forward``; the pickle never references TorchLens.
        """

        if self._prior is _NO_INSTANCE_FORWARD:
            return (getattr, (self._module, "forward"))
        return (operator.getitem, ((self._prior,), 0))


def install_session_forward_wrappers(trace: Trace, model: nn.Module) -> None:
    """Wrap every non-root submodule's ``forward`` for one capture session.

    The wrappers live exactly as long as the session:
    :func:`restore_session_forward_wrappers` (run by the session cleanup on
    every exit, ``BaseException`` included) puts each module's prior instance
    ``forward`` back by identity, or removes the instance attribute when there
    was none. A wrapper closes over its module and bound forward, so one left
    on a model between captures made ``copy.deepcopy`` produce a copy that ran
    the ORIGINAL's weights and sent its gradients there, made whole-model
    ``pickle``/``torch.save`` fail, and made a previously captured model crash
    a later capture that called it as an unregistered helper. Wrapping the
    current ``forward`` each session also picks up a forward the user (or a
    dispatch library) replaced after an earlier capture.

    Parameters
    ----------
    trace:
        Trace whose session owns the wrappers; the install record lives in
        ``trace._module_capture_ws.session_forward_wrappers``.
    model:
        Root module of the capture; its own ``forward`` stays undecorated.

    Returns
    -------
    None
        Submodule ``forward`` attributes are replaced in place.
    """

    from .model_prep import module_forward_decorator

    installed = trace._module_capture_ws.session_forward_wrappers
    for module in model.modules():
        if module is model:
            continue
        # Heal a decoration a cut-short cleanup or a mid-session shallow copy left.
        restore_undecorated_forward(module)
        if not hasattr(module, "forward"):
            continue
        prior = module.__dict__.get("forward", _NO_INSTANCE_FORWARD)
        original = module.forward
        wrapper = _SessionForward(
            module_forward_decorator(original, module), module, original, prior
        )
        mark_forward_call_decorated(wrapper)
        module.__dict__["forward"] = wrapper
        installed.append((module, prior, wrapper))


def restore_session_forward_wrappers(trace: Trace) -> None:
    """Put back every ``forward`` this session's wrappers replaced.

    A module whose ``forward`` no longer holds this session's wrapper (user
    code reassigned it during the capture) keeps the newer value, as an eager
    run would. The install record is emptied first, so a repeated call is a
    no-op and the trace never pins the modules.

    Parameters
    ----------
    trace:
        Trace whose session installed the wrappers.

    Returns
    -------
    None
        Module ``forward`` attributes are restored in place.
    """

    workspace = trace._module_capture_ws
    installed = workspace.session_forward_wrappers
    workspace.session_forward_wrappers = []
    for module, prior, wrapper in reversed(installed):
        if module.__dict__.get("forward") is not wrapper:
            continue
        if prior is _NO_INSTANCE_FORWARD:
            module.__dict__.pop("forward", None)
        else:
            module.__dict__["forward"] = prior
