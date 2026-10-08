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

import inspect
from types import MethodType
from typing import TYPE_CHECKING

from torch import nn

from ._tl import is_forward_call_decorated, mark_forward_call_decorated

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


def restore_undecorated_forward(module: nn.Module) -> None:
    """Undo a TorchLens ``forward`` decoration left on ``module``.

    Session wrappers are normally removed by the session cleanup, so this only
    acts on a leftover: a cleanup cut short, or a deepcopy taken while a
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
    # A wrapper copied from ANOTHER module instance (a deepcopy taken while a
    # capture held the source's wrappers) wraps the source's bound method;
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
        # Heal a decoration a cut-short cleanup or a mid-session deepcopy left.
        restore_undecorated_forward(module)
        if not hasattr(module, "forward"):
            continue
        prior = module.__dict__.get("forward", _NO_INSTANCE_FORWARD)
        wrapper = module_forward_decorator(module.forward, module)
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
