"""Register TorchLens's tensor gradient hooks with torch.

``tensor_tracking._install_tensor_grad_hook`` decides what is hooked and binds
the one shared fire function; :func:`register_tensor_grad_hook` puts it on the
tensor (or, for a tensor written in place again since its op, on that op's node).

A plain ``torch.Tensor`` with no ``__torch_function__`` override active takes
:func:`register_plain_tensor_hook`: the steps of ``torch.Tensor.register_hook``
through the C class's getsets. ``register_hook`` itself is Python and reads
``requires_grad``, ``_backward_hooks`` and ``grad_fn`` through ``torch.Tensor``,
where capture wraps and shadows those names, so it had to be marked and paused
and each read ran a TorchLens frame. The hook lands in the same
``_backward_hooks`` dict, keyed by a ``RemovableHandle`` id from torch's own
counter, so hook order, firing and removal are torch's.
``tests/test_backward_hook_install.py`` pins the body of torch's method so a
torch change to it fails loudly instead of drifting. Any other tensor keeps the
marked, paused ``register_hook``.

Capture keeps no handle for the hooks it never removes one by one:
:func:`add_plain_tensor_hook` takes the key from torch's ``RemovableHandle`` id
counter without building the handle object.

The fire path's fork-redirect and label-owner lookups and the payload-copy slot walk
live here too. Imports stay outside the capture modules' import cycle:
``tensor_tracking`` imports this module and passes in ``backward``'s node pre-hook
registrar.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.hooks import RemovableHandle

from ... import _state
from ..._data_classes._fork_lineage import _is_fork_relative
from ..._state import pause_logging
from ..._utils._internal_read import internal_scalar_read
from ..._utils._torch_compat import get_tensorbase_class
from .._entry_refusals import require_tensorbase_class

if TYPE_CHECKING:
    from ..._capture.session import LiveGradientReference

__all__ = (
    "_foreign_label_owner",
    "_fork_redirect_target",
    "_logged_copy_source_slot",
    "add_plain_tensor_hook",
    "raw_grad_fn",
    "raw_requires_grad",
    "register_plain_tensor_hook",
    "register_tensor_grad_hook",
    "takes_plain_hook_install",
)

# The C getset originals: TorchLens's own reads through them run no TorchLens
# frame, log no op and record no witnessed input fact (capture wraps and shadows
# these names on ``torch.Tensor``, never on the C class).
_TENSORBASE_CLS = require_tensorbase_class(get_tensorbase_class())
_HOOKS_OF = _TENSORBASE_CLS._backward_hooks.__get__
_SET_HOOKS = _TENSORBASE_CLS._backward_hooks.__set__
raw_grad_fn = _TENSORBASE_CLS.grad_fn.__get__
raw_requires_grad = _TENSORBASE_CLS.requires_grad.__get__
_HAS_TORCH_FUNCTION_UNARY = torch._C._has_torch_function_unary
_TENSOR = torch.Tensor


def takes_plain_hook_install(t: Any) -> bool:
    """Return whether ``t`` may take :func:`register_plain_tensor_hook`.

    A plain ``torch.Tensor`` with no active ``__torch_function__`` override:
    ``register_hook`` would dispatch to the override first, so such a tensor (and
    every subclass) keeps the ``register_hook`` call.
    """

    return type(t) is _TENSOR and not _HAS_TORCH_FUNCTION_UNARY(t)


def _plain_hooks_dict(t: torch.Tensor) -> OrderedDict[int, Any]:
    """Return ``t``'s gradient hook dict, creating and attaching it as torch does."""

    if not raw_requires_grad(t):
        raise RuntimeError("cannot register a hook on a tensor that doesn't require gradient")
    hooks = _HOOKS_OF(t)
    if hooks is None:
        hooks = OrderedDict()
        _SET_HOOKS(t, hooks)
        grad_fn = raw_grad_fn(t)
        if grad_fn is not None:
            grad_fn._register_hook_dict(t)
    return hooks


def register_plain_tensor_hook(
    t: torch.Tensor, hook: Callable[[torch.Tensor | None], torch.Tensor | None]
) -> RemovableHandle:
    """Register ``hook`` on ``t`` as ``torch.Tensor.register_hook`` does.

    Parameters
    ----------
    t:
        Plain tensor (:func:`takes_plain_hook_install` is true for it).
    hook:
        Gradient hook, called with the tensor's gradient.

    Returns
    -------
    RemovableHandle
        Torch's handle; ``remove()`` takes the hook out of the dict.

    Raises
    ------
    RuntimeError
        ``t`` does not require gradient (torch's own refusal).
    """

    hooks = _plain_hooks_dict(t)
    handle = RemovableHandle(hooks)
    hooks[handle.id] = hook
    return handle


def add_plain_tensor_hook(
    t: Any, hook: Callable[[torch.Tensor | None], torch.Tensor | None]
) -> bool:
    """Put ``hook`` on a plain ``t`` like :func:`register_plain_tensor_hook`, keeping no handle.

    Same dict, same position and a key from the same ``RemovableHandle`` id counter, so
    order and firing are unchanged; only the handle object is skipped, for hooks capture
    never removes one by one (they go with the tensor).

    Parameters
    ----------
    t:
        Tensor to hook.
    hook:
        Gradient hook, called with the tensor's gradient.

    Returns
    -------
    bool
        False, registering nothing, when ``t`` must take ``register_hook`` instead
        (:func:`takes_plain_hook_install` is false for it).
    """

    if type(t) is not _TENSOR or _HAS_TORCH_FUNCTION_UNARY(t):
        return False
    hooks = _plain_hooks_dict(t)
    hook_id = RemovableHandle.next_id
    RemovableHandle.next_id = hook_id + 1
    hooks[hook_id] = hook
    return True


def register_tensor_grad_hook(
    site: torch.Tensor | LiveGradientReference,
    hook: Callable[[torch.Tensor | None], torch.Tensor | None],
    register_slot_prehook: Callable[..., Any],
) -> Any:
    """Register TorchLens's gradient ``hook`` on ``site`` and return its removable handle.

    Parameters
    ----------
    site:
        Tensor whose gradient the hook receives, or a live reference to a tensor
        written in place again since its op: the hook then goes on the op's node,
        for the tensor's output slot.
    hook:
        Gradient hook (``None`` when no gradient arrived; returns the gradient to
        pass on).
    register_slot_prehook:
        ``backward._register_slot_gradient_prehook``, passed in so this module stays
        out of the backward import cycle.

    Returns
    -------
    Any
        The handle; ``remove()`` takes the hook off.
    """

    if isinstance(site, torch.Tensor) and takes_plain_hook_install(site):
        # torch's own ``register_hook`` steps through the C getsets: no TorchLens
        # frame and no torch call, so there is nothing to mark or pause.
        return register_plain_tensor_hook(site, hook)
    # torch's ``register_hook`` reads ``self.grad_fn`` internally, and the tensor can
    # be the user's registered state receiver (an in-place op output), so the read is
    # marked internal. Paused like every TorchLens-internal torch call (Critical
    # Invariant 2): the user's forward never makes this call, so it must not count.
    with pause_logging(), internal_scalar_read():
        if isinstance(site, torch.Tensor):
            return site.register_hook(hook)
        return register_slot_prehook(site.grad_fn, site.output_nr, hook)


def _fork_redirect_target(active_trace: Any) -> Any:
    """Return the fork relative whose managed backward is running, else ``active_trace``."""

    managed_trace = _state._active_trace
    if (
        active_trace is not None
        and managed_trace is not None
        and managed_trace is not active_trace
        and getattr(managed_trace, "_tl_active_backward_bracket", False)
        and _is_fork_relative(active_trace, managed_trace)
    ):
        return managed_trace
    return active_trace


def _foreign_label_owner(trace: Any, tensor_label: str, hooked_tensor_id: int) -> bool:
    """Whether ``trace``'s grad-hook owner map gives ``tensor_label`` to another tensor."""

    owner_map = trace.__dict__.get("_tl_grad_hook_owner_by_label")
    return (
        owner_map is not None and owner_map.get(tensor_label, hooked_tensor_id) != hooked_tensor_id
    )


# TorchLens's own payload copies: identity-gradient nodes with one graph edge.
_PAYLOAD_COPY_NODE_NAMES = frozenset({"CloneBackward0", "_DeferredPayloadCloneFnBackward"})
# An in-place op's payload is a copy of its safe copy; deeper chains are not minted.
_MAX_PAYLOAD_COPY_DEPTH = 4


def _logged_copy_source_slot(logged: torch.Tensor, grad_fn_handle: Any) -> int | None:
    """Return the output slot of ``grad_fn_handle`` that ``logged`` was copied from.

    The walk follows TorchLens's own copy nodes only: an in-place op's payload
    is a copy of the safe copy taken right after the op, so two clone nodes
    can sit between the payload and the op's node.

    Parameters
    ----------
    logged:
        Logged copy of an op output.
    grad_fn_handle:
        Autograd node the op produced, or ``None``.

    Returns
    -------
    int | None
        The slot when ``logged`` is a dead-end copy whose clone chain leads to
        ``grad_fn_handle``; ``None`` otherwise (no node, a detached copy, the
        copy is itself the op's live output, or the chain leads elsewhere).
    """

    if grad_fn_handle is None:
        return None
    with pause_logging(), internal_scalar_read():
        node = logged.grad_fn
        for _ in range(_MAX_PAYLOAD_COPY_DEPTH):
            if node is None or node is grad_fn_handle:
                return None
            if type(node).__name__ not in _PAYLOAD_COPY_NODE_NAMES:
                return None
            edges = [edge for edge in node.next_functions if edge[0] is not None]
            if len(edges) != 1:
                return None
            node, output_nr = edges[0]
            if node is grad_fn_handle:
                return int(output_nr)
    return None
