"""Shared facts about the in-forward ``torch.autograd.grad`` boundary op.

``backward.install_autograd_wrappers`` records a ``torch.autograd.grad`` call
made inside a capture as an ``autogradgrad`` boundary op whose callable is a
TorchLens recorder; ``validation._autograd_grad_replay`` re-derives that op's
gradients from the recorded subgraph. This module holds the recorder and what
both sides share: the recorder marker, the call signature and the engine-flag
config (it imports the wrapper machinery only when a call is recorded).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

AUTOGRAD_GRAD_RECORDER_ATTR = "__tl_autograd_grad_recorder__"
"""Marker set on the TorchLens callable that records an in-forward autograd.grad."""

AUTOGRAD_GRAD_FUNC_NAME = "autogradgrad"
"""Op type / func name of the boundary op."""

AUTOGRAD_GRAD_TRANSFORM_KIND = "autograd.grad"
"""``Op.transform_kind`` of the boundary op."""

_POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
_GRAD_SIGNATURE_FALLBACK = inspect.Signature(
    [
        inspect.Parameter("outputs", _POSITIONAL),
        inspect.Parameter("inputs", _POSITIONAL),
        inspect.Parameter("grad_outputs", _POSITIONAL, default=None),
        inspect.Parameter("retain_graph", _POSITIONAL, default=None),
        inspect.Parameter("create_graph", _POSITIONAL, default=False),
        inspect.Parameter("only_inputs", _POSITIONAL, default=True),
        inspect.Parameter("allow_unused", _POSITIONAL, default=None),
        inspect.Parameter("is_grads_batched", _POSITIONAL, default=False),
        inspect.Parameter("materialize_grads", _POSITIONAL, default=False),
    ]
)
_TENSOR_PARAMETERS = frozenset({"outputs", "inputs", "grad_outputs"})


def is_autograd_grad_recorder(func: Any) -> bool:
    """Return whether ``func`` is TorchLens's in-forward autograd.grad recorder.

    Parameters
    ----------
    func:
        An op's recorded callable.

    Returns
    -------
    bool
        ``True`` only for the recorder callable built at capture time.
    """

    return getattr(func, AUTOGRAD_GRAD_RECORDER_ATTR, False) is True


def grad_signature(grad_callable: Callable[..., Any]) -> inspect.Signature:
    """Return ``torch.autograd.grad``'s call signature, with a fixed fallback.

    Parameters
    ----------
    grad_callable:
        The (original) ``torch.autograd.grad`` callable.

    Returns
    -------
    inspect.Signature
        Its signature, or the torch 2.x signature when it cannot be read.
    """

    try:
        signature = inspect.signature(grad_callable)
    except (TypeError, ValueError):
        return _GRAD_SIGNATURE_FALLBACK
    if not {"outputs", "inputs"} <= set(signature.parameters):
        return _GRAD_SIGNATURE_FALLBACK
    return signature


def autograd_grad_call_config(
    grad_callable: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Return the engine flags of one ``torch.autograd.grad`` call.

    Parameters
    ----------
    grad_callable:
        The (original) ``torch.autograd.grad`` callable.
    args:
        Positional call arguments.
    kwargs:
        Keyword call arguments.

    Returns
    -------
    dict[str, Any]
        Explicitly passed scalar flags (``retain_graph``, ``create_graph``,
        ``allow_unused``, ...), never the tensor arguments.
    """

    try:
        bound = grad_signature(grad_callable).bind_partial(*args, **kwargs)
    except TypeError:
        return {}
    return {
        name: value
        for name, value in bound.arguments.items()
        if name not in _TENSOR_PARAMETERS and (value is None or isinstance(value, (bool, int)))
    }


def record_autograd_grad_boundary(
    engine: Callable[[], Any],
    grad_callable: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Record an in-forward ``torch.autograd.grad`` call as a boundary op.

    The call becomes one ``autogradgrad`` op per returned gradient
    (``transform_kind="autograd.grad"``, ``is_transform=True``), whose parents
    are the recorded tensors among its ``outputs``, ``inputs`` and
    ``grad_outputs``. The engine pass still runs inside, so the backward capture
    of that pass is unchanged. Validation re-derives the gradients from the
    recorded subgraph (``validation._autograd_grad_replay``).

    Parameters
    ----------
    engine:
        Runs the captured autograd engine pass and returns the gradients.
    grad_callable:
        The original ``torch.autograd.grad``, used to read the call's flags.
    args:
        Positional ``torch.autograd.grad`` arguments.
    kwargs:
        Keyword ``torch.autograd.grad`` arguments.

    Returns
    -------
    Any
        The gradients, as ``torch.autograd.grad`` returns them.
    """

    from .wrappers import _set_transform_metadata, torch_func_decorator

    # One shot: the recorder keeps no reference to the live call (its roots
    # hold the whole forward graph) once the engine pass has run, and it can
    # never be re-invoked as a replay; validation replays the subgraph instead.
    pending_engine = [engine]

    def autogradgrad(*_call_args: Any, **_call_kwargs: Any) -> Any:
        """Run the captured autograd engine pass (the boundary's interior)."""
        if not pending_engine:
            raise RuntimeError(
                "the autograd.grad boundary recorder runs once; validation replays its "
                "recorded subgraph instead"
            )
        return pending_engine.pop()()

    setattr(autogradgrad, AUTOGRAD_GRAD_RECORDER_ATTR, True)
    _set_transform_metadata(
        autogradgrad,
        transform_kind=AUTOGRAD_GRAD_TRANSFORM_KIND,
        tags=(AUTOGRAD_GRAD_TRANSFORM_KIND,),
        transform_config=autograd_grad_call_config(grad_callable, args, kwargs),
        inner_fn=autogradgrad,
    )
    return torch_func_decorator(autogradgrad, AUTOGRAD_GRAD_FUNC_NAME)(*args, **kwargs)
