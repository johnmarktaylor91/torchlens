"""Shared facts about the in-forward ``torch.autograd.backward`` boundary op.

``backward.install_autograd_wrappers`` records a ``torch.autograd.backward``
call made inside a capture (so also ``Tensor.backward``) as one
``autogradbackward`` boundary op per leaf gradient the pass writes;
``validation._autograd_backward_replay`` re-derives each written gradient from
the recorded subgraph. This module holds the recorder and what both sides
share: the recorder marker, the recorder's call signature and the engine-flag
config (it imports the wrapper machinery only when a call is recorded).

The engine's product is a side effect, ``leaf.grad``, so the recorder turns it
into op outputs: one recorder call per reached leaf, nested so that the
innermost call runs the engine pass and every call returns its own leaf's grad
once the pass has written it. An accumulating pass (the leaf already held a
grad) is recorded as an in-place op whose receiver is the previous grad.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import torch

AUTOGRAD_BACKWARD_RECORDER_ATTR = "__tl_autograd_backward_recorder__"
"""Marker set on the TorchLens callable that records an in-forward autograd.backward."""

AUTOGRAD_BACKWARD_FUNC_NAME = "autogradbackward"
"""Op type / func name of the boundary op."""

AUTOGRAD_BACKWARD_TRANSFORM_KIND = "autograd.backward"
"""``Op.transform_kind`` of the boundary op."""

_POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD

_BACKWARD_SIGNATURE_FALLBACK = inspect.Signature(
    [
        inspect.Parameter("tensors", _POSITIONAL),
        inspect.Parameter("grad_tensors", _POSITIONAL, default=None),
        inspect.Parameter("retain_graph", _POSITIONAL, default=None),
        inspect.Parameter("create_graph", _POSITIONAL, default=False),
        inspect.Parameter("grad_variables", _POSITIONAL, default=None),
        inspect.Parameter("inputs", _POSITIONAL, default=None),
    ]
)

_RECORDER_SIGNATURE = inspect.Signature(
    [
        inspect.Parameter("accumulate_into", _POSITIONAL),
        inspect.Parameter("tensors", _POSITIONAL),
        inspect.Parameter("grad_tensors", _POSITIONAL, default=None),
        inspect.Parameter("inputs", _POSITIONAL, default=None),
        inspect.Parameter("inplace", _POSITIONAL, default=False),
    ]
)
"""The recorded call: receiver (previous grad or None), roots, root grads, the one leaf."""

_TENSOR_PARAMETERS = frozenset({"tensors", "grad_tensors", "grad_variables", "inputs"})


def is_autograd_backward_recorder(func: Any) -> bool:
    """Return whether ``func`` is TorchLens's in-forward autograd.backward recorder.

    Parameters
    ----------
    func:
        An op's recorded callable.

    Returns
    -------
    bool
        ``True`` only for a recorder callable built at capture time.
    """

    return getattr(func, AUTOGRAD_BACKWARD_RECORDER_ATTR, False) is True


def recorder_signature() -> inspect.Signature:
    """Return the fixed call signature of the ``autogradbackward`` recorder."""

    return _RECORDER_SIGNATURE


def backward_signature(backward_callable: Callable[..., Any]) -> inspect.Signature:
    """Return ``torch.autograd.backward``'s call signature, with a fixed fallback.

    Parameters
    ----------
    backward_callable:
        The (original) ``torch.autograd.backward`` callable.

    Returns
    -------
    inspect.Signature
        Its signature, or the torch 2.x signature when it cannot be read.
    """

    try:
        signature = inspect.signature(backward_callable)
    except (TypeError, ValueError):
        return _BACKWARD_SIGNATURE_FALLBACK
    if "tensors" not in signature.parameters:
        return _BACKWARD_SIGNATURE_FALLBACK
    return signature


def autograd_backward_call_config(
    backward_callable: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Return the engine flags of one ``torch.autograd.backward`` call.

    Parameters
    ----------
    backward_callable:
        The (original) ``torch.autograd.backward`` callable.
    args:
        Positional call arguments.
    kwargs:
        Keyword call arguments.

    Returns
    -------
    dict[str, Any]
        Explicitly passed scalar flags (``retain_graph``, ``create_graph``), never
        the tensor arguments.
    """

    try:
        bound = backward_signature(backward_callable).bind_partial(*args, **kwargs)
    except TypeError:
        return {}
    return {
        name: value
        for name, value in bound.arguments.items()
        if name not in _TENSOR_PARAMETERS and (value is None or isinstance(value, (bool, int)))
    }


def _tensor_list(value: Any) -> list[torch.Tensor]:
    """Return the tensors of a tensor-or-sequence autograd argument."""

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


def reachable_leaves(roots: Sequence[torch.Tensor], restrict: Any) -> list[torch.Tensor]:
    """Return the leaf tensors whose ``.grad`` an engine pass from ``roots`` writes.

    Walks the autograd graph under ``roots`` to its ``AccumulateGrad`` nodes (each
    knows its ``variable``); a root that is itself a leaf requiring grad is written
    directly. ``restrict`` is the call's ``inputs`` argument: when given, the engine
    accumulates only into those tensors.

    Parameters
    ----------
    roots:
        Root tensors of the engine pass.
    restrict:
        The ``inputs`` argument of the call (a tensor, a sequence, or ``None``).

    Returns
    -------
    list[torch.Tensor]
        The leaves in discovery order, each once.
    """

    from .backward import _is_accumulate_grad

    leaves: list[torch.Tensor] = []
    leaf_ids: set[int] = set()
    stack: list[Any] = []

    def add_leaf(leaf: torch.Tensor) -> None:
        if id(leaf) not in leaf_ids:
            leaf_ids.add(id(leaf))
            leaves.append(leaf)

    for root in roots:
        grad_fn = root.grad_fn
        if grad_fn is not None:
            stack.append(grad_fn)
        elif root.requires_grad:
            add_leaf(root)
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if _is_accumulate_grad(node):
            variable = getattr(node, "variable", None)
            if isinstance(variable, torch.Tensor):
                add_leaf(variable)
            continue
        for next_fn, _ in getattr(node, "next_functions", ()) or ():
            if next_fn is not None:
                stack.append(next_fn)
    restricted = _tensor_list(restrict)
    if restricted:
        allowed = {id(tensor) for tensor in restricted}
        leaves = [leaf for leaf in leaves if id(leaf) in allowed]
    return leaves


def _leaf_recorder(
    trace: Any,
    leaf: torch.Tensor,
    previous: torch.Tensor | None,
    tensors: Any,
    grad_tensors: Any,
    config: dict[str, Any],
    inner: Callable[[], None],
) -> Callable[[], None]:
    """Return the zero-argument call that records one leaf's gradient as a boundary op.

    Parameters
    ----------
    trace:
        The active capture trace.
    leaf:
        The leaf whose ``.grad`` this op produces.
    previous:
        The leaf's grad before the pass (the in-place receiver), or ``None``.
    tensors, grad_tensors:
        The call's roots and root gradients (the op's recorded parents).
    config:
        The engine flags recorded as the op's transform config.
    inner:
        The next nested recorder call, or the engine pass itself.
    """

    from .wrappers import _set_transform_metadata, torch_func_decorator

    pending = [inner]

    def autogradbackward(
        accumulate_into: Any,
        tensors: Any,
        grad_tensors: Any = None,
        inputs: Any = None,
        inplace: bool = False,
    ) -> Any:
        """Run the nested pass (the boundary's interior) and return this leaf's grad."""

        if not pending:
            raise RuntimeError(
                "the autograd.backward boundary recorder runs once; validation replays its "
                "recorded subgraph instead"
            )
        from ..._state import pause_logging

        # The boundary is a leaf in the decoration nesting: whatever the engine
        # pass runs inside it (nested recorders, hooks) is not a sub-operation of
        # this recording, so the enclosing barcode is restored before the
        # decorator reads it.
        workspace = trace._wrapper_runtime_ws
        barcode = workspace.current_func_barcode
        try:
            pending.pop()()
            with pause_logging():
                written = leaf.grad
        finally:
            workspace.current_func_barcode = barcode
        return written

    setattr(autogradbackward, AUTOGRAD_BACKWARD_RECORDER_ATTR, True)
    _set_transform_metadata(
        autogradbackward,
        transform_kind=AUTOGRAD_BACKWARD_TRANSFORM_KIND,
        tags=(AUTOGRAD_BACKWARD_TRANSFORM_KIND,),
        transform_config=dict(config),
        inner_fn=autogradbackward,
    )
    decorated = torch_func_decorator(autogradbackward, AUTOGRAD_BACKWARD_FUNC_NAME)

    def call() -> None:
        decorated(
            previous,
            tensors,
            grad_tensors,
            inputs=(leaf,),
            inplace=previous is not None,
        )

    return call


def record_autograd_backward_boundary(
    trace: Any,
    engine: Callable[[], Any],
    backward_callable: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Record an in-forward ``torch.autograd.backward`` call as boundary ops.

    The call becomes one ``autogradbackward`` op per leaf gradient it writes
    (``transform_kind="autograd.backward"``, ``is_transform=True``), whose parents
    are the recorded tensors among its roots (``tensors``) and ``grad_tensors``
    plus, for a leaf that already held a grad, that previous grad as the in-place
    receiver. The engine pass still runs inside, so the backward capture of that
    pass is unchanged, and its dispatches are owned by the innermost op. Validation
    re-derives each gradient from the recorded subgraph
    (``validation._autograd_backward_replay``).

    Parameters
    ----------
    trace:
        The active capture trace.
    engine:
        Runs the captured autograd engine pass.
    backward_callable:
        The original ``torch.autograd.backward``, used to read the call's arguments.
    args:
        Positional ``torch.autograd.backward`` arguments.
    kwargs:
        Keyword ``torch.autograd.backward`` arguments.

    Returns
    -------
    Any
        What the engine pass returns (``torch.autograd.backward`` returns ``None``).
    """

    from ..._state import pause_logging

    try:
        bound = backward_signature(backward_callable).bind_partial(*args, **kwargs)
    except TypeError:
        return engine()
    tensors = bound.arguments.get("tensors")
    grad_tensors = bound.arguments.get("grad_tensors")
    if grad_tensors is None:
        grad_tensors = bound.arguments.get("grad_variables")
    with pause_logging():
        leaves = reachable_leaves(_tensor_list(tensors), bound.arguments.get("inputs"))
        previous = [leaf.grad for leaf in leaves]
    if not leaves:
        return engine()
    config = autograd_backward_call_config(backward_callable, args, kwargs)
    result: list[Any] = []

    def run_engine() -> None:
        result.append(engine())

    inner: Callable[[], None] = run_engine
    for index in reversed(range(len(leaves))):
        inner = _leaf_recorder(
            trace, leaves[index], previous[index], tensors, grad_tensors, config, inner
        )
    inner()
    return result[0] if result else None
