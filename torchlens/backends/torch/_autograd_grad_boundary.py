"""Shared facts about the in-forward autograd boundary ops.

``backward.install_autograd_wrappers`` records a ``torch.autograd.grad`` call
made inside a capture as an ``autogradgrad`` boundary op, and a
``torch.autograd.backward`` call (so also ``Tensor.backward``) as one
``autogradbackward`` boundary op per leaf gradient the pass writes; each op's
callable is a TorchLens recorder, and ``validation._autograd_grad_replay``
re-derives the ops' gradients from the recorded subgraph. This module holds the
recorders and what both sides share: the recorder markers, the call signatures
and the engine-flag configs (it imports the wrapper machinery only when a call
is recorded).

A backward's product is a side effect, ``leaf.grad``, so its recorder turns it
into op outputs: one recorder call per reached leaf, nested so that the
innermost call runs the engine pass and every call returns its own leaf's grad
once the pass has written it. An accumulating pass (the leaf already held a
grad) is recorded as an in-place op whose receiver is the previous grad, and
the engine's interior dispatches (the accumulation ``add_`` included) belong
to the innermost op, so the dispatch census balances without any exemption.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ... import _state
from ..._state import pause_logging
from ._backward_roots import _is_accumulate_grad

AUTOGRAD_GRAD_RECORDER_ATTR = "__tl_autograd_grad_recorder__"
"""Marker set on the TorchLens callable that records an in-forward autograd.grad."""

AUTOGRAD_GRAD_FUNC_NAME = "autogradgrad"
"""Op type / func name of the boundary op."""

AUTOGRAD_GRAD_TRANSFORM_KIND = "autograd.grad"
"""``Op.transform_kind`` of the boundary op."""

AUTOGRAD_BACKWARD_RECORDER_ATTR = "__tl_autograd_backward_recorder__"
"""Marker set on the TorchLens callable that records an in-forward autograd.backward."""

AUTOGRAD_BACKWARD_FUNC_NAME = "autogradbackward"
"""Op type / func name of the backward boundary op."""

AUTOGRAD_BACKWARD_TRANSFORM_KIND = "autograd.backward"
"""``Op.transform_kind`` of the backward boundary op."""

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
_BACKWARD_TENSOR_PARAMETERS = frozenset({"tensors", "grad_tensors", "grad_variables", "inputs"})
_RECORDER_SIGNATURE = inspect.Signature(
    [
        inspect.Parameter("accumulate_into", _POSITIONAL),
        inspect.Parameter("tensors", _POSITIONAL),
        inspect.Parameter("grad_tensors", _POSITIONAL, default=None),
        inspect.Parameter("inputs", _POSITIONAL, default=None),
        inspect.Parameter("inplace", _POSITIONAL, default=False),
    ]
)
"""The recorded backward call: receiver (previous grad or None), roots, root grads, one leaf."""


def _wrapper_tools() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return ``(_set_transform_metadata, torch_func_decorator)``, imported on first record."""

    from .wrappers import _set_transform_metadata, torch_func_decorator

    return _set_transform_metadata, torch_func_decorator


def _logging_trace() -> Any:
    """Return the trace whose logging is on, or ``None`` when no capture is logging."""

    active_trace, logging_enabled = _state.active_capture()
    return active_trace if logging_enabled else None


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
        The gradients, as ``torch.autograd.grad`` returns them (the engine pass
        runs unrecorded when no capture is logging).
    """

    if _logging_trace() is None:
        return engine()
    _set_transform_metadata, torch_func_decorator = _wrapper_tools()

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
        if name not in _BACKWARD_TENSOR_PARAMETERS
        and (value is None or isinstance(value, (bool, int)))
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

    leaves: dict[int, torch.Tensor] = {}
    stack: list[Any] = []
    for root in roots:
        if root.grad_fn is not None:
            stack.append(root.grad_fn)
        elif root.requires_grad:
            leaves.setdefault(id(root), root)
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if _is_accumulate_grad(node):
            variable = getattr(node, "variable", None)
            if isinstance(variable, torch.Tensor):
                leaves.setdefault(id(variable), variable)
            continue
        for next_fn, _ in getattr(node, "next_functions", ()) or ():
            if next_fn is not None:
                stack.append(next_fn)
    allowed = {id(tensor) for tensor in _tensor_list(restrict)}
    return [leaf for key, leaf in leaves.items() if not allowed or key in allowed]


@dataclass(frozen=True)
class _BackwardPass:
    """One recorded ``torch.autograd.backward`` call: its roots and the leaves it writes."""

    trace: Any
    tensors: Any
    grad_tensors: Any
    config: dict[str, Any]
    leaves: list[torch.Tensor]
    previous: list[torch.Tensor | None]


def _backward_pass(
    trace: Any, backward_callable: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> _BackwardPass | None:
    """Return the plan of one in-forward backward call, or ``None`` when it writes no leaf."""

    try:
        bound = backward_signature(backward_callable).bind_partial(*args, **kwargs)
    except TypeError:
        return None
    tensors = bound.arguments.get("tensors")
    grad_tensors = bound.arguments.get("grad_tensors")
    if grad_tensors is None:
        grad_tensors = bound.arguments.get("grad_variables")
    with pause_logging():
        leaves = reachable_leaves(_tensor_list(tensors), bound.arguments.get("inputs"))
        previous = [leaf.grad for leaf in leaves]
    if not leaves:
        return None
    config = autograd_backward_call_config(backward_callable, args, kwargs)
    return _BackwardPass(trace, tensors, grad_tensors, config, leaves, previous)


def _backward_recorder(
    backward_pass: _BackwardPass, leaf: torch.Tensor, inner: Callable[[], None]
) -> Callable[..., Any]:
    """Return the decorated recorder whose call produces ``leaf``'s gradient as an op."""

    _set_transform_metadata, torch_func_decorator = _wrapper_tools()
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
        # The boundary is a leaf in the decoration nesting: whatever the engine
        # pass runs inside it (nested recorders, hooks) is not a sub-operation of
        # this recording, so the enclosing barcode is restored before the
        # decorator reads it.
        workspace = backward_pass.trace._wrapper_runtime_ws
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
        transform_config=dict(backward_pass.config),
        inner_fn=autogradbackward,
    )
    return torch_func_decorator(autogradbackward, AUTOGRAD_BACKWARD_FUNC_NAME)


def _leaf_recorder(
    backward_pass: _BackwardPass, index: int, inner: Callable[[], None]
) -> Callable[[], None]:
    """Return the zero-argument call recording leaf ``index``'s gradient as a boundary op.

    The recorded call is ``(previous grad or None, roots, root grads, inputs=(leaf,),
    inplace=<accumulating>)``: the roots and root grads are the op's parents and a
    previous grad is its in-place receiver. ``inner`` is the next nested recorder
    call, or the engine pass itself.
    """

    leaf, previous = backward_pass.leaves[index], backward_pass.previous[index]
    decorated = _backward_recorder(backward_pass, leaf, inner)

    def call() -> None:
        decorated(
            previous,
            backward_pass.tensors,
            backward_pass.grad_tensors,
            inputs=(leaf,),
            inplace=previous is not None,
        )

    return call


def record_autograd_backward_boundary(
    traces_for_roots: Callable[[], Iterable[Any]],
    engine: Callable[[], Any],
    backward_callable: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run an in-forward ``torch.autograd.backward`` call, recorded as boundary ops.

    When a capture is logging and the call's roots reach it, the call becomes one
    ``autogradbackward`` op per leaf gradient it writes
    (``transform_kind="autograd.backward"``, ``is_transform=True``), whose parents
    are the recorded tensors among its roots (``tensors``) and ``grad_tensors``
    plus, for a leaf that already held a grad, that previous grad as the in-place
    receiver. The engine pass still runs inside, so the backward capture of that
    pass is unchanged, and its dispatches are owned by the innermost op. Validation
    re-derives each gradient from the recorded subgraph
    (``validation._autograd_grad_replay``). A backward on tensors the capture never
    saw runs unrecorded.

    Parameters
    ----------
    traces_for_roots:
        Returns the live traces whose graph the call's roots reach.
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

    trace = _logging_trace()
    if trace is None or not any(candidate is trace for candidate in traces_for_roots()):
        return engine()
    backward_pass = _backward_pass(trace, backward_callable, args, kwargs)
    if backward_pass is None:
        return engine()
    result: list[Any] = []

    def run_engine() -> None:
        result.append(engine())

    inner: Callable[[], None] = run_engine
    for index in reversed(range(len(backward_pass.leaves))):
        inner = _leaf_recorder(backward_pass, index, inner)
    inner()
    return result[0] if result else None
