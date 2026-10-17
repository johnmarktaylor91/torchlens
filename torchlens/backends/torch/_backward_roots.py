"""Autograd-entry roots: flattening and per-capture ownership.

``torch.autograd.backward`` and ``torch.autograd.grad`` take one root or a
sequence of roots. Each capture a backward reaches walks and hooks the graph
of every root whose graph reaches that capture's tensors, and only those, so a
multi-root backward over several captures records every node each capture owns
(the root helpers are split out of ``backward.py`` under its size ratchet).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


def _root_tensors(value: Any) -> tuple[torch.Tensor, ...]:
    """Flatten autograd root arguments into tensors.

    Parameters
    ----------
    value:
        Tensor or nested sequence passed to an autograd engine entry.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Tensor roots in left-to-right order.
    """

    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(_root_tensors(item))
        return tuple(tensors)
    return ()


def _autograd_roots_from_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    root_kwarg: str,
) -> Any:
    """Return root tensors from an autograd wrapper call.

    Parameters
    ----------
    args:
        Positional arguments passed to the wrapped function.
    kwargs:
        Keyword arguments passed to the wrapped function.
    root_kwarg:
        Keyword name that carries roots for this autograd entry.

    Returns
    -------
    Any
        Original root argument object, or ``None`` when absent.
    """

    if args:
        return args[0]
    return kwargs.get(root_kwarg)


def _roots_by_trace(
    tensors: tuple[torch.Tensor, ...],
    traces: tuple[Any, ...],
    traces_for_roots: Callable[[Any], tuple[Any, ...]],
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Return, per matched trace, the roots whose graphs reach that trace.

    Parameters
    ----------
    tensors:
        The autograd entry's root tensors, flattened in the caller's order.
    traces:
        Traces matched for the whole call, in discovery order.
    traces_for_roots:
        The registry walk that matches traces for a root object.

    Returns
    -------
    tuple[tuple[torch.Tensor, ...], ...]
        One tuple of roots per trace, in the caller's root order. A single
        root is every trace's root. A trace matched only through the roots
        together (a model-state boundary resumed by another root) walks every
        root that has a graph.
    """

    graph_roots = tuple(root for root in tensors if root.grad_fn is not None)
    if len(tensors) <= 1 or len(graph_roots) <= 1:
        first = tensors[0] if tensors else None
        return tuple((first,) if first is not None else () for _ in traces)
    reached = [traces_for_roots(root) for root in graph_roots]
    owned: list[tuple[torch.Tensor, ...]] = []
    for trace in traces:
        mine = tuple(
            root
            for root, matched in zip(graph_roots, reached, strict=True)
            if any(other is trace for other in matched)
        )
        owned.append(mine or graph_roots)
    return tuple(owned)
