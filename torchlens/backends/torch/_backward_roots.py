"""Autograd-entry roots: per-capture ownership and forward positions.

``torch.autograd.backward`` and ``torch.autograd.grad`` take one root or a
sequence of roots. Each capture a backward reaches walks and hooks the graph
of every root whose graph reaches that capture's tensors, and only those, so a
multi-root backward over several captures records every node each capture owns
(the root helpers are split out of ``backward.py`` under its size ratchet).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import torch

from ._tl import get_tensor_label


def _root_tensor_step_index(trace: Any, root: torch.Tensor) -> int | None:
    """Return a TorchLens ``step_index`` for an autograd root tensor.

    Parameters
    ----------
    trace:
        Trace that owns the root tensor.
    root:
        Root tensor passed to an autograd entry point.

    Returns
    -------
    int | None
        Root layer ``step_index`` when resolvable.
    """

    label = get_tensor_label(root)
    if label is None:
        return None
    lookup_labels = (getattr(trace, "_raw_to_final_layer_labels", {}).get(label, label), label)
    layer_lookup = getattr(trace, "layer_dict_all_keys", {})
    for lookup_label in lookup_labels:
        layer = layer_lookup.get(lookup_label)
        step_index = getattr(layer, "step_index", None)
        if isinstance(step_index, int):
            return step_index
    raw_match = re.search(r"_(\d+)_raw$", label)
    if raw_match is not None:
        return int(raw_match.group(1))
    final_match = re.search(r"_[1-9]\d*_(\d+)(?::[1-9]\d*)?$", label)
    if final_match is not None:
        return int(final_match.group(1))
    return None


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
