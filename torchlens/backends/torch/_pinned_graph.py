"""Measure the autograd graph a trace's graph-connected saved activations keep alive.

The default save keeps autograd history (``detach_saved_activations=False``):
every saved payload is a graph-connected copy, so the tensors autograd saved
for the backward of every op upstream of a saved site stay in memory for as
long as the trace holds the payload. ``Trace.pinned_autograd_memory`` discloses
those bytes (R2-37). The walk follows ``grad_fn.next_functions`` from each
attached payload once, reads each node's saved tensors through the same
best-effort reader the per-op ``autograd_memory`` uses, and counts every
storage once. Parameters and views of them are excluded: the model owns
them whether or not a trace pins the graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from ..._state import pause_logging
from ._ops_autograd import _add_autograd_saved_tensor
from ._ops_exhaustive import _iter_autograd_saved_candidates

__all__ = ["pinned_autograd_bytes"]


def pinned_autograd_bytes(payloads: Iterable[Any]) -> int:
    """Return the autograd-saved bytes reachable from graph-connected payloads.

    Parameters
    ----------
    payloads:
        Saved activation payloads; detached or non-tensor entries contribute
        nothing.

    Returns
    -------
    int
        Bytes of distinct non-Parameter tensors saved by the autograd nodes
        upstream of any payload, deduplicated by data pointer.
    """

    with pause_logging():
        stack = [
            payload.grad_fn
            for payload in payloads
            if isinstance(payload, torch.Tensor) and payload.grad_fn is not None
        ]
        # Hold every visited node: an id() alone could be reused by a freed
        # wrapper during the walk.
        visited: dict[int, Any] = {}
        seen_data_ptrs: set[int] = set()
        total = 0
        while stack:
            node = stack.pop()
            if node is None or id(node) in visited:
                continue
            visited[id(node)] = node
            for saved in _iter_autograd_saved_candidates(node):
                if isinstance(saved, torch.Tensor) and not _model_owned(saved):
                    added, _count = _add_autograd_saved_tensor(saved, seen_data_ptrs)
                    total += added
            stack.extend(
                next_node
                for next_node, _index in getattr(node, "next_functions", ())
                if next_node is not None
            )
    return total


def _model_owned(tensor: torch.Tensor) -> bool:
    """Return whether a saved tensor is a Parameter or a view of one.

    Parameters
    ----------
    tensor:
        Tensor an autograd node saved for backward (``addmm`` saves
        ``weight.t()``, a view of the Parameter).

    Returns
    -------
    bool
        ``True`` when the model owns the storage regardless of the trace.
    """

    return isinstance(tensor, nn.Parameter) or isinstance(
        getattr(tensor, "_base", None), nn.Parameter
    )
