"""Component edges as intervention selections, and single-receiver edge patching.

Patching edge ``s->t`` means: receiver ``t`` reads ``clean input - s_clean +
s_corrupt``, every other receiver reads what it reads in the patched
forward. Two existing intervention doors carry it, both on the replay
engine:

* ``m{L}`` and ``logits`` receivers own their pre-norm input edge (the op
  edge from the residual state into the receiver's norm), so the patch is
  one EDGE substitution of that consumed value (``edge_substitution``). The
  residual state itself, and every other consumer of it, stay clean.
* ``a{L}.h{H}<q|k|v>`` receivers share one norm with every head and role of
  their layer, so the patch goes one step later: the shared input edge is
  substituted on a scratch fork, and only head ``H``'s ``q`` (or ``k``, or
  ``v``) slice of the projection output it produces is written into the
  result (an ACT selection over the recipe's per-head facet).

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from ._errors import refuse

__all__ = ["patch_edges", "receiver_selection"]


def _input_edge_record(trace: Any, receiver: Any) -> Any:
    """Return the op edge that carries the receiver's pre-norm input into its norm."""

    from ._residual import _innermost_module

    matches = [
        record
        for record in trace.edges
        if str(trace.ops[str(record.parent_label)].label) == receiver.input_label
        and _innermost_module(trace.ops[str(record.child_label)]) == receiver.norm_address
    ]
    if len(matches) != 1:
        refuse(
            code="mi_edge_selection_unresolvable",
            message=f"Receiver {receiver.name!r} has {len(matches)} op edges from its input "
            f"{receiver.input_label!r} into its norm {receiver.norm_address!r}; one is needed "
            "to address the edge.",
            remedy="capture with tl.options.CaptureOptions(layers_to_save='all', "
            "intervention_ready=True) and rebuild the graph from that trace",
            receiver=receiver.name,
        )
    return matches[0]


def _head_spec(trace: Any, receiver: Any) -> Any:
    """Return the per-head facet spec (projection-output slice) of a q/k/v receiver."""

    facet = trace.modules[receiver.module_address].facets[receiver.kind]
    return facet.spec.select(2, int(receiver.head))


def receiver_selection(trace: Any, receiver: Any) -> Any:
    """Return the selection an edge patch into ``receiver`` writes.

    ``m{L}`` / ``logits``: the EDGE selection of the receiver's pre-norm
    input edge (values are residual-space tensors). ``a{L}.h{H}<q|k|v>``:
    the ACT selection of head ``H``'s role slice of the projection output
    (values are full projection-output tensors; only the slice is written).
    """

    if receiver.kind in ("q", "k", "v"):
        return _head_spec(trace, receiver).__selection__()
    return _input_edge_record(trace, receiver).__selection__()


def _single_receiver(edges: list[Any]) -> Any:
    """Return the one receiver every edge enters, refusing a mixed set."""

    receivers = {edge.receiver.name: edge.receiver for edge in edges}
    if len(receivers) != 1:
        refuse(
            code="mi_edge_patch_receivers_mixed",
            message=f"patch_edges got edges into {len(receivers)} receivers "
            f"({sorted(receivers)[:4]}); one call patches edges into one receiver.",
            remedy="call patch_edges once per receiver",
            receivers=sorted(receivers),
        )
    return next(iter(receivers.values()))


def patch_edges(trace: Any, edges: Iterable[Any], *, corrupt: Any) -> Any:
    """Return a fork of ``trace`` with edges into one receiver patched.

    Parameters
    ----------
    trace:
        The clean trace the edges' graph was built from, captured with
        ``layers_to_save='all'`` and ``intervention_ready=True``.
    edges:
        One or more :class:`ComponentEdge` objects, all into one receiver.
    corrupt:
        The :class:`ComponentGraph` of the corrupted input (same model,
        same shapes); each edge's corrupted contribution is
        ``corrupt.nodes[edge.sender.name].value``.

    Returns
    -------
    Trace
        A fork whose receiver reads ``clean input - sum(s_clean) +
        sum(s_corrupt)``; the input trace is not modified.
    """

    edges = list(edges)
    receiver = _single_receiver(edges)
    patched = receiver.value
    for edge in edges:
        corrupt_value = corrupt.nodes[edge.sender.name].value
        patched = patched - edge.sender.value + corrupt_value.to(patched.dtype)
    scratch = trace.fork()
    scratch.do(_input_edge_record(scratch, receiver).__selection__(), patched)
    if receiver.kind not in ("q", "k", "v"):
        return scratch
    spec = _head_spec(scratch, receiver)
    projection = scratch.ops[str(spec.home_label)].out
    if not isinstance(projection, torch.Tensor):
        refuse(
            code="mi_edge_selection_unresolvable",
            message=f"The projection output behind {receiver.name!r} was not recomputed.",
            remedy="capture with tl.options.CaptureOptions(layers_to_save='all', "
            "intervention_ready=True)",
            receiver=receiver.name,
        )
    result = trace.fork()
    result.do(_head_spec(result, receiver).__selection__(), projection.clone())
    return result
