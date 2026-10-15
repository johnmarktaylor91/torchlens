"""Component edges as intervention selections, and edge patching.

Patching edge ``s->t`` means: receiver ``t`` reads ``input - s + s_corrupt``,
every other receiver reads what it reads in the patched forward. When edges
into several receivers are patched in one call, ``input`` and ``s`` are the
values of the patched forward itself (the MIB circuit-track convention): a
receiver sees its patched senders' corrupted values while every other
sender carries whatever it computes in that forward, including the effects
of patches upstream of it. With one receiver nothing upstream is patched, so
``input`` and ``s`` are the clean values.

Two existing intervention doors carry each receiver's patch, both on the
replay engine:

* ``m{L}`` and ``logits`` receivers own their pre-norm input edge (the op
  edge from the residual state into the receiver's norm), so the patch is
  one EDGE substitution of that consumed value (``edge_substitution``). The
  residual state itself, and every other consumer of it, stay as computed.
* ``a{L}.h{H}<q|k|v>`` receivers share one norm with every head and role of
  their layer, so the patch goes one step later: the shared input edge is
  substituted on a scratch fork, and only head ``H``'s ``q`` (or ``k``, or
  ``v``) slice of the projection output it produces is written into the
  result (an ACT selection over the recipe's per-head facet).

Receivers are patched in topological order, one stage at a time (a layer's
attention receivers, then its MLP, then ``logits``). An edge substitution
stores a fixed value, so a later patch never revisits an earlier receiver,
and every receiver's patched value is computed from the forward as patched
so far: the component graph is rebuilt on the partly patched fork at the
start of each stage after the first write.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import math
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


def _stage_key(receiver: Any) -> tuple[float, int]:
    """Topological stage of a receiver: layer, then attention before MLP; logits last."""

    layer = math.inf if receiver.layer_index is None else float(receiver.layer_index)
    return (layer, 0 if receiver.kind in ("q", "k", "v") else 1)


def _stages(edges: list[Any]) -> list[list[tuple[Any, list[Any]]]]:
    """Group the edges by receiver, and the receivers into topological stages."""

    seen: set[str] = set()
    by_receiver: dict[str, tuple[Any, list[Any]]] = {}
    for edge in edges:
        if edge.name in seen:
            refuse(
                code="mi_edge_patch_edge_repeated",
                message=f"patch_edges got edge {edge.name!r} more than once; each edge is "
                "patched at most once.",
                remedy="pass each edge once, for example list(dict.fromkeys(edges))",
                edge=edge.name,
            )
        seen.add(edge.name)
        by_receiver.setdefault(edge.receiver.name, (edge.receiver, []))[1].append(edge)
    stages: dict[tuple[float, int], list[tuple[Any, list[Any]]]] = {}
    for receiver, receiver_edges in by_receiver.values():
        stages.setdefault(_stage_key(receiver), []).append((receiver, receiver_edges))
    return [stages[key] for key in sorted(stages)]


def _write_receiver(owner: Any, source: Any, receiver: Any, patched: torch.Tensor) -> None:
    """Write ``patched`` as the receiver's input into ``owner`` (forked from ``source``'s state)."""

    if receiver.kind not in ("q", "k", "v"):
        owner.do(_input_edge_record(owner, receiver).__selection__(), patched)
        return
    scratch = source.fork()
    scratch.do(_input_edge_record(scratch, receiver).__selection__(), patched)
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
    owner.do(_head_spec(owner, receiver).__selection__(), projection.clone())


def patch_edges(trace: Any, edges: Iterable[Any], *, corrupt: Any) -> Any:
    """Return a fork of ``trace`` with the given edges patched to their corrupted senders.

    Parameters
    ----------
    trace:
        The clean trace the edges' graph was built from, captured with
        ``layers_to_save='all'`` and ``intervention_ready=True``.
    edges:
        :class:`ComponentEdge` objects, into one receiver or several. A
        circuit runs as one call: pass every edge outside it,
        ``[e for name, e in graph.edges.items() if name not in kept]``.
        No edges returns an unpatched fork.
    corrupt:
        The :class:`ComponentGraph` of the corrupted input (same model,
        same shapes); each edge's corrupted contribution is
        ``corrupt.nodes[edge.sender.name].value``.

    Returns
    -------
    Trace
        A fork whose every patched receiver reads ``input - sum(s) +
        sum(s_corrupt)`` over its patched edges, with ``input`` and ``s``
        taken from the patched forward itself (the clean values when one
        receiver is patched); the input trace is not modified.
    """

    from ._edges import component_graph

    stages = _stages(list(edges))
    result = None
    for stage in stages:
        live = None if result is None else component_graph(result)
        for receiver, receiver_edges in stage:
            if live is None:
                patched = receiver.value
                senders = {edge.sender.name: edge.sender.value for edge in receiver_edges}
            else:
                patched = live.receivers[receiver.name].value
                senders = {
                    edge.sender.name: live.nodes[edge.sender.name].value for edge in receiver_edges
                }
            for edge in receiver_edges:
                corrupt_value = corrupt.nodes[edge.sender.name].value
                patched = patched - senders[edge.sender.name] + corrupt_value.to(patched.dtype)
            source = trace if result is None else result
            owner = trace.fork() if result is None else result
            _write_receiver(owner, source, receiver, patched)
            result = owner
    return trace.fork() if result is None else result
