"""Pair autograd ``GradFn`` nodes with the forward ops whose outputs they belong to.

A forward op records the identity of its output's ``grad_fn`` (``grad_fn_object_id``).
A backward walk pairs each node it discovers through that identity, but a walk that
runs DURING the forward (``torch.autograd.grad`` or ``backward`` inside ``forward``)
runs before any layer exists, so its discoveries carry no op label. The projection
rebuild therefore re-pairs every unpaired discovery against the finalized layers with
the same lookup the walk uses, so the pairing does not depend on when the walk ran.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import replace
from typing import Any

from ..._utils._record_group_table import record_member

__all__ = ("_layer_by_grad_fn_id", "_layer_grad_fn_pairing_rank", "_pair_discovered_with_layers")


def _layer_grad_fn_pairing_rank(layer: Any) -> int:
    """Return the preference rank for a forward op grad-fn pairing candidate.

    Parameters
    ----------
    layer:
        Forward Op/Layer candidate sharing a grad-fn object id.

    Returns
    -------
    int
        Lower rank is preferred; compute ops outrank boundary sentinels.
    """

    is_compute = record_member(layer, "is_compute_op", False) or record_member(
        layer,
        "is_compute_layer",
        False,
    )
    return 0 if is_compute else 1


def _layer_by_grad_fn_id(trace: Any) -> dict[int, str]:
    """Build a mapping from forward grad_fn_handle identity to final layer label.

    Parameters
    ----------
    trace:
        Trace whose layers were captured during the forward pass.

    Returns
    -------
    dict[int, str]
        Mapping from ``id(grad_fn_handle)`` to final layer label. This is the
        ``GradFn.op_label`` contract: always the layer label, never a
        pass-qualified op label, also on multi-pass layers. Consumers that need
        the exact pass resolve it with ``_paired_pass_op``.
    """
    mapping: dict[int, str] = {}
    ranks: dict[int, int] = {}
    for layer in trace._layer_list:
        grad_fn_object_id = record_member(layer, "grad_fn_object_id", None)
        if grad_fn_object_id is None:
            continue
        candidate_rank = _layer_grad_fn_pairing_rank(layer)
        current_rank = ranks.get(grad_fn_object_id)
        if current_rank is not None and current_rank <= candidate_rank:
            continue
        mapping[grad_fn_object_id] = layer.layer_label
        ranks[grad_fn_object_id] = candidate_rank
    return mapping


def _pair_discovered_with_layers(trace: Any, discovered: MutableMapping[int, Any]) -> None:
    """Give each discovery with no op label the layer that records its node, in place.

    Parameters
    ----------
    trace:
        Trace whose finalized layers record their outputs' ``grad_fn`` identities.
    discovered:
        ``GradFnDiscovered`` events keyed by node identity (the fold state's map).
        A discovery that already names an op, or whose node no layer records, is
        left as it is.
    """

    unpaired = [object_id for object_id, event in discovered.items() if event.op_label is None]
    if not unpaired:
        return
    layer_lookup = _layer_by_grad_fn_id(trace)
    for object_id in unpaired:
        layer_label = layer_lookup.get(object_id)
        if layer_label is not None:
            discovered[object_id] = replace(discovered[object_id], op_label=layer_label)
