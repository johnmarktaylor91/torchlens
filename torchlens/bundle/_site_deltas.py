"""Per-pass member deltas behind the bundle diff picture (``Bundle.draw(graph="diff")``).

The diff picture draws one node per op, so a multi-pass layer (a module
called twice, a recurrent cell) draws one node per pass, as ``trace.draw``
does unrolled. Its values are computed per pass here, keyed by site: a
single-pass layer's site is its supergraph node name and pass ``k`` of a
multi-pass layer is ``"<node>:<k>"``, the row-key convention of
``Bundle.compare`` and ``Bundle.rank_changes``. A layer-level value (a stored
comparison) joins the passes instead (``_deltas._tensor_field``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

from ._aligned_sites import site_row_key
from ._deltas import _member_distances, _pass_ops, _tensor_field

if TYPE_CHECKING:
    from . import Bundle


@dataclass(frozen=True)
class SiteDeltas:
    """Per-site distances of the compared members from the baseline.

    Attributes
    ----------
    values:
        Site key to member name to distance. Members without a tensor at a
        site are omitted there.
    node_for_label:
        Member layer label to its supergraph node name.
    """

    values: Mapping[str, Mapping[str, float]]
    node_for_label: Mapping[str, str]

    def key(self, op: Any) -> str:
        """Return the site key of one drawn op (``relu_1_2:2`` for a pass of a multi-pass layer)."""

        layer_label = str(getattr(op, "layer_label", ""))
        node = self.node_for_label.get(layer_label, layer_label)
        return node + site_row_key(op)[len(layer_label) :]

    def value(self, op: Any, member: str) -> float:
        """Return ``member``'s distance at ``op``'s site, ``0.0`` when it has none."""

        return float(self.values.get(self.key(op), {}).get(member, 0.0))


def site_deltas(
    bundle: Bundle,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    baseline_name: str,
    on: Literal["out", "grad"],
    members: Sequence[str],
) -> SiteDeltas:
    """Compute per-pass distances of ``members`` from an already-gated baseline.

    Pass ``k`` of a member's layer is compared with pass ``k`` of the
    baseline's layer at the same supergraph node (``metric`` over the ``on``
    field); a pass one side lacks has no value and the baseline reads ``0.0``.

    Returns
    -------
    SiteDeltas
        The per-site values and the label-to-node map the renderer looks them up by.
    """

    supergraph = bundle.supergraph
    values: dict[str, dict[str, float]] = {}
    # Keyed by label rather than ``id(layer)``: the structural alignment and
    # the supergraph materialize independent layer objects.
    node_for_label = {
        str(layer.layer_label): str(node_name)
        for node_name, node in supergraph.nodes.items()
        for layer in node.layer_refs.values()
    }
    for node_name in supergraph.topological_order:
        node = supergraph.nodes[node_name]
        reference_layer = node.layer_refs.get(baseline_name)
        if reference_layer is None:
            continue
        member_passes = {
            name: _ops_by_pass(node.layer_refs[name]) for name in members if name in node.layer_refs
        }
        reference_passes = _ops_by_pass(reference_layer)
        for pass_number, reference_op in reference_passes.items():
            reference = _tensor_field(reference_op, on)
            if reference is None:
                continue
            site = str(node_name) if len(reference_passes) == 1 else f"{node_name}:{pass_number}"
            candidates = {
                name: _tensor_field(passes[pass_number], on) if pass_number in passes else None
                for name, passes in member_passes.items()
            }
            site_values = _member_distances(reference, candidates, baseline_name, metric, site)
            if site_values:
                values[site] = site_values
    return SiteDeltas(values=values, node_for_label=node_for_label)


def _ops_by_pass(layer: Any) -> dict[int, Any]:
    """Return a layer's ops keyed by pass number; an op stands for itself as pass 1."""

    ops = _pass_ops(layer) or [layer]
    return {int(getattr(op, "pass_number", 1) or 1): op for op in ops}
