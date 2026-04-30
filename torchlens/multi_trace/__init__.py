"""Internal multi-trace analysis helpers for TorchLens intervention bundles.

Internal surface:

* :class:`NodeView` -- per-node view returned by ``Bundle.node(...)``.
* :func:`compare_topology` / :class:`TopologyDiff` -- pairwise structural
  diff between two ``ModelLog`` graphs.
* :class:`Supergraph` / :class:`SupergraphNode` -- the union graph data
  structure produced by :func:`build_supergraph`.
* Metric primitives (cosine_distance, relative_l2, pearson_correlation_distance,
  relative_l1_scalar) and :func:`resolve_metric`.

This subpackage is intentionally not part of the top-level TorchLens public API.
"""

from __future__ import annotations

from .metrics import (
    METRIC_REGISTRY,
    cosine_distance,
    pearson_correlation_distance,
    relative_l1_scalar,
    relative_l2,
    resolve_metric,
)
from .node_view import NodeView
from .topology import (
    Supergraph,
    SupergraphNode,
    TopologyDiff,
    build_supergraph,
    compare_topology,
)


__all__ = [
    "METRIC_REGISTRY",
    "NodeView",
    "Supergraph",
    "SupergraphNode",
    "TopologyDiff",
    "build_supergraph",
    "compare_topology",
    "cosine_distance",
    "pearson_correlation_distance",
    "relative_l1_scalar",
    "relative_l2",
    "resolve_metric",
]
