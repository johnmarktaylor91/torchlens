"""Multi-trace analysis for TorchLens.

Public surface:

* :class:`TraceBundle` -- container for N ``ModelLog`` instances, with a
  union supergraph and per-node accessors.
* :class:`NodeView` -- per-node view returned by ``bundle[node_name]``.
* :func:`bundle` -- thin factory wrapper around :class:`TraceBundle` for
  ergonomic construction (``tl.bundle([ml1, ml2, ml3])``).
* :func:`compare_topology` / :class:`TopologyDiff` -- pairwise structural
  diff between two ``ModelLog`` graphs.
* :class:`Supergraph` / :class:`SupergraphNode` -- the union graph data
  structure produced by :func:`build_supergraph`.
* Metric primitives (cosine_distance, relative_l2, pearson_correlation_distance,
  relative_l1_scalar) and :func:`resolve_metric`.

Visualization, counterfactual branch enumeration, intervention APIs, and
streaming aggregate are deferred to a Phase 2 dispatch -- see
``.project-context/todos.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from .bundle import TraceBundle
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

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..data_classes.model_log import ModelLog


def bundle(
    traces: List["ModelLog"],
    names: Optional[List[str]] = None,
    groups: Optional[Dict[str, List[str]]] = None,
) -> TraceBundle:
    """Construct a :class:`TraceBundle` from a list of ``ModelLog`` instances.

    Thin wrapper around the :class:`TraceBundle` constructor; offered so
    user code can read as ``tl.bundle([ml1, ml2, ml3])`` rather than
    ``tl.TraceBundle([ml1, ml2, ml3])``.
    """

    return TraceBundle(traces, names=names, groups=groups)


__all__ = [
    "METRIC_REGISTRY",
    "NodeView",
    "Supergraph",
    "SupergraphNode",
    "TopologyDiff",
    "TraceBundle",
    "build_supergraph",
    "bundle",
    "compare_topology",
    "cosine_distance",
    "pearson_correlation_distance",
    "relative_l1_scalar",
    "relative_l2",
    "resolve_metric",
]
