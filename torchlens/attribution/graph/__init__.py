"""Attribution graphs: direct linear effects between transcoder features, errors, tokens and logits.

Documented-unstable. ``build_graph`` reads one prompt's capture with a set
of transcoders and returns an :class:`AttributionGraph` whose every feature
and logit node is accounted for by its incoming edges plus named constant
terms. Doc of record: ``docs/reference/attribution.md``, section
Attribution graphs.
"""

from ._build import build_graph
from ._result import NODE_KINDS, AttributionGraph, GraphNode

__all__ = ["AttributionGraph", "GraphNode", "NODE_KINDS", "build_graph"]
