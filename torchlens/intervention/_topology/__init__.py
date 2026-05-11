"""Topology comparison and supergraph construction for intervention bundles."""

from __future__ import annotations

from .topology import (
    Supergraph,
    SupergraphNode,
    TopologyDiff,
    build_supergraph,
    compare_topology,
)

__all__ = [
    "Supergraph",
    "SupergraphNode",
    "TopologyDiff",
    "build_supergraph",
    "compare_topology",
]
