"""Semantic identity goldens for the RenderIR-backed rank renderer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import torchlens as tl

from test_render_dotid_cert10 import _ModuleDictQuoteKeyModel

pydot = pytest.importorskip("pydot")

_GOLDEN = Path(__file__).parent / "golden" / "rank_render_ir_semantics.json"


def _clean(value: str | None) -> str:
    """Return a stable unquoted pydot value."""

    return "" if value is None else value.strip('"')


def _semantic_record(source: str) -> dict[str, Any]:
    """Extract renderer semantics independently of DOT statement formatting."""

    graphs = pydot.graph_from_dot_data(source)
    assert graphs
    nodes: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []
    regions: list[dict[str, str]] = []

    def visit(graph: pydot.Graph) -> None:
        """Collect semantic records from one graph and its nested regions."""

        for node in graph.get_nodes():
            name = _clean(node.get_name())
            if name in {"", "graph", "node", "edge", "\\n"}:
                continue
            attrs = node.get_attributes()
            nodes.append(
                {
                    "name": name,
                    "shape": _clean(attrs.get("shape")),
                    "style": _clean(attrs.get("style")),
                    "fillcolor": _clean(attrs.get("fillcolor")),
                }
            )
        for edge in graph.get_edges():
            attrs = edge.get_attributes()
            edges.append(
                {
                    "source": _clean(edge.get_source()),
                    "target": _clean(edge.get_destination()),
                    "style": _clean(attrs.get("style")),
                    "color": _clean(attrs.get("color")),
                    "arrowsize": _clean(attrs.get("arrowsize")),
                }
            )
        for subgraph in graph.get_subgraphs():
            attrs = subgraph.get_attributes()
            regions.append(
                {
                    "name": _clean(subgraph.get_name()),
                    "style": _clean(attrs.get("style")),
                    "penwidth": _clean(attrs.get("penwidth")),
                }
            )
            visit(subgraph)

    visit(graphs[0])
    return {
        "nodes": sorted(nodes, key=lambda item: item["name"]),
        "edges": sorted(edges, key=lambda item: (item["source"], item["target"])),
        "regions": sorted(regions, key=lambda item: item["name"]),
    }


@pytest.mark.smoke
def test_rank_render_ir_semantic_golden(tmp_path: Path) -> None:
    """Pin rank node, edge, and nested-region semantic identity."""

    trace = tl.trace(_ModuleDictQuoteKeyModel(), torch.randn(2, 3))
    try:
        source = trace.draw(
            vis_outpath=str(tmp_path / "rank_semantics"),
            vis_save_only=True,
            vis_fileformat="svg",
            vis_node_placement="rank",
            order_siblings=False,
        )
    finally:
        trace.cleanup()
    assert source is not None
    assert _semantic_record(source) == json.loads(_GOLDEN.read_text())
