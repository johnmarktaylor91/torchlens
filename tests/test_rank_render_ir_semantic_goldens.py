"""Semantic identity goldens for the RenderIR-backed rank renderer.

Regenerate deliberately with ``TORCHLENS_UPDATE_RANK_RENDER_IR=1`` (the
update run reports SKIP, never green; re-run without the flag to verify).

Governance adjudication (b10 R78 round-3): this family is ENVIRONMENT-
SENSITIVE — the semantic record is extracted from DOT emitted through the
``graphviz`` python package (whose quoting rules this family deliberately
stresses via ``_ModuleDictQuoteKeyModel``) and parsed back through ``pydot``,
so its bytes can drift with either emitter/parser version even when torchlens
behavior is unchanged. The golden resolves through ``tests/_oracle_env.py``
with the family-scoped fingerprint extension ``("graphviz", "pydot")``,
fail-closed off-canonical like the other governed byte families.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from _oracle_env import (
    flag_armed,
    guard_wrap_state_for_golden_update,
    require_env_golden,
    require_update_reason,
    resolve_env_golden,
    write_provenance,
)
from test_render_dotid_cert10 import _ModuleDictQuoteKeyModel

import torchlens as tl

pydot = pytest.importorskip("pydot")

_GOLDEN = Path(__file__).parent / "golden" / "rank_render_ir_semantics.json"
_UPDATE_ENV = "TORCHLENS_UPDATE_RANK_RENDER_IR"
#: Direct byte-generators of this family's golden (see module docstring).
_EMITTER_PACKAGES = ("graphviz", "pydot")


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

    regen = flag_armed(os.environ, _UPDATE_ENV)
    if regen:
        # Generation is in-process: refuse to render golden bytes on a torch
        # already wrapped by earlier tests (SF-53), and require the WHY
        # before the capture runs.
        guard_wrap_state_for_golden_update(_UPDATE_ENV)
        require_update_reason(_UPDATE_ENV)
    trace = tl.trace(_ModuleDictQuoteKeyModel(), torch.randn(2, 3))
    try:
        source = trace.draw(
            path=str(tmp_path / "rank_semantics"),
            vis_save_only=True,
            file_format="svg",
            vis_node_placement="rank",
            order_siblings=False,
        )
    finally:
        trace.cleanup()
    assert source is not None
    if regen:
        # This golden previously had NO documented regen path (b10 R78-8c):
        # hand edits were the only option. Update writes then SKIPS so a
        # regeneration run never reports a vacuous green.
        golden_path, _ = resolve_env_golden(_GOLDEN.parent, _GOLDEN.name, _EMITTER_PACKAGES)
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(
            json.dumps(_semantic_record(source), indent=1, sort_keys=True) + "\n"
        )
        write_provenance(
            golden_path.parent,
            "tests/test_rank_render_ir_semantic_goldens.py",
            _UPDATE_ENV,
            require_update_reason(_UPDATE_ENV),
        )
        pytest.skip(f"updated rank-render-ir golden; re-run without {_UPDATE_ENV} to verify")
    golden_path = require_env_golden(
        _GOLDEN.parent, _GOLDEN.name, _UPDATE_ENV, extra_packages=_EMITTER_PACKAGES
    )
    assert _semantic_record(source) == json.loads(golden_path.read_text())
