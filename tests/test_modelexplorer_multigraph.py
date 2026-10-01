"""Multi-graph collection tests: rolled projection + feedback overlay (D10).

Feed-forward captures emit NO rolled graph at all (rolling changes nothing);
recurrent captures append the disclosed rolled DAG projection whose
back-edges leave ``incomingEdges`` (dagre silently reverses cyclic edges)
and ride the named "recurrent feedback" edge-overlay carrier instead.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


class _FeedForward(nn.Module):
    """No module reuse, no recurrence: rolled must NOT be emitted."""

    def __init__(self) -> None:
        """Two distinct layers."""

        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plain chain."""

        return self.b(torch.relu(self.a(x)))


class _Recurrent(nn.Module):
    """A three-iteration cell: the rolled/feedback case."""

    def __init__(self) -> None:
        """One reused cell."""

        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Iterate the same cell."""

        h = x
        for _ in range(3):
            h = torch.tanh(self.cell(h))
        return h


@pytest.fixture(scope="module")
def recurrent_payload() -> Any:
    """Export the recurrent toy once per module."""

    log = tl.trace(_Recurrent().eval(), torch.randn(2, 4))
    try:
        yield tl.export.to_model_explorer_dict(log)
    finally:
        log.cleanup()


def test_feed_forward_emits_single_graph() -> None:
    """Zero multi-pass ops -> exactly one graph, no rolled projection."""

    log = tl.trace(_FeedForward().eval(), torch.randn(2, 4))
    payload = tl.export.to_model_explorer_dict(log)
    assert [graph["id"] for graph in payload["graphs"]] == ["00-execution"]


def test_recurrent_appends_rolled_graph(recurrent_payload: dict[str, Any]) -> None:
    """Multi-pass ops append the rolled projection after the exact graph."""

    assert [graph["id"] for graph in recurrent_payload["graphs"]] == [
        "00-execution",
        "01-rolled",
    ]
    assert recurrent_payload["graphSorting"] == "name_asc"
    # Physical order equals name_asc order (the component predates the
    # sorting key, memo D10).
    ids = [graph["id"] for graph in recurrent_payload["graphs"]]
    assert ids == sorted(ids)


def test_rolled_graph_is_acyclic_with_feedback_overlay(
    recurrent_payload: dict[str, Any],
) -> None:
    """Back-edges leave incomingEdges and ride the feedback overlay (D10)."""

    rolled = recurrent_payload["graphs"][1]
    node_index = {node["id"]: position for position, node in enumerate(rolled["nodes"])}
    for node in rolled["nodes"]:
        for edge in node.get("incomingEdges", []):
            assert node_index[edge["sourceNodeId"]] < node_index[node["id"]], (
                "rolled incomingEdges must stay forward-only"
            )
    overlays = rolled["tasksData"]["edgeOverlaysDataListLeftPane"]
    assert overlays[0]["type"] == "edge_overlays"
    feedback = overlays[0]["overlays"][0]
    assert feedback["name"] == "recurrent feedback"
    assert feedback["edges"], "the recurrent capture must carry feedback edges"
    for edge in feedback["edges"]:
        assert edge["sourceNodeId"] in node_index
        assert edge["targetNodeId"] in node_index
    assert rolled["groupNodeAttributes"][""]["feedback_edges"] == str(len(feedback["edges"]))


def test_rolled_nodes_disclose_pass_counts(recurrent_payload: dict[str, Any]) -> None:
    """Rolled multi-pass nodes carry passes + total_* aggregation keys."""

    rolled = recurrent_payload["graphs"][1]
    cell = next(node for node in rolled["nodes"] if node["label"] == "linear")
    attrs = {attr["key"]: attr["value"] for attr in cell["attrs"]}
    assert attrs["passes"] == "3"
    assert "time" not in attrs  # per-pass value never masquerades on rolled
    assert attrs["torchlens_label"]
    rows = tl.export.validate_model_explorer_payload(recurrent_payload)
    assert rows.ok, rows.failures


def test_include_rolled_false_suppresses_projection() -> None:
    """include_rolled=False keeps the exact graph only."""

    log = tl.trace(_Recurrent().eval(), torch.randn(2, 4))
    payload = tl.export.to_model_explorer_dict(log, include_rolled=False)
    assert [graph["id"] for graph in payload["graphs"]] == ["00-execution"]
