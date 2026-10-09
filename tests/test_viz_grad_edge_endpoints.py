"""Gradient edges and forward-to-grad_fn ties must attach to declared DOT nodes.

Graphviz silently creates a plain oval for every edge endpoint that names no
declared node. A gradient arrow whose endpoint name drifts from the forward
node's name therefore renders as a stray oval floating beside the graph instead
of attaching to the forward op. These tests render small models with saved
gradients in every forward view and require every edge endpoint to be a node
the renderer declared.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._render_common import GRADIENT_ARROW_COLOR

pydot = pytest.importorskip("pydot")

_DEFAULT_STATEMENTS = frozenset({"node", "edge", "graph"})


class _BlockModel(nn.Module):
    """Feedforward model with a nested submodule for collapsed views."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.block = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 4))
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""

        y = self.block(x) * 2
        return self.head(y + x.sum())


class _RecurrentModel(nn.Module):
    """Model that reuses one cell so its layers carry several passes."""

    def __init__(self) -> None:
        """Initialize the shared cell."""

        super().__init__()
        self.cell = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the cell three times."""

        for _ in range(3):
            x = torch.tanh(self.cell(x))
        return x * 2


def _traced_with_grads(model: nn.Module) -> tl.Trace:
    """Return a trace of ``model`` with saved gradients from one backward pass.

    Parameters
    ----------
    model:
        Model to trace.

    Returns
    -------
    tl.Trace
        Trace whose forward ops carry saved gradients.
    """

    torch.manual_seed(0)
    trace = tl.trace(
        model,
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
    )
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    return trace


def _unquote(name: str) -> str:
    """Return a DOT identifier without surrounding quotes.

    Parameters
    ----------
    name:
        Identifier as pydot reports it.

    Returns
    -------
    str
        Bare identifier.
    """

    name = name.strip()
    if len(name) >= 2 and name[0] == name[-1] == '"':
        return name[1:-1]
    return name


def _regions(graph: Any) -> Iterator[Any]:
    """Yield ``graph`` and every nested subgraph.

    Parameters
    ----------
    graph:
        Parsed pydot graph or subgraph.

    Yields
    ------
    Any
        Each region in depth-first order.
    """

    yield graph
    for subgraph in graph.get_subgraphs():
        yield from _regions(subgraph)


def _endpoint(name: Any) -> str:
    """Return the node identifier of an edge endpoint, dropping any port.

    Parameters
    ----------
    name:
        Endpoint as pydot reports it.

    Returns
    -------
    str
        Node identifier.
    """

    bare = _unquote(str(name))
    if not str(name).strip().startswith('"') and ":" in bare:
        return bare.split(":", 1)[0]
    return bare


def _dot_nodes_and_edges(dot: str) -> tuple[set[str], list[tuple[str, str, dict[str, str]]]]:
    """Return the declared node names and the edges of a DOT source.

    Parameters
    ----------
    dot:
        DOT source returned by a draw call.

    Returns
    -------
    tuple[set[str], list[tuple[str, str, dict[str, str]]]]
        Declared node names and ``(tail, head, attributes)`` edge triples.
    """

    graphs = pydot.graph_from_dot_data(dot)
    assert graphs, "DOT parser returned no graph"
    declared: set[str] = set()
    edges: list[tuple[str, str, dict[str, str]]] = []
    for region in _regions(graphs[0]):
        for node in region.get_nodes():
            name = _unquote(node.get_name())
            if name not in _DEFAULT_STATEMENTS:
                declared.add(name)
        for edge in region.get_edges():
            attrs = {key: _unquote(str(value)) for key, value in edge.get_attributes().items()}
            edges.append((_endpoint(edge.get_source()), _endpoint(edge.get_destination()), attrs))
    return declared, edges


def _assert_edges_attach(dot: str, *, expect_gradient_edges: bool) -> None:
    """Require every edge endpoint to be a declared node.

    Parameters
    ----------
    dot:
        DOT source returned by a draw call.
    expect_gradient_edges:
        Whether the render must contain gradient-colored edges, so the check
        cannot pass vacuously.
    """

    declared, edges = _dot_nodes_and_edges(dot)
    undeclared = sorted(
        {endpoint for tail, head, _ in edges for endpoint in (tail, head)} - declared
    )
    assert not undeclared, f"edges name undeclared nodes (Graphviz draws stray ovals): {undeclared}"
    if expect_gradient_edges:
        gradient_edges = [
            edge for edge in edges if edge[2].get("color", "").upper() == GRADIENT_ARROW_COLOR
        ]
        assert gradient_edges, "render carries no gradient edges; the check would be vacuous"


_FORWARD_VIEWS: dict[str, dict[str, Any]] = {
    "unrolled": {"vis_mode": "unrolled"},
    "rolled": {"vis_mode": "rolled"},
    "unrolled_collapsed": {"vis_mode": "unrolled", "vis_call_depth": 1},
    "rolled_collapsed": {"vis_mode": "rolled", "vis_call_depth": 1},
}

_MODELS: dict[str, Callable[[], nn.Module]] = {
    "block": _BlockModel,
    "recurrent": _RecurrentModel,
}


@pytest.mark.smoke
def test_grad_edges_attach_to_forward_nodes_unrolled_and_rolled(tmp_path: Path) -> None:
    """Saved-gradient arrows in draw() name the forward nodes in both views."""

    trace = _traced_with_grads(_BlockModel())
    try:
        for view in ("unrolled", "rolled"):
            dot = trace.draw(
                vis_outpath=str(tmp_path / view),
                vis_save_only=True,
                vis_fileformat="dot",
                vis_mode=view,
            )
            _assert_edges_attach(dot, expect_gradient_edges=True)
    finally:
        trace.cleanup()


@pytest.mark.parametrize("model_name", sorted(_MODELS))
@pytest.mark.parametrize("view_name", sorted(_FORWARD_VIEWS))
def test_grad_edges_attach_in_every_forward_view(
    tmp_path: Path, model_name: str, view_name: str
) -> None:
    """Every forward view, including multi-pass and collapsed, attaches gradient arrows."""

    trace = _traced_with_grads(_MODELS[model_name]())
    try:
        dot = trace.draw(
            vis_outpath=str(tmp_path / f"{model_name}_{view_name}"),
            vis_save_only=True,
            vis_fileformat="dot",
            **_FORWARD_VIEWS[view_name],
        )
        _assert_edges_attach(dot, expect_gradient_edges=True)
    finally:
        trace.cleanup()


@pytest.mark.parametrize("model_name", sorted(_MODELS))
def test_grad_edges_attach_with_repeat_folding(tmp_path: Path, model_name: str) -> None:
    """Folded repeat runs keep gradient arrows on rendered nodes."""

    trace = _traced_with_grads(_MODELS[model_name]())
    try:
        dot = trace.draw(
            vis_outpath=str(tmp_path / f"{model_name}_folded"),
            vis_save_only=True,
            vis_fileformat="dot",
            fold_repeats=True,
        )
        _assert_edges_attach(dot, expect_gradient_edges=False)
    finally:
        trace.cleanup()


@pytest.mark.smoke
@pytest.mark.parametrize("model_name", sorted(_MODELS))
def test_combined_ties_attach_to_forward_nodes(tmp_path: Path, model_name: str) -> None:
    """draw_combined forward-to-grad_fn ties and backward edges name declared nodes."""

    trace = _traced_with_grads(_MODELS[model_name]())
    try:
        dot = trace.draw_combined(
            vis_outpath=str(tmp_path / f"{model_name}_combined"),
            vis_save_only=True,
            vis_fileformat="dot",
        )
        _assert_edges_attach(dot, expect_gradient_edges=True)
        declared, edges = _dot_nodes_and_edges(dot)
        ties = [edge for edge in edges if edge[2].get("constraint") == "false"]
        if model_name == "block":
            assert ties, "feedforward combined render carries no forward-to-grad_fn ties"
    finally:
        trace.cleanup()


@pytest.mark.parametrize("model_name", sorted(_MODELS))
@pytest.mark.parametrize("view", ["unrolled", "rolled"])
def test_backward_graph_edges_attach(tmp_path: Path, model_name: str, view: str) -> None:
    """draw_backward edges name declared grad_fn nodes in both views."""

    trace = _traced_with_grads(_MODELS[model_name]())
    try:
        dot = trace.draw_backward(
            vis_outpath=str(tmp_path / f"{model_name}_backward_{view}"),
            vis_save_only=True,
            vis_fileformat="dot",
            vis_mode=view,
        )
        _assert_edges_attach(dot, expect_gradient_edges=False)
    finally:
        trace.cleanup()
