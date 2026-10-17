"""An attribution graph painted onto its trace by ``tl.attribution.overlay``.

Pre-registered bars (fixtures F1 and F3): with the default influence, a
rendered node gets a value exactly when it is the output of a module (or
carries a layer label) holding feature or error nodes, and the value is the
``abs_sum`` of those nodes' influence computed independently in numpy (closed
form ``w @ ((I - A)^-1 - I)`` on the row-normalised absolute adjacency) to
1e-12; with ``graph_value="activation"`` the painted sites are exactly those
holding feature nodes, valued by the sum of absolute activations.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from tests import _ag1_fixtures as fx

TOL = 1e-12
DOC = Path(__file__).resolve().parents[1] / "docs" / "reference" / "attribution.md"


def _build(name: str) -> tuple[object, tl.attribution.graph.AttributionGraph]:
    n_layers = 3 if name == "F3" else 2
    model = fx.build_model(n_layers)
    transcoders = {"F1": fx.exact_transcoders, "F3": fx.cross_layer_transcoders}[name](model)
    trace = tl.trace(model, torch.tensor([fx.TOKENS]))
    graph = tl.attribution.build_graph(
        trace,
        transcoders,
        embedding="embed",
        frozen=fx.frozen_names(n_layers),
        max_n_logits=2,
        desired_logit_prob=0.99,
    )
    return trace, graph


@pytest.fixture(scope="module")
def built() -> Iterator[dict[str, tuple[object, tl.attribution.graph.AttributionGraph]]]:
    pairs = {name: _build(name) for name in ("F1", "F3")}
    try:
        yield pairs
    finally:
        pairs.clear()


def _numpy_influence(graph) -> np.ndarray:
    adjacency = graph.adjacency_matrix.numpy().astype(np.float64)
    absolute = np.abs(adjacency)
    normalized = absolute / np.maximum(absolute.sum(axis=1, keepdims=True), 1e-10)
    n = adjacency.shape[0]
    weights = np.zeros(n)
    n_logits = len(graph.node_indices("logit"))
    weights[n - n_logits :] = graph.logit_probabilities.numpy()
    return weights @ (np.linalg.inv(np.eye(n) - normalized) - np.eye(n))


def _site_of(node, sites) -> str | None:
    """The graph site a rendered node shows, or None (inputs and outputs show none)."""

    if node.is_input or node.is_output:
        return None
    if node.layer_label in sites:
        return node.layer_label
    return next((m for m in node.output_of_modules or () if m in sites), None)


def _check_painting(trace, color_by, expected: dict[str, float]) -> int:
    painted = 0
    for label in trace.layer_labels:
        node = trace[label]
        site = _site_of(node, expected)
        got = color_by(node)
        if site is None:
            assert got is None, label
        else:
            assert got is not None and abs(got - expected[site]) <= TOL, (label, got)
            painted += 1
    return painted


@pytest.mark.parametrize("name", ["F1", "F3"])
def test_graph_overlay_colours_exactly_the_node_sites(built, name) -> None:
    trace, graph = built[name]
    influence = _numpy_influence(graph)
    expected: dict[str, float] = {}
    for index, node in enumerate(graph.nodes):
        if node.kind in ("feature", "error"):
            expected[node.site] = expected.get(node.site, 0.0) + abs(influence[index])
    assert len(expected) >= 2
    painted = _check_painting(trace, tl.attribution.overlay(trace, graph), expected)
    assert painted >= len(expected)


@pytest.mark.parametrize("name", ["F1", "F3"])
def test_graph_overlay_activation_colours_feature_sites(built, name) -> None:
    trace, graph = built[name]
    activations = graph.activation_values.numpy()
    expected: dict[str, float] = {}
    for index in graph.node_indices("feature"):
        site = graph.nodes[index].site
        expected[site] = expected.get(site, 0.0) + abs(float(activations[index]))
    color_by = tl.attribution.overlay(trace, graph, graph_value="activation")
    assert _check_painting(trace, color_by, expected) >= len(expected)


def test_graph_overlay_reduce_applies_across_a_site(built) -> None:
    trace, graph = built["F1"]
    influence = _numpy_influence(graph)
    expected: dict[str, float] = {}
    for index, node in enumerate(graph.nodes):
        if node.kind in ("feature", "error"):
            expected[node.site] = max(expected.get(node.site, -np.inf), influence[index])
    _check_painting(trace, tl.attribution.overlay(trace, graph, reduce="max"), expected)


def test_graph_overlay_refuses_an_unknown_graph_value(built) -> None:
    trace, graph = built["F1"]
    with pytest.raises(tl.attribution.AttributionError) as excinfo:
        tl.attribution.overlay(trace, graph, graph_value="gradient")
    assert excinfo.value.fields["code"] == "attribution_overlay_argument_invalid"
    assert "Remedy:" in str(excinfo.value)


def test_graph_value_is_refused_for_other_sources(built) -> None:
    trace, _graph = built["F1"]
    label = trace.layer_labels[1]
    with pytest.raises(tl.attribution.AttributionError) as excinfo:
        tl.attribution.overlay(trace, {label: 1.0}, graph_value="influence")
    assert excinfo.value.fields["code"] == "attribution_overlay_argument_invalid"
    assert tl.attribution.overlay(trace, {label: 1.0})(trace[label]) == 1.0


def test_graph_overlay_refuses_a_graph_from_another_trace(built) -> None:
    _trace, graph = built["F1"]
    other = tl.trace(nn.Sequential(nn.Linear(2, 2)), torch.ones(1, 2))
    with pytest.raises(tl.attribution.AttributionError) as excinfo:
        tl.attribution.overlay(other, graph)
    assert excinfo.value.fields["code"] == "attribution_overlay_graph_mismatch"
    assert "Remedy:" in str(excinfo.value)


def test_docs_graph_example_overlay_lines_run(tmp_path) -> None:
    fences = re.findall(r"```python\n(.*?)\n```", DOC.read_text(encoding="utf-8"), re.DOTALL)
    (fence,) = [f for f in fences if "tl.attribution.overlay(trace, graph" in f]
    namespace: dict[str, object] = {"DOCS_TMPDIR": str(tmp_path), "__name__": "__docs__"}
    exec(compile(fence, str(DOC), "exec"), namespace)  # noqa: S102 -- the doc's own example
    color_by = namespace["color_by"]
    trace = namespace["trace"]
    assert any(color_by(trace[label]) is not None for label in trace.layer_labels)
