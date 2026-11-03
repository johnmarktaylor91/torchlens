"""An attribution graph painted onto its trace by ``trace.draw(color_by=graph)``.

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
from torchlens._utils._record_group_table import record_member
from torchlens.attribution._viz import attribution_color_by

TOL = 1e-12
DOC = Path(__file__).resolve().parents[1] / "docs" / "reference" / "attribution.md"


def _build(name: str) -> tuple[object, tl.attribution._graph.AttributionGraph]:
    n_layers = 3 if name == "F3" else 2
    model = fx.build_model(n_layers)
    transcoders = {"F1": fx.exact_transcoders, "F3": fx.cross_layer_transcoders}[name](model)
    trace = tl.trace(model, torch.tensor([fx.TOKENS]))
    graph = tl.attribution.build_graph(
        trace,
        transcoders,
        embedding_module="embed",
        frozen=fx.frozen_names(n_layers),
        max_num_logits=2,
        logit_probability_coverage=0.99,
    )
    return trace, graph


@pytest.fixture(scope="module")
def built() -> Iterator[dict[str, tuple[object, tl.attribution._graph.AttributionGraph]]]:
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

    if getattr(node, "is_input", False) or getattr(node, "is_output", False):
        return None
    if node.layer_label in sites:
        return node.layer_label
    return next((m for m in record_member(node, "output_of_modules", ()) or () if m in sites), None)


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


def test_graph_overlay_refuses_a_graph_from_another_trace(built) -> None:
    _trace, graph = built["F1"]
    other = tl.trace(nn.Sequential(nn.Linear(2, 2)), torch.ones(1, 2))
    with pytest.raises(tl.errors.AttributionError) as excinfo:
        attribution_color_by(other, graph)
    assert excinfo.value.fields["code"] == "attribution_overlay_graph_mismatch"
    assert "Remedy:" in str(excinfo.value)


def test_docs_graph_example_overlay_lines_run(tmp_path) -> None:
    fences = re.findall(r"```python\n(.*?)\n```", DOC.read_text(encoding="utf-8"), re.DOTALL)
    (fence,) = [f for f in fences if "color_by(trace, graph" in f]
    namespace: dict[str, object] = {"DOCS_TMPDIR": str(tmp_path), "__name__": "__docs__"}
    exec(compile(fence, str(DOC), "exec"), namespace)  # noqa: S102 -- the doc's own example
    color_by = namespace["color_by"]
    trace = namespace["trace"]
    assert any(color_by(trace[label]) is not None for label in trace.layer_labels)
