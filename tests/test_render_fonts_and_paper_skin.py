"""Backward and combined graphs use the theme font; the paper skin renders without warnings.

Forward ``draw`` sets the theme's font family on the graph, node and edge
defaults. ``draw_backward`` and ``draw_combined`` set none, so Graphviz fell
back to its serif default for every label there (finding F19). The ``paper``
skin set ``colorscheme=paired12`` although every colour TorchLens emits is hex
or named, so the scheme changed nothing and made Graphviz warn on every paper
render with a legend (finding F27).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization.themes import THEME_PRESETS

pydot = pytest.importorskip("pydot")


def _traced_with_grads() -> tl.Trace:
    """Return a small trace with saved gradients from one backward pass.

    Returns
    -------
    tl.Trace
        Trace with forward ops and grad_fn metadata.
    """

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    trace = tl.trace(
        model,
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
    )
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    return trace


def _default_fontnames(dot: str) -> dict[str, Any]:
    """Return the ``fontname`` of the graph, node and edge default statements.

    Parameters
    ----------
    dot:
        DOT source.

    Returns
    -------
    dict[str, Any]
        ``fontname`` per statement kind (``None`` when unset).
    """

    graph = pydot.graph_from_dot_data(dot)[0]
    node_defaults: dict[str, Any] = {}
    for attrs in graph.get_node_defaults() or ():
        node_defaults.update(attrs)
    edge_defaults: dict[str, Any] = {}
    for attrs in graph.get_edge_defaults() or ():
        edge_defaults.update(attrs)
    return {
        "graph": graph.get_attributes().get("fontname"),
        "node": node_defaults.get("fontname"),
        "edge": edge_defaults.get("fontname"),
    }


@pytest.mark.parametrize("entry", ["draw_backward", "draw_combined"])
def test_backward_and_combined_graphs_declare_the_theme_font(tmp_path: Path, entry: str) -> None:
    """Graph, node and edge defaults name the default theme's font family."""

    family = THEME_PRESETS["torchlens"].typography.family
    trace = _traced_with_grads()
    try:
        dot = getattr(trace, entry)(
            vis_outpath=str(tmp_path / entry),
            vis_save_only=True,
            vis_fileformat="dot",
        )
    finally:
        trace.cleanup()
    fonts = _default_fontnames(dot)
    assert {kind: str(value).strip('"') for kind, value in fonts.items()} == {
        "graph": family,
        "node": family,
        "edge": family,
    }


def test_paper_skin_renders_with_a_legend_without_graphviz_warnings(tmp_path: Path) -> None:
    """A paper render with a legend sets no colour scheme and Graphviz reports nothing."""

    trace = tl.trace(nn.Sequential(nn.Linear(3, 3), nn.ReLU()), torch.randn(1, 3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dot = trace.draw(
            vis_save_only=True,
            vis_fileformat="svg",
            vis_outpath=str(tmp_path / "paper"),
            vis_theme="paper",
            show_legend=True,
        )
    assert "colorscheme" not in dot
    assert (tmp_path / "paper.svg").exists()
    assert [str(w.message) for w in caught if "color" in str(w.message).lower()] == []
