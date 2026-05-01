"""Tests for Phase 7 visualization theme presets."""

from __future__ import annotations

from pathlib import Path

import graphviz
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import VisualizationOptions


@pytest.mark.parametrize("theme", ["paper", "dark", "colorblind", "high_contrast"])
def test_theme_preset_renders(tmp_path: Path, theme: str) -> None:
    """Each public theme preset should render a Graphviz graph."""

    log = tl.log_forward_pass(nn.Sequential(nn.Linear(3, 3), nn.ReLU()), torch.randn(1, 3))

    dot = log.render_graph(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / theme),
        vis_theme=theme,
        show_legend=True,
    )

    assert "TorchLens legend" in dot
    assert (tmp_path / f"{theme}.svg").exists()


def test_visualization_options_convenience_knobs_return_graph(tmp_path: Path) -> None:
    """Grouped convenience knobs should forward to render_graph."""

    log = tl.log_forward_pass(nn.Linear(2, 2), torch.randn(1, 2))
    options = VisualizationOptions(
        view="unrolled",
        output_path=str(tmp_path / "paper.svg"),
        save_only=True,
        file_format="svg",
        for_paper=True,
        font_size=14,
        dpi=120,
        return_graph=True,
    )

    graph = log.render_graph(**tl.options.visualization_to_render_kwargs(options))

    assert isinstance(graph, graphviz.Digraph)
    assert graph.graph_attr["dpi"] == "120"
    assert graph.node_attr["fontsize"] == "14"
