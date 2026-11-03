"""Tests for Phase 7 visualization theme presets."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import VisualizationOptions


@pytest.mark.parametrize("theme", ["paper", "dark", "colorblind", "high_contrast"])
def test_theme_preset_renders(tmp_path: Path, theme: str) -> None:
    """Each public theme preset should render a Graphviz graph."""

    log = tl.trace(nn.Sequential(nn.Linear(3, 3), nn.ReLU()), torch.randn(1, 3))

    dot = log.draw(
        file_format="svg",
        path=str(tmp_path / theme),
        theme=theme,
        show_legend=True,
    ).dot

    assert 'ID="tl_key"' in dot  # the key under the header
    assert (tmp_path / f"{theme}.svg").exists()


def test_visualization_options_convenience_knobs_forward_to_draw(tmp_path: Path) -> None:
    """Grouped convenience knobs should forward to draw."""

    log = tl.trace(nn.Linear(2, 2), torch.randn(1, 2))
    options = VisualizationOptions(
        rolled=False,
        container_path=str(tmp_path / "paper.png"),
        file_format="png",
        theme="paper",
        font_size=14,
        dpi=120,
    )

    graph = log.draw(**tl.options.visualization_to_render_kwargs(options)).dot

    assert isinstance(graph, str)
    assert re.search(r'\bdpi="?120"?', graph)
    assert re.search(r'\bfontsize="?14"?', graph)

    # dpi is a raster-only knob (vizmech D23): vector targets drop it so the
    # coordinate space is not silently rescaled.
    vector_options = VisualizationOptions(
        rolled=False,
        container_path=str(tmp_path / "paper.svg"),
        file_format="svg",
        theme="paper",
        font_size=14,
        dpi=120,
    )
    vector_graph = log.draw(**tl.options.visualization_to_render_kwargs(vector_options)).dot
    assert not re.search(r'\bdpi="?\d', vector_graph)
