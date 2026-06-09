"""Tests for source-code panel visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl


def _render_dot(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    """Render a Trace to DOT using a temporary SVG output path.

    Parameters
    ----------
    log:
        Model log to render.
    tmp_path:
        Temporary directory supplied by pytest.
    **kwargs:
        Additional render options.

    Returns
    -------
    str
        DOT source returned by ``Trace.draw``.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


def _render_svg_output(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    """Render to SVG and return the composed output file's contents.

    The code panel is composed beside the graph as a separate render, so its text
    lives in the saved SVG file rather than the graph's returned DOT source.

    Parameters
    ----------
    log:
        Model log to render.
    tmp_path:
        Temporary directory supplied by pytest.
    **kwargs:
        Additional render options.

    Returns
    -------
    str
        Contents of the rendered SVG file.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    out_path = tmp_path / "graph"
    log.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(out_path),
        **kwargs,
    )
    return (tmp_path / "graph.svg").read_text(encoding="utf-8")


class _CodePanelModel(nn.Module):
    """Small model with inspectable source for code-panel tests."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return self.linear(x).relu()


def test_code_panel_false_no_panel(tmp_path: Path) -> None:
    """Default rendering should not include a code panel in the graph or output."""

    log = tl.trace(_CodePanelModel(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path)
    svg = _render_svg_output(log, tmp_path)

    assert "cluster_torchlens_code_panel" not in dot
    assert "__tl_code_panel_node" not in dot
    assert "Source code" not in svg


def test_code_panel_does_not_distort_graph_dot(tmp_path: Path) -> None:
    """A code panel is composed separately, so it never enters the graph's DOT.

    The graph layout must be byte-for-byte identical whether or not a code panel
    is requested -- the panel is a side-by-side composition, not a subgraph.
    """

    log = tl.trace(_CodePanelModel(), torch.randn(1, 4))
    plain_dot = _render_dot(log, tmp_path)
    paneled_dot = _render_dot(log, tmp_path, code_panel=True)

    assert "cluster_torchlens_code_panel" not in paneled_dot
    assert "def forward" not in paneled_dot
    assert plain_dot == paneled_dot


def test_code_panel_true_emits_forward_source(tmp_path: Path) -> None:
    """The True shorthand should render forward source into the composed output."""

    log = tl.trace(_CodePanelModel(), torch.randn(1, 4))
    svg = _render_svg_output(log, tmp_path, code_panel=True)

    assert "def forward" in svg
    assert "return self.linear(x).relu()" in svg


def test_code_panel_class_emits_class_source(tmp_path: Path) -> None:
    """The class mode should render the model class definition."""

    log = tl.trace(_CodePanelModel(), torch.randn(1, 4))
    svg = _render_svg_output(log, tmp_path, code_panel="class")

    assert "class _CodePanelModel" in svg


def test_code_panel_init_plus_forward(tmp_path: Path) -> None:
    """The init+forward mode should include both method definitions."""

    log = tl.trace(_CodePanelModel(), torch.randn(1, 4))
    svg = _render_svg_output(log, tmp_path, code_panel="init+forward")

    assert "def __init__" in svg
    assert "def forward" in svg


def test_code_panel_callable_overrides(tmp_path: Path) -> None:
    """Callable code-panel options should use returned text verbatim."""

    model = _CodePanelModel()
    log = tl.trace(model, torch.randn(1, 4))
    svg = _render_svg_output(log, tmp_path, code_panel=lambda model: "CUSTOM_TEXT_TOKEN")

    assert "CUSTOM_TEXT_TOKEN" in svg


def test_code_panel_html_escape(tmp_path: Path) -> None:
    """Code-panel text should escape HTML metacharacters in the rendered output."""

    model = _CodePanelModel()
    log = tl.trace(model, torch.randn(1, 4))
    svg = _render_svg_output(log, tmp_path, code_panel=lambda model: "x < y > z & q")

    assert "x &lt; y &gt; z &amp; q" in svg


def test_code_panel_truncates_long_source(tmp_path: Path) -> None:
    """Long code-panel text should be capped with a truncation marker."""

    model = _CodePanelModel()
    log = tl.trace(model, torch.randn(1, 4))
    source_text = "\n".join(f"line {idx}" for idx in range(125))
    svg = _render_svg_output(log, tmp_path, code_panel=lambda model: source_text)

    assert "... 5 more lines" in svg
    assert "line 119" in svg
    assert "line 120" not in svg
