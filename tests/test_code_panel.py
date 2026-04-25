"""Tests for source-code panel visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl


def _render_dot(log: tl.ModelLog, tmp_path: Path, **kwargs: Any) -> str:
    """Render a ModelLog to DOT using a temporary SVG output path.

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
        DOT source returned by ``ModelLog.render_graph``.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.render_graph(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


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
    """Default rendering should not include a code-panel cluster."""

    log = tl.log_forward_pass(_CodePanelModel(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path)

    assert "cluster_torchlens_code_panel" not in dot
    assert "__tl_code_panel_node" not in dot


def test_code_panel_true_emits_forward_source(tmp_path: Path) -> None:
    """The True shorthand should render forward source."""

    log = tl.log_forward_pass(_CodePanelModel(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path, code_panel=True)

    assert "def forward" in dot
    assert "return self.linear(x).relu()" in dot


def test_code_panel_class_emits_class_source(tmp_path: Path) -> None:
    """The class mode should render the model class definition."""

    log = tl.log_forward_pass(_CodePanelModel(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path, code_panel="class")

    assert "class _CodePanelModel" in dot


def test_code_panel_init_plus_forward(tmp_path: Path) -> None:
    """The init+forward mode should include both method definitions."""

    log = tl.log_forward_pass(_CodePanelModel(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path, code_panel="init+forward")

    assert "def __init__" in dot
    assert "def forward" in dot


def test_code_panel_callable_overrides(tmp_path: Path) -> None:
    """Callable code-panel options should use returned text verbatim."""

    model = _CodePanelModel()
    log = tl.log_forward_pass(model, torch.randn(1, 4))
    dot = _render_dot(log, tmp_path, code_panel=lambda model: "CUSTOM_TEXT_TOKEN")

    assert "CUSTOM_TEXT_TOKEN" in dot


def test_code_panel_html_escape(tmp_path: Path) -> None:
    """Code-panel text should escape Graphviz HTML metacharacters."""

    model = _CodePanelModel()
    log = tl.log_forward_pass(model, torch.randn(1, 4))
    dot = _render_dot(log, tmp_path, code_panel=lambda model: "x < y > z & q")

    assert "x &lt; y &gt; z &amp; q" in dot


def test_code_panel_truncates_long_source(tmp_path: Path) -> None:
    """Long code-panel text should be capped with a truncation marker."""

    model = _CodePanelModel()
    log = tl.log_forward_pass(model, torch.randn(1, 4))
    source_text = "\n".join(f"line {idx}" for idx in range(125))
    dot = _render_dot(log, tmp_path, code_panel=lambda model: source_text)

    assert "... 5 more lines" in dot
    assert "line 119" in dot
    assert "line 120" not in dot
