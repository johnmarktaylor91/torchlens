"""Tests for source-code panel visualization."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._visualization import code_panel
from torchlens._visualization.code_panel import (
    MAX_CODE_PANEL_LINE_CHARS,
    _wrap_source_line,
    capture_model_source_code,
)


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
        file_format="svg",
        path=str(tmp_path / "graph"),
        **kwargs,
    ).dot


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
        file_format="svg",
        path=str(out_path),
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


def test_code_panel_true_emits_forward_source(tmp_path: Path) -> None:
    """The True shorthand should render forward source into the composed output."""

    log = tl.trace(_CodePanelModel(), torch.randn(1, 4))
    svg = _render_svg_output(log, tmp_path, code_panel=True)

    assert "def forward" in svg
    assert "return self.linear(x).relu()" in svg


def test_source_stripped_artifact_visualizes_without_crashing(tmp_path: Path) -> None:
    """A source-stripped bundle must degrade to a placeholder panel, never crash."""

    log = tl.trace(_CodePanelModel(), torch.randn(1, 4))
    bundle_path = tmp_path / "stripped.tlspec"
    log.save(bundle_path, include_source=False, overwrite=True)
    loaded = tl.load(bundle_path)

    # Requesting a source panel on the reloaded, source-stripped trace must not
    # raise; it renders a "not embedded" placeholder instead of the source.
    svg = _render_svg_output(loaded, tmp_path, code_panel="forward")
    assert "not embedded" in svg
    assert "def forward" not in svg


def test_code_panel_truncates_long_source(tmp_path: Path) -> None:
    """Long code-panel text should be capped with a truncation marker."""

    model = _CodePanelModel()
    log = tl.trace(model, torch.randn(1, 4))
    source_text = "\n".join(f"line {idx}" for idx in range(125))
    svg = _render_svg_output(log, tmp_path, code_panel=lambda model: source_text)

    assert "... 5 more lines" in svg
    assert "line 119" in svg
    assert "line 120" not in svg


def test_wrap_source_line_pathological_indent_still_wraps() -> None:
    """Indent nearly as wide as the cap falls back to a shallow hanging indent."""

    line = " " * (MAX_CODE_PANEL_LINE_CHARS - 4) + " ".join(["token"] * 30)
    pieces = _wrap_source_line(line)

    assert all(len(piece) <= MAX_CODE_PANEL_LINE_CHARS for piece in pieces)
    assert sum(piece.count("token") for piece in pieces) == 30


def _count_source_reads(monkeypatch: Any) -> dict[str, int]:
    """Count ``inspect.getsourcelines`` calls made by the code-panel module.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture used to install the counting wrapper.

    Returns
    -------
    dict[str, int]
        Mutable counter updated under the ``"n"`` key on every call.
    """

    counter = {"n": 0}
    real = inspect.getsourcelines

    def counting(obj: Any) -> Any:
        counter["n"] += 1
        return real(obj)

    monkeypatch.setattr(code_panel.inspect, "getsourcelines", counting)
    return counter


class _MemoProbe(nn.Module):
    """Model whose class/init/forward source the memo tests read."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the linear projection of ``x``."""

        return self.linear(x)


def test_source_capture_is_memoized_per_class(monkeypatch: Any) -> None:
    """Repeat captures of one class read the source files exactly once each."""

    baseline = capture_model_source_code(_MemoProbe())  # warm the memo
    counter = _count_source_reads(monkeypatch)

    for _ in range(5):
        # Fresh instances too: the memo keys on the class, not the instance.
        assert capture_model_source_code(_MemoProbe()) == baseline

    assert counter["n"] == 0


def test_source_capture_tolerates_unmemoizable_targets() -> None:
    """Targets that cannot be weakly referenced still resolve, uncached."""

    for target in (len, object.__init__, 5, None, "text"):
        assert code_panel._get_source_or_empty(target) == ""
