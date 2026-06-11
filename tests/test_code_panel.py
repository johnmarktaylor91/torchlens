"""Tests for source-code panel visualization."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.visualization.code_panel import (
    MAX_CODE_PANEL_LINE_CHARS,
    _wrap_source_line,
    render_code_panel_svg,
)

# Design advance width of the monospace fonts rasterizers resolve "Courier" to
# (Courier/Nimbus Mono/Liberation Mono/Noto Sans Mono are all 0.600 em;
# DejaVu Sans Mono is 0.602 em). Used to compute true rendered text extents,
# independently of whatever sizing factor the panel implementation uses.
_TRUE_MONOSPACE_ADVANCE_EM = 0.602


def _panel_geometry(svg: str) -> tuple[float, float, list[tuple[float, str]]]:
    """Parse a code-panel SVG's canvas width, box right edge, and text runs.

    Parameters
    ----------
    svg:
        SVG document produced by ``render_code_panel_svg``.

    Returns
    -------
    tuple[float, float, list[tuple[float, str]]]
        Canvas width in points, panel-box right edge in points, and one
        ``(right_extent, content)`` pair per monospace text run, where
        ``right_extent`` is the run's true rendered right edge computed from
        monospace font metrics.
    """

    width_match = re.search(r'width="([\d.]+)pt"', svg)
    assert width_match is not None
    canvas_width = float(width_match.group(1))
    box_match = re.search(r'<path fill="#fafafa"[^>]*\bd="([^"]+)"', svg)
    assert box_match is not None
    box_right = max(float(x) for x, _y in re.findall(r"(-?[\d.]+),(-?[\d.]+)", box_match.group(1)))
    runs: list[tuple[float, str]] = []
    for text_match in re.finditer(r"<text\b([^>]*)>([^<]*)</text>", svg):
        attrs, raw_content = text_match.group(1), text_match.group(2)
        if 'font-family="Courier' not in attrs:
            continue
        x_match = re.search(r'\bx="(-?[\d.]+)"', attrs)
        size_match = re.search(r'\bfont-size="([\d.]+)"', attrs)
        assert x_match is not None and size_match is not None
        content = html.unescape(raw_content)
        right_extent = float(x_match.group(1)) + len(content) * _TRUE_MONOSPACE_ADVANCE_EM * float(
            size_match.group(1)
        )
        runs.append((right_extent, content))
    return canvas_width, box_right, runs


def _panel_code_lines(svg: str) -> list[str]:
    """Return the displayed code lines of a panel SVG, nbsp-normalized.

    Parameters
    ----------
    svg:
        SVG document produced by ``render_code_panel_svg``.

    Returns
    -------
    list[str]
        Monospace text-run contents with non-breaking spaces normalized to
        regular spaces, excluding the source-link row.
    """

    _, _, runs = _panel_geometry(svg)
    return [
        content.replace("\xa0", " ")
        for _, content in runs
        if content not in ("Open source", "Source code")
    ]


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


def test_code_panel_long_line_fits_inside_panel_box() -> None:
    """A long single line must never extend past the panel box or canvas.

    Graphviz under-measures Courier text, so without explicit monospace
    sizing the rendered glyph run overruns the panel box and gets clipped at
    the SVG viewport during composition/rasterization.
    """

    long_line = (
        "def forward(self, x: torch.Tensor, "
        "attention_mask: torch.Tensor | None = None) -> torch.Tensor:"
    )
    assert len(long_line) <= MAX_CODE_PANEL_LINE_CHARS
    svg = render_code_panel_svg(long_line)
    canvas_width, box_right, runs = _panel_geometry(svg)

    assert runs, "expected monospace text runs in the panel SVG"
    for right_extent, content in runs:
        assert right_extent <= box_right, (
            f"text run {content!r} extends to {right_extent:.1f}pt, past the "
            f"panel box right edge at {box_right:.1f}pt"
        )
        assert right_extent <= canvas_width, (
            f"text run {content!r} extends to {right_extent:.1f}pt, past the "
            f"SVG canvas edge at {canvas_width:.1f}pt"
        )


def test_code_panel_wraps_lines_beyond_char_cap() -> None:
    """Lines beyond the character cap wrap with a hanging indent, unclipped."""

    words = " ".join(f"word{idx:03d}" for idx in range(40))
    long_line = f"        result = compute({words})"
    assert len(long_line) > MAX_CODE_PANEL_LINE_CHARS
    svg = render_code_panel_svg(long_line)
    canvas_width, box_right, runs = _panel_geometry(svg)
    code_lines = [line for line in _panel_code_lines(svg) if line.strip()]

    assert len(code_lines) > 1, "expected the long line to wrap into multiple rows"
    assert all(len(line) <= MAX_CODE_PANEL_LINE_CHARS for line in code_lines)
    assert code_lines[0].startswith("        result = compute(")
    assert all(line.startswith(" " * 12) for line in code_lines[1:])
    rejoined = " ".join(line.strip() for line in code_lines)
    for idx in range(40):
        assert f"word{idx:03d}" in rejoined
    for right_extent, content in runs:
        assert right_extent <= box_right, f"wrapped run {content!r} overruns the panel box"
        assert right_extent <= canvas_width, f"wrapped run {content!r} overruns the canvas"


def test_wrap_source_line_short_line_unchanged() -> None:
    """Lines within the cap pass through wrapping untouched."""

    line = "    return self.linear(x).relu()"
    assert _wrap_source_line(line) == [line]


def test_wrap_source_line_hard_splits_monster_tokens() -> None:
    """An unbreakable token longer than the cap is hard-split, never clipped."""

    line = "x" * 400
    pieces = _wrap_source_line(line)

    assert len(pieces) > 1
    assert all(len(piece) <= MAX_CODE_PANEL_LINE_CHARS for piece in pieces)
    assert "".join(piece.lstrip(" ") for piece in pieces) == line


def test_wrap_source_line_pathological_indent_still_wraps() -> None:
    """Indent nearly as wide as the cap falls back to a shallow hanging indent."""

    line = " " * (MAX_CODE_PANEL_LINE_CHARS - 4) + " ".join(["token"] * 30)
    pieces = _wrap_source_line(line)

    assert all(len(piece) <= MAX_CODE_PANEL_LINE_CHARS for piece in pieces)
    assert sum(piece.count("token") for piece in pieces) == 30
