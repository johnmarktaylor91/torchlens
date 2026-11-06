"""Text drawn in a browser stays inside its node box, its key column and the page.

Graphviz measures text with every glyph advance rounded to a 96 dpi pixel; a
browser draws the face's real advances, about 3 % wider at 14 pt. Measured on
the GPT-2 drawing (Graphviz 2.43): an embedding node's long row touched its
border, the header's edit line was cut off at the page edge, and the key's
"module run as one op" ran into the next swatch. These tests read the SVG
Graphviz writes and place each text at its real advances in the bundled face.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from example_models import TinyReluAdd
from torch import nn

import torchlens as tl
from torchlens._visualization._header_block import KeySwatch, _key_rows_html
from torchlens._visualization._label_fit import fitted_cell_width
from torchlens._visualization._text_metrics import graphviz_text_width, text_width
from torchlens._visualization.node_spec import render_lines_to_html
from torchlens._visualization.themes import resolve_theme

pytestmark = pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz dot not installed")

#: GPT-2's ``transformer.wte`` constructor row, the row that touched its border.
_LONG_ROW = "num_embeddings=50257, embedding_dim=768, padding_idx=-1"
_ATTR = re.compile(r'([a-zA-Z_:][-a-zA-Z0-9_:.]*)="([^"]*)"')


def _attrs(tag: str) -> dict[str, str]:
    """Return the attributes of one SVG tag."""

    return dict(_ATTR.findall(tag))


def _drawn_right(text_tag: str, body: str) -> float:
    """Return a start-anchored SVG text's right end at the bundled face's real advances."""

    attrs = _attrs(text_tag)
    size = float(attrs.get("font-size", "14"))
    return float(attrs["x"]) + text_width(body, size, attrs.get("font-weight") == "bold")


def test_graphviz_measure_is_the_pixel_rounded_advance() -> None:
    """The rounding model reproduces the widths Graphviz 2.43 laid the GPT-2 drawing out with."""

    assert graphviz_text_width(_LONG_ROW, 14.0) == pytest.approx(390.0, abs=1.0)
    assert graphviz_text_width("(1, 5, 768), 15.4 KB", 14.0) == pytest.approx(117.0, abs=1.0)
    assert text_width(_LONG_ROW, 14.0) - graphviz_text_width(_LONG_ROW, 14.0) > 10.0


def test_only_a_long_row_widens_its_cell() -> None:
    """Ordinary labels keep their bytes; a long row holds its cell at the drawn width."""

    short = render_lines_to_html(["embedding_1_2", "(1, 5, 768), 15.4 KB", "@transformer.wte"])
    assert "WIDTH=" not in short
    long = render_lines_to_html(["embedding_1_2", "(1, 5, 768), 15.4 KB", _LONG_ROW])
    width = fitted_cell_width([[(_LONG_ROW, 14.0, False)]])
    assert width is not None and width >= text_width(_LONG_ROW, 14.0)
    assert f'<TR><TD WIDTH="{width}" ALIGN="CENTER"><B>embedding_1_2</B>' in long


def test_long_row_draws_inside_its_box(tmp_path: Path) -> None:
    """In the SVG, the long row's real-advance right end sits inside the node's border."""

    def add_row(node: Any, spec: Any) -> Any:
        """Give every op box the long constructor row."""

        return spec.replace(lines=[*spec.lines, _LONG_ROW], shape="box")

    trace = tl.trace(TinyReluAdd(), torch.randn(2, 3))
    try:
        path = trace.draw(file_format="svg", node_spec_fn=add_row, path=str(tmp_path / "rows")).path
    finally:
        trace.cleanup()
    svg = Path(path).read_text(encoding="utf-8")
    checked = 0
    for group in re.findall(r'<g id="node\d+" class="node[^"]*">(.*?)</g>', svg, flags=re.S):
        polygon = re.search(r'<polygon [^>]*points="([^"]*)"', group)
        row = re.search(r"(<text [^>]*>)" + re.escape(_LONG_ROW.replace("-", "&#45;")), group)
        if polygon is None or row is None:
            continue
        right_border = max(float(point.split(",")[0]) for point in polygon.group(1).split())
        assert _drawn_right(row.group(1), _LONG_ROW) <= right_border - 2.0
        checked += 1
    assert checked >= 2


class _ALongModelNameThatSetsTheWidthOfItsOwnDrawingHeader(nn.Module):
    """A one-relu model whose class name, the header's first line, is wider than its graph."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``relu(x)``."""

        return torch.relu(x)


def test_header_line_fits_the_page(tmp_path: Path) -> None:
    """A header line wider than the graph fits the SVG page at real advances."""

    trace = tl.trace(_ALongModelNameThatSetsTheWidthOfItsOwnDrawingHeader(), torch.randn(2, 3))
    try:
        path = trace.draw(file_format="svg", path=str(tmp_path / "header")).path
    finally:
        trace.cleanup()
    svg = Path(path).read_text(encoding="utf-8")
    width = float(re.search(r'viewBox="[\d.]+ [\d.]+ ([\d.]+) ', svg).group(1))
    assert re.search(rf'<svg width="{round(width)}pt"', svg)
    head = svg[: svg.index('<g id="node')]
    name = _ALongModelNameThatSetsTheWidthOfItsOwnDrawingHeader.__name__
    title = re.search(r"(<text [^>]*>)" + name, head)
    assert title is not None
    # Graph units start 4 pt in from the page edge; keep the 4 pt margin on the right.
    assert _drawn_right(title.group(1), name) + 4.0 <= width - 4.0 + 0.01
    page = re.search(r'<polygon fill="[^"]*" stroke="transparent" points="([^"]*)"', svg)
    assert page is not None
    assert max(float(p.split(",")[0]) for p in page.group(1).split()) == pytest.approx(width - 4)


def test_key_word_cells_hold_their_measured_width() -> None:
    """Each key word cell holds its column's measured width, so no word reaches the next swatch."""

    theme = resolve_theme(None)
    swatches = tuple(
        KeySwatch(text=text, shape="box", fill="#FFFFFF", border="#000000")
        for text in ("input", "output", "module", "module run as one op", "collapsed module")
    )
    html = _key_rows_html(swatches, theme)
    widths = [int(value) for value in re.findall(r'<TD ALIGN="LEFT" WIDTH="(\d+)">', html)]
    assert len(widths) == len(swatches)
    size = theme.typography.secondary_size
    assert widths[0] >= text_width("module run as one op", size) + 2.0  # column 1 of row 2
    table = int(re.search(r'FIXEDSIZE="TRUE" ALIGN="LEFT" WIDTH="(\d+)"', html).group(1))
    assert table == sum(widths[:3]) + 3 * 22 + 3 * 7


class _MLP(nn.Module):
    """Expand, GELU, project, dropout: GELU and dropout draw as module boxes."""

    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Linear(4, 8)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(8, 4)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class _WithMLP(nn.Module):
    """A model holding the block at ``mlp``."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = _MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.mlp(x)


def test_key_words_end_before_the_next_swatch(tmp_path: Path) -> None:
    """In the SVG key, every word's real-advance right end stops short of the next swatch."""

    trace = tl.trace(_WithMLP().eval(), torch.randn(1, 4))
    try:
        path = trace.draw(file_format="svg", module="mlp", path=str(tmp_path / "key")).path
    finally:
        trace.cleanup()
    svg = Path(path).read_text(encoding="utf-8")
    head = svg[: svg.index('<g id="node')]
    images = [_attrs(tag) for tag in re.findall(r"<image [^>]*>", head)]
    words = re.findall(r'(<text [^>]*font-size="10.00"[^>]*>)([^<]+)</text>', head)
    assert any(body == "module run as one op" for _tag, body in words)
    checked = 0
    for tag, body in words:
        attrs = _attrs(tag)
        x, y = float(attrs["x"]), float(attrs["y"])
        right = _drawn_right(tag, body)
        nearby = [
            float(image["x"])
            for image in images
            if float(image["x"]) > x
            and abs(float(image["y"]) + float(image["height"]) / 2 - y) < 12
        ]
        if nearby:
            assert right <= min(nearby) - 2.0, (body, right, min(nearby))
            checked += 1
    assert checked >= 2
