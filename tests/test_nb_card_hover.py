"""Hover values on notebook cards, with a complete static floor.

The card must read completely with scripts blocked (GitHub previews,
untrusted notebooks, exported files): every rendered grid cell carries its
source index and value as static attributes. The optional script only adds
a hover readout line, and it is inert-safe: no network, no dynamic code,
no markup writes.
"""

from __future__ import annotations

import itertools
import re
from html.parser import HTMLParser

import torch
from torch import nn

import torchlens as tl
from torchlens.notebook._grid import array_grid_html
from torchlens.notebook.cardtree import (
    _CARD_HOVER_SCRIPT,
    Card,
    CardHtml,
    CardText,
    render_card_html,
)

_SCRIPT = re.compile(r"<script>(.*?)</script>", re.S)


class _Cells(HTMLParser):
    """Collect every grid cell's attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.cells: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "td":
            self.cells.append(dict(attrs))


def _cells(html: str) -> list[dict[str, str | None]]:
    parser = _Cells()
    parser.feed(_SCRIPT.sub("", html))
    return parser.cells


def _index(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.strip("[]").split(", "))


def _card_with_grid(tensor: torch.Tensor) -> str:
    grid = array_grid_html(tensor)
    return render_card_html(Card(title="t", children=(CardHtml(grid.html),)))


def test_static_html_carries_every_value_without_script() -> None:
    """Untruncated grid: each element's index and value is in the static HTML."""

    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 7.0
    cells = _cells(_card_with_grid(tensor))
    assert len(cells) == tensor.numel()
    seen = {}
    for cell in cells:
        index = _index(cell["data-tl-index"])
        seen[index] = cell["data-tl-value"]
        # The native tooltip carries the same index and value with scripts blocked.
        assert cell["title"].startswith(f"{cell['data-tl-index']} = {cell['data-tl-value']}")
    assert set(seen) == set(itertools.product(range(2), range(3), range(4)))
    for index, value in seen.items():
        assert value == f"{float(tensor[index]):.6g}"


def test_truncated_grid_maps_cells_to_source_indices() -> None:
    """Truncated grid: real cells name their true source index; bands name none."""

    torch.manual_seed(3)
    tensor = torch.randn(300, 300)
    cells = _cells(_card_with_grid(tensor))
    real = [cell for cell in cells if "data-tl-index" in cell]
    bands = [cell for cell in cells if "data-tl-index" not in cell]
    assert real and bands
    tail_seen = False
    for cell in real:
        row, col = _index(cell["data-tl-index"])
        tail_seen = tail_seen or row > 150 or col > 150
        # Every real cell carries its number, including cells drawn with an
        # out-of-bounds motif.
        assert cell["data-tl-value"] == f"{float(tensor[row, col]):.6g}"
        assert cell["data-tl-value"] in cell["title"]
    assert tail_seen, "tail edges must map to the end of the axis"


def test_nonfinite_cells_carry_index_and_motif() -> None:
    """NaN and Inf cells keep their index and say what they are."""

    tensor = torch.tensor([[1.0, float("nan")], [float("inf"), -2.0]])
    cells = {cell["data-tl-index"]: cell for cell in _cells(_card_with_grid(tensor))}
    assert set(cells) == {"[0, 0]", "[0, 1]", "[1, 0]", "[1, 1]"}
    assert "nan" in cells["[0, 1]"]["data-tl-value"].lower()
    assert "inf" in cells["[1, 0]"]["data-tl-value"].lower()


def test_script_is_inert_safe() -> None:
    """No network, no dynamic code, no markup writes, no timers, no storage."""

    (body,) = _SCRIPT.findall(_CARD_HOVER_SCRIPT)
    forbidden = (
        "fetch",
        "XMLHttpRequest",
        "WebSocket",
        "EventSource",
        "sendBeacon",
        "import",
        "eval",
        "Function",
        "setTimeout",
        "setInterval",
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "createElement",
        "src",
        "href",
        "http",
        "//",
        "location",
        "cookie",
        "Storage",
        "postMessage",
        "window",
    )
    for token in forbidden:
        assert token not in body, token
    # The only write is textContent on the card's own readout lines.
    assert re.findall(r"\.(\w+)\s*=", body) == ["textContent", "textContent"]
    assert "previousElementSibling" in body


def test_script_rides_only_cards_with_grids() -> None:
    """A card with no grid emits no script; a grid card emits exactly one."""

    plain = render_card_html(Card(title="t", children=(CardText("x"),)))
    assert "<script" not in plain
    grid = _card_with_grid(torch.ones(2, 2))
    assert grid.count("<script>") == 1
    assert grid.endswith(_CARD_HOVER_SCRIPT)


def test_layer_card_has_hover_values_and_readout() -> None:
    """A captured layer card carries the cells, the readout line and the script."""

    trace = tl.trace(nn.Sequential(nn.Linear(4, 3), nn.ReLU()), torch.randn(2, 4))
    label = next(label for label in trace.layer_labels if label.startswith("relu"))
    html = trace[label]._repr_html_()
    assert 'class="tl-grid-hover"' in html
    assert html.count("<script>") == 1
    cells = _cells(html)
    assert len(cells) == 6
    out = trace[label].out
    for cell in cells:
        assert cell["data-tl-value"] == f"{float(out[_index(cell['data-tl-index'])]):.6g}"
