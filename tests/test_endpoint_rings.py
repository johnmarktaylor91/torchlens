"""Input and output marks under a colour scale.

With no colour channel the input is a pale green oval and the output a salmon
one, as released. Under ``color_by=`` the fill belongs to the data: the two
endpoints move their role to a ring (the skin's input or output colour as a
2.5 pt border on the page fill, or on their scale colour when they carry a
value), and the key lists ``input`` and ``output`` as those rings. ``size_by=``
alone changes no fill. A focused draw's ``from`` and ``to`` stubs follow the
same rule.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._endpoint_rings import (
    ENDPOINT_RING_PENWIDTH,
    endpoint_ring_role,
    endpoint_ring_swatch,
)
from torchlens._visualization._header_block import KeySwatch, key_swatches
from torchlens._visualization._legend import drawn_legend_roles
from torchlens._visualization.themes import DARK_PAGE, resolve_theme

pytestmark = pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz dot not installed")

_INPUT = "#98FB98"
_OUTPUT = "#ff9999"


class _SmallMLP(nn.Module):
    """Two linears around a ReLU module (the focus target)."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack."""

        return self.fc2(self.relu(self.fc1(x)))


@pytest.fixture(scope="module")
def mlp_log() -> Any:
    """One traced small MLP for every test here."""

    torch.manual_seed(0)
    log = tl.trace(_SmallMLP(), torch.randn(3, 4))
    try:
        yield log
    finally:
        log.cleanup()


def _draw(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    """Return the dot source of one SVG draw."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.draw(file_format="svg", path=str(tmp_path / "graph"), **kwargs).dot


def _body(dot: str) -> str:
    """Return the dot source without the header's ``graph [`` line."""

    return "\n".join(line for line in dot.splitlines() if not line.strip().startswith("graph ["))


def _header(dot: str) -> str:
    """Return the header's ``graph [`` line."""

    return next(line for line in dot.splitlines() if line.strip().startswith("graph ["))


def _ring_lines(dot: str, ink: str) -> list[str]:
    """Return the node lines ringed in ``ink`` at the endpoint width."""

    pattern = re.compile(rf'color="{re.escape(ink)}"[^\n]*?penwidth={ENDPOINT_RING_PENWIDTH}')
    return [line for line in _body(dot).splitlines() if pattern.search(line)]


def _fill_count(dot: str, fill: str) -> int:
    """Return how many drawn nodes are filled with ``fill``."""

    return len(re.findall(rf'fillcolor="{re.escape(fill)}"', _body(dot)))


def test_plain_draw_keeps_the_released_endpoint_fills(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """No colour channel: green input fill, salmon output fill, no ring anywhere."""

    dot = _draw(mlp_log, tmp_path)
    assert _fill_count(dot, _INPUT) == 1
    assert _fill_count(dot, _OUTPUT) == 1
    assert not _ring_lines(dot, _INPUT) and not _ring_lines(dot, _OUTPUT)
    assert f"penwidth={ENDPOINT_RING_PENWIDTH}" not in _body(dot)


def test_size_by_alone_keeps_the_released_endpoint_fills(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """``size_by`` touches no fill, so the endpoints stay filled, not ringed."""

    dot = _draw(mlp_log, tmp_path, size_by="bytes")
    assert _fill_count(dot, _INPUT) == 1
    assert _fill_count(dot, _OUTPUT) == 1
    assert not _ring_lines(dot, _INPUT) and not _ring_lines(dot, _OUTPUT)


def test_color_by_moves_the_endpoint_roles_to_rings(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """Under ``color_by`` no node wears a role fill; each endpoint wears its ring once."""

    dot = _draw(mlp_log, tmp_path, color_by="bytes")
    assert _fill_count(dot, _INPUT) == 0
    assert _fill_count(dot, _OUTPUT) == 0
    assert len(_ring_lines(dot, _INPUT)) == 1
    assert len(_ring_lines(dot, _OUTPUT)) == 1
    # Each ring sits on one node only: no node carries both inks.
    assert _ring_lines(dot, _INPUT) != _ring_lines(dot, _OUTPUT)


def test_color_by_key_lists_the_endpoints_as_rings(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """The drawn key names input and output, and its entries are the rings."""

    dot = _draw(mlp_log, tmp_path, color_by="bytes", show_legend=True)
    header = _header(dot)
    assert ">input<" in header and ">output<" in header
    theme = resolve_theme("torchlens")
    swatches = key_swatches(
        theme, frozenset({"input", "output"}), has_module=False, endpoint_rings=True
    )
    by_text = {swatch.text: swatch for swatch in swatches}
    assert by_text["input"] == KeySwatch(
        "input", "oval", theme.default_fill, _INPUT, ENDPOINT_RING_PENWIDTH
    )
    assert by_text["output"] == KeySwatch(
        "output", "oval", theme.default_fill, _OUTPUT, ENDPOINT_RING_PENWIDTH
    )
    # Without a colour scale the same roles are the released filled ovals.
    plain = {
        s.text: s for s in key_swatches(theme, frozenset({"input", "output"}), has_module=False)
    }
    assert plain["input"].fill.lower() == _INPUT.lower() and plain["input"].line == 1.0
    assert plain["output"].fill.lower() == _OUTPUT.lower() and plain["output"].line == 1.0


def test_focus_stubs_ring_under_color_by(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """A ``module=`` draw's from/to stubs ring instead of filling under ``color_by``."""

    plain = _draw(mlp_log, tmp_path / "plain", module="relu")
    assert _fill_count(plain, _INPUT) == 1 and _fill_count(plain, _OUTPUT) == 1
    scaled = _draw(mlp_log, tmp_path / "scaled", module="relu", color_by="bytes")
    assert _fill_count(scaled, _INPUT) == 0 and _fill_count(scaled, _OUTPUT) == 0
    (input_stub,) = _ring_lines(scaled, _INPUT)
    (output_stub,) = _ring_lines(scaled, _OUTPUT)
    # A stub carries no value, so it keeps the page fill inside its ring.
    for line in (input_stub, output_stub):
        assert re.search(r"fillcolor=(white|\"#FFFFFF\")", line, re.IGNORECASE), line


def test_dark_skin_rings_sit_on_the_dark_page(mlp_log: tl.Trace, tmp_path: Path) -> None:
    """On the dark skin a valueless endpoint stub keeps the dark page inside its ring."""

    dot = _draw(mlp_log, tmp_path, module="relu", color_by="bytes", theme="dark")
    (input_stub,) = _ring_lines(dot, _INPUT)
    assert f'fillcolor="{DARK_PAGE}"' in input_stub


def test_ring_detection_needs_the_ring_width() -> None:
    """The key counts a ring by colour and width; a thin role-coloured border is not one."""

    theme = resolve_theme("torchlens")
    ring = {"shape": "oval", "fillcolor": "white", "color": _INPUT, "penwidth": 2.5}
    assert endpoint_ring_role(ring, theme) == "input"
    assert "input" in drawn_legend_roles([ring], theme)
    thin = dict(ring, penwidth=None)
    assert endpoint_ring_role(thin, theme) is None
    assert "input" not in drawn_legend_roles([thin], theme)
    other = KeySwatch("module", "frame", "white", "black", 1.5)
    assert endpoint_ring_swatch(theme, other) is other
