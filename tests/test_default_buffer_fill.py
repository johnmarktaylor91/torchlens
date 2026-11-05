"""The default skin draws buffer nodes white, as the released TorchLens did.

The released look fills only inputs, outputs, weights and true/false tests; a
buffer is a white cylinder, its shape the cue. A later palette filled buffers
Okabe-Ito orange (``#E69F00``) in the default skin. These tests pin the white
fill on every path that paints a buffer under the default skin (the unrolled
and rolled drawings, a Bundle's merged drawing and the key) and check that a
skin which colours buffers on purpose (``colorblind``) still does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._header_block import key_swatches
from torchlens._visualization.themes import THEME_PRESETS

#: The Okabe-Ito hues; none is a default-skin fill.
_OKABE_ITO = ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
_FILL_RE = re.compile(r'fillcolor="?([^"\s\]]+)')


class _ScaledNorm(nn.Module):
    """A Linear run twice through a registered buffer, then an eval BatchNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.register_buffer("scale", torch.full((4,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            x = self.lin(x) * self.scale
        return self.bn(x)


def _trace(shift: float = 0.0) -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(_ScaledNorm().eval(), torch.randn(2, 4) + shift)


def _cylinder_fills(dot: str) -> list[str]:
    """Return the fill of every drawn cylinder (buffer) node."""

    fills = []
    for line in dot.splitlines():
        if "shape=cylinder" in line:
            match = _FILL_RE.search(line)
            fills.append(match.group(1) if match else "")
    return fills


def _okabe_ito_in(dot: str) -> list[str]:
    return [hue for hue in _OKABE_ITO if hue.lower() in dot.lower()]


@pytest.mark.parametrize("rolled", [False, True], ids=["unrolled", "rolled"])
def test_default_skin_draws_buffers_white(rolled: bool, tmp_path: Path) -> None:
    """Every buffer cylinder is white and no Okabe-Ito hue appears anywhere."""

    trace = _trace()
    dot = trace.draw(
        path=str(tmp_path / "default"), file_format="gv", show_buffers="always", rolled=rolled
    ).dot
    fills = _cylinder_fills(dot)
    assert fills, "the drawing has no buffer cylinder to check"
    assert set(fills) == {"white"}, fills
    assert _okabe_ito_in(dot) == []


def test_default_skin_bundle_draws_buffers_white(tmp_path: Path) -> None:
    """A Bundle's merged drawing paints buffers like the single-trace drawing."""

    bundle = tl.Bundle({"a": _trace(), "b": _trace(shift=1.0)}, baseline="a")
    path = tmp_path / "bundle.gv"
    drawing = bundle.draw(str(path))
    dot = getattr(drawing, "dot", None) or path.read_text()
    fills = _cylinder_fills(dot)
    assert fills, "the bundle drawing has no buffer cylinder to check"
    assert set(fills) == {"white"}, fills
    assert _okabe_ito_in(dot) == []


def test_default_skin_key_shows_a_white_buffer_cylinder() -> None:
    """The key's buffer entry is a white cylinder, like the node it explains."""

    swatches = key_swatches(THEME_PRESETS["torchlens"], frozenset({"buffer"}), has_module=False)
    buffer_rows = [(s.shape, s.fill) for s in swatches if s.text == "buffer"]
    assert buffer_rows == [("cylinder", "white")]


def test_colorblind_skin_keeps_its_buffer_colour(tmp_path: Path) -> None:
    """The opt-in colorblind skin still fills buffers with its Okabe-Ito orange."""

    dot = (
        _trace()
        .draw(
            path=str(tmp_path / "cb"), file_format="gv", show_buffers="always", theme="colorblind"
        )
        .dot
    )
    assert set(_cylinder_fills(dot)) == {"#E69F00"}
