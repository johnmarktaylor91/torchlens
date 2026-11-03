"""Skins own all of a render's ink, and the legend lists only what is drawn.

Defects pinned here (UI sprint visual track, renderer findings):

- the dark skin left edges, edge labels and module-box borders black and
  module boxes white on its near-black page, and kept its near-white node
  text on light semantic fills (about 1.2:1 on the input fill);
- the role legend listed every category whatever the graph held;
- the legend had no plain-op row, and buffers were filled like plain ops,
  so the buffer swatch and a plain op looked the same in every skin;
- the header's capture advisory printed the model file's absolute path.

White-page skins must keep their historical edge and module-box DOT.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._header_block import key_swatches
from torchlens._visualization._legend import drawn_legend_roles
from torchlens._visualization.lenses.audit import palette_distinguishability
from torchlens._visualization.themes import THEME_PRESETS


class _InnerRes(nn.Module):
    """A residual block: one Linear plus an add, so its module box holds two ops."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + x


class _NestedRes(nn.Module):
    """Two nested module boxes around the residual block."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(_InnerRes())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Stateful(nn.Module):
    """A registered buffer read by a plain op."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.full((3,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x * self.scale)


class _Branch(nn.Module):
    """A tensor-driven if; capture records a scalar-escape advisory at the ``if``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x * 2
        return x - 1


def _draw(model: nn.Module, x: torch.Tensor, tmp_path: Path, name: str, **kwargs: object) -> str:
    """Trace ``model`` and return the DOT source of an SVG render."""

    torch.manual_seed(0)
    trace = tl.trace(model, x)
    return str(
        trace.draw(
            file_format="svg",
            path=str(tmp_path / name),
            **kwargs,
        )
    )


def _edge_lines(dot: str) -> list[str]:
    """Return the drawn edge statements (the legend's invisible tie excluded)."""

    return [line for line in dot.splitlines() if " -> " in line and "style=invis" not in line]


def _fills(skin: str) -> tuple[str, str]:
    """Return the plain-op fill and the key's buffer fill under ``skin``."""

    from torchlens._visualization.node_spec import NodeSpec
    from torchlens._visualization.themes import apply_theme_to_spec

    theme = THEME_PRESETS[skin]
    plain = str(apply_theme_to_spec(NodeSpec(["x"], fillcolor="white"), theme).fillcolor)
    buffer = next(s.fill for s in key_swatches(theme, frozenset({"buffer"}), has_module=False))
    return plain, buffer


@pytest.mark.parametrize("skin", sorted(THEME_PRESETS))
def test_plain_op_row_and_buffer_fill_are_distinguishable(skin: str) -> None:
    """Every skin's buffer fill differs from its plain-op fill under CVD."""

    plain, buffer = _fills(skin)
    findings = palette_distinguishability({"plain op": plain, "buffer": buffer})
    assert len(findings) == 4
    assert all(finding.passed for finding in findings), [f.detail for f in findings]


def test_drawn_roles_ignore_layout_helpers_and_collapsed_boxes() -> None:
    """Invisible helpers and default-fill collapsed boxes are not plain ops."""

    theme = THEME_PRESETS["torchlens"]
    roles = drawn_legend_roles(
        [
            {"shape": "point", "fillcolor": "white"},
            {"shape": "oval", "style": "invis", "fillcolor": "white"},
            {"shape": "box3d", "fillcolor": "white"},
        ],
        theme,
    )
    assert roles == frozenset()


def test_as_node_intervention_keeps_the_site_inside_its_module_boxes(tmp_path: Path) -> None:
    """The site-to-hook edge sits in the same cluster as the hook-to-child edge."""

    torch.manual_seed(0)
    trace = tl.trace(
        _NestedRes(),
        torch.randn(1, 3),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    fork = trace.fork("edited")
    fork.do(fork[tl.units("linear_1_1:1", [(0, 0)])], tl.zero_ablate())
    dot = str(
        fork.draw(
            file_format="svg",
            path=str(tmp_path / "as_node"),
            intervention_mode="as_node",
        )
    )
    into_hook = [line for line in _edge_lines(dot) if "-> intervention_hook_" in line]
    out_of_hook = [
        line for line in _edge_lines(dot) if line.lstrip().startswith("intervention_hook_")
    ]
    assert len(into_hook) == 1 and len(out_of_hook) == 1

    def depth(line: str) -> int:
        return len(line) - len(line.lstrip("\t"))

    assert depth(into_hook[0]) == depth(out_of_hook[0]) > 1


@pytest.mark.parametrize("skin", ["dark"])
def test_rank_path_page_follows_the_skin(skin: str, tmp_path: Path) -> None:
    """The rank engine paints the dark page; white pages stay implicit, as before."""

    dot = _draw(
        nn.Sequential(nn.Linear(3, 3), nn.ReLU()),
        torch.randn(1, 3),
        tmp_path,
        f"rank_{skin}",
        theme=skin,
        layout="rank",
    )
    page = THEME_PRESETS[skin].graph["bgcolor"]
    if skin == "dark":
        assert f'bgcolor="{page}"' in dot
    else:
        assert "bgcolor" not in dot
