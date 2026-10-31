"""The gradient and bundle pictures take the skin like ``draw`` (JMT, 29 Oct 2026).

Pins here:

- ``Trace.draw_backward(theme=...)`` and the functional
  ``torchlens._visualization.draw_backward(visualization=VisualizationOptions(theme=...))``
  paint the skin's page, text ink and typography; an unknown skin refuses;
- ``_visualization.show_bundle_graph(theme=..., show_legend=True)`` uses the skin's page and
  Helvetica face, ``draw``'s node shapes and fills, the key under the header,
  and member lists only on nodes and arrows not every member runs;
- the dark bundle diff caption takes the page's text ink.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _visualization
from torchlens._visualization import draw_backward
from torchlens._visualization.bundle_diff import bundle_diff
from torchlens._visualization.themes import THEME_PRESETS
from torchlens.options import VisualizationOptions


class _Block(nn.Module):
    """A weighted leaf module inside a nested module, then a plain add."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


def _backward_trace() -> tl.Trace:
    torch.manual_seed(0)
    trace = tl.trace(
        _Block(),
        torch.randn(1, 3),
        capture=tl.options.CaptureOptions(save_grads=True, backward_ready=True),
    )
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    return trace


def _bundle() -> tl.Bundle:
    torch.manual_seed(0)
    model, inputs = _Block(), torch.randn(1, 3)
    clean = tl.trace(model, inputs, capture=tl.options.CaptureOptions(intervention_ready=True))
    edited = clean.fork("intervened")
    edited.do(tl.units("linear_1_1:1", [(0, 0)]).resolve(edited), tl.zero_ablate())
    return tl.Bundle({"clean": clean, "intervened": edited}, baseline="clean")


def test_draw_backward_takes_the_skin(tmp_path: Path) -> None:
    """The dark skin's page and ink reach the gradient picture."""

    dark = THEME_PRESETS["dark"]
    dot = _backward_trace().draw_backward(
        path=str(tmp_path / "bwd_dark"),
        vis_save_only=True,
        file_format="svg",
        theme="dark",
    )
    assert f'bgcolor="{dark.graph["bgcolor"]}"' in dot
    assert f"<FONT COLOR='{dark.default_font}'>" in dot
    assert (tmp_path / "bwd_dark.svg").exists()


def test_draw_backward_default_skin_is_unchanged_in_kind(tmp_path: Path) -> None:
    """The default skin keeps a white page and the pinned face."""

    dot = _backward_trace().draw_backward(
        path=str(tmp_path / "bwd"), vis_save_only=True, file_format="svg"
    )
    assert "bgcolor=white" in dot
    assert "fontname=Helvetica" in dot


def test_draw_backward_refuses_an_unknown_skin(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _backward_trace().draw_backward(
            path=str(tmp_path / "bwd_bad"), vis_save_only=True, theme="sparkly"
        )


def test_functional_draw_backward_reads_the_grouped_theme(tmp_path: Path) -> None:
    options = VisualizationOptions(
        container_path=str(tmp_path / "bwd_fn"), save_only=True, file_format="svg", theme="dark"
    )
    dot = draw_backward(_backward_trace(), visualization=options)
    assert f'bgcolor="{THEME_PRESETS["dark"].graph["bgcolor"]}"' in dot


def test_bundle_picture_follows_draw(tmp_path: Path) -> None:
    """Skin page and face, draw's shapes and fills, the key, member lists only where they differ."""

    dot = _visualization.show_bundle_graph(
        _bundle(),
        path=str(tmp_path / "bundle_dark"),
        vis_save_only=True,
        file_format="svg",
        theme="dark",
        show_legend=True,
    )
    assert dot is not None
    assert f'bgcolor="{THEME_PRESETS["dark"].graph["bgcolor"]}"' in dot
    assert "fontname=Helvetica" in dot
    assert "Times" not in dot
    assert "[clean,intervened]" not in dot  # both members run every node and arrow
    assert 'label="clean,intervened"' not in dot
    assert 'ID="tl_key"' in dot
    for word in (">input<", ">output<", ">trainable weights<"):
        assert word in dot
    svg = (tmp_path / "bundle_dark.svg").read_text()
    assert "data:image/png;base64," in svg  # the key's swatches are embedded


def test_bundle_member_lists_appear_where_members_differ(tmp_path: Path) -> None:
    """A node only one member runs keeps its member list."""

    torch.manual_seed(0)
    small = tl.trace(nn.Sequential(nn.Linear(3, 3)), torch.randn(1, 3))
    large = tl.trace(nn.Sequential(nn.Linear(3, 3), nn.ReLU()), torch.randn(1, 3))
    dot = _visualization.show_bundle_graph(
        tl.Bundle({"small": small, "large": large}, baseline="small"),
        path=str(tmp_path / "bundle_diff_members"),
        vis_save_only=True,
        file_format="svg",
    )
    assert dot is not None
    assert "[large]" in dot


def test_dark_bundle_diff_caption_uses_page_ink(tmp_path: Path) -> None:
    dot = bundle_diff(
        _bundle(),
        left="clean",
        right="intervened",
        path=str(tmp_path / "diff_dark"),
        vis_save_only=True,
        file_format="svg",
        theme="dark",
    )
    assert f'fontcolor="{THEME_PRESETS["dark"].default_font}"' in str(dot)
