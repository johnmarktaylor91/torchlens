"""Regenerate the committed encoding/label-suppression reference renders.

Mirrors ``render_collapse_reference.py``: every image under
``docs/images/encoding`` referenced from ``docs/reference/encoding.md`` is
produced by THIS script, so the gallery is regenerable and reviewable
in-repo.

Usage::

    python scripts/render_encoding_reference.py

Sections:

* SUPPRESSION (L5 M4 merge evidence): before/after pairs for the checked
  suppression of redundant constructor-arg rows on the visual-pack trio
  (conv, mini transformer, attention node_style).
* CHANNELS (L5 M5): color_by timing render, size_by taper motif on a CNN
  and a transformer, stacking(a) lockstep column diagram, and the explicit
  annotation for a non-lockstep loop (whose auto request refuses).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.experimental import node_styles

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "encoding"


class SmallCNN(nn.Module):
    """Conv stack with a taper (the size_by motif model)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return self.head(self.pool(x).flatten(1))


class MiniTransformerBlock(nn.Module):
    """One pre-norm transformer block (attention + MLP)."""

    def __init__(self, dim: int = 16, heads: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attended
        return x + self.fc2(torch.relu(self.fc1(self.norm2(x))))


class LockstepDecoder(nn.Module):
    """Two-cell lockstep loop: the classic stacked-timestep diagram."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(8, 8)
        self.cell_a = nn.Linear(8, 8)
        self.cell_b = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for _ in range(3):
            x = torch.tanh(self.cell_a(x))
            x = torch.tanh(self.cell_b(x))
        return x


class ChainedLoops(nn.Module):
    """Sequential loops: the auto license REFUSES; explicit annotation works."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.relu(self.a(x))
        for _ in range(3):
            x = torch.relu(self.b(x))
        return x


def _render(trace: tl.Trace, name: str, **kwargs: object) -> None:
    trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(OUTPUT_DIR / name),
        **kwargs,
    )
    print(f"rendered {name}.svg")


def render_suppression_evidence() -> None:
    """M4 merge evidence: before/after pairs on the visual-pack trio."""

    torch.manual_seed(0)
    cnn = tl.trace(SmallCNN(), torch.randn(1, 3, 16, 16))
    try:
        _render(cnn, "suppression_conv_after")
        _render(cnn, "suppression_conv_before", show_redundant_args=True)
    finally:
        cnn.cleanup()
    transformer = tl.trace(MiniTransformerBlock(), torch.randn(1, 6, 16))
    try:
        _render(transformer, "suppression_transformer_after")
        _render(transformer, "suppression_transformer_before", show_redundant_args=True)
        _render(
            transformer,
            "suppression_attention_style_after",
            node_spec_fn=node_styles.attention_node_mode,
        )
        _render(
            transformer,
            "suppression_attention_style_before",
            node_spec_fn=node_styles.attention_node_mode,
            show_redundant_args=True,
        )
    finally:
        transformer.cleanup()


def render_channel_gallery() -> None:
    """M5 sample graphs: color_by / size_by / stack_by on reasonable networks."""

    torch.manual_seed(0)
    cnn = tl.trace(SmallCNN(), torch.randn(1, 3, 16, 16))
    try:
        _render(cnn, "color_by_time", color_by="time")
        _render(cnn, "size_by_dims_cnn", size_by="dims")
    finally:
        cnn.cleanup()
    transformer = tl.trace(MiniTransformerBlock(), torch.randn(1, 6, 16))
    try:
        _render(transformer, "size_by_dims_transformer", size_by="dims")
    finally:
        transformer.cleanup()
    decoder = tl.trace(LockstepDecoder(), torch.randn(2, 8))
    try:
        _render(decoder, "stack_by_auto_lockstep", stack_by=True, direction="leftright")
    finally:
        decoder.cleanup()
    chained = tl.trace(ChainedLoops(), torch.randn(2, 8))
    try:
        try:
            _render(chained, "stack_by_auto_refused", stack_by=True)
        # The reference generator records any renderer refusal for inspection.
        except Exception as error:  # noqa: BLE001
            print(f"stack_by=True on chained loops refused as designed: {error}")
        _render(
            chained,
            "stack_by_explicit_chained",
            stack_by="pass_index",
            direction="leftright",
        )
    finally:
        chained.cleanup()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_suppression_evidence()
    render_channel_gallery()
    # Sanity: every emitted svg is non-trivial.
    for svg in sorted(OUTPUT_DIR.glob("*.svg")):
        assert svg.stat().st_size > 1024, svg
    print(f"done -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
