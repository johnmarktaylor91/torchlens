"""The transformer lens draws one attention block in full and says which.

Folding repeated blocks collapses every block into a box, which hides the
attention wiring the lens exists to show (and GPT-2's per-block cache
outputs make the run fold illegal, leaving a 12-block strip). The lens
instead focuses the drawing on the first repeated block that holds
attention, names it in the caption, and leaves an explicit ``module=`` alone.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._visualization import lenses
from torchlens._visualization.lenses._attention import first_attention_block


class _Attention(nn.Module):
    """q/k/v projections into scaled_dot_product_attention."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)

    def forward(self, x: Any) -> Any:
        query, key, value = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)


class _Block(nn.Module):
    """Attention plus a one-layer MLP, residual on both."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()
        self.mlp = nn.Linear(8, 8)

    def forward(self, x: Any) -> Any:
        x = x + self.attn(x)
        return x + self.mlp(x)


class _Stack(nn.Module):
    """Three identical blocks, then a head."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(3)])
        self.head = nn.Linear(8, 2)

    def forward(self, x: Any) -> Any:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class _Single(nn.Module):
    """One attention module and a head: no repeated block."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()
        self.head = nn.Linear(8, 2)

    def forward(self, x: Any) -> Any:
        return self.head(self.attn(x))


def test_first_attention_block_finds_the_repeated_block() -> None:
    """The walk stops at the first block whose class repeats among its siblings."""

    log = tl.trace(_Stack(), torch.randn(1, 4, 8))
    try:
        assert first_attention_block(log) == ("blocks.0", "_Block", 3)
    finally:
        log.cleanup()


def test_transformer_lens_focuses_the_first_block_and_says_so() -> None:
    """The draw focuses on blocks.0 and the caption names it and the count."""

    log = tl.trace(_Stack(), torch.randn(1, 4, 8))
    try:
        resolution = lenses.resolve_lens(log, "transformer")
        assert resolution.draw_kwargs["module"] == "blocks.0"
        assert any("first of 3 _Block blocks" in line for line in resolution.disclosure)
    finally:
        log.cleanup()


def test_transformer_lens_keeps_an_explicit_module() -> None:
    """An explicit module= wins and the focus caption line is not added."""

    log = tl.trace(_Stack(), torch.randn(1, 4, 8))
    try:
        resolution = lenses.resolve_lens(log, "transformer", {"module": "blocks.2"})
        assert resolution.draw_kwargs["module"] == "blocks.2"
        assert not any("first of" in line for line in resolution.disclosure)
    finally:
        log.cleanup()


def test_transformer_lens_without_repeated_block_draws_everything() -> None:
    """Attention outside any repeated block leaves the whole graph in view."""

    log = tl.trace(_Single(), torch.randn(1, 4, 8))
    try:
        assert first_attention_block(log) is None
        resolution = lenses.resolve_lens(log, "transformer")
        assert resolution.draw_kwargs.get("module") is None
    finally:
        log.cleanup()
