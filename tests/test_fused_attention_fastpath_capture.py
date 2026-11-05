"""Capture, replay and ground truth run one attention program (wave-3 stress F4).

``nn.MultiheadAttention`` takes its fused ``_native_multi_head_attention`` fast path
in eval mode only while no input or parameter requires a gradient under grad mode.
The fused kernel is one opaque op, and whether it runs depends on grad mode and on
the input's ``requires_grad``. Capture holds torch's public fast-path switch off for
its forward, validation holds it for the ground truth and the fast live run for its
native forward, so every one of them runs the unfused program, and the trace records
the hold as a capture advisory. Capture never forces ``requires_grad`` on the frozen
weights, so the replay and the ground truth are the same program kernel for kernel
and agree bit for bit, in float64 as in float32.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils._torch_compat import (
    get_mha_fastpath_switch_support,
)

WIDTH = 8
BATCH = 3


class FastPathAttention(nn.Module):
    """``nn.MultiheadAttention`` set up for its inference fast path (eval, frozen, batch-first)."""

    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(WIDTH, 2, batch_first=True)
        self.attention.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attention(x, x, x, need_weights=False)
        return out


def _model_and_input() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return FastPathAttention().eval(), torch.randn(BATCH, 3, WIDTH)


def _needs_switch() -> None:
    if not get_mha_fastpath_switch_support():
        pytest.skip("torch build without the public torch.backends.mha fast-path switch")


def test_capture_keeps_the_frozen_attention_weights_frozen() -> None:
    """The frozen weights stay frozen inside the captured forward and after it."""

    model, x = _model_and_input()
    seen: list[bool] = []
    handle = model.attention.register_forward_pre_hook(
        lambda module, args: seen.extend(p.requires_grad for p in module.parameters())
    )
    try:
        tl.trace(model, [x])
    finally:
        handle.remove()
    assert seen and not any(seen)
    assert not any(p.requires_grad for p in model.parameters())
