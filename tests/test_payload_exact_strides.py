"""Saved payloads keep the exact strides of a dense source tensor.

ATen picks kernels from strides, size-1 dims included: xcit's LPI conv input,
a ``(1, C, H, W)`` view with batch stride ``C``, is NCHW to
``suggest_memory_format`` but ``is_contiguous(channels_last)``. TorchLens used
to copy it as channels_last (batch stride ``C*H*W``), so the isolated replay
ran the channels-last conv kernel and missed the saved output by 4.8e-7
(``xcit_*`` forward_replay failures, rung-2 menagerie triage). Copies now use
``preserve_format``: exact strides for every dense source. A non-dense view
(a ``split`` slice) is still copied compactly; that residual is pinned below.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.utils.tensor_utils import safe_copy


class _LpiLike(nn.Module):
    """Token tensor ``(B, N, C)`` permuted to a ``(B, C, H, W)`` view, then a depthwise conv."""

    def __init__(self, channels: int = 16, side: int = 6) -> None:
        super().__init__()
        self.side = side
        self.proj = nn.Linear(channels, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(tokens)
        batch, _, channels = tokens.shape
        grid = tokens.permute(0, 2, 1).reshape(batch, channels, self.side, self.side)
        return self.conv(grid)


def _conv_input_payload(trace: tl.Trace) -> torch.Tensor:
    conv = next(op for op in trace.layer_list if op.func_name == "conv2d")
    return conv.saved_args[0]


def test_safe_copy_keeps_size1_dim_strides_of_a_dense_view() -> None:
    """The NCHW-to-ATen view keeps batch stride ``C`` (it used to become ``C*H*W``)."""

    grid = torch.randn(1, 36, 16).permute(0, 2, 1).reshape(1, 16, 6, 6)
    assert grid.stride() == (16, 1, 96, 16)
    copy = safe_copy(grid, detach_tensor=True)
    assert copy.stride() == grid.stride()
    assert torch.equal(copy, grid)


def test_captured_conv_input_has_the_live_strides_and_validates() -> None:
    """The saved conv input keeps the live view's strides ``(C, 1, W*C, C)``."""

    torch.manual_seed(0)
    model = _LpiLike().eval()
    tokens = torch.randn(1, 36, 16)
    trace = tl.trace(
        model, tokens, capture=CaptureOptions(layers_to_save="all", save_arg_values=True)
    )
    assert _conv_input_payload(trace).stride() == (16, 1, 96, 16)
    assert tl.validation.validate_forward_pass(model, tokens) is True


def test_standard_contiguous_single_channel_stays_standard() -> None:
    """An NCHW ``C=1`` tensor keeps standard strides (the BC-ResNet view case)."""

    mono = torch.randn(2, 1, 5, 7)
    copy = safe_copy(mono, detach_tensor=True)
    assert copy.stride() == mono.stride() == (35, 35, 7, 1)


def test_channels_last_source_stays_channels_last() -> None:
    """A genuinely channels-last activation is copied channels-last."""

    image = torch.randn(2, 3, 5, 7).contiguous(memory_format=torch.channels_last)
    copy = safe_copy(image, detach_tensor=True)
    assert copy.stride() == image.stride()
    assert copy.is_contiguous(memory_format=torch.channels_last)


def test_non_dense_view_is_still_copied_compactly() -> None:
    """Residual: a ``split`` slice has no dense layout, so its copy is compact.

    Exact replay of such a view would need its strides recorded and rebuilt
    with ``empty_strided`` (mambaout's GELU on a ``split_with_sizes`` slice
    misses its saved output by 7e-7 this way); until then this pins the
    current behavior so a change is deliberate.
    """

    base = torch.randn(1, 4, 4, 32)
    left, _ = base.split([16, 16], dim=-1)
    assert left.stride() == (512, 128, 32, 1)
    copy = safe_copy(left, detach_tensor=True)
    assert copy.stride() == (256, 64, 16, 1)
    assert torch.equal(copy, left)
